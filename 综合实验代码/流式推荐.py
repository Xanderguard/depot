# -*- coding: utf-8 -*-
"""
实时推荐引擎 — Redis 因子向量 + numpy 矩阵运算
数据流: Kafka → 提取学生ID → Redis取因子向量 → numpy点积算分 → 写MySQL

Redis 中预缓存的数据 (由 model_train.py 写入):
  factor:user:{personid}  → 用户隐因子向量 [f0, f1, ...]
  factor:item:{courseid}  → 物品隐因子向量 [f0, f1, ...]
  rec:course_names        → {courseid: course_name}
  rec:hot_courses         → [{courseid, count, avg}, ...]  冷启动备用
  model:rank              → 向量维度
"""

import builtins
import json
import numpy as np
import pandas as pd
import pymysql
import redis
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# ---- 运行环境 ----
KAFKA_HOST = "node1:9092"
KAFKA_TOPIC = "edu_exam_data"
MYSQL_HOST = "node1"
MYSQL_USER = "root"
MYSQL_PASS = "123456"
MYSQL_DB = "online_edu"
REDIS_HOST = "node1"
REDIS_PORT = 6379
REDIS_DB = 0
CHECKPOINT_DIR = "file:///root/online_education/checkpoint"
FILE_PERSON = "file:///root/online_education/data/b1_t_stat_person.xlsx"
RECOMMEND_NUM = 5


def _get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def _get_connection():
    return pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER,
                           password=MYSQL_PASS, database=MYSQL_DB, charset="utf8")


def load_support_data():
    """从 Redis 和 Excel 加载推荐所需的基础数据"""
    redis_client = _get_redis()

    # 物品因子矩阵: {courseid: [f0, f1, ...]}
    item_vectors = {}
    course_name_map = {}
    try:
        name_data = redis_client.get("rec:course_names")
        if name_data:
            course_name_map = json.loads(name_data)
    except:
        pass

    rank = int(redis_client.get("model:rank") or 20)

    # 扫描所有 factor:item:* 键
    keys = redis_client.keys("factor:item:*")
    for key in keys:
        courseid = int(key.decode().split(":")[-1])
        try:
            vector = json.loads(redis_client.get(key))
            item_vectors[courseid] = np.array(vector, dtype=np.float32)
        except:
            pass
    item_ids = sorted(item_vectors.keys())

    # 课程热度（冷启动备用）
    hot_courses = []
    try:
        hot_data = redis_client.get("rec:hot_courses")
        if hot_data:
            hot_courses = json.loads(hot_data)
    except:
        pass

    # 学生名称映射
    person_pd = pd.read_excel(FILE_PERSON)
    student_map = {}
    for _, row in person_pd.iterrows():
        if row["role"] == 2:
            student_map[int(row["personid"])] = row["user_name"] or row["login_name"]

    redis_client.close()

    # 构建物品因子矩阵: (n_items, rank)
    item_matrix = np.array([item_vectors[cid] for cid in item_ids], dtype=np.float32)
    course_name_list = [course_name_map.get(str(cid), f"课程{cid}") for cid in item_ids]

    print(f"  物品因子: {len(item_vectors)} 个 ({item_matrix.shape[1]}维)")
    print(f"  课程名称: {len(course_name_map)} 门 | 学生映射: {len(student_map)} 人 | 冷启动: {len(hot_courses)} 门")
    return item_ids, item_matrix, course_name_list, student_map, hot_courses, rank


# 注意: Kafka 消息里所有字段都是字符串格式，先 String 再 cast
EXAM_SCHEMA = StructType([
    StructField("personid", StringType()),
    StructField("clazzid", StringType()),
    StructField("score", StringType()),
])


def process_batch(batch_df, batch_id, item_ids, item_matrix, course_names,
                  student_map, hot_courses, rank):
    """
    每批次: 提取学生 → Redis取用户向量 → numpy矩阵乘法 → 取Top-N → 写MySQL
    """
    if batch_df.rdd.isEmpty():
        return

    unique_students = batch_df.select("personid").distinct() \
        .filter(col("personid").isNotNull()) \
        .withColumnRenamed("personid", "person_id").collect()

    print(f"\n  [推荐批次 {batch_id}] {len(unique_students)} 个学生")

    redis_client = _get_redis()
    connection = _get_connection()
    cursor = connection.cursor()
    insert_count = 0
    hot_count = 0

    for row in unique_students:
        if row["person_id"] is None:
            continue
        personid = int(row["person_id"])
        student_name = student_map.get(personid, f"学生{personid}")

        # 从 Redis 取用户因子向量
        user_data = redis_client.get(f"factor:user:{personid}")
        if user_data is not None:
            # 有因子 → numpy 批量算分
            user_vec = np.array(json.loads(user_data), dtype=np.float32)
            scores = np.dot(user_vec, item_matrix.T)  # (rank,) · (rank, n_items) → (n_items,)
            top_indices = np.argsort(scores)[::-1][:RECOMMEND_NUM]

            for rank_order, idx in enumerate(top_indices, 1):
                courseid = item_ids[idx]
                score = float(scores[idx])
                confidence = builtins.round(0.95 - (rank_order - 1) * 0.05, 4)
                course_name = course_names[idx]
                cursor.execute(
                    "INSERT INTO recommend_tracking "
                    "(student_name, course_name, predict_score, confidence, rank_order) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (student_name, course_name, builtins.round(score, 1), confidence, rank_order))
                insert_count += 1
        else:
            # 冷启动：用课程热度
            for rank_order, course in enumerate(hot_courses[:RECOMMEND_NUM], 1):
                course_name = course.get("course_name", f"课程{course['courseid']}")
                cursor.execute(
                    "INSERT INTO recommend_tracking "
                    "(student_name, course_name, predict_score, confidence, rank_order) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (student_name, course_name,
                     builtins.round(float(course.get("avg", 60)), 1),
                     builtins.round(0.5 - (rank_order - 1) * 0.05, 4), rank_order))
                hot_count += 1
                insert_count += 1

    connection.commit()
    cursor.close()
    connection.close()
    redis_client.close()

    print(f"  [推荐批次 {batch_id}] 模型预测 {insert_count - hot_count} 条 "
          f"+ 冷启动 {hot_count} 条 → 共 {insert_count} 条")


def main():
    print("=" * 60)
    print("  实时推荐 — Redis因子向量 + numpy矩阵运算")
    print("=" * 60)

    # 加载基础数据
    item_ids, item_matrix, course_names, student_map, hot_courses, rank = load_support_data()

    # 补充冷启动课程的名称
    course_name_map = {}
    try:
        r = _get_redis()
        name_data = r.get("rec:course_names")
        if name_data:
            course_name_map = json.loads(name_data)
        r.close()
    except:
        pass
    for course in hot_courses:
        course["course_name"] = course_name_map.get(str(course["courseid"]), f"课程{course['courseid']}")

    spark = SparkSession.builder \
        .appName("OnlineEdu-StreamRecommend") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    kafka_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_HOST) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 50000) \
        .option("failOnDataLoss", "false").load()

    parsed = kafka_stream.select(
        from_json(col("value").cast("string"), EXAM_SCHEMA).alias("data")
    ).select("data.*") \
     .withColumn("personid", col("personid").cast("long")) \
     .withColumn("score", col("score").cast("double"))

    query = parsed.writeStream \
        .foreachBatch(lambda df, bid: process_batch(
            df, bid, item_ids, item_matrix, course_names,
            student_map, hot_courses, rank)) \
        .trigger(processingTime="10 seconds") \
        .option("checkpointLocation", CHECKPOINT_DIR + "/recommend") \
        .start()

    print("\n  实时推荐已启动 (numpy矩阵运算)...")
    query.awaitTermination()


if __name__ == "__main__":
    main()
