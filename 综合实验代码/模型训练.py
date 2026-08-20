# -*- coding: utf-8 -*-
"""
ALS 协同过滤推荐模型训练
读取 Excel → 构建评分矩阵 → ALS训练 → 提取因子矩阵缓存到 Redis

缓存到 Redis 的数据:
  factor:user:{personid}  → 用户隐因子向量 (JSON数组, rank维)
  factor:item:{courseid}  → 物品隐因子向量 (JSON数组, rank维)
  rec:hot_courses         → 课程热度排行 (冷启动备用)
  rec:course_names        → 课程ID→名称映射
  model:rank              → 隐因子维度
"""

import builtins
import json
import numpy as np
import pandas as pd
import redis
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS

# ---- 运行环境 ----
REDIS_HOST = "node1"
REDIS_PORT = 6379
REDIS_DB = 0

FILE_CLAZZ = "file:///root/online_education/data/b1_t_stat_clazz.xlsx"
FILE_COURSE = "file:///root/online_education/data/b1_t_stat_course.xlsx"
FILE_EXAM = "file:///root/online_education/data/b1_t_stat_exam_answer.xlsx"

ALS_MAX_ITER = 10
ALS_REG_PARAM = 0.05
ALS_RANK = 20


def main():
    print("=" * 50)
    print("  ALS 推荐模型训练 → Redis 缓存因子向量")
    print("=" * 50)

    spark = SparkSession.builder \
        .appName("OnlineEdu-ModelTrain") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # ---- 1. 加载数据 ----
    print("\n[1] 加载数据...")
    course_df = spark.createDataFrame(pd.read_excel(FILE_COURSE)) \
        .select("courseid", "name").withColumnRenamed("name", "course_name")

    clazz_df = spark.createDataFrame(pd.read_excel(FILE_CLAZZ)) \
        .select("clazzid", "courseid")

    exam_df = spark.createDataFrame(pd.read_excel(FILE_EXAM)) \
        .select("personid", "clazzid", "score") \
        .filter(col("score").isNotNull() & (col("score") >= 0) & (col("score") <= 100)) \
        .withColumn("personid", col("personid").cast("long")) \
        .withColumn("clazzid", col("clazzid").cast("long")) \
        .withColumn("score", col("score").cast("double"))

    # 补全 courseid
    exam_with_course = exam_df.join(clazz_df, "clazzid", "inner") \
        .select("personid", "courseid", "score")

    # 构建评分矩阵
    rating_df = exam_with_course.groupBy("personid", "courseid").agg(
        round(avg("score"), 1).alias("avg_score"),
        count("*").alias("exam_count")
    ).filter(col("exam_count") >= 1).cache()

    # 归一化
    score_stats = rating_df.agg(min("avg_score"), max("avg_score")).collect()[0]
    score_min, score_max = score_stats[0], score_stats[1]
    score_range = score_max - score_min if score_max != score_min else 1
    rating_df = rating_df.withColumn("norm_score",
        (col("avg_score") - lit(score_min)) / lit(score_range))

    student_count = rating_df.select("personid").distinct().count()
    course_count = rating_df.select("courseid").distinct().count()
    print(f"  评分矩阵: {rating_df.count()} 条 | 学生:{student_count} 课程:{course_count}")

    # ---- 2. ALS 训练 ----
    print("\n[2] 训练 ALS...")
    train_data, test_data = rating_df.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        maxIter=ALS_MAX_ITER, regParam=ALS_REG_PARAM, rank=ALS_RANK,
        userCol="personid", itemCol="courseid", ratingCol="norm_score",
        coldStartStrategy="drop", nonnegative=True,
    )
    model = als.fit(train_data)

    # 评估
    predictions = model.transform(test_data).na.drop()
    pred_df = predictions.select("prediction", "norm_score").collect()
    rmse = np.sqrt(np.mean([(r[0] - r[1]) ** 2 for r in pred_df]))
    print(f"  RMSE (归一化): {builtins.round(rmse, 4)}")

    # ---- 3. 提取因子矩阵 → 缓存到 Redis ----
    print("\n[3] 缓存因子向量到 Redis...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    # 用户因子: [personid, [f0, f1, ..., f19]]
    user_factors = model.userFactors.collect()
    user_count = 0
    for row in user_factors:
        personid = int(row["id"])
        vector = [float(v) for v in row["features"]]
        redis_client.setex(f"factor:user:{personid}", 86400,
                           json.dumps(vector, ensure_ascii=False))
        user_count += 1
    print(f"  用户因子: {user_count} 个 (TTL=24h)")

    # 物品因子: [courseid, [f0, f1, ..., f19]]
    item_factors = model.itemFactors.collect()
    item_count = 0
    item_ids = []
    for row in item_factors:
        courseid = int(row["id"])
        vector = [float(v) for v in row["features"]]
        redis_client.setex(f"factor:item:{courseid}", 86400,
                           json.dumps(vector, ensure_ascii=False))
        item_ids.append(courseid)
        item_count += 1
    print(f"  物品因子: {item_count} 个 (TTL=24h)")

    # 模型参数
    redis_client.setex("model:rank", 86400, str(ALS_RANK))
    redis_client.setex("model:rmse", 86400, str(builtins.round(rmse, 4)))

    # 课程名称映射
    course_name_map = {}
    for row in course_df.collect():
        course_name_map[int(row["courseid"])] = row["course_name"]
    redis_client.setex("rec:course_names", 86400,
                       json.dumps(course_name_map, ensure_ascii=False))
    print(f"  课程名称映射: {len(course_name_map)} 门")

    # 课程热度（冷启动用）
    hot_list = []
    hot_df = rating_df.groupBy("courseid").agg(
        count("*").alias("student_count"),
        round(avg("avg_score"), 1).alias("avg_score_val")
    ).orderBy(desc("student_count")).collect()
    for row in hot_df:
        hot_list.append({
            "courseid": int(row["courseid"]),
            "count": int(row["student_count"]),
            "avg": float(row["avg_score_val"])
        })
    redis_client.setex("rec:hot_courses", 7200,
                       json.dumps(hot_list, ensure_ascii=False))
    print(f"  课程热度: {len(hot_list)} 门 (TTL=2h)")

    redis_client.close()

    # ---- 4. 保存完整模型到磁盘（备用） ----
    model.write().overwrite().save("file:///root/online_education/models/als_model")
    print(f"\n[4] 完整模型已保存到 models/als_model")

    print("\n[完成] Redis 缓存：用户因子 {user_count} + 物品因子 {item_count} = "
          f"{user_count + item_count} 个向量")
    spark.stop()


if __name__ == "__main__":
    main()
