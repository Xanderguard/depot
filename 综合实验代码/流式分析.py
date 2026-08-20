# -*- coding: utf-8 -*-
"""
在线教育实时分析系统
数据流: Kafka 考试数据 → 关联离线维度表 → 清洗 → 6条聚合流 → MySQL 排名表
参照: Java 版 ExamAnswerJoinAnalysis + HBaseStatsWriter
"""

import pandas as pd
import pymysql
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# ---- 运行环境: Hadoop 3.2.1 / Spark 3.4.2 / MySQL 5.7 / Python 3.8 ----
KAFKA_HOST = "node1:9092"
KAFKA_TOPIC = "edu_exam_data"
MYSQL_HOST = "node1"
MYSQL_USER = "root"
MYSQL_PASS = "123456"
MYSQL_DB = "online_edu"
CHECKPOINT_DIR = "file:///root/online_education/checkpoint"
FILE_PERSON = "file:///root/online_education/data/b1_t_stat_person.xlsx"
FILE_COURSE = "file:///root/online_education/data/b1_t_stat_course.xlsx"
FILE_CLAZZ = "file:///root/online_education/data/b1_t_stat_clazz.xlsx"
PASS_LINE = 60


# ================================================================
# MySQL 写入
# ================================================================
def _get_connection():
    return pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER,
                           password=MYSQL_PASS, database=MYSQL_DB, charset="utf8")

def _write_to_mysql(table_name, column_list, row_list):
    """每批次 TRUNCATE + INSERT 全量替换"""
    connection = _get_connection()
    cursor = connection.cursor()
    cursor.execute(f"TRUNCATE TABLE {table_name}")
    placeholders = ",".join(["%s"] * len(column_list))
    for row in row_list:
        cursor.execute(
            f"INSERT INTO {table_name} ({','.join(column_list)}) VALUES ({placeholders})", row)
    connection.commit()
    cursor.close()
    connection.close()


# ================================================================
# 6 个排名写入函数（从聚合 DataFrame 取 top10 / 分布，写 MySQL）
# ================================================================
def _write_course_rank(agg_df):
    """课程及格率 Top10（排除全0分脏数据）"""
    top10 = agg_df.filter((col("exam_count") >= 30) & (col("total_score") > 0)) \
        .withColumn("pass_rate_pct", round(col("pass_count") * 100.0 / col("exam_count"), 1)) \
        .fillna(0) \
        .withColumn("rank_order", row_number().over(Window.orderBy(desc("pass_rate_pct")))) \
        .filter(col("rank_order") <= 10) \
        .select("rank_order", "course_name", "pass_rate_pct", "exam_count").collect()
    _write_to_mysql("course_pass_rank",
        ["rank_order", "course_name", "pass_rate", "exam_count"],
        [(row.rank_order, row.course_name, float(row.pass_rate_pct), int(row.exam_count))
         for row in top10])
    print(f"[课程Top10] {len(top10)}条", end="  ")


def _write_class_rank(agg_df):
    """班级平均分 Top10（排除全0分脏数据）"""
    top10 = agg_df.filter((col("exam_count") >= 30) & (col("total_score") > 0)) \
        .withColumn("avg_score_val", round(col("total_score") / col("exam_count"), 1)) \
        .fillna(0) \
        .withColumn("rank_order", row_number().over(Window.orderBy(desc("avg_score_val")))) \
        .filter(col("rank_order") <= 10) \
        .select("rank_order", "class_name", "avg_score_val", "exam_count").collect()
    _write_to_mysql("class_score_rank",
        ["rank_order", "class_name", "avg_score", "exam_count"],
        [(row.rank_order, row.class_name, float(row.avg_score_val), int(row.exam_count))
         for row in top10])
    print(f"[班级Top10] {len(top10)}条", end="  ")


def _write_teacher_rank(agg_df):
    """教师综合分 Top10 (综合分 = 均分*0.5 + 及格率*0.5)"""
    top10 = agg_df.filter((col("exam_count") >= 20) & (col("total_score") > 0)) \
        .withColumn("avg_score_val", round(col("total_score") / col("exam_count"), 1)) \
        .withColumn("pass_rate_val", round(col("pass_count") * 100.0 / col("exam_count"), 1)) \
        .fillna(0) \
        .withColumn("composite", round(col("avg_score_val") * 0.5 + col("pass_rate_val") * 0.5, 1)) \
        .withColumn("rank_order", row_number().over(Window.orderBy(desc("composite")))) \
        .filter(col("rank_order") <= 10) \
        .select("rank_order", "teacher_name", "avg_score_val", "pass_rate_val", "composite", "exam_count").collect()
    _write_to_mysql("teacher_score_rank",
        ["rank_order", "teacher_name", "avg_score", "pass_rate", "score", "exam_count"],
        [(row.rank_order, row.teacher_name, float(row.avg_score_val),
          float(row.pass_rate_val), float(row.composite), int(row.exam_count)) for row in top10])
    print(f"[教师Top10] {len(top10)}条")


def _write_exam_dist(agg_df):
    """全部课程难度四档分布（排除全0分脏数据）"""
    base = agg_df.filter((col("exam_count") >= 30) & (col("total_score") > 0)) \
        .withColumn("avg_score_val", round(col("total_score") / col("exam_count"), 1)) \
        .withColumn("pass_rate_val", round(col("pass_count") * 100.0 / col("exam_count"), 1)) \
        .fillna(0)

    distribution = base.withColumn("difficulty_level",
        when(col("avg_score_val") < 55, "困难")
        .when(col("avg_score_val") < 65, "较难")
        .when(col("avg_score_val") < 75, "中等")
        .otherwise("较易")
    ).groupBy("difficulty_level").agg(
        count("*").alias("course_count"),
        round(avg("pass_rate_val"), 1).alias("avg_pass_rate")
    ).orderBy(
        when(col("difficulty_level") == "困难", 1)
        .when(col("difficulty_level") == "较难", 2)
        .when(col("difficulty_level") == "中等", 3)
        .otherwise(4)
    ).collect()
    _write_to_mysql("exam_difficulty_dist",
        ["difficulty", "course_count", "avg_pass_rate"],
        [(row.difficulty_level, int(row.course_count), float(row.avg_pass_rate))
         for row in distribution])
    print(f"[难度分布] {len(distribution)}档", end="  ")


def _write_semester_summary(agg_df):
    """各学期整体概览"""
    summary = agg_df.orderBy("semester_str").collect()
    _write_to_mysql("semester_summary",
        ["semester", "avg_score", "course_count", "total_exam"],
        [(row.semester_str, float(row.avg_score), int(row.course_count), int(row.total_exam))
         for row in summary])
    print(f"[学期汇总] {len(summary)}条")


# ================================================================
# 主流程
# ================================================================
def main():
    # 1. 创建 SparkSession
    spark = SparkSession.builder \
        .appName("ExamAnswerJoinAnalysis") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # 2. 加载离线维度表（批读取）
    print("加载离线维度表...")
    course_pd = pd.read_excel(FILE_COURSE)
    course_df = spark.createDataFrame(course_pd) \
        .select("courseid", "name", "personid") \
        .withColumnRenamed("name", "course_name") \
        .withColumnRenamed("personid", "teacher_id").cache()

    clazz_pd = pd.read_excel(FILE_CLAZZ)
    clazz_df = spark.createDataFrame(clazz_pd) \
        .select("clazzid", "courseid", "name", "student_count", "semester") \
        .withColumnRenamed("name", "clazz_name").cache()

    person_pd = pd.read_excel(FILE_PERSON)
    person_df = spark.createDataFrame(person_pd) \
        .select("personid", "user_name").dropDuplicates(["personid"])

    student_df = person_df \
        .withColumnRenamed("personid", "student_personid") \
        .withColumnRenamed("user_name", "student_name").cache()

    teacher_df = person_df \
        .withColumnRenamed("personid", "teacher_personid") \
        .withColumnRenamed("user_name", "teacher_name").cache()

    print(f"  课程:{course_df.count()} 班级:{clazz_df.count()} 人员:{person_df.count()}")

    # 3. 从 Kafka 读取流数据
    exam_schema = StructType([
        StructField("id", StringType(), True),
        StructField("insert_time", StringType(), True),
        StructField("create_time", StringType(), True),
        StructField("last_modify_time", StringType(), True),
        StructField("personid", StringType(), True),
        StructField("courseid", StringType(), True),
        StructField("exam_id", StringType(), True),
        StructField("status", StringType(), True),
        StructField("answer_time", StringType(), True),
        StructField("score", StringType(), True),
        StructField("piyue_person_id", StringType(), True),
        StructField("ip", StringType(), True),
        StructField("piyue_time", StringType(), True),
        StructField("isDeleted", StringType(), True),
        StructField("fid", StringType(), True),
        StructField("clazzid", StringType(), True),
        StructField("answerid", StringType(), True),
    ])

    print("连接 Kafka...")
    kafka_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_HOST) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 100000) \
        .option("failOnDataLoss", "false").load()

    # 解析 JSON → 删掉全空列 → 类型转换
    answer_df = kafka_stream \
        .selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), exam_schema).alias("data")) \
        .select("data.*") \
        .drop("courseid", "status", "piyue_person_id", "ip", "piyue_time", "isDeleted")

    answer_df = answer_df \
        .withColumn("personid", col("personid").cast("long")) \
        .withColumn("clazzid", col("clazzid").cast("long")) \
        .withColumn("score", col("score").cast("double"))

    # 4. 流批 Join: 答题→班级→课程→教师→学生
    print("构建流批 Join...")
    step1 = answer_df.join(clazz_df, answer_df["clazzid"] == clazz_df["clazzid"], "inner")
    step2 = step1.join(course_df, clazz_df["courseid"] == course_df["courseid"], "inner")
    step3 = step2.join(teacher_df, course_df["teacher_id"] == teacher_df["teacher_personid"], "inner")
    joined_df = step3.join(student_df, answer_df["personid"] == student_df["student_personid"], "inner")

    # 5. 清洗 → 统一输出列
    print("数据清洗...")
    clean_df = joined_df.select(
        col("course_name"),
        col("clazz_name").alias("class_name"),
        col("teacher_name"),
        col("student_name"),
        col("semester"),
        answer_df["score"].alias("score"),
    ).filter(col("score").isNotNull() & (col("score") >= 0) & (col("score") <= 100))

    # 6. 控制台实时打印清洗后的数据（5条样本 + 行数）
    clean_df.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, bid: print(
            f"\n{'='*50}\n[样本-批次{bid}] {df.count()}条\n{'='*50}",
            df.show(5, truncate=False))) \
        .trigger(processingTime="10 seconds") \
        .option("checkpointLocation", CHECKPOINT_DIR + "/console") \
        .start()

    # 7. 6条聚合流 — outputMode("complete") 实现跨批次自动累加，写 MySQL
    trigger_interval = "10 seconds"

    # 1) 课程及格率 Top10
    clean_df.groupBy("course_name").agg(
        count("*").alias("exam_count"),
        sum("score").alias("total_score"),
        sum(when(col("score") >= PASS_LINE, 1)).alias("pass_count")
    ).writeStream.outputMode("complete") \
        .foreachBatch(lambda df, bid: _write_course_rank(df)) \
        .trigger(processingTime=trigger_interval) \
        .option("checkpointLocation", CHECKPOINT_DIR + "/agg_course").start()

    # 2) 班级平均分 Top10
    clean_df.groupBy("class_name").agg(
        count("*").alias("exam_count"),
        sum("score").alias("total_score"),
        sum(when(col("score") >= PASS_LINE, 1)).alias("pass_count")
    ).writeStream.outputMode("complete") \
        .foreachBatch(lambda df, bid: _write_class_rank(df)) \
        .trigger(processingTime=trigger_interval) \
        .option("checkpointLocation", CHECKPOINT_DIR + "/agg_class").start()

    # 3) 教师综合分 Top10
    clean_df.groupBy("teacher_name").agg(
        count("*").alias("exam_count"),
        sum("score").alias("total_score"),
        sum(when(col("score") >= PASS_LINE, 1)).alias("pass_count")
    ).writeStream.outputMode("complete") \
        .foreachBatch(lambda df, bid: _write_teacher_rank(df)) \
        .trigger(processingTime=trigger_interval) \
        .option("checkpointLocation", CHECKPOINT_DIR + "/agg_teacher").start()

    # 4) 课程难度四档分布
    clean_df.groupBy("course_name").agg(
        count("*").alias("exam_count"),
        sum("score").alias("total_score"),
        max("score").alias("max_score"),
        sum(when(col("score") >= PASS_LINE, 1)).alias("pass_count")
    ).writeStream.outputMode("complete") \
        .foreachBatch(lambda df, bid: _write_exam_dist(df)) \
        .trigger(processingTime=trigger_interval) \
        .option("checkpointLocation", CHECKPOINT_DIR + "/agg_exam").start()

    # 5) 学期汇总
    clean_df.withColumn("semester_str", col("semester").cast("string")) \
        .groupBy("semester_str").agg(
            round(avg("score"), 1).alias("avg_score"),
            approx_count_distinct("course_name").alias("course_count"),
            count("*").alias("total_exam")
        ).writeStream.outputMode("complete") \
        .foreachBatch(lambda df, bid: _write_semester_summary(df)) \
        .trigger(processingTime=trigger_interval) \
        .option("checkpointLocation", CHECKPOINT_DIR + "/agg_semester_summary").start()

    print("6条聚合流已启动")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
