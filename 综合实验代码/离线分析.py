# -*- coding: utf-8 -*-
"""
离线分析 — 对 MySQL 推荐结果进行分析 (需求5)
读 recommend_tracking → 写 3 张 FineBI 友好的分析表:
  recommend_summary      — 推荐总览 (一行: 覆盖率/置信度/预测分)
  recommend_course_top10 — 热门推荐课程排行
  recommend_rank_dist    — 推荐排名分布
"""

import builtins
import pandas as pd
import pymysql
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

MYSQL_URL = "jdbc:mysql://node1:3306/online_edu?useSSL=false&characterEncoding=utf8"
MYSQL_HOST = "node1"
MYSQL_USER = "root"
MYSQL_PASS = "123456"
MYSQL_DB = "online_edu"
FILE_PERSON = "file:///root/online_education/data/b1_t_stat_person.xlsx"


def _get_connection():
    return pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER,
                           password=MYSQL_PASS, database=MYSQL_DB, charset="utf8")


def main():
    spark = SparkSession.builder \
        .appName("OfflineAnalysis") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    jdbc_opts = {
        "url": MYSQL_URL,
        "user": MYSQL_USER,
        "password": MYSQL_PASS,
        "driver": "com.mysql.jdbc.Driver",
    }

    print("=" * 60)
    print("  推荐效果离线分析")
    print("=" * 60)

    rec_df = spark.read.jdbc(
        url=jdbc_opts["url"], table="recommend_tracking", properties=jdbc_opts)
    total_records = rec_df.count()

    if total_records == 0:
        print("  推荐表为空，请先运行 streaming_recommend.py")
        spark.stop()
        return

    recommended_students = rec_df.select("student_name").distinct().count()
    person_pd = pd.read_excel(FILE_PERSON)
    total_students = len(person_pd[person_pd["role"] == 2])
    coverage = builtins.round(recommended_students * 100.0 / total_students, 1)

    # ---- 表1: 推荐总览 (一行数据，FineBI 直接拖指标卡) ----
    print("\n[1] 推荐总览")
    stats = rec_df.agg(
        round(avg("predict_score"), 1).alias("avg_predict_score"),
        round(max("predict_score"), 1).alias("max_predict_score"),
        round(min("predict_score"), 1).alias("min_predict_score"),
        round(avg("confidence"), 4).alias("avg_confidence"),
        round(max("confidence"), 4).alias("max_confidence"),
        round(min("confidence"), 4).alias("min_confidence"),
    ).collect()[0]

    db = _get_connection()
    cur = db.cursor()
    cur.execute("TRUNCATE TABLE recommend_summary")
    cur.execute(
        "INSERT INTO recommend_summary "
        "(total_records, recommended_students, total_students, coverage_rate, "
        "avg_predict_score, max_predict_score, min_predict_score, "
        "avg_confidence, max_confidence, min_confidence) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (int(total_records), int(recommended_students), int(total_students),
         float(coverage),
         float(stats["avg_predict_score"]), float(stats["max_predict_score"]),
         float(stats["min_predict_score"]),
         float(stats["avg_confidence"]), float(stats["max_confidence"]),
         float(stats["min_confidence"])))
    db.commit()
    print(f"  总记录:{total_records} 覆盖:{recommended_students}/{total_students}={coverage}%")
    print(f"  预测分: avg={stats['avg_predict_score']} max={stats['max_predict_score']} min={stats['min_predict_score']}")
    print(f"  置信度: avg={stats['avg_confidence']} max={stats['max_confidence']} min={stats['min_confidence']}")

    # ---- 表2: 热门推荐课程 Top10 (FineBI 柱状图) ----
    print("\n[2] 热门推荐课程 Top10")
    top_courses = rec_df.groupBy("course_name").agg(
        count("*").alias("recommend_count"),
    ).orderBy(desc("recommend_count")).limit(10)

    top_courses.write.jdbc(
        url=jdbc_opts["url"], table="recommend_course_top10",
        mode="overwrite", properties=jdbc_opts)
    top_courses.show(10, truncate=False)

    # ---- 表3: 推荐排名分布 (FineBI 饼图/柱状图) ----
    print("\n[3] 推荐排名分布")
    rank_dist = rec_df.groupBy("rank_order").agg(
        count("*").alias("student_count"),
        round(avg("predict_score"), 1).alias("avg_score"),
    ).orderBy("rank_order")

    rank_dist.write.jdbc(
        url=jdbc_opts["url"], table="recommend_rank_dist",
        mode="overwrite", properties=jdbc_opts)
    rank_dist.show(5, truncate=False)

    cur.close()
    db.close()
    print("\n离线分析完成")
    spark.stop()


if __name__ == "__main__":
    main()
