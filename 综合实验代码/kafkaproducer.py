# -*- coding: utf-8 -*-
"""
把考试答题数据从 Excel 读出来，一条条发到 Kafka
模拟在线教育平台实时产生数据的过程
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer

# ---- 写死的配置 ----
KAFKA_HOST = "node1:9092"
KAFKA_TOPIC = "edu_exam_data"
FILE_EXAM = "../data/b1_t_stat_exam_answer.xlsx"
FILE_CLAZZ = "../data/b1_t_stat_clazz.xlsx"


def main():
    print("=" * 50)
    print("  考试数据 -> Kafka 生产者")
    print(f"  Kafka: {KAFKA_HOST}")
    print(f"  Topic: {KAFKA_TOPIC}")
    print("=" * 50)

    # 1. 连接 Kafka
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_HOST,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        max_request_size=10485760,
    )

    # 2. 读取数据
    print("\n[1] 读取 Excel 数据...")
    exam_df = pd.read_excel(FILE_EXAM)

    # 只去掉 score 为空或明显异常的数据，其余全部原样发送
    exam_df = exam_df.dropna(subset=["personid", "score"])
    exam_df = exam_df[(exam_df["score"] >= 0) & (exam_df["score"] <= 100)]

    print(f"  有效数据: {len(exam_df)} 条")

    # 3. 逐条发送（全部17个字段，全部转字符串，跟 Java 版一致）
    print("\n[2] 开始发送...")
    all_cols = list(exam_df.columns)
    total = len(exam_df)
    success = 0

    for i, (_, row) in enumerate(exam_df.iterrows(), 1):
        # 所有字段转字符串，NaN → null
        record = {}
        for col_name in all_cols:
            val = row[col_name]
            if pd.isna(val):
                record[col_name] = None
            else:
                record[col_name] = str(val)

        try:
            producer.send(KAFKA_TOPIC, key=str(row["personid"]).encode(), value=record)
            success += 1
        except Exception as e:
            print(f"  发送失败: {e}")

        if i % 500 == 0:
            print(f"  进度: {i}/{total} ({i*100//total}%)")
            producer.flush()

        time.sleep(0.01)

    producer.flush()
    producer.close()

    print(f"\n[完成] 成功发送 {success}/{total} 条")
    print("=" * 50)


if __name__ == "__main__":
    main()
