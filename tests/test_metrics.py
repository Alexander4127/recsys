from datetime import datetime
from unittest import TestCase

from pyspark.sql import SparkSession
from pyspark.sql.types import (FloatType, StringType, StructField, StructType,
                               TimestampType)

from sponge_bob_magic.metrics.metrics import Metrics

LOG_SCHEMA = StructType([
    StructField("user_id", StringType()),
    StructField("item_id", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("context", StringType()),
    StructField("relevance", FloatType())
])

REC_SCHEMA = StructType([
    StructField("user_id", StringType()),
    StructField("item_id", StringType()),
    StructField("context", StringType()),
    StructField("relevance", FloatType())
])


class TestMetrics(TestCase):
    def setUp(self):
        self.spark = (
            SparkSession.builder
            .master("local[1]")
            .config("spark.driver.memory", "512m")
            .getOrCreate()
        )

    def test_hit_rate_at_k(self):
        metrics = Metrics()
        recommendations = self.spark.createDataFrame(
            data=[
                ["user1", "item1", "day", 1.0],
                ["user1", "item2", "night", 2.0],
                ["user2", "item1", "night", 4.0],
                ["user2", "item2", "night", 3.0],
                ["user3", "item1", "day", 5.0],
                ["user1", "item3", "night", 1.0]
            ],
            schema=REC_SCHEMA
        )
        ground_truth = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item7", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertEqual(
            metrics.hit_rate_at_k(recommendations, ground_truth, 10),
            1 / 3
        )
        self.assertEqual(
            metrics.hit_rate_at_k(recommendations, ground_truth, 1),
            0.0
        )