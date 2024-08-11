from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from src.utils import timing_decorator


@timing_decorator
def load_and_preprocess_data(spark: SparkSession, filepath: str):
    # Load the data
    df = spark.read.json(filepath)

    # Select relevant columns and create binary label
    df = df.select("reviewText", "overall")
    df = df.withColumn("label", when(col("overall") >= 4, 1.0).otherwise(0.0))

    return df


@timing_decorator
def split_data(df, test_size=0.2, seed=42):
    return df.randomSplit([1 - test_size, test_size], seed=seed)