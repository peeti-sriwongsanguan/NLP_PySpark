import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import findspark
findspark.init()

from pyspark.sql import SparkSession
from src.data_processing import load_and_preprocess_data, split_data
from src.model import build_pipeline, train_model, evaluate_model
from src.utils import timing_decorator

@timing_decorator
def main():
    # Create Spark session
    spark = SparkSession.builder \
        .appName("AmazonReviewSentiment") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    # Set log level to ERROR to suppress WARN messages
    spark.sparkContext.setLogLevel("ERROR")

    # Load and preprocess data
    df = load_and_preprocess_data(spark, "data/reviews_Automotive_5.json.gz")

    # Split the data
    train_df, test_df = split_data(df)

    # Build the pipeline
    pipeline = build_pipeline()

    # Train the model
    model = train_model(pipeline, train_df)

    # Evaluate the model
    auc, accuracy = evaluate_model(model, test_df)

    print(f"Area Under ROC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()