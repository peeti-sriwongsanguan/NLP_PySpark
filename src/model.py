from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from src.utils import timing_decorator


@timing_decorator
def build_pipeline():
    tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="rawFeatures")
    idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10)
    return Pipeline(stages=[tokenizer, remover, countVectorizer, idf, lr])


@timing_decorator
def train_model(pipeline, train_df):
    return pipeline.fit(train_df)


@timing_decorator
def evaluate_model(model, test_df):
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    correct_preds = predictions.filter(col("label") == col("prediction")).count()
    total_preds = predictions.count()
    accuracy = correct_preds / total_preds

    return auc, accuracy