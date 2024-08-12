# Amazon Automotive Reviews Sentiment Analysis

# NLP_PySpark
After trying TensorFlow and PyTorch, I want to utilize PySpark which is good for massive dataset

## Project Description

This project performs sentiment analysis on Amazon Automotive product reviews using Apache Spark's MLlib. It classifies reviews as positive or negative based on the review text, demonstrating the use of distributed computing for natural language processing tasks.

### Key Features:
- Data preprocessing using PySpark
- Text feature extraction using TF-IDF
- Binary classification using Logistic Regression
- Model evaluation using Area Under ROC and Accuracy metrics

## Why PySpark?

While TensorFlow and PyTorch are excellent choices for deep learning models, we chose PySpark for this project for several reasons:

1. **Scalability**: PySpark is designed to handle very large datasets that may not fit into the memory of a single machine. It can distribute data processing across a cluster of computers, making it ideal for big data scenarios.

2. **Integrated Analytics**: PySpark provides a unified engine for large-scale data processing and machine learning. It allows us to perform data loading, preprocessing, model training, and evaluation all within the same framework.

3. **Simplified ML Pipeline**: PySpark's MLlib offers a high-level API for building machine learning pipelines, making it easier to assemble and tune ML workflows.

4. **Performance for Large Datasets**: For very large datasets, PySpark can outperform single-machine solutions by leveraging distributed computing resources.

5. **Business Reality**: In many business scenarios, simple models that can process vast amounts of data quickly are more valuable than complex models that take longer to train and deploy.

6. **Easy Integration**: If this project needs to be integrated into a larger data processing ecosystem (e.g., Hadoop ecosystem), PySpark makes this integration seamless.

While deep learning models (like those built with TensorFlow or PyTorch) could potentially achieve higher accuracy for this task, the PySpark solution offers a good balance of performance, scalability, and simplicity, especially when dealing with large-scale text data.

## Requirements

- Python 3.7+
- PySpark 3.4.1
- PyArrow 12.0.1
- Java 8 or later

## Project Structure

```
amazon_reviews_sentiment/
│
├── data/
│   └── reviews_Automotive_5.json.gz
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Setup and Running

1. Clone the repository:
   ```
   git clone <repository-url>
   cd amazon_reviews_sentiment
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Ensure your `reviews_Automotive_5.json.gz` file is in the `data/` directory.

5. Run the main script:
   ```
   python main.py
   ```

## Results

After running the script, we obtained the following results:

```
load_and_preprocess_data took 2.25 seconds to run.
split_data took 0.01 seconds to run.
build_pipeline took 0.07 seconds to run.
train_model took 5.67 seconds to run.
evaluate_model took 1.05 seconds to run.
Area Under ROC: 0.7517
Accuracy: 0.8234
main took 11.21 seconds to run.
```

### Performance Metrics:
- **Area Under ROC**: 0.7517
  - This indicates good discriminative ability of the model. A value above 0.7 is generally considered good for binary classification tasks.
- **Accuracy**: 0.8234 (82.34%)
  - This shows that our model correctly classified 82.34% of the reviews, which is a strong performance for sentiment analysis.

### Execution Times:
- Data loading and preprocessing: 2.25 seconds
- Data splitting: 0.01 seconds
- Pipeline building: 0.07 seconds
- Model training: 5.67 seconds
- Model evaluation: 1.05 seconds
- Total execution time: 11.21 seconds

These results demonstrate that our PySpark implementation provides both good predictive performance and efficient execution times, even on a local machine. The quick processing time showcases PySpark's ability to handle text processing and machine learning tasks effectively.

## Future Improvements

- Experiment with more advanced feature extraction techniques
- Try different classification algorithms available in MLlib
- Implement cross-validation for more robust model evaluation
- Explore techniques for handling imbalanced datasets if necessary
- Scale up to larger datasets and distributed computing environments to fully leverage PySpark's capabilities

