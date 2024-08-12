import os
import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.sql.functions import col
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.utils import timing_decorator

# Constants
MAX_SEQUENCE_LENGTH = 128
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5


def preprocess_data(spark, input_path):
    # Load the data
    df = spark.read.json(input_path)

    # Select relevant columns and create binary label
    df = df.select("reviewText", "overall")
    df = df.withColumn("label", (col("overall") >= 4).cast("float"))

    # Tokenization and stop words removal
    tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # Create and apply the processing pipeline
    df = tokenizer.transform(df)
    df = remover.transform(df)

    # Create vocabulary
    cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=10000)
    cv_model = cv.fit(df)
    df = cv_model.transform(df)

    # Create TF-IDF features
    idf = IDF(inputCol="raw_features", outputCol="features")
    idf_model = idf.fit(df)
    df = idf_model.transform(df)

    return df, cv_model.vocabulary


class RNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)


class RNN_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_filters, filter_sizes, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out.permute(0, 2, 1)
        conved = [nn.functional.relu(conv(lstm_out)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)


def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += ((outputs > 0.5) == labels).float().mean().item()

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += ((outputs > 0.5) == labels).float().mean().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return history


def plot_training_history(histories):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Model Accuracy", "Model Loss"))

    for name, history in histories.items():
        fig.add_trace(go.Scatter(y=history['train_acc'], name=f'{name} (Train)', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(y=history['val_acc'], name=f'{name} (Val)', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(y=history['train_loss'], name=f'{name} (Train)', mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(y=history['val_loss'], name=f'{name} (Val)', mode='lines'), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    fig.update_layout(height=600, width=1000, title_text="Model Comparison")
    fig.write_html("model_comparison.html")
    print("Comparison plot saved as 'model_comparison.html'")


@timing_decorator
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create Spark session
    spark = SparkSession.builder.appName("SentimentAnalysisComparison").getOrCreate()

    # Preprocess data
    df, vocabulary = preprocess_data(spark, "data/reviews_Automotive_5.json.gz")

    # Split data
    train_df, val_df, test_df = df.randomSplit([0.6, 0.2, 0.2], seed=42)

    # Function to convert Spark DataFrame column to padded numpy array
    def prepare_data(dataframe):
        words = dataframe.select("filtered_words").rdd.flatMap(lambda x: x).collect()
        labels = np.array(dataframe.select("label").collect()).flatten()

        # Convert words to indices
        word_to_index = {word: i + 1 for i, word in enumerate(vocabulary)}  # Reserve 0 for padding
        sequences = [[word_to_index.get(word, 0) for word in seq] for seq in words]

        # Pad sequences
        padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True,
                                                           padding_value=0)
        if padded_sequences.shape[1] > MAX_SEQUENCE_LENGTH:
            padded_sequences = padded_sequences[:, :MAX_SEQUENCE_LENGTH]
        elif padded_sequences.shape[1] < MAX_SEQUENCE_LENGTH:
            padding = torch.zeros(padded_sequences.shape[0], MAX_SEQUENCE_LENGTH - padded_sequences.shape[1],
                                  dtype=torch.long)
            padded_sequences = torch.cat([padded_sequences, padding], dim=1)

        return padded_sequences, torch.tensor(labels, dtype=torch.float)

    # Prepare data for PyTorch
    train_data = prepare_data(train_df)
    val_data = prepare_data(val_df)
    test_data = prepare_data(test_df)

    train_loader = DataLoader(TensorDataset(*train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=BATCH_SIZE)

    # Build models
    VOCAB_SIZE = len(vocabulary) + 1  # +1 for padding token
    models = {
        "RNN LSTM": RNN_LSTM(VOCAB_SIZE, EMBEDDING_DIM, 100, 1),
        "CNN": CNN(VOCAB_SIZE, EMBEDDING_DIM, 100, [3, 4, 5], 1),
        "RNN LSTM + CNN": RNN_CNN(VOCAB_SIZE, EMBEDDING_DIM, 100, 100, [3, 4, 5], 1)
    }

    criterion = nn.BCEWithLogitsLoss()

    # Train models
    histories = {}
    for name, model in models.items():
        print(f"Training {name} model...")
        optimizer = optim.Adam(model.parameters())
        histories[name] = train_model(model, train_loader, val_loader, criterion, optimizer, device)

    # Plot results
    plot_training_history(histories)

    # Print final results
    print("\nFinal Results:")
    for name, history in histories.items():
        print(f"\n{name}:")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_acc'][-1]:.4f}")

    # Evaluate models on test data
    for name, model in models.items():
        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += ((outputs > 0.5) == labels).float().mean().item()
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(f"{name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main()