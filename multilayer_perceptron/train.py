import argparse
import os
import sys
import pandas as pd
import numpy as np

from utils.data_loader import load_data
from utils.neural_network import NeuralNetwork
from utils.neuron import Predict


def to_one_hot(y, num_classes=2):
    """
    Convert labels to one-hot encoding.

    Args:
        y (np.ndarray): Labels as integers or binary array of shape (1, n_samples) or (n_samples,)
        num_classes (int): Number of classes

    Returns:
        np.ndarray: One-hot encoded labels of shape (num_classes, n_samples)
    """
    # ensure y is a flat array of integer class labels
    y_flat = y.flatten().astype(int)
    n_samples = y_flat.shape[0]
    one_hot = np.zeros((num_classes, n_samples), dtype=int)
    one_hot[y_flat, np.arange(n_samples)] = 1
    return one_hot


def standardize_features(X_train, X_test):
    """
    Standardize features by subtracting mean and dividing by standard deviation.
    Uses training data statistics to transform both training and test data.
    """
    mean = np.mean(X_train, axis=1, keepdims=True)
    std = np.std(X_train, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (X_train - mean) / std, (X_test - mean) / std


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--layer", type=int, nargs="+", default=[24, 24],
        help="number of neurons in each hidden layer (default: `--layer 24 24`)"
    )
    p.add_argument(
        "--learning_rate", type=float, default=0.01,
        help="learning rate for gradient descent (default: 0.01)"
    )
    p.add_argument(
        "--epochs", type=int, default=100,
        help="number of training epochs (default: 100)"
    )
    return p.parse_args()


def main() -> None:
    args = get_args()

    # Ensure train_validation_split has been run
    if not (os.path.isfile("data/train_data.csv") and os.path.isfile("data/val_data.csv")):
        print("Error: 'data/train_data.csv' and 'data/val_data.csv' not found.")
        print("Please run 'train_validation_split.py' first to generate these files.")
        sys.exit(1)

    # Load pre-saved train/validation CSVs
    train_df = load_data("data/train_data.csv")
    val_df   = load_data("data/val_data.csv")

    # Remove any non-data rows (e.g. header rows imported as data)
    for df in (train_df, val_df):
        # keep only rows where diagnosis is 'B' or 'M'
        df.drop(df[~df['diagnosis'].isin(['B','M'])].index, inplace=True)
        # convert feature columns to numeric and drop any rows with NaNs
        feature_cols = [c for c in df.columns if c not in ['id','diagnosis']]
        df[feature_cols] = df[feature_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))
        df.dropna(inplace=True)

    # Prepare datasets
    X_train_raw = train_df.drop(["id","diagnosis"], axis=1).values.T
    y_train_labels = train_df["diagnosis"].map({"B":0,"M":1}).values.reshape(1,-1)
    y_train = to_one_hot(y_train_labels, num_classes=2)

    X_val_raw = val_df.drop(["id","diagnosis"], axis=1).values.T
    y_val_labels = val_df["diagnosis"].map({"B":0,"M":1}).values.reshape(1,-1)
    y_val = to_one_hot(y_val_labels, num_classes=2)

    # Scale
    X_train, X_val = standardize_features(X_train_raw, X_val_raw)

    # Train
    nn = NeuralNetwork(
        X_train, y_train,
        X_val, y_val,
        n_hidden=args.layer,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    parameters = nn.deep_neural_network()

    # Save model
    model_data = {
        'parameters': parameters,
        'architecture': args.layer,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }
    os.makedirs("trained_model", exist_ok=True)
    np.save("trained_model/trained_model.npy", model_data)
    print("Model saved to 'trained_model/trained_model.npy'")

    # Final evaluation
    preds = Predict(X_val, parameters).predict()
    y_val_class = np.argmax(y_val, axis=0)
    acc = np.mean(preds.flatten() == y_val_class)
    print(f"\nFinal validation accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
