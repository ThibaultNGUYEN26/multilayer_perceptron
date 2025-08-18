import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.neural_network import NeuralNetwork
from utils.neuron import Predict
from utils.neural_network_visualizer import NeuralNetworkVisualizer


def to_one_hot(y, num_classes=2) -> np.ndarray:
    """
    Convert integer class labels to one-hot encoded format.
    One-hot encoding is a process that transforms categorical labels into a binary matrix representation.
    Each label is represented as a vector of zeros with a single one at the index corresponding to the class.
    For example, for 3 classes, label 1 becomes [0, 1, 0].
    One-hot encoding is useful for multi-class classification problems.

    Args:
        y (np.ndarray): Labels as integers or binary array of shape (1, n_samples) or (n_samples,).
        num_classes (int): Number of unique classes.

    Returns:
        np.ndarray: One-hot encoded labels of shape (num_classes, n_samples).
    """
    # ensure y is a flat array of integer class labels
    y_flat = y.flatten().astype(int)
    n_samples = y_flat.shape[0]
    one_hot = np.zeros((num_classes, n_samples), dtype=int)
    one_hot[y_flat, np.arange(n_samples)] = 1

    return one_hot


def standardize_features(X_train, X_val) -> tuple:
    """
    Standardize features by subtracting mean and dividing by standard deviation.
    Uses training data statistics to transform both training and validation data.

    Args:
        X_train (np.ndarray): Training features of shape (n_features, n_samples)
        X_val (np.ndarray): Validation features of shape (n_features, n_samples)

    Returns:
        tuple: (standardized_X_train, standardized_X_val)
    """
    mean = np.mean(X_train, axis=1, keepdims=True)
    std = np.std(X_train, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (X_train - mean) / std, (X_val - mean) / std


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for training a neural network.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir", "-d", type=str, default="data",
        help="Directory containing train_data.csv and val_data.csv (default: data)"
    )
    p.add_argument(
        "--layers", "-l", type=int, nargs="+", default=[24, 24],
        help="number of neurons in each hidden layer (default: `--layers 24 24`)"
    )
    p.add_argument(
        "--learning-rate", "-lr", type=float, default=0.01,
        help="learning rate for gradient descent (default: 0.01)"
    )
    p.add_argument(
        "--epochs", "-e", type=int, default=100,
        help="number of training epochs (default: 100)"
    )
    p.add_argument(
        "--early-stopping", "-es", type=int, default=0,
        help="Enable early stopping with specified patience (number of epochs to wait for improvement) (default: 0)"
    )
    p.add_argument(
        "--model", "-m", type=str, default="trained_model/gradient_descent.npy",
        help="Save trained model to this file (default: trained_model/gradient_descent.npy)",
    )
    p.add_argument(
        "--optimizer", "-o", type=str, choices=['gradient_descent', 'nesterov'], default='gradient_descent',
        help="Optimizer to use for training (default: gradient_descent)"
    )
    p.add_argument(
        "--visualize", "-v", action='store_true',
        help="Enable real-time visualization of training progress"
    )
    return p.parse_args()


def main() -> None:
    args = get_args()

    # Build paths using the data directory argument
    train_data_path = os.path.join(args.data_dir, "train_data.csv")
    val_data_path = os.path.join(args.data_dir, "val_data.csv")

    # Ensure train_validation_split has been run
    if not (os.path.isfile(train_data_path) and os.path.isfile(val_data_path)):
        print(f"Error: '{train_data_path}' and '{val_data_path}' not found.")
        print("Please run 'train_validation_split.py' first to generate these files.")
        sys.exit(1)

    # Load pre-saved train/validation CSVs
    train_df = load_data(train_data_path)
    val_df   = load_data(val_data_path)

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

    # Initialize visualizer if requested
    visualizer = None
    if args.visualize:
        layer_sizes = [X_train.shape[0]] + args.layers + [y_train.shape[0]]
        visualizer = NeuralNetworkVisualizer(layer_sizes, activation_threshold=0.55).show()

    # Train
    neural_network = NeuralNetwork(
        X_train, y_train,
        X_val, y_val,
        n_hidden=args.layers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        optimizer=args.optimizer,
        visualizer=visualizer
    )
    parameters, training_history = neural_network.deep_neural_network()

    # Save model with training history
    model_data = {
        'parameters': parameters,
        'training_history': training_history,
        'architecture': args.layers,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'optimizer': args.optimizer,
    }

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    np.save(args.model, model_data)
    print(f"Model saved to '{args.model}'")

    # Final evaluation
    preds = Predict(X_val, parameters).predict()
    y_val_class = np.argmax(y_val, axis=0)
    acc = np.mean(preds.flatten() == y_val_class)
    print(f"\nFinal validation accuracy: {acc:.3f}")

    if visualizer:
        while plt.get_fignums():
            plt.pause(0.5)
if __name__ == "__main__":
    main()
