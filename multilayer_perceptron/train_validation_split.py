import numpy as np
import pandas as pd

from utils.data_loader import load_data
from utils.plot import TrainValidationDistributionPlot


def train_validation_split(df, train_size=0.8, val_size=0.2, random_state=42):
    """
    Split dataset into training and validation sets with stratification by diagnosis.

    Args:
        df (pd.DataFrame): Input dataframe
        train_size (float): Proportion of training data (default: 0.8)
        val_size (float): Proportion of validation data (default: 0.2)
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df)
    """
    # Validate proportions
    if abs(train_size + val_size - 1.0) > 1e-6:
        raise ValueError(f"Proportions must sum to 1.0, got {train_size + val_size}")

    np.random.seed(random_state)

    # Separate by diagnosis
    benign = df[df["diagnosis"] == "B"]
    malignant = df[df["diagnosis"] == "M"]

    # Calculate sizes for each class
    n_benign_val = int(len(benign) * val_size)
    n_benign_train = len(benign) - n_benign_val

    n_malignant_val = int(len(malignant) * val_size)
    n_malignant_train = len(malignant) - n_malignant_val

    # Generate random indices for benign samples
    benign_indices = np.random.permutation(len(benign))
    benign_val_indices = benign_indices[:n_benign_val]
    benign_train_indices = benign_indices[n_benign_val:]

    # Generate random indices for malignant samples
    malignant_indices = np.random.permutation(len(malignant))
    malignant_val_indices = malignant_indices[:n_malignant_val]
    malignant_train_indices = malignant_indices[n_malignant_val:]

    # Split benign data
    benign_train = benign.iloc[benign_train_indices]
    benign_val = benign.iloc[benign_val_indices]

    # Split malignant data
    malignant_train = malignant.iloc[malignant_train_indices]
    malignant_val = malignant.iloc[malignant_val_indices]

    # Combine splits and shuffle
    train_df = pd.concat([benign_train, malignant_train]).sample(frac=1, random_state=random_state)
    val_df = pd.concat([benign_val, malignant_val]).sample(frac=1, random_state=random_state)

    return train_df, val_df


def train_test_split_custom(df, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets with stratification by diagnosis.
    This is the original function for backward compatibility.

    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (train_df, test_df)
    """
    np.random.seed(random_state)

    # Separate by diagnosis
    benign = df[df["diagnosis"] == "B"]
    malignant = df[df["diagnosis"] == "M"]

    # Calculate test sizes for each class
    n_benign_test = int(len(benign) * test_size)
    n_malignant_test = int(len(malignant) * test_size)

    # Randomly sample test indices
    benign_indices = np.random.choice(len(benign), n_benign_test, replace=False)
    malignant_indices = np.random.choice(len(malignant), n_malignant_test, replace=False)

    # Split benign data
    benign_test = benign.iloc[benign_indices]
    benign_train = benign.drop(benign.index[benign_indices])

    # Split malignant data
    malignant_test = malignant.iloc[malignant_indices]
    malignant_train = malignant.drop(malignant.index[malignant_indices])

    # Combine splits
    train_df = pd.concat([benign_train, malignant_train]).sample(frac=1, random_state=random_state)
    test_df = pd.concat([benign_test, malignant_test]).sample(frac=1, random_state=random_state)

    return train_df, test_df


def print_split_info(train_df, val_df=None):
    """
    Print information about the dataset splits.

    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame, optional): Validation dataframe
    """
    total_samples = len(train_df)
    if val_df is not None:
        total_samples += len(val_df)

    print("Dataset Split Information:")
    print("-" * 50)

    # Training set info
    train_benign = len(train_df[train_df["diagnosis"] == "B"])
    train_malignant = len(train_df[train_df["diagnosis"] == "M"])
    print(f"Training:   {len(train_df):3d} samples ({len(train_df)/total_samples*100:.1f}%) - "
          f"Benign: {train_benign:3d}, Malignant: {train_malignant:3d}")

    # Validation set info
    if val_df is not None:
        val_benign = len(val_df[val_df["diagnosis"] == "B"])
        val_malignant = len(val_df[val_df["diagnosis"] == "M"])
        print(f"Validation: {len(val_df):3d} samples ({len(val_df)/total_samples*100:.1f}%) - "
              f"Benign: {val_benign:3d}, Malignant: {val_malignant:3d}")

    print(f"Total:      {total_samples:3d} samples")
    print("-" * 50)


if __name__ == "__main__":
    # Load the dataset
    df = load_data("data/data.csv")
    print()

    # Test the train/validation split
    train_df, val_df = train_validation_split(
        df, train_size=0.8, val_size=0.2, random_state=42
    )

    print_split_info(train_df, val_df)
    print()

    # Save the split datasets
    train_df.to_csv("data/train_data.csv", index=False)
    val_df.to_csv("data/val_data.csv", index=False)
    print("Split datasets saved:")
    print("- data/train_data.csv")
    print("- data/val_data.csv")
    print()

    # Plot the train/validation distribution
    plot = TrainValidationDistributionPlot(train_df, val_df)
    plot.plot()
