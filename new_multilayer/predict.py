import argparse
import numpy as np
import pandas as pd
import sys

from data_loader import load_data
from neuron import Predict


def binary_cross_entropy_loss(y_true, y_pred):
    """
    Calculate binary cross-entropy loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))


def standardize_features_with_saved_stats(X, stats):
    """
    Standardize features using previously saved statistics.
    """
    mean = stats['mean']
    std = stats['std']
    return (X - mean) / std


def to_one_hot(y, num_classes=2):
    """
    Convert labels to one-hot encoding.
    """
    y_flat = y.flatten().astype(int)
    n_samples = y_flat.shape[0]
    one_hot = np.zeros((num_classes, n_samples), dtype=int)
    one_hot[y_flat, np.arange(n_samples)] = 1
    return one_hot


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for prediction.
    """
    p = argparse.ArgumentParser(description="Make predictions using trained neural network")
    p.add_argument(
        "--model",
        type=str,
        default="trained_model.npy",
        help="Path to trained model file (default: trained_model.npy)"
    )
    p.add_argument(
        "--data",
        type=str,
        default="data.csv",
        help="Path to data file for prediction (default: data.csv)"
    )
    p.add_argument(
        "--output",
        type=str,
        help="Path to save predictions as CSV file (optional)"
    )
    p.add_argument(
        "--show-predictions",
        type=int,
        default=10,
        help="Number of predictions to display (1 to max, default: 10)"
    )
    return p.parse_args()


def main() -> None:
    args = get_args()

    # Load the trained model
    try:
        model_data = np.load(args.model, allow_pickle=True).item()
        parameters = model_data['parameters']
        print("Model loaded successfully!")
        print(f"Architecture: {model_data.get('architecture', 'Unknown')}")
        print(f"Learning rate: {model_data.get('learning_rate', 'Unknown')}")
        print(f"Training epochs: {model_data.get('epochs', model_data.get('iterations', 'Unknown'))}")
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train a model first using train.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load the data for prediction
    try:
        df = load_data(args.data)
        print(f"\nLoaded data: {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: Data file '{args.data}' not found!")
        sys.exit(1)

    # Validate show-predictions arg
    max_show = len(df)
    if args.show_predictions < 1 or args.show_predictions > max_show:
        print(f"Error: --show-predictions must be between 1 and {max_show}")
        sys.exit(1)
    n_show = args.show_predictions

    # Prepare features
    X_raw = df.drop(["id", "diagnosis"], axis=1).values.T
    mean = np.mean(X_raw, axis=1, keepdims=True)
    std = np.std(X_raw, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    X_scaled = (X_raw - mean) / std

    # Make predictions
    predictor = Predict(X_scaled, parameters)
    predictions = predictor.predict()
    probabilities = predictor.predict_proba()

    # Convert to labels
    pred_labels = ["Benign" if p == 0 else "Malignant" for p in predictions.flatten()]

    # Build results DataFrame
    results_df = df.copy()
    results_df['predicted_class'] = pred_labels
    results_df['predicted_label'] = predictions.flatten()
    results_df['benign_probability'] = probabilities[0, :]
    results_df['malignant_probability'] = probabilities[1, :]

    # Summary
    print(f"\nPrediction Summary:")
    print(f"Predicted Benign: {sum(predictions.flatten() == 0)} samples")
    print(f"Predicted Malignant: {sum(predictions.flatten() == 1)} samples")

    # Display first N predictions
    print(f"\nFirst {n_show} predictions:")
    print("ID       | Actual  | Predicted | Benign Prob | Malignant Prob")
    print("-" * 65)
    actual_labels = df['diagnosis']
    for i in range(n_show):
        sample_id = df.iloc[i]['id']
        actual = actual_labels.iloc[i]
        pred_class = pred_labels[i]
        benign_prob = probabilities[0, i]
        malignant_prob = probabilities[1, i]
        print(f"{sample_id:8} | {actual:7} | {pred_class:9} | {benign_prob:11.4f} | {malignant_prob:14.4f}")

    # Evaluate if ground truth available
    if 'diagnosis' in df.columns:
        actual_ints = df['diagnosis'].map({'B':0,'M':1}).values
        y_true_one_hot = to_one_hot(actual_ints.reshape(1, -1), num_classes=2)
        bce_loss = binary_cross_entropy_loss(y_true_one_hot, probabilities)
        accuracy = np.mean(predictions.flatten() == actual_ints)
        print(f"\nEvaluation Results:")
        print(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")
        print(f"Overall Accuracy: {accuracy:.3f}")
        tb = sum((actual_ints==0)&(predictions.flatten()==0))
        tm = sum((actual_ints==1)&(predictions.flatten()==1))
        fb = sum((actual_ints==1)&(predictions.flatten()==0))
        fm = sum((actual_ints==0)&(predictions.flatten()==1))
        print(f"\nDetailed Results:")
        print(f"True Benign: {tb}")
        print(f"True Malignant: {tm}")
        print(f"False Benign: {fb}")
        print(f"False Malignant: {fm}")

    # Save CSV if requested
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to: {args.output}")

if __name__ == "__main__":
    main()
