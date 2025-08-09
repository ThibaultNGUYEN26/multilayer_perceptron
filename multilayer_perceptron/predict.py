import argparse
import numpy as np
import pandas as pd
import sys

from utils.data_loader import load_data
from utils.neuron import Predict
from utils.neural_network import binary_cross_entropy_loss


def standardize_features_with_saved_stats(X, stats) -> np.ndarray:
    """
    Standardize features using previously saved statistics.

    Args:
        X (np.ndarray): Features to standardize
        stats (dict): Dictionary containing 'mean' and 'std' for standardization

    Returns:
        np.ndarray: Standardized features
    """
    mean = stats['mean']
    std = stats['std']

    return (X - mean) / std


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for prediction.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    p = argparse.ArgumentParser(description="Make predictions using trained neural network")
    p.add_argument(
        "--model", "-m",
        type=str,
        default="trained_model/trained_model.npy",
        help="Path to trained model file (default: trained_model/trained_model.npy)"
    )
    p.add_argument(
        "--data", "-d",
        type=str,
        default="data/data.csv",
        help="Path to data file for prediction (default: data/data.csv)"
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save predictions as CSV file (optional)"
    )
    p.add_argument(
        "--show-predictions", "-s",
        type=int,
        default=10,
        help="Number of predictions to display (1 to max, default: 10)"
    )
    return p.parse_args()


def main() -> None:
    """
    Main function to load model, data, and make predictions.
    Displays predictions and saves results to CSV if specified.
    """
    args = get_args()

    # Load the trained model
    try:
        model_data = np.load(args.model, allow_pickle=True).item()
        parameters = model_data['parameters']
        print("Model loaded successfully!")
        print(f"Architecture: {model_data.get('architecture', 'Unknown')}")
        print(f"Learning rate: {model_data.get('learning_rate', 'Unknown')}")
        print(f"Training epochs: {model_data.get('epochs', model_data.get('iterations', 'Unknown'))}")

        # Show training history summary if available
        if 'training_history' in model_data:
            history = model_data['training_history']
            print(f"Final training accuracy: {history[-1, 1]:.4f}")
            print(f"Final validation accuracy: {history[-1, 3]:.4f}")
            print(f"Best validation accuracy: {np.max(history[:, 3]):.4f}")

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
    print(" ID      | Actual  | Predicted | Benign Prob | Malignant Prob | Result")
    print("-" * 80)
    actual_labels = df['diagnosis']
    for i in range(n_show):
        sample_id = df.iloc[i]['id']
        actual = actual_labels.iloc[i]
        pred_class = pred_labels[i]
        benign_prob = probabilities[0, i]
        malignant_prob = probabilities[1, i]
        result = "Correct" if (actual == 'M' and pred_class == 'Malignant') or (actual == 'B' and pred_class == 'Benign') else "Wrong"
        print(f"{sample_id:8} | {actual:7} | {pred_class:9} | {benign_prob:11.4f} | {malignant_prob:14.4f} | {result}")

    # Evaluate if ground truth available
    if 'diagnosis' in df.columns:
        # Convert actual labels to integers
        actual_ints = df['diagnosis'].map({'B':0,'M':1}).values
        # Convert actual labels to float for loss calculation
        y_true = actual_ints.astype(float)
        # Get predicted probabilities for the malignant class
        p_malignant = probabilities[1, :] if probabilities.ndim == 2 else probabilities.ravel()
        # Calculate binary cross-entropy loss
        bce_loss = binary_cross_entropy_loss(y_true, p_malignant)
        # Calculate accuracy
        accuracy = np.mean(predictions.flatten() == actual_ints)
        print(f"\nEvaluation Results:")
        print(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")
        print(f"Overall Accuracy: {accuracy:.3f}")
        # Detailed results
        tb = sum((actual_ints==0)&(predictions.flatten()==0))
        tm = sum((actual_ints==1)&(predictions.flatten()==1))
        fb = sum((actual_ints==1)&(predictions.flatten()==0))
        fm = sum((actual_ints==0)&(predictions.flatten()==1))
        print(f"\nDetailed Results:")
        print(f"Correct number of Benign: {tb}")
        print(f"Correct number of Malignant: {tm}")
        print(f"Wrong number of Benign: {fb}")
        print(f"Wrong number of Malignant: {fm}")

    # Save all predictions to predictions.csv
    all_predictions = pd.DataFrame({
        'ID': df['id'],
        'Actual': df['diagnosis'],
        'Predicted': pred_labels,
        'Benign_Prob': probabilities[0, :],
        'Malignant_Prob': probabilities[1, :],
        'Result': ['Correct' if (actual == 'M' and pred == 'Malignant') or (actual == 'B' and pred == 'Benign') else 'Wrong'
                  for actual, pred in zip(df['diagnosis'], pred_labels)]
    })

    # Save predictions to CSV only if output flag is provided
    if args.output:
        all_predictions.to_csv(args.output, index=False)
        print(f"\nAll {len(all_predictions)} predictions saved to: {args.output}")

if __name__ == "__main__":
    main()
