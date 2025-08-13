import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils.plot import ModelComparisonPlot

def load_model_data(model_path):
    """
    Load model data from .npy file.

    Args:
        model_path (str): Path to the model file.
    """
    try:
        data = np.load(model_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def print_model_summary(model1_data, model2_data, model1_name, model2_name) -> None:
    """
    Print summary comparison of both models.

    Args:
        model1_data (dict): Data for the first model.
        model2_data (dict): Data for the second model.
        model1_name (str): Name of the first model.
        model2_name (str): Name of the second model.
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)

    def print_model_info(data, name):
        print(f"\n{name}:")
        print(f"  Optimizer: {data.get('optimizer', 'gradient_descent')}")
        print(f"  Learning Rate: {data.get('learning_rate', '0.01')}")
        if 'layers' in data:
            if isinstance(data['layers'], list):
                architecture = ' '.join(map(str, data['layers']))
            else:
                architecture = str(data['layers'])
        elif 'architecture' in data:
            if isinstance(data['architecture'], list):
                architecture = ' '.join(map(str, data['architecture']))
            else:
                architecture = str(data['architecture'])
        print(f"  Architecture: {architecture}")
        print(f"  Epochs Completed: {data.get('epochs_completed', len(data['training_history']))}")
        print(f"  Final Train Acc: {data['training_history'][-1, 1]:.4f}")
        print(f"  Final Val Acc: {data['training_history'][-1, 3]:.4f}")
        print(f"  Final Train Loss: {data['training_history'][-1, 0]:.4f}")
        print(f"  Final Val Loss: {data['training_history'][-1, 2]:.4f}")

    print_model_info(model1_data, f"Model 1 ({model1_name})")
    print_model_info(model2_data, f"Model 2 ({model2_name})")

    # Comparison
    print(f"\nCOMPARISON:")
    val_acc_diff = model2_data['training_history'][-1, 3] - model1_data['training_history'][-1, 3]
    val_loss_diff = model2_data['training_history'][-1, 2] - model1_data['training_history'][-1, 2]

    if val_acc_diff > 0:
        print(f"  Model 2 has {val_acc_diff:.4f} higher validation accuracy")
    elif val_acc_diff < 0:
        print(f"  Model 1 has {abs(val_acc_diff):.4f} higher validation accuracy")
    else:
        print(f"  Both models have the same validation accuracy")

    if val_loss_diff < 0:
        print(f"  Model 2 has {abs(val_loss_diff):.4f} lower validation loss")
    elif val_loss_diff > 0:
        print(f"  Model 1 has {val_loss_diff:.4f} lower validation loss")
    else:
        print(f"  Both models have the same validation loss")

def get_args():
    parser = argparse.ArgumentParser(description='Compare learning curves from two models')
    parser.add_argument(
        'model1', type=str,
        help='Path to first model file (.npy)'
    )
    parser.add_argument(
        'model2', type=str,
        help='Path to second model file (.npy)'
    )
    return parser.parse_args()

def main():
    args = get_args()

    # Check if files exist
    if not os.path.exists(args.model1):
        print(f"Error: Model file '{args.model1}' not found")
        return

    if not os.path.exists(args.model2):
        print(f"Error: Model file '{args.model2}' not found")
        return

    # Load model data
    model1_data = load_model_data(args.model1)
    model2_data = load_model_data(args.model2)

    if model1_data is None or model2_data is None:
        print("Failed to load one or both models")
        return

    # Extract model names
    model1_name = os.path.basename(args.model1)
    model2_name = os.path.basename(args.model2)

    # Print summary
    print_model_summary(model1_data, model2_data, model1_name, model2_name)

    # Plot comparison
    ModelComparisonPlot(model1_data, model2_data, model1_name, model2_name).plot_comparison()

if __name__ == "__main__":
    main()
