import argparse
import numpy as np

# ANSI color codes
class Colors:
    BLUE = '\033[94m'      # Blue for descriptions
    GREEN = '\033[92m'     # Green for values
    WHITE = '\033[97m'     # White for titles
    YELLOW = '\033[93m'    # Yellow for section headers
    RED = '\033[91m'       # Red for warnings
    ORANGE = '\033[38;5;208m'  # Orange for moderate warnings
    EOC = '\033[0m'      # EOC to default color
    BOLD = '\033[1m'       # Bold text

def calculate_convergence_metrics(history) -> dict:
    """
    Calculate convergence-related metrics.

    Args:
        history (np.ndarray): Training history with columns [train_loss, train_acc, val_loss, val_acc]

    Returns:
        dict: A dictionary containing convergence metrics.
    """
    val_loss = history[:, 2]
    train_loss = history[:, 0]
    val_acc = history[:, 3]
    train_acc = history[:, 1]

    metrics = {}

    # Loss convergence
    metrics['final_loss_variance'] = np.var(val_loss[-10:]) if len(val_loss) >= 10 else np.var(val_loss)
    metrics['loss_plateau_epochs'] = count_plateau_epochs(val_loss)

    # Accuracy stability
    metrics['accuracy_stability'] = 1.0 - np.std(val_acc[-10:]) if len(val_acc) >= 10 else 1.0 - np.std(val_acc)

    # Learning rate effectiveness
    early_improvement = val_acc[min(9, len(val_acc)-1)] - val_acc[0] if len(val_acc) > 1 else 0
    total_improvement = val_acc[-1] - val_acc[0] if len(val_acc) > 1 else 0
    metrics['early_learning_rate'] = early_improvement / min(10, len(val_acc))
    metrics['overall_learning_rate'] = total_improvement / len(val_acc) if len(val_acc) > 0 else 0

    return metrics

def count_plateau_epochs(values, threshold=0.001) -> int:
    """
    Count epochs where improvement is below threshold.

    Args:
        values (np.ndarray): Array of metric values (e.g., loss or accuracy).
        threshold (float): Improvement threshold to consider as plateau (default: 0.001).

    Returns:
        int: Number of epochs where improvement is below the threshold.
    """
    if len(values) < 2:
        return 0

    plateau_count = 0
    for i in range(1, len(values)):
        if abs(values[i] - values[i-1]) < threshold:
            plateau_count += 1
    return plateau_count

def calculate_overfitting_metrics(history) -> dict:
    """
    Calculate detailed overfitting metrics.

    Args:
        history (np.ndarray): Training history with columns [train_loss, train_acc, val_loss, val_acc]

    Returns:
        dict: A dictionary containing overfitting metrics.
    """
    train_acc = history[:, 1]
    val_acc = history[:, 3]
    train_loss = history[:, 0]
    val_loss = history[:, 2]

    metrics = {}

    # Generalization gap
    metrics['final_acc_gap'] = train_acc[-1] - val_acc[-1]
    metrics['final_loss_gap'] = val_loss[-1] - train_loss[-1]
    metrics['max_acc_gap'] = np.max(train_acc - val_acc)
    metrics['max_loss_gap'] = np.max(val_loss - train_loss)

    # Overfitting onset detection
    acc_gaps = train_acc - val_acc
    overfitting_start = -1
    for i in range(len(acc_gaps)):
        if acc_gaps[i] > 0.05:  # 5% gap threshold
            overfitting_start = i + 1
            break
    metrics['overfitting_onset_epoch'] = overfitting_start

    # Validation performance degradation
    if len(val_acc) >= 10:
        best_val_acc_idx = np.argmax(val_acc)
        if best_val_acc_idx < len(val_acc) - 5:  # Peak not in last 5 epochs
            degradation = val_acc[best_val_acc_idx] - val_acc[-1]
            metrics['val_acc_degradation'] = degradation
        else:
            metrics['val_acc_degradation'] = 0.0
    else:
        metrics['val_acc_degradation'] = 0.0

    return metrics

def calculate_learning_dynamics(history) -> dict:
    """
    Analyze learning dynamics and phases.

    Args:
        history (np.ndarray): Training history with columns [train_loss, train_acc, val_loss, val_acc]

    Returns:
        dict: A dictionary containing learning dynamics metrics.
    """
    val_acc = history[:, 3]
    train_acc = history[:, 1]
    val_loss = history[:, 2]

    metrics = {}

    # Learning phases
    if len(val_acc) >= 3:
        # Initial learning phase (first 25% or 10 epochs, whichever is smaller)
        initial_phase_end = min(len(val_acc) // 4, 10)
        initial_improvement = val_acc[initial_phase_end-1] - val_acc[0] if initial_phase_end > 0 else 0
        metrics['initial_phase_improvement'] = initial_improvement

        # Middle learning phase
        mid_start = initial_phase_end
        mid_end = min(len(val_acc) * 3 // 4, len(val_acc) - 5)
        if mid_end > mid_start:
            mid_improvement = val_acc[mid_end-1] - val_acc[mid_start-1]
            metrics['middle_phase_improvement'] = mid_improvement
        else:
            metrics['middle_phase_improvement'] = 0.0

        # Final learning phase
        final_start = max(mid_end, len(val_acc) - 10)
        final_improvement = val_acc[-1] - val_acc[final_start-1] if final_start > 0 else 0
        metrics['final_phase_improvement'] = final_improvement

    # Learning velocity (rate of change)
    if len(val_acc) >= 2:
        acc_changes = np.diff(val_acc)
        metrics['avg_learning_velocity'] = np.mean(acc_changes)
        metrics['learning_acceleration'] = np.mean(np.diff(acc_changes)) if len(acc_changes) >= 2 else 0

    # Loss reduction efficiency
    if len(val_loss) >= 2:
        loss_reduction = val_loss[0] - val_loss[-1]
        metrics['total_loss_reduction'] = loss_reduction
        metrics['loss_reduction_per_epoch'] = loss_reduction / len(val_loss)

    return metrics

def calculate_performance_consistency(history) -> dict:
    """
    Calculate performance consistency metrics.

    Args:
        history (np.ndarray): Training history with columns [train_loss, train_acc, val_loss, val_acc]

    Returns:
        dict: A dictionary containing performance consistency metrics.
    """
    val_acc = history[:, 3]
    val_loss = history[:, 2]

    metrics = {}

    # Accuracy consistency (coefficient of variation)
    if len(val_acc) > 1:
        metrics['accuracy_cv'] = np.std(val_acc) / np.mean(val_acc) if np.mean(val_acc) > 0 else float('inf')

        # Rolling window consistency (last 20% of training)
        window_size = max(len(val_acc) // 5, 5)
        if len(val_acc) >= window_size:
            recent_acc = val_acc[-window_size:]
            metrics['recent_accuracy_consistency'] = 1.0 - (np.std(recent_acc) / np.mean(recent_acc))
        else:
            metrics['recent_accuracy_consistency'] = 1.0 - (np.std(val_acc) / np.mean(val_acc))

    # Loss consistency
    if len(val_loss) > 1:
        metrics['loss_cv'] = np.std(val_loss) / np.mean(val_loss) if np.mean(val_loss) > 0 else float('inf')

    # Monotonicity (how often validation accuracy improves)
    if len(val_acc) >= 2:
        improvements = np.sum(np.diff(val_acc) > 0)
        metrics['improvement_ratio'] = improvements / (len(val_acc) - 1)

    return metrics

def analyze_training_history(model_path, verbose=False) -> None:
    """
    Analyze and display comprehensive training history metrics.

    Args:
        model_path (str): Path to the trained model file.
        verbose (bool): If True, display additional information.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If there is an error loading the model data.
    """

    # Load model and extract training history
    try:
        model_data = np.load(model_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if 'training_history' not in model_data:
        print("No training history found in the model file.")
        return

    history = model_data['training_history']
    epochs = len(history)

    print("=" * 80)
    print(f"{Colors.WHITE}{Colors.BOLD}COMPREHENSIVE TRAINING HISTORY ANALYSIS{Colors.EOC}")
    print("=" * 80)

    # Basic info
    print(f"{Colors.WHITE}Model Configuration:{Colors.EOC}")
    print(f"  Total Epochs: {Colors.GREEN}{epochs}{Colors.EOC}")
    print(f"  Architecture: {Colors.GREEN}{model_data.get('architecture', 'Unknown')}{Colors.EOC}")
    print(f"  Learning Rate: {Colors.GREEN}{model_data.get('learning_rate', 'Unknown')}{Colors.EOC}")

    # Final metrics
    print(f"\n" + "="*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}FINAL PERFORMANCE METRICS (Epoch {epochs}){Colors.EOC}")
    print(f"="*50)
    if verbose:
        print(f"{Colors.BLUE}  📊 These are the performance metrics achieved at the end of training.{Colors.EOC}")
        print(f"{Colors.BLUE}     They show your model's final capability on both training and validation data.{Colors.EOC}")
    print(f"  Training Loss:       {Colors.GREEN}{history[-1, 0]:.6f}{Colors.EOC}")
    print(f"  Training Accuracy:   {Colors.GREEN}{history[-1, 1]:.6f} ({history[-1, 1]*100:.2f}%){Colors.EOC}")
    print(f"  Validation Loss:     {Colors.GREEN}{history[-1, 2]:.6f}{Colors.EOC}")
    print(f"  Validation Accuracy: {Colors.GREEN}{history[-1, 3]:.6f} ({history[-1, 3]*100:.2f}%){Colors.EOC}")

    # Best metrics
    best_train_acc_epoch = np.argmax(history[:, 1]) + 1
    best_val_acc_epoch = np.argmax(history[:, 3]) + 1
    min_train_loss_epoch = np.argmin(history[:, 0]) + 1
    min_val_loss_epoch = np.argmin(history[:, 2]) + 1

    print(f"\n" + "="*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}PEAK PERFORMANCE METRICS{Colors.EOC}")
    print(f"="*50)
    if verbose:
        print(f"{Colors.BLUE}  🏆 These show the best performance your model achieved during training.{Colors.EOC}")
        print(f"{Colors.BLUE}     If peak validation performance occurred early, you might have overtrained.{Colors.EOC}")
    print(f"  Best Training Accuracy:   {Colors.GREEN}{np.max(history[:, 1]):.6f} ({np.max(history[:, 1])*100:.2f}%) at Epoch {best_train_acc_epoch}{Colors.EOC}")
    print(f"  Best Validation Accuracy: {Colors.GREEN}{np.max(history[:, 3]):.6f} ({np.max(history[:, 3])*100:.2f}%) at Epoch {best_val_acc_epoch}{Colors.EOC}")
    print(f"  Lowest Training Loss:     {Colors.GREEN}{np.min(history[:, 0]):.6f} at Epoch {min_train_loss_epoch}{Colors.EOC}")
    print(f"  Lowest Validation Loss:   {Colors.GREEN}{np.min(history[:, 2]):.6f} at Epoch {min_val_loss_epoch}{Colors.EOC}")

    # Calculate comprehensive metrics
    convergence_metrics = calculate_convergence_metrics(history)
    overfitting_metrics = calculate_overfitting_metrics(history)
    learning_dynamics = calculate_learning_dynamics(history)
    consistency_metrics = calculate_performance_consistency(history)

    # Convergence Analysis
    print(f"\n" + "="*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}CONVERGENCE ANALYSIS{Colors.EOC}")
    print(f"="*50)
    if verbose:
        print(f"{Colors.BLUE}  🎯 Convergence metrics tell you how stable and effective your training was.{Colors.EOC}")
        print(f"{Colors.BLUE}     Low variance and few plateau epochs indicate good convergence.{Colors.EOC}")
    print(f"  Loss Variance (last 10 epochs): {Colors.GREEN}{convergence_metrics['final_loss_variance']:.8f}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Measures how much validation loss fluctuated in final epochs{Colors.EOC}")
        print(f"{Colors.BLUE}       Lower values (< 0.001) indicate stable convergence{Colors.EOC}")
    print(f"  Plateau Epochs Count:           {Colors.GREEN}{convergence_metrics['loss_plateau_epochs']}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Number of epochs with minimal loss improvement (< 0.001){Colors.EOC}")
        print(f"{Colors.BLUE}       High counts suggest learning rate might be too low{Colors.EOC}")
    print(f"  Accuracy Stability:             {Colors.GREEN}{convergence_metrics['accuracy_stability']:.6f}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → How consistent validation accuracy was (1.0 = perfect stability){Colors.EOC}")
        print(f"{Colors.BLUE}       Values > 0.95 indicate very stable training{Colors.EOC}")
    print(f"  Early Learning Rate:            {Colors.GREEN}{convergence_metrics['early_learning_rate']:.6f}/epoch{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Average accuracy improvement per epoch in first 10 epochs{Colors.EOC}")
        print(f"{Colors.BLUE}       Higher values indicate effective initial learning{Colors.EOC}")
    print(f"  Overall Learning Rate:          {Colors.GREEN}{convergence_metrics['overall_learning_rate']:.6f}/epoch{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Average accuracy improvement per epoch across all training{Colors.EOC}")
        print(f"{Colors.BLUE}       Positive values show overall improvement trend{Colors.EOC}")

    # Overfitting Analysis
    print(f"\n" + "="*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}OVERFITTING ANALYSIS{Colors.EOC}")
    print(f"="*50)
    if verbose:
        print(f"{Colors.BLUE}  ⚠️  Overfitting occurs when your model memorizes training data instead of learning patterns.{Colors.EOC}")
        print(f"{Colors.BLUE}     Large gaps between training and validation performance indicate overfitting.{Colors.EOC}")
    print(f"  Final Accuracy Gap (Train-Val): {Colors.GREEN}{overfitting_metrics['final_acc_gap']:.6f}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Training accuracy minus validation accuracy at final epoch{Colors.EOC}")
        print(f"{Colors.BLUE}       Gaps > 0.05 suggest overfitting, > 0.10 is significant concern{Colors.EOC}")
    print(f"  Final Loss Gap (Val-Train):     {Colors.GREEN}{overfitting_metrics['final_loss_gap']:.6f}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Validation loss minus training loss at final epoch{Colors.EOC}")
        print(f"{Colors.BLUE}       Positive values indicate validation is harder (normal){Colors.EOC}")
    print(f"  Maximum Accuracy Gap:           {Colors.GREEN}{overfitting_metrics['max_acc_gap']:.6f}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Largest train-validation accuracy gap observed during training{Colors.EOC}")
        print(f"{Colors.BLUE}       Shows worst overfitting point{Colors.EOC}")
    print(f"  Maximum Loss Gap:               {Colors.GREEN}{overfitting_metrics['max_loss_gap']:.6f}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → Largest validation-training loss gap observed{Colors.EOC}")

    if overfitting_metrics['overfitting_onset_epoch'] > 0:
        print(f"  Overfitting Onset:              {Colors.GREEN}Epoch {overfitting_metrics['overfitting_onset_epoch']}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → First epoch where accuracy gap exceeded 5%{Colors.EOC}")
            print(f"{Colors.BLUE}       Consider stopping training around this point{Colors.EOC}")
    else:
        print(f"  Overfitting Onset:              {Colors.GREEN}Not detected{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → No significant overfitting detected (accuracy gap stayed < 5%){Colors.EOC}")

    print(f"  Validation Accuracy Degradation: {Colors.GREEN}{overfitting_metrics['val_acc_degradation']:.6f}{Colors.EOC}")
    if verbose:
        print(f"{Colors.BLUE}     → How much validation accuracy dropped from its peak{Colors.EOC}")
        print(f"{Colors.BLUE}       Values > 0.02 suggest you trained too long{Colors.EOC}")
        print(f"{Colors.BLUE}       Overfitting : [gap > 15%: Severe], [gap > 10%: Significant], [gap > 5%: Mild], [gap < 5%: Good generalization]{Colors.EOC}")

    # Overfitting assessment
    acc_gap = overfitting_metrics['final_acc_gap']
    if acc_gap > 0.15:
        print(f"  Assessment: {Colors.RED}🔴 Severe overfitting detected!{Colors.EOC}")
    elif acc_gap > 0.10:
        print(f"  Assessment: {Colors.ORANGE}🟠 Significant overfitting detected{Colors.EOC}")
    elif acc_gap > 0.05:
        print(f"  Assessment: {Colors.YELLOW}🟡 Mild overfitting detected{Colors.EOC}")
    else:
        print(f"  Assessment: {Colors.GREEN}🟢 Good generalization{Colors.EOC}")

    # Learning Dynamics
    print(f"\n" + "="*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}LEARNING DYNAMICS{Colors.EOC}")
    print(f"="*50)
    if verbose:
        print(f"{Colors.BLUE}  📈 Learning dynamics show how your model learned over time.{Colors.EOC}")
        print(f"{Colors.BLUE}     Healthy patterns show strong initial learning, sustained middle phase, and gentle final phase.{Colors.EOC}")
    if 'initial_phase_improvement' in learning_dynamics:
        print(f"  Initial Phase Improvement (first 25%): {Colors.GREEN}{learning_dynamics['initial_phase_improvement']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Accuracy gained in early training phase{Colors.EOC}")
            print(f"{Colors.BLUE}       Good: > 0.05, Poor: < 0.01{Colors.EOC}")
        print(f"  Middle Phase Improvement (middle 50%): {Colors.GREEN}{learning_dynamics['middle_phase_improvement']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Accuracy gained during sustained learning phase{Colors.EOC}")
            print(f"{Colors.BLUE}       Shows if model continues learning after initial gains{Colors.EOC}")
        print(f"  Final Phase Improvement (last 25%):    {Colors.GREEN}{learning_dynamics['final_phase_improvement']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Accuracy gained in final training phase{Colors.EOC}")
            print(f"{Colors.BLUE}       Negative values indicate overfitting in late training{Colors.EOC}")

    if 'avg_learning_velocity' in learning_dynamics:
        print(f"  Average Learning Velocity:             {Colors.GREEN}{learning_dynamics['avg_learning_velocity']:.6f}/epoch{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Average rate of accuracy change per epoch{Colors.EOC}")
            print(f"{Colors.BLUE}       Positive = improving, negative = degrading{Colors.EOC}")
        print(f"  Learning Acceleration:                 {Colors.GREEN}{learning_dynamics['learning_acceleration']:.6f}/epoch²{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → How the learning rate itself changed over time{Colors.EOC}")
            print(f"{Colors.BLUE}       Negative values normal as learning slows near convergence{Colors.EOC}")

    if 'total_loss_reduction' in learning_dynamics:
        print(f"  Total Loss Reduction:                  {Colors.GREEN}{learning_dynamics['total_loss_reduction']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → How much validation loss decreased from start to finish{Colors.EOC}")
            print(f"{Colors.BLUE}       Larger positive values indicate more effective training{Colors.EOC}")
        print(f"  Loss Reduction per Epoch:              {Colors.GREEN}{learning_dynamics['loss_reduction_per_epoch']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Average loss reduction achieved per training epoch{Colors.EOC}")
            print(f"{Colors.BLUE}       Measures training efficiency{Colors.EOC}")

    # Performance Consistency
    print(f"\n" + "="*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}PERFORMANCE CONSISTENCY{Colors.EOC}")
    print(f"="*50)
    if verbose:
        print(f"{Colors.BLUE}  📊 Consistency metrics measure how stable your training was.{Colors.EOC}")
        print(f"{Colors.BLUE}     Consistent training leads to more reliable and reproducible results.{Colors.EOC}")
    if 'accuracy_cv' in consistency_metrics:
        print(f"  Accuracy Coefficient of Variation: {Colors.GREEN}{consistency_metrics['accuracy_cv']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Ratio of accuracy standard deviation to mean{Colors.EOC}")
            print(f"{Colors.BLUE}       Lower values (< 0.1) indicate more consistent training{Colors.EOC}")
        print(f"  Recent Accuracy Consistency:       {Colors.GREEN}{consistency_metrics['recent_accuracy_consistency']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → How consistent accuracy was in final 20% of training{Colors.EOC}")
            print(f"{Colors.BLUE}       Values near 1.0 indicate stable convergence{Colors.EOC}")

    if 'loss_cv' in consistency_metrics:
        print(f"  Loss Coefficient of Variation:     {Colors.GREEN}{consistency_metrics['loss_cv']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Ratio of loss standard deviation to mean{Colors.EOC}")
            print(f"{Colors.BLUE}       Lower values indicate more stable loss optimization{Colors.EOC}")

    if 'improvement_ratio' in consistency_metrics:
        print(f"  Improvement Ratio:                 {Colors.GREEN}{consistency_metrics['improvement_ratio']:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Fraction of epochs where validation accuracy improved{Colors.EOC}")
            print(f"{Colors.BLUE}       Higher values (> 0.4) indicate consistent learning progress{Colors.EOC}")

    # Learning progress analysis
    if len(history) >= 10:
        val_acc = history[:, 3]
        recent_val_acc = np.mean(val_acc[-5:])
        early_val_acc = np.mean(val_acc[:5])
        improvement = recent_val_acc - early_val_acc

        print(f"\n" + "="*50)
        print(f"{Colors.YELLOW}{Colors.BOLD}LEARNING PROGRESS SUMMARY{Colors.EOC}")
        print(f"="*50)
        if verbose:
            print(f"{Colors.BLUE}  📋 Overall learning progress from start to finish of training.{Colors.EOC}")
            print(f"{Colors.BLUE}     Compares early performance to recent performance to assess total improvement.{Colors.EOC}")
        print(f"  Early Validation Accuracy (first 5 epochs): {Colors.GREEN}{early_val_acc:.6f}{Colors.EOC}")
        print(f"  Recent Validation Accuracy (last 5 epochs): {Colors.GREEN}{recent_val_acc:.6f}{Colors.EOC}")
        print(f"  Total Improvement:                          {Colors.GREEN}{improvement:.6f}{Colors.EOC}")
        if verbose:
            print(f"{Colors.BLUE}     → Difference between recent and early validation accuracy{Colors.EOC}")
            print(f"{Colors.BLUE}       Shows total learning achieved during training{Colors.EOC}")
            print(f"{Colors.BLUE}       [improvement > 0.1: Excellent], [> 0.05: Good], [> 0.01: Moderate], [<= 0.01: Poor]{Colors.EOC}")

        if improvement > 0.1:
            print(f"  Progress Assessment: {Colors.GREEN}🟢 Excellent learning progress{Colors.EOC}")
        elif improvement > 0.05:
            print(f"  Progress Assessment: {Colors.YELLOW}🟡 Good learning progress{Colors.EOC}")
        elif improvement > 0.01:
            print(f"  Progress Assessment: {Colors.ORANGE}🟠 Moderate learning progress{Colors.EOC}")
        else:
            print(f"  Progress Assessment: {Colors.RED}🔴 Poor learning progress{Colors.EOC}")

    # Training recommendations
    print(f"\n" + "="*50)
    print(f"{Colors.YELLOW}{Colors.BOLD}TRAINING RECOMMENDATIONS{Colors.EOC}")
    print(f"="*50)
    if verbose:
        print(f"{Colors.BLUE}  💡 Actionable suggestions based on your training patterns.{Colors.EOC}")
        print(f"{Colors.BLUE}     These recommendations can help improve your next training run.{Colors.EOC}")

    recommendations = []

    if overfitting_metrics['final_acc_gap'] > 0.1:
        recommendations.append("• Consider reducing model complexity or adding regularization")
        recommendations.append("• Try early stopping at peak validation performance")

    if convergence_metrics['loss_plateau_epochs'] > epochs * 0.3:
        recommendations.append("• Consider increasing learning rate for faster convergence")

    if learning_dynamics.get('final_phase_improvement', 0) < 0.001 and epochs < 50:
        recommendations.append("• Model may benefit from more training epochs")

    if consistency_metrics.get('improvement_ratio', 0) < 0.3:
        recommendations.append("• Consider adjusting learning rate schedule or optimizer")

    if len(recommendations) == 0:
        recommendations.append(f"{Colors.GREEN}🎉 Training appears well-optimized!{Colors.EOC}")

    for rec in recommendations:
        print(f"  {rec}")

    print(f"\n" + "="*80)

def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for training history analysis.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze training history")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="trained_model/gradient_descent.npy",
        help="Path to trained model file (default: trained_model/gradient_descent.npy)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed descriptions for each metric"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    analyze_training_history(args.model, args.verbose)
