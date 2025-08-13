import matplotlib.pyplot as plt
import mplcursors
import numpy as np


def on_key(event) -> None:
    """
    Close the figure when 'escape' is pressed.

    Args:
        event : The key press event.
    """
    if event.key == 'escape':
        plt.close(event.canvas.figure)


class DistributionPlot:
    def __init__(self, df) -> None:
        """
        Initialize the DistributionPlot class with a DataFrame.

        Args:
            df : pandas DataFrame
        """
        self.df = df.copy()

    def plot(self) -> None:
        """
        Plot the distribution of diagnosis labels in the DataFrame.
        This method creates a bar chart showing the count of benign and malignant diagnoses.
        """
        # Map labels
        self.df['diagnosis_lbl'] = self.df['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})
        counts = self.df['diagnosis_lbl'].value_counts()
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color=["#51C759", "#EE5E59"], edgecolor="k", alpha=0.8)
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Diagnosis")

        # Interactive totals
        cursor = mplcursors.cursor(bars, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f"Total: {bars[sel.index].get_height()}")

		# Connect key press event to close the plot
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()


class CorrelationPlot:
    def __init__(self, df) -> None:
        """
        Initialize the CorrelationPlot class with a DataFrame.

        Args:
            df : pandas DataFrame
        """
        self.df = df.copy()

    def plot(self) -> None:
        """
        Plot the correlation of features with the diagnosis label.
        This method creates a horizontal bar chart showing the correlation of each feature with the diagnosis label.
        """
        df = self.df.copy()
        df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
        corrs = df.corr(numeric_only=True)['diagnosis'].drop('diagnosis').sort_values()

        # Color gradient
        normed = np.linspace(0.3, 0.8, len(corrs))
        colors = plt.cm.Reds(normed)

        # Figure size
        height = len(corrs) * 0.3
        fig, ax = plt.subplots(figsize=(8, height))
        bars = ax.barh(corrs.index, corrs.values, color=colors)
        ax.set_xlabel("Correlation with Diagnosis")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Correlation with Diagnosis (Benign=0, Malignant=1)")

        # Add correlation values on bars
        cursor = mplcursors.cursor(bars, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f"Correlation: {bars[sel.index].get_width()}")

        ax.margins(y=0.01)

		# Connect key press event to close the plot
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

class StatisticalSummaryPlot:
    def __init__(self, df) -> None:
        """
        Initialize the StatisticalSummaryPlot class with a DataFrame.

        Args:
            df : pandas DataFrame
        """
        self.df = df.copy()

    def plot(self) -> None:
        """
        Plot statistical summary including means of _mean, _se, and _worst features by diagnosis, plus dataset summary.
        Data is normalized before plotting to show standardized feature distributions.
        """

        # Get numeric features (excluding id and diagnosis)
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_features:
            numeric_features.remove('id')

        if not numeric_features:
            print("No numeric features found for statistical summary.")
            return

        # NORMALIZE THE DATA BEFORE PLOTTING
        df_normalized = self.df.copy()

        # Extract feature data for normalization (transpose for standardization)
        X_raw = df_normalized[numeric_features].values.T  # Shape: (n_features, n_samples)

        # Standardize features: (X - mean) / std
        mean = np.mean(X_raw, axis=1, keepdims=True)
        std = np.std(X_raw, axis=1, keepdims=True)
        std = np.where(std == 0, 1, std)  # Prevent division by zero
        X_normalized = (X_raw - mean) / std

        # Put normalized data back into dataframe
        df_normalized[numeric_features] = X_normalized.T

        # Define base features and suffix groups
        base_features = [
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
        ]
        mean_feats   = [f + '_mean'    for f in base_features if f + '_mean'    in numeric_features]
        se_feats     = [f + '_se'      for f in base_features if f + '_se'      in numeric_features]
        worst_feats  = [f + '_worst'   for f in base_features if f + '_worst'   in numeric_features]

        # Calculate descriptive stats on NORMALIZED data
        benign_stats    = df_normalized[df_normalized['diagnosis'] == 'B'][numeric_features].describe()
        malignant_stats = df_normalized[df_normalized['diagnosis'] == 'M'][numeric_features].describe()

        # Prepare subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        width = 0.35

        # Plot 1: Mean (_mean) comparison
        x1 = np.arange(len(mean_feats))
        mb = benign_stats.loc['mean', mean_feats]
        mm = malignant_stats.loc['mean', mean_feats]
        bars1_benign = ax1.bar(x1 - width/2, mb, width, label='Benign', color='#51C759', alpha=0.7)
        bars1_malignant = ax1.bar(x1 + width/2, mm, width, label='Malignant', color='#EE5E59', alpha=0.7)
        ax1.set_xticks(x1)
        ax1.set_xticklabels(mean_feats, rotation=45, ha='right')
        ax1.set_ylabel('Normalized Mean Value')
        ax1.set_title('Normalized Mean (_mean) Comparison by Diagnosis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # mplcursors for Plot 1
        cursor1 = mplcursors.cursor([bars1_benign, bars1_malignant], hover=True)
        @cursor1.connect("add")
        def on_add1(sel):
            label = 'Benign' if sel.artist == bars1_benign else 'Malignant'
            feature = mean_feats[sel.index]
            value = sel.artist[sel.index].get_height()
            sel.annotation.set_text(f"{label}\n{feature}\n{value:.2f}")

        # Plot 2: SE (_se) comparison
        x2 = np.arange(len(se_feats))
        sb = benign_stats.loc['mean', se_feats]
        sm = malignant_stats.loc['mean', se_feats]
        bars2_benign = ax2.bar(x2 - width/2, sb, width, label='Benign', color='#51C759', alpha=0.7)
        bars2_malignant = ax2.bar(x2 + width/2, sm, width, label='Malignant', color='#EE5E59', alpha=0.7)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(se_feats, rotation=45, ha='right')
        ax2.set_ylabel('Normalized Average _se Value')
        ax2.set_title('Normalized Standard Error (_se) Comparison by Diagnosis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # mplcursors for Plot 2
        cursor2 = mplcursors.cursor([bars2_benign, bars2_malignant], hover=True)
        @cursor2.connect("add")
        def on_add2(sel):
            label = 'Benign' if sel.artist == bars2_benign else 'Malignant'
            feature = se_feats[sel.index]
            value = sel.artist[sel.index].get_height()
            sel.annotation.set_text(f"{label}\n{feature}\n{value:.2f}")

        # Plot 3: Worst (_worst) comparison
        x3 = np.arange(len(worst_feats))
        wb = benign_stats.loc['mean', worst_feats]
        wm = malignant_stats.loc['mean', worst_feats]
        bars3_benign = ax3.bar(x3 - width/2, wb, width, label='Benign', color='#51C759', alpha=0.7)
        bars3_malignant = ax3.bar(x3 + width/2, wm, width, label='Malignant', color='#EE5E59', alpha=0.7)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(worst_feats, rotation=45, ha='right')
        ax3.set_ylabel('Normalized Mean Worst Value')
        ax3.set_title('Normalized Worst (_worst) Comparison by Diagnosis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # mplcursors for Plot 3
        cursor3 = mplcursors.cursor([bars3_benign, bars3_malignant], hover=True)
        @cursor3.connect("add")
        def on_add3(sel):
            label = 'Benign' if sel.artist == bars3_benign else 'Malignant'
            feature = worst_feats[sel.index]
            value = sel.artist[sel.index].get_height()
            sel.annotation.set_text(f"{label}\n{feature}\n{value:.2f}")

        # Plot 4: Dataset summary
        total_features   = len([f for f in base_features if any(
                 f + suffix in numeric_features for suffix in ['_mean', '_se', '_worst'])])
        benign_count     = len(self.df[self.df['diagnosis'] == 'B'])
        malignant_count  = len(self.df[self.df['diagnosis'] == 'M'])
        categories = ['Total Features', 'Benign Samples', 'Malignant Samples']
        values     = [total_features, benign_count, malignant_count]
        colors     = ['#FFA500', '#51C759', '#EE5E59']

        bars = ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Dataset Summary')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 400)
        ax4.set_yticks(np.arange(0, 401, 50))

        # mplcursors for Plot 4
        cursor4 = mplcursors.cursor(bars, hover=True)
        @cursor4.connect("add")
        def on_add4(sel):
            sel.annotation.set_text(f"{categories[sel.index]}: {values[sel.index]}")

        # Close on Escape
        def on_key(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

class TrainValidationDistributionPlot:
    def __init__(self, train_df, val_df) -> None:
        """
        Initialize the TrainValidationDistributionPlot class with training and validation DataFrames.
        """
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()

    def plot(self) -> None:
        """
        Plot the distribution of diagnosis labels in the training and validation DataFrames.
        This method creates a pie chart showing the proportion of benign and malignant diagnoses in both datasets.
        """
        # Map labels
        lbl_map = {'B': 'Benign', 'M': 'Malignant'}
        for df in (self.train_df, self.val_df):
            if 'diagnosis_lbl' not in df.columns:
                df['diagnosis_lbl'] = df['diagnosis'].map(lbl_map)

        # Count occurrences
        counts = [
            self.train_df['diagnosis_lbl'].value_counts().get('Benign', 0),
            self.train_df['diagnosis_lbl'].value_counts().get('Malignant', 0),
            self.val_df['diagnosis_lbl'].value_counts().get('Benign', 0),
            self.val_df['diagnosis_lbl'].value_counts().get('Malignant', 0)
        ]
        # Create a pie chart
        labels = ['Train Benign', 'Train Malignant', 'Val Benign', 'Val Malignant']
        colors = ['#51C759', '#EE5E59', '#B8E4A0', '#F5B1AE']

        # Create a pie chart with hover functionality
        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, _ = ax.pie(counts, labels=labels, startangle=90, colors=colors, wedgeprops=dict(edgecolor='k'))
        ax.set_title("Train vs Validation Diagnosis Distribution")

        annot = ax.annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        annot.set_visible(False)

        # Hover functionality
        def hover(event):
            if event.inaxes == ax:
                for i, w in enumerate(wedges):
                    contains, _ = w.contains(event)
                    if contains:
                        annot.xy = (event.xdata, event.ydata)
                        annot.set_text(f"{labels[i]}: {counts[i]}")
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

		# Connect key press event to close the plot
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()


class TrainAccuracyLossPlot:
    def __init__(self, train_loss, train_acc) -> None:
        """
        Initialize the TrainAccuracyLossPlot class with training loss and accuracy data.

        Args:
            train_loss (list): List of training loss values.
            train_acc (list): List of training accuracy values.
        """
        self.train_loss = train_loss
        self.train_acc = train_acc

    def plot(self) -> None:
        """
        Plot the training loss and accuracy over iterations.
        This method creates a figure with two subplots: one for training loss and one for training accuracy.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        # Plot training loss
        ax1.plot(self.train_loss, label='Training Loss', color='blue')
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss Over Iterations")
        ax1.legend()

        # Plot training accuracy
        ax2.plot(self.train_acc, label='Training Accuracy', color='green')
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy Over Iterations")
        ax2.legend()

        # Connect key press event to close the plot
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()


class TrainValidationPlot:
    def __init__(self, training_history) -> None:
        """
        Initialize the TrainValidationPlot class with training history data.

        Args:
            training_history (np.ndarray): Array with shape (iterations, 4) containing
                                         [train_loss, train_acc, val_loss, val_acc]
        """
        self.training_history = training_history

    def plot(self) -> None:
        """
        Plot the training and validation loss and accuracy over iterations.
        This method creates a figure with two subplots: one for loss and one for accuracy.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        iterations = range(1, len(self.training_history) + 1)

        # Plot loss
        ax1.plot(iterations, self.training_history[:, 0], label='Training Loss', color='blue', linewidth=2)
        ax1.plot(iterations, self.training_history[:, 2], label='Validation Loss', color='red', linewidth=2)
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(iterations, self.training_history[:, 1], label='Training Accuracy', color='blue', linewidth=2)
        ax2.plot(iterations, self.training_history[:, 3], label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Connect key press event to close the plot
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

class ModelComparisonPlot:
    def __init__(self, model1_data, model2_data, model1_name, model2_name) -> None:
        """
        Initialize the plotter with default figure size.

        Args:
            model1_data (dict): Data for the first model containing 'training_history'.
            model2_data (dict): Data for the second model containing 'training_history'.
            model1_name (str): Name of the first model (used for labeling).
            model2_name (str): Name of the second model (used for labeling).
        """
        self.model1_data = model1_data
        self.model2_data = model2_data
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.figsize = (14, 10)


    def plot_comparison(self) -> None:
        """
        Plot learning curves comparison for two models.
        """

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Model Comparison: Learning Curves', fontsize=16, fontweight='bold')

        # Get training histories
        history1 = self.model1_data['training_history']
        history2 = self.model2_data['training_history']

        epochs1 = range(1, len(history1) + 1)
        epochs2 = range(1, len(history2) + 1)

        # Use model file names as labels (remove .npy extension)
        label1 = self.model1_name.replace('.npy', '')
        label2 = self.model2_name.replace('.npy', '')

        # Plot Training Loss
        self._plot_metric(ax1, epochs1, history1[:, 0], epochs2, history2[:, 0], label1, label2, 'Training Loss', 'Loss')
        # Plot Validation Loss
        self._plot_metric(ax2, epochs1, history1[:, 2], epochs2, history2[:, 2], label1, label2, 'Validation Loss', 'Loss')
        # Plot Training Accuracy
        self._plot_metric(ax3, epochs1, history1[:, 1], epochs2, history2[:, 1], label1, label2, 'Training Accuracy', 'Accuracy')
        # Plot Validation Accuracy
        self._plot_metric(ax4, epochs1, history1[:, 3], epochs2, history2[:, 3], label1, label2, 'Validation Accuracy', 'Accuracy')

        # Connect key press event to close the plot
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()

    def _plot_metric(self, ax, epochs1, data1, epochs2, data2, label1, label2, title, ylabel) -> None:
        """
        Helper method to plot a single metric comparison.

        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            epochs1 (range): Range of epochs for the first model.
            data1 (np.ndarray): Data for the first model.
            epochs2 (range): Range of epochs for the second model.
            data2 (np.ndarray): Data for the second model.
            label1 (str): Label for the first model.
            label2 (str): Label for the second model.
            title (str): Title of the plot.
            ylabel (str): Y-axis label.
        """
        ax.plot(epochs1, data1, 'b-', label=label1, linewidth=2)
        ax.plot(epochs2, data2, 'r-', label=label2, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
