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
