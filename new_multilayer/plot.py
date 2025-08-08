import matplotlib.pyplot as plt
import mplcursors
import numpy as np


def on_key(event):
    """
    Close the figure when 'escape' is pressed.

    Parameters:
    event : The key press event.
    """
    if event.key == 'escape':
        plt.close(event.canvas.figure)


class DistributionPlot:
    """
    Plot distribution of diagnosis labels in a DataFrame.
    This class creates a bar chart showing the count of benign and malignant diagnoses.

    Parameters:
    df : pandas DataFrame
    """
    def __init__(self, df):
        self.df = df.copy()

    def plot(self):
        # Map labels
        self.df['diagnosis_lbl'] = self.df['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})
        counts = self.df['diagnosis_lbl'].value_counts()

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
    """
    Plot horizontal bar chart of feature correlations with diagnosis.
    This class computes the correlation of each feature with the diagnosis label and displays it in a horizontal bar chart.
    The Pearson correlation coefficient is calculated for each feature and gives the value between -1 and 1.
    1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation.

	Parameters:
    df : pandas DataFrame
    """
    def __init__(self, df):
        self.df = df.copy()

    def plot(self):
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

        cursor = mplcursors.cursor(bars, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(f"Correlation: {bars[sel.index].get_width()}")

        ax.margins(y=0.01)

		# Connect key press event to close the plot
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()


class TrainTestDistributionPlot:
    """
    Plot a pie chart showing train vs test diagnosis distributions.
    This class visualizes the distribution of benign and malignant diagnoses in both training and testing datasets.

    Parameters:
    train_df : pandas DataFrame for training data
	test_df : pandas DataFrame for testing data
    """
    def __init__(self, train_df, test_df):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

    def plot(self):
        lbl_map = {'B': 'Benign', 'M': 'Malignant'}
        for df in (self.train_df, self.test_df):
            if 'diagnosis_lbl' not in df.columns:
                df['diagnosis_lbl'] = df['diagnosis'].map(lbl_map)

        counts = [
            self.train_df['diagnosis_lbl'].value_counts().get('Benign', 0),
            self.train_df['diagnosis_lbl'].value_counts().get('Malignant', 0),
            self.test_df['diagnosis_lbl'].value_counts().get('Benign', 0),
            self.test_df['diagnosis_lbl'].value_counts().get('Malignant', 0)
        ]
        labels = ['Train Benign', 'Train Malignant', 'Test Benign', 'Test Malignant']
        colors = ['#51C759', '#EE5E59', '#B8E4A0', '#F5B1AE']

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, _ = ax.pie(counts, labels=labels, startangle=90, colors=colors, wedgeprops=dict(edgecolor='k'))
        ax.set_title("Train vs Test Diagnosis Distribution")

        annot = ax.annotate('', xy=(0, 0), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        annot.set_visible(False)

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
