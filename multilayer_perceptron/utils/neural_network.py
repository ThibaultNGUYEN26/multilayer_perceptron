import numpy as np

from .neuron import Initialisation, Model, Gradients, Update, Predict
from utils.plot import TrainValidationPlot

def binary_cross_entropy_loss(y_true, y_pred) -> float:
    """
    Calculate binary cross-entropy loss.

    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_pred (np.ndarray): Predicted probabilities for the positive class

    Returns:
        float: Binary cross-entropy loss
    """
    eps = 1e-15
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate binary cross-entropy loss
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class NeuralNetwork:

    def __init__(self, X_train, y_train, X_val, y_val, n_hidden, learning_rate, epochs) -> None:
        """
        Initialize the neural network with training and validation data, hidden layer sizes,
        learning rate, and number of epochs.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels (one-hot encoded).
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels (one-hot encoded).
            n_hidden (tuple): Sizes of hidden layers.
            learning_rate (float): Learning rate for parameter updates.
            epochs (int): Number of training epochs.
        """
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_val       = X_val
        self.y_val       = y_val
        self.n_hidden    = n_hidden
        self.learning_rate = learning_rate
        self.epochs      = epochs

    def accuracy_score(self, y_true, y_pred) -> float:
        """
        Calculate the accuracy score.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            float: Accuracy score
        """
        return np.mean(y_true == y_pred)

    def deep_neural_network(self) -> np.ndarray:
        """
        Train a deep neural network using forward and backward propagation.
        This method initializes the parameters, performs forward propagation,
        computes gradients, updates parameters, and tracks training loss and accuracy.
        It displays training and validation metrics at each epoch and visualizes the results.

        returns:
            np.ndarray: Final parameters of the trained neural network.
        """

        # Define the dimensions of the neural network
        dimensions = list(self.n_hidden)
        dimensions.insert(0, self.X_train.shape[0])
        dimensions.append(self.y_train.shape[0])
        np.random.seed(1)

        # Initialize parameters
        parametres = Initialisation(dimensions).initialisation()

        # Initialize an array to store training history (loss and accuracy for both train and val)
        training_history = np.zeros((int(self.epochs), 4))  # train_loss, train_acc, val_loss, val_acc

        C = len(parametres) // 2
        # Iterate through the number of epochs for training
        for i in range(self.epochs):

            # Forward propagation on training data
            activations = Model(self.X_train, parametres).forward_propagation()
            # Backward propagation
            gradients = Gradients(self.y_train, parametres, activations).back_propagation()
            # Update parameters
            parametres = Update(gradients, parametres, self.learning_rate).update()

            # Training metrics
            Af_train = activations['A' + str(C)]
            training_history[i, 0] = binary_cross_entropy_loss(self.y_train, Af_train)
            y_pred_train = Predict(self.X_train, parametres).predict()
            y_true_train_class = np.argmax(self.y_train, axis=0)
            training_history[i, 1] = self.accuracy_score(y_true_train_class, y_pred_train.flatten())

            # Validation metrics
            activations_val = Model(self.X_val, parametres).forward_propagation()
            Af_val = activations_val['A' + str(C)]
            training_history[i, 2] = binary_cross_entropy_loss(self.y_val, Af_val)
            y_pred_val = Predict(self.X_val, parametres).predict()
            y_true_val_class = np.argmax(self.y_val, axis=0)
            training_history[i, 3] = self.accuracy_score(y_true_val_class, y_pred_val.flatten())

            # Display metrics at each epoch as required
            print(f"Epoch {i+1:3d}/{self.epochs}: "
                  f"Train Loss: {training_history[i, 0]:.4f}, Train Acc: {training_history[i, 1]:.4f}, "
                  f"Val Loss: {training_history[i, 2]:.4f}, Val Acc: {training_history[i, 3]:.4f}")

        # plot training and validation accuracy and loss
        TrainValidationPlot(training_history).plot()

        return parametres
