import numpy as np

from .neuron import Initialisation, Model, Gradients, Update, Predict
from utils.plot import TrainValidationPlot

def categorical_cross_entropy_loss(y_true, y_pred) -> float:
    """
    Calculate categorical cross-entropy loss.

    Args:
        y_true (np.ndarray): True labels (one-hot encoded)
        y_pred (np.ndarray): Predicted probabilities (softmax output)

    Returns:
        float: Categorical cross-entropy loss
    """
    # Clip predictions to avoid log(0)
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    # Calculate categorical cross-entropy loss
    return -np.mean(np.sum(y_true * np.log(p), axis=0))

class NeuralNetwork:

    def __init__(self, X_train, y_train, X_val, y_val, n_hidden, learning_rate, epochs, early_stopping=0, optimizer='gradient_descent') -> None:
        """
        Initialize the neural network with training and validation data, hidden layer sizes,
        learning rate, number of epochs, and early stopping patience.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels (one-hot encoded).
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels (one-hot encoded).
            n_hidden (tuple): Sizes of hidden layers.
            learning_rate (float): Learning rate for parameter updates.
            epochs (int): Number of training epochs.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping (0 = disabled).
        """
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_val       = X_val
        self.y_val       = y_val
        self.n_hidden    = n_hidden
        self.learning_rate = learning_rate
        self.epochs      = epochs
        self.early_stopping = early_stopping
        self.optimizer = optimizer

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

    def deep_neural_network(self) -> tuple:
        """
        Train a deep neural network using forward and backward propagation.
        This method initializes the parameters, performs forward propagation,
        computes gradients, updates parameters, and tracks training loss and accuracy.
        It displays training and validation metrics at each epoch and visualizes the results.
        Supports early stopping to prevent overfitting.

        returns:
            tuple: (parameters, training_history) of the trained neural network.
        """

        # Define the dimensions of the neural network
        dimensions = list(self.n_hidden)
        dimensions.insert(0, self.X_train.shape[0])
        dimensions.append(self.y_train.shape[0])
        np.random.seed(1)

        # Initialize parameters
        parameters = Initialisation(dimensions).initialisation()

        # Initialize training history as a list (we don't know final length due to early stopping)
        training_history = []

        C = len(parameters) // 2

        best_val_acc = -np.inf
        stale = 0
        patience = int(self.early_stopping)
        best_params = None
        if self.optimizer == 'nesterov':
            velocity = {}
            look_ahead_params = None

        # Iterate through the number of epochs for training
        for i in range(self.epochs):
            if self.optimizer == 'nesterov':
                if look_ahead_params is not None:
                    forward_params = look_ahead_params
                else:
                    forward_params = parameters
                # Forward propagation on training data with Nesterov momentum
                activations = Model(self.X_train, forward_params).forward_propagation()
                # Backward propagation
                gradients  = Gradients(self.y_train, forward_params, activations).back_propagation()
                # Update parameters using Nesterov acceleration gradient
                parameters, velocity, look_ahead_params = Update(gradients, parameters, self.learning_rate, velocity).nesterov_acceleration_gradient()
            else:
                    # Forward propagation on training data
                    activations = Model(self.X_train, parameters).forward_propagation()
                    # Backward propagation
                    gradients = Gradients(self.y_train, parameters, activations).back_propagation()
                    # Update parameters using gradient descent
                    parameters = Update(gradients, parameters, self.learning_rate).gradient_descent()

            # Training metrics
            if self.optimizer == 'nesterov':
                # Recalculate activations with current parameters for metrics
                train_activations = Model(self.X_train, parameters).forward_propagation()
                Af_train = train_activations['A' + str(C)]
            else:
                Af_train = activations['A' + str(C)]
            train_loss = categorical_cross_entropy_loss(self.y_train, Af_train)
            y_pred_train = Predict(self.X_train, parameters).predict()
            y_true_train_class = np.argmax(self.y_train, axis=0)
            train_acc = self.accuracy_score(y_true_train_class, y_pred_train.flatten())

            # Validation metrics
            activations_val = Model(self.X_val, parameters).forward_propagation()
            Af_val = activations_val['A' + str(C)]
            val_loss = categorical_cross_entropy_loss(self.y_val, Af_val)
            y_pred_val = Predict(self.X_val, parameters).predict()
            y_true_val_class = np.argmax(self.y_val, axis=0)
            val_acc = self.accuracy_score(y_true_val_class, y_pred_val.flatten())

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {k: v.copy() for k,v in parameters.items()}
                stale = 0
            else:
                stale += 1
                if patience > 0 and stale >= patience:
                    print(f"Early stopping at epoch {i+1} (best val_acc={best_val_acc:.4f})")
                    if best_params is not None:
                        parameters = {k: v.copy() for k,v in best_params.items()}
                    break

            # Store metrics in training history
            training_history.append([train_loss, train_acc, val_loss, val_acc])

            # Display metrics at each epoch as required
            print(f"Epoch {i+1:3d}/{self.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Convert training history to numpy array
        training_history = np.array(training_history)

        # plot training and validation accuracy and loss
        TrainValidationPlot(training_history).plot()

        return parameters, training_history
