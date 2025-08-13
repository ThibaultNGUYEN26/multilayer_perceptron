import numpy as np

class Initialisation:
    def __init__(self, dimensions) -> None:
        """
        Initialize the Initialisation class with the dimensions of the neural network.

        Args:
            dimensions (list): List of integers representing the size of each layer.
        """
        self.dimensions = dimensions

    def initialisation(self) -> dict:
        """
        Initializes the parameters for the neural network.

        Returns:
            dict: A dictionary containing the initialized weights and biases for each layer.
        """
        # Initialize parameters for each layer
        parameters = {}
        C = len(self.dimensions)

        # Randomly initialize weights and biases for each layer
        np.random.seed(1)

        # Loop through each layer to initialize weights and biases
        for c in range(1, C):
            parameters['W' + str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c - 1])
            parameters['b' + str(c)] = np.random.randn(self.dimensions[c], 1)

        return parameters


class Model:
    def __init__(self, X, parameters) -> None:
        """
        Initialize the Model class with input features and parameters.

        Args:
            X (np.ndarray): Input features for the neural network.
            parameters (dict): Dictionary containing the weights and biases of the neural network.
        """
        self.X = X
        self.parameters = parameters

    def forward_propagation(self) -> dict:
        """
        Performs forward propagation through the neural network.

        Returns:
            dict: A dictionary containing the activations for each layer.
        """
        # Initialize the activations dictionary
        activations = {'A0': self.X}
        # Number of layers in the neural network
        C = len(self.parameters) // 2

        # Loop through each layer to compute activations
        for c in range(1, C + 1):
            Z = self.parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + self.parameters['b' + str(c)]
            activations['Z' + str(c)] = Z

            # Apply softmax to the output layer, sigmoid to hidden layers
            if c == C:  # Output layer
                activations['A' + str(c)] = self._softmax(Z)
            else:  # Hidden layers
                activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

        return activations

    def _softmax(self, Z):
        """
        Compute the softmax activation function.

        Args:
            Z (np.ndarray): Linear activations of the output layer.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        # Subtract max for numerical stability
        Z_stable = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


class Gradients:
    def __init__(self, y, parameters, activations) -> None:
        """
        Initialize the Gradients class with true labels, parameters, and activations.

        Args:
            y (np.ndarray): True labels for the training data.
            parameters (dict): Dictionary containing the weights and biases of the neural network.
            activations (dict): Dictionary containing the activations from forward propagation.
        """
        self.y = y
        self.parameters = parameters
        self.activations = activations

    def back_propagation(self) -> dict:
        """
        Computes the gradients for the parameters using backpropagation.

        Returns:
            dict: A dictionary containing the gradients for weights and biases.
        """
        # Initialize the gradients dictionary
        m = self.y.shape[1]
        # Number of layers in the neural network
        C = len(self.parameters) // 2

        # For softmax + cross-entropy, the gradient of the output layer is simplified
        dZ = self.activations['A' + str(C)] - self.y
        gradients = {}

        # Loop through each layer to compute gradients
        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = 1 / m * np.dot(dZ, self.activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(self.parameters['W' + str(c)].T, dZ) * self.activations['A' + str(c - 1)] * (1 - self.activations['A' + str(c - 1)])

        return gradients


class Update:
    def __init__(self, gradients, parameters, learning_rate, velocity=None) -> None:
        """
        Initialize the Update class with gradients, parameters, and learning rate.

        Args:
            gradients (dict): Dictionary containing the gradients for weights and biases.
            parameters (dict): Dictionary containing the current weights and biases of the neural network.
            learning_rate (float): Learning rate for updating the parameters.
        """
        self.gradients = gradients
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.velocity = velocity if velocity is not None else {}

    def gradient_descent(self) -> dict:
        """
        Updates the parameters of the neural network using the computed gradients.

        Returns:
            dict: Updated parameters of the neural network.
        """
        # Update the parameters using the gradients and learning rate
        C = len(self.parameters) // 2

        # Loop through each layer to update weights and biases
        for c in range(1, C + 1):
            self.parameters['W' + str(c)] = self.parameters['W' + str(c)] - self.learning_rate * self.gradients['dW' + str(c)]
            self.parameters['b' + str(c)] = self.parameters['b' + str(c)] - self.learning_rate * self.gradients['db' + str(c)]

        return self.parameters


    def nesterov_acceleration_gradient(self, momentum=0.9) -> tuple:
        """
        Updates the parameters using Nesterov accelerated gradient descent.

        Args:
            momentum (float): Momentum factor for Nesterov acceleration (default: 0.9).

        Returns:
            tuple: (updated_parameters, updated_velocity, look_ahead_parameters)
        """
        C = len(self.parameters) // 2

        # Initialize velocity if not exists or if it's empty
        if not hasattr(self, 'velocity') or not self.velocity:
            self.velocity = {}
            for c in range(1, C + 1):
                self.velocity['vW' + str(c)] = np.zeros_like(self.parameters['W' + str(c)])
                self.velocity['vb' + str(c)] = np.zeros_like(self.parameters['b' + str(c)])

        # Store the current parameters for look-ahead computation
        look_ahead_params = {}

        # Loop through each layer to update weights and biases with Nesterov acceleration
        for c in range(1, C + 1):
            # Update velocity
            self.velocity['vW' + str(c)] = momentum * self.velocity['vW' + str(c)] + self.learning_rate * self.gradients['dW' + str(c)]
            self.velocity['vb' + str(c)] = momentum * self.velocity['vb' + str(c)] + self.learning_rate * self.gradients['db' + str(c)]

            # Update parameters
            self.parameters['W' + str(c)] -= self.velocity['vW' + str(c)]
            self.parameters['b' + str(c)] -= self.velocity['vb' + str(c)]

            # Compute look-ahead parameters for next iteration
            look_ahead_params['W' + str(c)] = self.parameters['W' + str(c)] - momentum * self.velocity['vW' + str(c)]
            look_ahead_params['b' + str(c)] = self.parameters['b' + str(c)] - momentum * self.velocity['vb' + str(c)]

        return self.parameters, self.velocity, look_ahead_params


class Predict:
    def __init__(self, X, parameters) -> None:
        """
        Initialize the Predict class with input features and parameters.

        Args:
            X (np.ndarray): Input features for making predictions.
            parameters (dict): Dictionary containing the weights and biases of the neural network.
        """
        self.X = X
        self.parameters = parameters

    def predict(self) -> np.ndarray:
        """
        Makes predictions using the trained neural network.

        Returns:
            np.ndarray: Predicted class indices based on the highest probability from softmax output.
        """
        # Perform forward propagation to get the activations
        activations = Model(self.X, self.parameters).forward_propagation()
        # Get the number of layers
        C = len(self.parameters) // 2
        # Get the activations of the output layer (softmax probabilities)
        Af = activations['A' + str(C)]

        # Return the class with highest probability
        return np.argmax(Af, axis=0).reshape(1, -1)

    def predict_proba(self) -> np.ndarray:
        """
        Returns the class probabilities using the trained neural network.

        Returns:
            np.ndarray: Predicted class probabilities from softmax output.
        """
        # Perform forward propagation to get the activations
        activations = Model(self.X, self.parameters).forward_propagation()
        # Get the number of layers
        C = len(self.parameters) // 2
        # Get the activations of the output layer (softmax probabilities)
        return activations['A' + str(C)]
