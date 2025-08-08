import numpy as np

class Initialisation:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def initialisation(self):
        W1 = np.random.randn(self.n1, self.n0)
        b1 = np.random.randn(self.n1, 1)

        W2 = np.random.randn(self.n2, self.n1)
        b2 = np.random.randn(self.n2, 1)

        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        return parameters

class Model:
    def __init__(self, X, parameters):
        self.X = X
        self.W1 = parameters["W1"]
        self.b1 = parameters["b1"]
        self.W2 = parameters["W2"]
        self.b2 = parameters["b2"]

    def forward_propagation(self):
        Z1 = self.W1.dot(self.X) + self.b1
        A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation
        Z2 = self.W2.dot(A1) + self.b2
        A2 = 1 / (1 + np.exp(-Z2))

        activations = {
            "A1": A1,
            "A2": A2
        }

        return activations

class Loss:
    def __init__(self, A, y):
        self.A = A
        self.y = y

    def log_loss(self):
        eps = 1e-15
        A_clipped = np.clip(self.A, eps, 1 - eps)
        loss = - (self.y * np.log(A_clipped) + (1 - self.y) * np.log(1 - A_clipped))
        return np.mean(loss)

class Gradients:
    def __init__(self, X, y, activations, parameters):
        self.A1 = activations["A1"]
        self.A2 = activations["A2"]
        self.W2 = parameters["W2"]
        self.X = X
        self.y = y

    def back_propagation(self):
        m = self.y.shape[1]
        dZ2 = self.A2 - self.y
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * self.A1 * (1 - self.A1)
        dW1 = 1 / m * dZ1.dot(self.X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return gradients

class Update:
    def __init__(self, gradients, parameters, learning_rate):
        self.dW1 = gradients["dW1"]
        self.db1 = gradients["db1"]
        self.dW2 = gradients["dW2"]
        self.db2 = gradients["db2"]

        self.W1 = parameters["W1"]
        self.b1 = parameters["b1"]
        self.W2 = parameters["W2"]
        self.b2 = parameters["b2"]

        self.learning_rate = learning_rate

    def update(self):
        W1 = self.W1 - self.learning_rate * self.dW1
        b1 = self.b1 - self.learning_rate * self.db1
        W2 = self.W2 - self.learning_rate * self.dW2
        b2 = self.b2 - self.learning_rate * self.db2

        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        return parameters

class Predict:
    def __init__(self, X, parameters):
        self.X = X
        self.W1 = parameters["W1"]
        self.b1 = parameters["b1"]
        self.W2 = parameters["W2"]
        self.b2 = parameters["b2"]

    def predict(self):
        activations = Model(self.X, self.parameters).forward_propagation()
        A2 = activations["A2"]
        return A2 >= 0.5
