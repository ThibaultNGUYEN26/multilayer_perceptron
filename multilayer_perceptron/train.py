import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

# Load train and test sets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Separate features and labels
X_train = train_data.drop(columns=['Diagnosis']).values.T
y_train = train_data['Diagnosis'].values.reshape(1, -1)

X_test = test_data.drop(columns=['Diagnosis']).values.T
y_test = test_data['Diagnosis'].values.reshape(1, -1)

feature_columns = [
    "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
    "Compactness_Mean", "Concavity_Mean", "ConcavePoints_Mean", "Symmetry_Mean", "FractalDimension_Mean",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE",
    "Compactness_SE", "Concavity_SE", "ConcavePoints_SE", "Symmetry_SE", "FractalDimension_SE",
    "Worst_Radius", "Worst_Texture", "Worst_Perimeter", "Worst_Area", "Worst_Smoothness",
    "Worst_Compactness", "Worst_Concavity", "Worst_ConcavePoints", "Worst_Symmetry", "Worst_FractalDimension"
]


def initialisation(dimensions):

    parameters = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parameters


def softmax(Z):

    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))

    return expZ / np.sum(expZ, axis=0, keepdims=True)


def forward_propagation(X, parameters):

    activations = {'A0': X}

    C = len(parameters) // 2

    for c in range(1, C + 1):

        Z = parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameters['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    # Final layer uses softmax
    Z_final = parameters['W' + str(C)].dot(activations['A' + str(C - 1)]) + parameters['b' + str(C)]
    A_final = softmax(Z_final)
    activations['A' + str(C)] = A_final

    return activations


def one_hot_encoding(y, num_classes=2):

    # Flatten y so it becomes shape (m,)
    y_flat = y.flatten().astype(int)
    # Create an identity matrix of size num_classes,
    # and index into it with y_flat
    y_oh = np.eye(num_classes)[y_flat].T  # shape: (num_classes, m)

    return y_oh


def back_propagation(y_oh, parameters, activations):

  m = y_oh.shape[1]
  C = len(parameters) // 2

  dZ = activations['A' + str(C)] - y_oh
  gradients = {}

  for c in reversed(range(1, C + 1)):
    gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
    gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    if c > 1:
      dZ = np.dot(parameters['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

  return gradients


def update(gradients, parameters, learning_rate):

    C = len(parameters) // 2

    for c in range(1, C + 1):
        parameters['W' + str(c)] -= learning_rate * gradients['dW' + str(c)]
        parameters['b' + str(c)] -= learning_rate * gradients['db' + str(c)]

    return parameters


def deep_neural_network(X, y, X_test, y_test, hidden_layers=(16, 16, 16), learning_rate=0.01, n_iter=3000):

    # initialisation parameters
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(2)
    np.random.seed(1)

    y_oh = one_hot_encoding(y, num_classes=2)
    y_test_oh = one_hot_encoding(y_test, num_classes=2)
    parameters = initialisation(dimensions)

    training_history = np.zeros((n_iter, 2))
    testing_history = np.zeros((n_iter, 2))

    C = len(parameters) // 2

    # gradient descent
    for i in range(n_iter):
        activations = forward_propagation(X, parameters)
        gradients = back_propagation(y_oh, parameters, activations)
        parameters = update(gradients, parameters, learning_rate)
        Af_train = activations['A' + str(C)]

        # train_loss = log_loss(y.flatten(), Af_train.flatten())
        train_loss = log_loss(y.flatten(), Af_train.T, labels=[0,1])
        y_pred_train = np.argmax(Af_train, axis=0)
        train_acc = accuracy_score(y.flatten(), y_pred_train)

        # Evaluate on Test Set
        activations_test = forward_propagation(X_test, parameters)
        Af_test = activations_test['A' + str(C)]
        val_loss = log_loss(y_test.flatten(), Af_test.T, labels=[0,1])
        y_pred_test = np.argmax(Af_test, axis=0)
        val_acc = accuracy_score(y_test.flatten(), y_pred_test.flatten())

        training_history[i] = [train_loss, train_acc]
        testing_history[i] = [val_loss, val_acc]

        # Print formatted metrics every epoch
        print(f"epoch {i+1}/{n_iter} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    # Plot Learning Curves
    plt.figure(figsize=(12, 4))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='Train Loss', color='blue')
    plt.plot(testing_history[:, 0], label='Test Loss', linestyle='dashed', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend(loc="upper right")

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='Train Accuracy', color='blue')
    plt.plot(testing_history[:, 1], label='Test Accuracy', linestyle='dashed', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Time")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


    print(f"\nTrain set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")

    print("\nFinal Train Accuracy:", training_history[-1, 1])
    print("Final Test Accuracy:", testing_history[-1, 1])

    # Save trained parameters
    np.save("trained_params.npy", parameters)
    print("\nModel parameters saved to 'trained_params.npy'")

    return parameters

if __name__ == "__main__":
    # Train the Model
    trained_params = deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers=(16, 16, 16), learning_rate=0.1, n_iter=1000)
