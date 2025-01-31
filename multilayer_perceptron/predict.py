import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from main import forward_propagation

# Load the trained model parameters
def load_parameters(filepath="trained_params.npy"):
    return np.load(filepath, allow_pickle=True).item()

# Load test data
test_data = pd.read_csv("test_data.csv")
X_test = test_data.drop(columns=['Diagnosis']).values.T
y_test = test_data['Diagnosis'].values.reshape(1, -1)

# Load trained parameters
parameters = load_parameters()

# Perform predictions
activations = forward_propagation(X_test, parameters)
y_pred_proba = activations['A' + str(len(parameters) // 2)]
y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary labels

# Compute binary cross-entropy loss
loss = log_loss(y_test.flatten(), y_pred_proba.flatten())

# Compute accuracy
accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())

# Display results
print(f"Test Set Evaluation:")
print(f"- Binary Cross-Entropy Loss: {loss:.4f}")
print(f"- Accuracy: {accuracy:.4f}")
