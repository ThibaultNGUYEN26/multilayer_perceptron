import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from train import forward_propagation

def load_parameters(filepath="trained_params.npy"):
    return np.load(filepath, allow_pickle=True).item()

# Load test data
test_data = pd.read_csv("test_data.csv")
X_test = test_data.drop(columns=['Diagnosis']).values.T
y_test = test_data['Diagnosis'].values.reshape(1, -1)

# Load trained parameters
parameters = load_parameters()

# Perform forward pass
activations = forward_propagation(X_test, parameters)

# The final activation is shape (2, m)
A_final = activations['A' + str(len(parameters) // 2)]  # shape: (2, m)

# Transpose to shape (m, 2), so each row corresponds to a sample
y_pred_proba = A_final.T  # shape: (m, 2)

# Evaluate log-loss (cross-entropy)
loss = log_loss(y_test.flatten(), y_pred_proba, labels=[0,1])

# Convert probabilities to class predictions via argmax
y_pred = np.argmax(y_pred_proba, axis=1)  # shape: (m,)

accuracy = accuracy_score(y_test.flatten(), y_pred)

# Display results
print(f"Test Set Evaluation with Softmax final layer:")
print(f"- Cross-Entropy Loss: {loss:.4f}")
print(f"- Accuracy: {accuracy:.4f}")
