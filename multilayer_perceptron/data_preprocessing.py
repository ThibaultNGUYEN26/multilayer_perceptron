import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Database:
    1. ID number
    2. Diagnosis
Mean Features:
    3. Radius Mean
    4. Texture Mean
    5. Perimeter Mean
    6. Area Mean
    7. Smoothness Mean
    8. Compactness Mean
    9. Concavity Mean
    10. Concave Points Mean
    11. Symmetry Mean
    12. Fractal Dimension Mean
Standard Error (SE) Features:
    13. Radius SE
    14. Texture SE
    15. Perimeter SE
    16. Area SE
    17. Smoothness SE
    18. Compactness SE
    19. Concavity SE
    20. Concave Points SE
    21. Symmetry SE
    22. Fractal Dimension SE
Worst (Largest Mean of Three Largest Values) Features:
    23. Worst Radius
    24. Worst Texture
    25. Worst Perimeter
    26. Worst Area
    27. Worst Smoothness
    28. Worst Compactness
    29. Worst Concavity
    30. Worst Concave Points
    31. Worst Symmetry
    32. Worst Fractal Dimension
"""

# Define column names based on dataset description
column_names = [
    "ID_number", "Diagnosis",
    "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
    "Compactness_Mean", "Concavity_Mean", "ConcavePoints_Mean", "Symmetry_Mean", "FractalDimension_Mean",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE",
    "Compactness_SE", "Concavity_SE", "ConcavePoints_SE", "Symmetry_SE", "FractalDimension_SE",
    "Worst_Radius", "Worst_Texture", "Worst_Perimeter", "Worst_Area", "Worst_Smoothness",
    "Worst_Compactness", "Worst_Concavity", "Worst_ConcavePoints", "Worst_Symmetry", "Worst_FractalDimension"
]

# Load dataset
file_path = "data.csv"
df = pd.read_csv(file_path, names=column_names, header=None)

# Drop ID column
if "ID_number" in df.columns:
    df.drop(columns=["ID_number"], inplace=True)

# Convert Diagnosis to binary (M = 1, B = 0)
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

# Display basic dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values Per Column:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values.")

# **Class Distribution**
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Diagnosis"], palette="coolwarm")
plt.title("Class Distribution (0 = Benign, 1 = Malignant)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

# **Feature Correlation Heatmap**
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# **Pairplot for Selected Features**
sns.pairplot(df, vars=["Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean"], hue="Diagnosis", palette="coolwarm")
plt.show()

# **Distribution of Key Features**
features_to_plot = ["Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean"]
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df, x=feature, hue="Diagnosis", kde=True, palette="coolwarm", bins=30)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

# **Check Highly Correlated Features**
correlation_matrix = df.corr().abs()
high_corr_features = correlation_matrix[correlation_matrix > 0.8].stack().reset_index()
high_corr_features = high_corr_features[high_corr_features['level_0'] != high_corr_features['level_1']]
print("\nHighly Correlated Features (r > 0.8):")
print(high_corr_features.drop_duplicates(subset=['level_0']))

# Split features and labels
X = df.drop(columns=["Diagnosis"])  # Features
y = df["Diagnosis"]  # Labels

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80-20) while preserving class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Display dataset shapes
print(f"\nTrain Set: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Test Set: X_test = {X_test.shape}, y_test = {y_test.shape}")

# Convert train/test sets to DataFrames
train_data = pd.DataFrame(X_train, columns=X.columns)
train_data["Diagnosis"] = y_train.values

test_data = pd.DataFrame(X_test, columns=X.columns)
test_data["Diagnosis"] = y_test.values

# Save train & test datasets
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
print("\nTrain and Test sets saved as CSV files!")
