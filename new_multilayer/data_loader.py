import pandas as pd

FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
]

COLS = (
    ["id", "diagnosis"]
    + [f"{f}_mean"  for f in FEATURES]
    + [f"{f}_se"    for f in FEATURES]
    + [f"{f}_worst" for f in FEATURES]
)

def load_data(path="data.csv") -> pd.DataFrame:
    """
    Load the dataset from the specified path.
    Args:
        path (str): Path to the CSV file containing the dataset.
    Returns:
        pd.DataFrame: DataFrame containing the dataset with appropriate column names.
    """
    return pd.read_csv(path, names=COLS)
