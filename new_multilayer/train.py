import pandas as pd
from sklearn.model_selection import train_test_split

from plot import DistributionPlot, CorrelationPlot, TrainTestDistributionPlot
from new_multilayer.neural_network import ArtificialNeuron


def main():
    # Define the column names
    features = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]
    cols = ["id", "diagnosis"] \
         + [f"{f}_mean" for f in features] \
         + [f"{f}_se"   for f in features] \
         + [f"{f}_worst" for f in features]

    # Load the data
    df = pd.read_csv("data.csv", names=cols)

    # Quick check
    print(df.head())

    # Generate plots using plot.py classes
    DistributionPlot(df).plot()
    CorrelationPlot(df).plot()

    # Prepare features and target
    X = df.drop(["id", "diagnosis"], axis=1).values
    y = df["diagnosis"].map({"B": 0, "M": 1}).values.reshape(-1, 1)

    # Split the data
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["diagnosis"],
    )

    X_train = train_df.drop(["id", "diagnosis"], axis=1).values
    y_train = train_df["diagnosis"].map({"B": 0, "M": 1}).values.reshape(-1, 1)
    X_test  = test_df .drop(["id", "diagnosis"], axis=1).values
    y_test  = test_df ["diagnosis"].map({"B": 0, "M": 1}).values.reshape(-1, 1)

    TrainTestDistributionPlot(train_df, test_df).plot()

    # Train the artificial neuron
    neuron = ArtificialNeuron(X_train, y_train, learning_rate=0.01, n_iter=100)
    W, b = neuron.artificial_neuron()


if __name__ == "__main__":
    main()
