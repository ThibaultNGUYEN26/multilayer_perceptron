from data_loader import load_data

from plot import DistributionPlot, CorrelationPlot

def main() -> None:
    """
    Main function to load the dataset and plot distributions and correlations.
    """
    df = load_data()
    print(df.head())
    DistributionPlot(df).plot()
    CorrelationPlot(df).plot()

if __name__ == "__main__":
    main()
