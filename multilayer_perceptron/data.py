from utils.data_loader import load_data

from utils.plot import DistributionPlot, CorrelationPlot, StatisticalSummaryPlot

def main() -> None:
    """
    Main function to load the dataset and plot distributions and correlations.
    """
    df = load_data()
    print(df.head())

    # Basic data exploration plots
    DistributionPlot(df).plot()
    CorrelationPlot(df).plot()
    StatisticalSummaryPlot(df).plot()

if __name__ == "__main__":
    main()
