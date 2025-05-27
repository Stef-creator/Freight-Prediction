import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose


def compute_seasonal_features(filepath, column, showplot=False):
    """
    Decompose a time series into trend and seasonal components.

    Args:
        filepath (str): Path to CSV file with a 'date' column and the target column.
        column (str): Column name to decompose (e.g., 'bpi', 'brent_price').
        showplot (bool): If True, display decomposition plots.

    Returns:
        pd.DataFrame: DataFrame with ['date', f'{column}_trend', f'{column}_seasonal']
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.set_index('date')[[column]]  

    # Weekly frequency, interpolate gaps
    df = df.resample('W-MON').mean().interpolate()

    # Decompose using additive model (assumes linear trend + seasonal pattern)
    result = seasonal_decompose(df[column], model='additive', period=52)
    df[f'{column}_trend'] = result.trend
    df[f'{column}_seasonal'] = result.seasonal
    df.reset_index(inplace=True)

    # Optional plot
    if showplot:
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axs[0].plot(df['date'], df[column], label='Original')
        axs[1].plot(df['date'], df[f'{column}_trend'], label='Trend', color='green')
        axs[2].plot(df['date'], df[f'{column}_seasonal'], label='Seasonal', color='orange')
        for ax in axs:
            ax.legend(loc='upper left')
            ax.grid(True)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return df[['date', f'{column}_trend', f'{column}_seasonal']]


def compute_bpi_volatility(filepath='data/processed/bpi.csv', showplots=False):
    """
    Compute and return BPI volatility as a rolling 4-week standard deviation.

    Args:
        filepath (str): Path to a CSV file containing 'date' and 'bpi' columns.
        showplots (bool): If True, display plots of BPI and its volatility.

    Returns:
        pd.DataFrame: Weekly BPI volatility (resampled to Mondays).
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Rolling 4-week standard deviation as volatility
    df['bpi_volatility'] = df['bpi'].rolling(window=4).std()

    if showplots:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)

        # Plot raw BPI
        ax1.plot(df['date'], df['bpi'], label='BPI')
        ax1.set_ylabel('Raw BPI')
        ax1.set_title('Raw BPI')
        ax1.legend()
        ax1.grid(True)

        # Plot rolling volatility
        ax2.plot(df['date'], df['bpi_volatility'], label='BPI Volatility', color='darkred')
        ax2.set_ylabel('4-Week Rolling Std Dev')
        ax2.set_title('BPI Volatility')
        ax2.legend()
        ax2.grid(True)

        # Format x-axis
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Drop NaNs and resample to weekly Mondays
    df = df.dropna(subset=['bpi_volatility'])
    bpi_volatility = df[['date', 'bpi_volatility']]
    bpi_volatility = bpi_volatility.set_index('date').resample('W-MON').mean().reset_index()

    return bpi_volatility

def before_interpolate(filepath='data/processed/all_data.csv', showplot=False):
    """
    Load and visualize Gulf/PNW prices before interpolation.

    Args:
        filepath (str): Path to the CSV file.
        showplot (bool): If True, display a line plot.

    Returns:
        None
    """
    df = pd.read_csv(filepath)
    df = df[(df['date'] >= '2021-07-27') & (df['date'] <= '2024-12-02')]
    df['date'] = pd.to_datetime(df['date'])
    df['Gulf'] = df['Gulf'].str.replace(',', '').astype(float)
    df['PNW'] = df['PNW'].str.replace(',', '').astype(float)

    if showplot:
        plt.figure(figsize=(12, 5))
        plt.plot(df['date'], df['Gulf'], label='Gulf')
        plt.plot(df['date'], df['PNW'], label='PNW')
        plt.xlabel('Date')
        plt.ylabel('Gulf and PNW Freight Price')
        plt.title('Gulf and PNW Freight Price')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return None


def interpolate(filepath='data/processed/all_data.csv', showplot=False):
    """
    Interpolate missing values in Gulf and PNW columns.

    Args:
        filepath (str): Path to the CSV file.
        showplot (bool): If True, plot interpolated values.

    Returns:
        pd.DataFrame: DataFrame with interpolated 'Gulf' and 'PNW'.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[(df['date'] >= '2021-07-27') & (df['date'] <= '2024-12-02')]

    df['Gulf'] = df['Gulf'].astype(str).str.replace(',', '').astype(float)
    df['PNW'] = df['PNW'].astype(str).str.replace(',', '').astype(float)
    df['Gulf'] = df['Gulf'].interpolate()
    df['PNW'] = df['PNW'].interpolate()

    if showplot:
        plt.figure(figsize=(12, 5))
        plt.plot(df['date'], df['Gulf'], label='Gulf')
        plt.plot(df['date'], df['PNW'], label='PNW')
        plt.xlabel('Date')
        plt.ylabel('Freight Price')
        plt.title('Gulf and PNW Freight Prices (Interpolated)')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df
