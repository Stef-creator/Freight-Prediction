import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def compute_bpi_volatility(filepath='data/processed/bpi.csv', showplots=False):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # 💡 Create volatility: rolling standard deviation over a 4-week window
    df['bpi_volatility'] = df['bpi'].rolling(window=4).std()

    # Plot the volatility
    if showplots == True:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

        # --- First plot: Gulf & PNW ---
        ax1.plot(df['date'], df['bpi'], label='BPI')
        ax1.set_ylabel('Raw BPI')
        ax1.set_title('Raw BPI')
        ax1.legend()
        ax1.grid(True)

        # --- Second plot: BPI or any other metric ---
        ax2.plot(df['date'], df['bpi_volatility'] , label='bpi_volatility', color='darkred')
        ax2.set_ylabel('BPI Volatility (Rolling 4-Week Std Dev)')
        ax2.set_title('BPI Volatility (Rolling 4-Week Std Dev)')
        ax2.legend()
        ax2.grid(True)

        # --- Format shared x-axis ---
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
    df = df.dropna(subset=['bpi_volatility'])
    bpi_volatility = df[['date', 'bpi_volatility']]
    bpi_volatility = bpi_volatility.set_index('date').resample('W-MON').mean().reset_index()

    return bpi_volatility
