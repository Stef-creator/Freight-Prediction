import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates

def compute_seasonal_features(filepath, column, showplot=False):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Set weekly frequency
    df = df.set_index('date').resample('W-MON').mean().interpolate()

    result = seasonal_decompose(df[column], model='additive', period=52)
    df[f'{column}_trend'] = result.trend
    df[f'{column}_seasonal'] = result.seasonal
    df.reset_index(inplace=True)

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
