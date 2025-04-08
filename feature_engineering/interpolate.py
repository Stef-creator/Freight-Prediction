import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def heat_map(filepath = 'data/processed/all_data.csv'):
    df = pd.read_csv(filepath)
    sns.heatmap(df.isna(), cbar=False)
    plt.title('Missing Data Heatmap')
    plt.show()
    return df

def before_interpolate(filepath = 'data/processed/all_data.csv', showplot = False):
    df = pd.read_csv(filepath)
    df = df[(df['date'] >= '2021-07-27') & (df['date'] <= '2024-12-02')]
    df['date'] = pd.to_datetime(df['date']) 
    df['Gulf'] = df['Gulf'].str.replace(',', '').astype(float)
    df['PNW'] = df['PNW'].str.replace(',', '').astype(float)

    if showplot == True:
        plt.figure(figsize=(12, 5))

        plt.plot(df['date'], df['Gulf'], label='Gulf')
        plt.plot(df['date'], df['PNW'], label='PNW')

        plt.xlabel('Date')
        plt.ylabel('Gulf and PNW Freight Price')
        plt.title('Gulf and PNW Freight Price')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Every 2 months
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as 'Jan 2022'
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return


def interpolate(filepath='data/processed/all_data.csv', showplot=False):
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

