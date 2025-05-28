import pandas as pd

def fetch_brent(filepath='data/raw/Europe_Brent_Spot_Price_FOB.csv'):
    """
    Load and clean Europe Brent Spot Price (FOB) data from EIA CSV.

    Args:
        filepath (str): Path to the raw CSV file downloaded from EIA.

    Returns:
        pd.DataFrame: Cleaned Brent price data with ['date', 'brent_price'].
                      Also saves to 'data/processed/brent.csv'.
    """
    df = pd.read_csv(filepath, skiprows=4)

    df = df.rename(columns={
        'Day': 'date',
        'Europe Brent Spot Price FOB  Dollars per Barrel': 'brent_price'
    })

    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'brent_price']].sort_values('date')
    df.to_csv('data/processed/brent.csv', index=False)

    return df
