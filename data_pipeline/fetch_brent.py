import pandas as pd

def fetch_brent(filepath = 'data/raw/Europe_Brent_Spot_Price_FOB.csv'):
    df = pd.read_csv(filepath, skiprows=4)
    df = df.rename(columns={
        'Day': 'date',
        'Europe Brent Spot Price FOB  Dollars per Barrel': 'brent_price'
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'brent_price']]
    df = df.sort_values('date')
    df.to_csv('data/processed/brent.csv', index=False)
    return df
