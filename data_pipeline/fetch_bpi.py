import pandas as pd

def fetch_bpi(filepath = 'data/raw/Baltic Panamax (BPI).xlsx'):
    df = pd.read_excel(filepath)
    df = df.drop(columns=['Index Name', 'Index Short Name', 'ISIN','Change (%)', 'Open', 'High', 'Low', 'Turnover'])
    df = df.rename(columns={
        'Date': 'date',
        'Close': 'bpi'
    })
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df = df[['date', 'bpi']]
    df = df.sort_values('date')
    df.to_csv('data/processed/bpi.csv', index=False)
    return df

