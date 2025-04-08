import pandas as pd

def fetch_corn(filepath = 'data/raw/US Corn.xlsx'):
    df = pd.read_excel(filepath)
    df = df.drop(columns=['Commodity Name', 'Change (%)', 'High', 'Low', 'Open', 'Volume'])
    df = df.rename(columns={
        'Date': 'date',
        'Value': 'corn_price'
    })
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df = df[['date', 'corn_price']]
    df = df.sort_values('date')
    df.to_csv('data/processed/corn.csv', index=False)
    return df

def fetch_wheat(filepath = 'data/raw/US Wheat.xlsx'):
    df = pd.read_excel(filepath)
    df = df.drop(columns=['Commodity Name', 'Change (%)', 'High', 'Low', 'Open', 'Volume'])
    df = df.rename(columns={
        'Date': 'date',
        'Value': 'wheat_price'
    })
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df = df[['date', 'wheat_price']]
    df = df.sort_values('date')
    df.to_csv('data/processed/wheat.csv', index=False)
    return df
