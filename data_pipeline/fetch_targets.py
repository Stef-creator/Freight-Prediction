import pandas as pd

def fetch_targets(filepath = 'data/raw/T&M Grain Transport Cost Index Calculations.xlsx'):    
    df = pd.read_excel(filepath, skiprows=6)
    df = df[['Date', 'Gulf', 'PNW']]
    df = df.rename(columns={
        'Date': 'date',
        'Gulf': 'Gulf',
        'PNW': 'PNW'
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df = df.sort_values('date')
    df.to_csv('data/processed/targets.csv', index=False)
    return df
