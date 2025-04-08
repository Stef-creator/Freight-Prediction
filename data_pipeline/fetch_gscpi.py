import pandas as pd

def fetch_gscpi(filepath = 'data/raw/gscpi_data.xls'):    
    df = pd.read_excel(filepath, sheet_name= 'GSCPI Monthly Data', engine='xlrd', skiprows=5, header=None)
    df = df.iloc[:, :2]  
    df.columns = ['date', 'gscpi']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df = df.sort_values('date')
    df.to_csv('data/processed/gscpi.csv', index=False)
    return df

