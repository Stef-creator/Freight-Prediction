import pandas as pd

def fetch_targets(filepath='data/raw/T&M Grain Transport Cost Index Calculations.xlsx'):
    """
    Load and clean target freight price data (Gulf and PNW) from Excel.

    Args:
        filepath (str): Path to the raw Excel file.

    Returns:
        pd.DataFrame: Cleaned target data with ['date', 'Gulf', 'PNW'].
                      Also saves to 'data/processed/targets.csv'.
    """
    df = pd.read_excel(filepath, skiprows=6)
    df = df[['Date', 'Gulf', 'PNW']]

    df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    df.to_csv('data/processed/targets.csv', index=False)
    return df
