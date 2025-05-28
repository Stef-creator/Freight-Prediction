import pandas as pd

def fetch_bpi(filepath='data/raw/Baltic Panamax (BPI).xlsx'):
    """
    Load and clean the Baltic Panamax Index (BPI) Excel file.

    Args:
        filepath (str): Path to the raw Excel file.

    Returns:
        pd.DataFrame: Cleaned BPI dataset with columns ['date', 'bpi'].
                      Also saves a processed CSV to 'data/processed/bpi.csv'.
    """
    df = pd.read_excel(filepath)

    df = df.drop(columns=[
        'Index Name', 'Index Short Name', 'ISIN',
        'Change (%)', 'Open', 'High', 'Low', 'Turnover'
    ])

    df = df.rename(columns={
        'Date': 'date',
        'Close': 'bpi'
    })

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df = df[['date', 'bpi']].sort_values('date')

    df.to_csv('data/processed/bpi.csv', index=False)
    return df
