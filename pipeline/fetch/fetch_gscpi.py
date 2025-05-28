import pandas as pd

def fetch_gscpi(filepath='data/raw/gscpi_data.xls'):
    """
    Load and clean GSCPI (Global Supply Chain Pressure Index) monthly data.

    Args:
        filepath (str): Path to the raw Excel file downloaded from NY Fed.

    Returns:
        pd.DataFrame: Cleaned GSCPI data with ['date', 'gscpi'].
                      Also saves to 'data/processed/gscpi.csv'.
    """
    df = pd.read_excel(
        filepath,
        sheet_name='GSCPI Monthly Data',
        engine='xlrd',
        skiprows=5,
        header=None
    )

    df = df.iloc[:, :2]
    df.columns = ['date', 'gscpi']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df.to_csv('data/processed/gscpi.csv', index=False)

    return df
