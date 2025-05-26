import pandas as pd

def fetch_corn(filepath='data/raw/US Corn.xlsx'):
    """
    Load and clean U.S. corn price data from Excel.

    Args:
        filepath (str): Path to the corn price Excel file.

    Returns:
        pd.DataFrame: Cleaned corn price data with ['date', 'corn_price'].
                      Also saves to 'data/processed/corn.csv'.
    """
    df = pd.read_excel(filepath)
    df = df.drop(columns=['Commodity Name', 'Change (%)', 'High', 'Low', 'Open', 'Volume'])
    df = df.rename(columns={'Date': 'date', 'Value': 'corn_price'})
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df = df[['date', 'corn_price']].sort_values('date')
    df.to_csv('data/processed/corn.csv', index=False)
    return df


def fetch_wheat(filepath='data/raw/US Wheat.xlsx'):
    """
    Load and clean U.S. wheat price data from Excel.

    Args:
        filepath (str): Path to the wheat price Excel file.

    Returns:
        pd.DataFrame: Cleaned wheat price data with ['date', 'wheat_price'].
                      Also saves to 'data/processed/wheat.csv'.
    """
    df = pd.read_excel(filepath)
    df = df.drop(columns=['Commodity Name', 'Change (%)', 'High', 'Low', 'Open', 'Volume'])
    df = df.rename(columns={'Date': 'date', 'Value': 'wheat_price'})
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    df = df[['date', 'wheat_price']].sort_values('date')
    df.to_csv('data/processed/wheat.csv', index=False)
    return df
