import pandas as pd


def fetch_awaiting(filepath='data/raw/Containerships Awaiting Berths at U.S. Ports.csv'):
    """
    Load and transform containerships awaiting berth data.

    Args:
        filepath (str): Path to the raw UTF-16 CSV file.

    Returns:
        pd.DataFrame: Processed long-format data with ['date', 'ships_waiting'].
                      Also saves to 'data/processed/waiting.csv'.
    """
    df = pd.read_csv(filepath, encoding='utf-16', delimiter='\t', skiprows=1)
    df = df.rename(columns={df.columns[0]: 'report_date', df.columns[1]: 'port'})
    df_long = df.melt(id_vars=['report_date', 'port'], var_name='date', value_name='ships_waiting')
    df_long['date'] = pd.to_datetime(df_long['date'], format='%m/%d/%y', errors='coerce')
    df_long = df_long.dropna(subset=['date']).drop(columns=['report_date'])
    df_us_ports = df_long[df_long['port'] == 'All U.S. Container Ports'].drop(columns=['port'])
    df_us_ports = df_us_ports.sort_values('date')
    df_us_ports.to_csv('data/processed/waiting.csv', index=False)
    return df_us_ports


def fetch_anchored(filepath='data/raw/Containerships Anchored off US Ports.csv'):
    """
    Load and transform anchored containerships data.

    Args:
        filepath (str): Path to the raw UTF-16 CSV file.

    Returns:
        pd.DataFrame: Aggregated daily data with ['date', 'ships_anchored'].
                      Also saves to 'data/processed/anchored.csv'.
    """
    df = pd.read_csv(filepath, encoding='utf-16', delimiter='\t', skiprows=1)
    df = df.rename(columns={df.columns[0]: 'report_date', df.columns[1]: 'port'})
    df_long = df.melt(id_vars=['report_date', 'port'], var_name='date', value_name='ships_anchored')
    df_long['date'] = pd.to_datetime(df_long['date'], format='%m/%d/%y', errors='coerce')
    df_long = df_long.dropna(subset=['date']).drop(columns=['report_date'])
    df_long = df_long.groupby('date', as_index=False)['ships_anchored'].sum().sort_values('date')
    df_long.to_csv('data/processed/anchored.csv', index=False)
    return df_long


def fetch_capacity(filepath='data/raw/Capacity of Containerships.csv'):
    """
    Load and clean capacity of containerships data.

    Args:
        filepath (str): Path to the raw UTF-16 CSV file.

    Returns:
        pd.DataFrame: Long-format weekly capacity data with ['date', 'ship_cap'].
                      Also saves to 'data/processed/capacity.csv'.
    """
    df = pd.read_csv(filepath, encoding='utf-16', delimiter='\t', skiprows=1)
    df = df.rename(columns={df.columns[0]: 'report_date', df.columns[1]: 'date'})
    df = df.drop(columns=['report_date', 'Year of Date'])
    df_long = df.melt(id_vars=['date'], var_name='month_col', value_name='ship_cap')
    df_long['date'] = pd.to_datetime(df_long['date'], format='%d/%m/%Y', errors='coerce')
    df_long['ship_cap'] = df_long['ship_cap'].str.replace(',', '').astype(float)
    df_long = df_long.drop(columns=['month_col']).dropna(subset=['ship_cap'])
    df_long = df_long.sort_values('date')
    df_long.to_csv('data/processed/capacity.csv', index=False)
    return df_long
