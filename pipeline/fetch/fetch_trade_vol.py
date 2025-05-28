import pandas as pd

def fetch_tvol(filepath='data/raw/CPB-World-Trade-Monitor-December-2024.xlsx'):
    """
    Extract global trade volume from CPB's World Trade Monitor Excel sheet.

    Args:
        filepath (str): Path to the raw Excel file.

    Returns:
        pd.DataFrame: Cleaned monthly trade volume data with ['date', 'trade_vol'].
                      Also saves to 'data/processed/tvol.csv'.
    """
    df = pd.read_excel(filepath, header=None)

    # Extract date and value rows from known positions
    date_row = df.iloc[3, 5:]
    data_row = df.iloc[7, 5:]

    # Convert 'YYYYmMM' to datetime
    date_row = pd.to_datetime(
        date_row.astype(str).str.replace('m', '-'),
        format='%Y-%m',
        errors='coerce'
    )

    df_clean = pd.concat([date_row, data_row], axis=1)
    df_clean = df_clean.rename(columns={
        df_clean.columns[0]: 'date',
        df_clean.columns[1]: 'trade_vol'
    })

    df_clean = df_clean.sort_values('date')
    df_clean.to_csv('data/processed/tvol.csv', index=False)
    return df_clean
