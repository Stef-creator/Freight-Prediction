import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import os


def load_dataset(name, parse_date_col='date'):
    """
    Load a processed CSV file from the data/processed folder.

    Args:
        name (str): Name of the dataset (without .csv)
        parse_date_col (str): Column to parse as datetime

    Returns:
        pd.DataFrame
    """
    path = f'data/processed/{name}.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run fetch first.')
    df = pd.read_csv(path, parse_dates=[parse_date_col])
    return df


def load_all_data():
    """
    Load all processed datasets, resample to weekly Mondays,
    forward-fill where needed, and merge into one dataframe.

    Returns:
        pd.DataFrame: Merged dataset saved to data/processed/all_data.csv
    """
    # Load individual datasets
    anchored = load_dataset('anchored')
    bpi = load_dataset('bpi')
    brent = load_dataset('brent')
    capacity = load_dataset('capacity')
    corn = load_dataset('corn')
    gscpi = load_dataset('gscpi')
    targets = load_dataset('targets')
    tvol = load_dataset('tvol')
    waiting = load_dataset('waiting')
    wheat = load_dataset('wheat')

    # Resample to weekly frequency
    bpi = bpi.set_index('date').resample('W-MON').mean().reset_index()
    brent = brent.set_index('date').resample('W-MON').mean().reset_index()
    corn = corn.set_index('date').resample('W-MON').mean().reset_index()
    wheat = wheat.set_index('date').resample('W-MON').mean().reset_index()

    gscpi = gscpi.set_index('date').resample('W-MON').ffill().reset_index()
    tvol = tvol.set_index('date').resample('W-MON').ffill().reset_index()

    anchored = anchored.set_index('date').resample('W-MON').ffill().reset_index()
    capacity = capacity.set_index('date').resample('W-MON').ffill().reset_index()
    waiting = waiting.set_index('date').resample('W-MON').ffill().reset_index()

    targets = targets.set_index('date').resample('W-MON').ffill().reset_index()

    # Merge all datasets on 'date'
    df = (
        anchored
        .merge(brent, on='date', how='outer')
        .merge(bpi, on='date', how='outer')
        .merge(capacity, on='date', how='outer')
        .merge(corn, on='date', how='outer')
        .merge(gscpi, on='date', how='outer')
        .merge(targets, on='date', how='outer')
        .merge(tvol, on='date', how='outer')
        .merge(waiting, on='date', how='outer')
        .merge(wheat, on='date', how='outer')
    )

    # Final sort and save
    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv('data/processed/all_data.csv', index=False)
    print('✅ All datasets merged and saved to data/processed/all_data.csv')

    return df
