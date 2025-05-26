import sys
import os

# Ensure root directory is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from utils.preprocessing import (
    interpolate,
    compute_bpi_volatility,
    compute_seasonal_features
)


def run_all_feature_engineering():
    """
    Run complete feature engineering pipeline:
    - Interpolates Gulf & PNW prices
    - Computes BPI volatility
    - Extracts seasonal + trend components from BPI and Brent
    - Merges all engineered features into one aligned DataFrame
    - Saves final output to 'data/processed/processed.csv'

    Returns:
        pd.DataFrame: Final merged and cleaned dataset
    """

    # --- Seasonal decomposition ---
    bpi_seasonal = compute_seasonal_features(
        filepath='data/processed/bpi.csv',
        column='bpi'
    )

    brent_seasonal = compute_seasonal_features(
        filepath='data/processed/brent.csv',
        column='brent_price'
    )

    # --- Interpolation and volatility ---
    processed = interpolate()
    bpi_volatility = compute_bpi_volatility()

    # --- Date alignment ---
    for df in [processed, bpi_volatility, brent_seasonal, bpi_seasonal]:
        df['date'] = pd.to_datetime(df['date'])

    # --- Merge all features ---
    df = (
        processed
        .merge(bpi_volatility, on='date', how='inner')
        .merge(brent_seasonal, on='date', how='inner')
        .merge(bpi_seasonal, on='date', how='inner')
    )

    df = df.dropna()
    df.to_csv('data/processed/processed.csv', index=False)

    return df


if __name__ == '__main__':
    df = run_all_feature_engineering()
    print('✅ Missing Gulf and PNW interpolated')
    print('✅ Feature engineered!')
    print('✅ Processed data created and saved to data/processed/processed.csv')
