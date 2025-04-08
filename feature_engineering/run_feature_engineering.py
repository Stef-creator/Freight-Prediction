import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from feature_engineering.interpolate import interpolate
from feature_engineering.generate_volatility import compute_bpi_volatility
from feature_engineering.seasonal_decomp import compute_seasonal_features

def run_all_feature_engineering():   

    bpi_seasonal = compute_seasonal_features(
        filepath='data/processed/bpi.csv',
        column='bpi',
    )

    brent_seasonal = compute_seasonal_features(
    filepath='data/processed/brent.csv',
    column='brent_price',
    )
    
    processed, bpi_volatility = interpolate(), compute_bpi_volatility()

    processed['date'] = pd.to_datetime(processed['date'])
    bpi_volatility['date'] = pd.to_datetime(bpi_volatility['date'])
    brent_seasonal['date'] = pd.to_datetime(brent_seasonal['date'])
    bpi_seasonal['date'] = pd.to_datetime(bpi_seasonal['date'])    

    df = processed \
        .merge(bpi_volatility, on='date', how='inner') \
        .merge(brent_seasonal, on='date', how='inner') \
        .merge(bpi_seasonal, on='date', how='inner') \
    
    df = df.dropna()
    df.to_csv('data/processed/processed.csv', index=False)

    return df




if __name__ == '__main__':
    df = run_all_feature_engineering()
    print('✅ Missing Gulf and PNW interpolated \n✅ Feature engineered! \n✅ Processed data created!')