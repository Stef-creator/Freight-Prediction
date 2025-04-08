import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


def run_arimax_lagged_exog(filepath='data/processed/processed.csv', target='Gulf'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # --- Prepare dataframe ---
    exog_vars = ['bpi_trend', 'PNW', 'brent_price_trend', 'ships_anchored']
    df = df[['date', target] + exog_vars].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    # --- Shift exogenous variables by 1 week backward (lagged predictors) ---
    for col in exog_vars:
        df[col] = df[col].shift(1)

    # Drop rows with NaNs after lagging
    df = df.dropna().reset_index(drop=True)

    # --- Train/Test Split ---
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # --- Auto ARIMA order selection ---
    print('Selecting ARIMA order with lagged exogenous variables...')
    auto_model = auto_arima(train['y'], exogenous=train[exog_vars],
                            seasonal=False, stepwise=True, suppress_warnings=True, trace=True)
    best_order = auto_model.order
    print(f'Best ARIMA Order: {best_order}')

    # --- Fit SARIMAX on full dataset (with lagged exog) ---
    model = SARIMAX(df['y'], exog=df[exog_vars], order=best_order)
    results = model.fit(disp=False)

    # --- Full in-sample prediction ---
    df['predicted'] = results.predict(start=0, end=len(df) - 1, exog=df[exog_vars])

    # --- Evaluation on test period ---
    test_compare = df.iloc[split_idx:]
    mae = mean_absolute_error(test_compare['y'], test_compare['predicted'])
    r2 = r2_score(test_compare['y'], test_compare['predicted'])

    print(f'\n📊 Evaluation on Test Set:')
    print(f'MAE: {mae:.2f}')
    print(f'R²: {r2:.3f}')

    # --- Plot full prediction ---
    plt.figure(figsize=(12, 5))
    plt.plot(df['ds'], df['y'], label='Actual', linewidth=2)
    plt.plot(df['ds'], df['predicted'], label='Predicted', linestyle='--')
    plt.axvline(df['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    plt.title(f'ARIMAX (Lagged Exog): Full Prediction for {target}')
    plt.xlabel('Date')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    plot_path = os.path.join(results_dir, f'{target}_arimax_lagged_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Save metrics
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- ARIMAX Lagged ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'ARIMAX Lagged MAE: {mae:.2f}\n')
        f.write(f'ARIMAX Lagged R²: {r2:.3f}\n')

    return results

