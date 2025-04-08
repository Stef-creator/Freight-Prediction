import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


def run_arimax_model(filepath='data/processed/processed.csv', target='Gulf'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # --- Prepare dataframe ---
    exog_features = ['bpi', 'PNW', 'brent_price', 'corn_price', 'ships_anchored']
    df = df[['date', target] + exog_features].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    # --- Train/Test Split ---
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # --- Auto ARIMA order selection ---
    print('Running auto_arima to determine best order...')
    auto_model = auto_arima(train['y'], exogenous=train[exog_features], seasonal=False,
                             stepwise=True, suppress_warnings=True, trace=True)

    best_order = auto_model.order
    print(f'Best ARIMA Order: {best_order}')

    # --- Fit SARIMAX model on full data ---
    model = SARIMAX(df['y'], exog=df[exog_features], order=best_order)
    results = model.fit(disp=False)

    # --- Predict full dataset (in-sample) ---
    forecast = results.predict(start=0, end=len(df) - 1, exog=df[exog_features])
    df['predicted'] = forecast

    # --- Evaluation on test set only ---
    test_compare = df.iloc[split_idx:]
    mae = mean_absolute_error(test_compare['y'], test_compare['predicted'])
    r2 = r2_score(test_compare['y'], test_compare['predicted'])
    print(f'Mean Absolute Error (MAE) on Test Set: {mae:.2f}')
    print(f'R² Score on Test Set: {r2:.3f}')

    # --- Plot full predictions ---
    plt.figure(figsize=(12, 5))
    plt.plot(df['ds'], df['y'], label='Actual', linewidth=2)
    plt.plot(df['ds'], df['predicted'], label='Predicted', linestyle='--')
    plt.axvline(df['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    plt.title(f'ARIMAX: Actual vs Predicted {target}')
    plt.xlabel('Date')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    plot_path = os.path.join(results_dir, f'{target}_arimax_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Save metrics
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- ARIMAX ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'ARIMAX MAE: {mae:.2f}\n')
        f.write(f'ARIMAX R²: {r2:.3f}\n')

    return results

