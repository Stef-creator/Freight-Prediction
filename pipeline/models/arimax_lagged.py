import sys
import os
import datetime
import joblib

# Add root path to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


def run_arimax_lagged_exog(filepath='data/processed/processed.csv', target='Gulf'):
    """
    Train and evaluate an ARIMAX model with 1-week lagged exogenous regressors for forecasting.

    Steps:
    1. Load processed dataset selecting target and specific exogenous variables.
    2. Lag all exogenous variables by one week to avoid contemporaneous leakage.
    3. Perform time-aware train/test split (80/20).
    4. Use auto_arima to select optimal ARIMA(p,d,q) parameters on training data.
    5. Fit SARIMAX model on full data with selected order and lagged exog.
    6. Generate in-sample predictions.
    7. Evaluate performance on test set with MAE and R².
    8. Save prediction plot and append results to log file.

    Args:
        filepath (str): Path to processed dataset CSV.
        target (str): Target column name.

    Returns:
        results: SARIMAX fitted results object.
    """

    #  Load and prepare data 
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    exog_vars = ['bpi_trend', 'PNW', 'brent_price_trend', 'ships_anchored']
    df = df[['date', target] + exog_vars].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    #  Lag exogenous variables by 1 week 
    for col in exog_vars:
        df[col] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)

    #  Train/Test Split (80/20) 
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    #  Auto ARIMA for order selection 
    auto_model = auto_arima(
        train['y'],
        exogenous=train[exog_vars],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True
    )
    best_order = auto_model.order

    #  Fit SARIMAX model 
    model = SARIMAX(df['y'], exog=df[exog_vars], order=best_order)
    results = model.fit(disp=False)

    #  Forecast the test period 
    df['predicted'] = results.predict(start=0, end=len(df) - 1, exog=df[exog_vars])

    #  Evaluate test set 
    test_compare = df.iloc[split_idx:]
    mae = mean_absolute_error(test_compare['y'], test_compare['predicted'])
    r2 = r2_score(test_compare['y'], test_compare['predicted'])

    #  Define directories 
    plots_dir = 'reports/plots'
    models_dir = 'reports/models_saved'
    metrics_dir = 'reports/models'

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save prediction plot
    plt.figure(figsize=(12, 5))
    plt.plot(df['ds'], df['y'], label='Actual', linewidth=2)
    plt.plot(df['ds'], df['predicted'], label='Predicted', linestyle='--')
    plt.axvline(df['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    plt.title(f'ARIMAX (Lagged Exog): Actual vs Predicted {target}')
    plt.xlabel('Date')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f'{target}_arimax_lagged_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Save the trained model 
    model_path = os.path.join(models_dir, f'{target}_arimax_lagged_model.joblib')
    joblib.dump(model, model_path)

    # Save metrics
    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- ARIMAX Lagged ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best ARIMA Order: {best_order}\n')
        f.write(f'ARIMAX Lagged MAE: {mae:.2f}\n')
        f.write(f'ARIMAX Lagged R²: {r2:.3f}\n')

    return results
