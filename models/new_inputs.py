import sys
import os
import datetime
import joblib

# Add project root to sys.path
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
    """
    Train and evaluate an ARIMAX model for 1-week-ahead forecasting with contemporaneous exogenous regressors.

    This function:
    1. Loads processed data containing the target and exogenous variables.
    2. Splits data into train/test sets preserving temporal order.
    3. Uses auto_arima to find the best ARIMA order on training data.
    4. Fits SARIMAX model on the full dataset using the optimal order.
    5. Predicts the full time series including test set.
    6. Evaluates test set predictions with MAE and R².
    7. Saves performance metrics and prediction plot.

    Args:
        filepath (str): Path to the processed dataset CSV.
        target (str): Target variable column name.

    Returns:
        results: Fitted SARIMAX model results object.
    """

    #  Load and prepare data 
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    exog_vars = ['bpi', 'brent_price', 'corn_price', 'brent_price_seasonal', 'ships_anchored']
    df = df[['date', target] + exog_vars].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    #  Train/Test split (80/20) 
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

    #  Fit SARIMAX on full data 
    model = SARIMAX(df['y'], exog=df[exog_vars], order=best_order)
    results = model.fit(disp=False)

    #  Predict full range 
    df['predicted'] = results.predict(start=0, end=len(df) - 1, exog=df[exog_vars])

    #  Evaluate on test set 
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
    plt.title(f'ARIMAX: Actual vs Predicted {target}')
    plt.xlabel('Date')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, f'{target}_arimax_prediction_plot.png'))
    plt.close()

    # Save the trained model 
    model_path = os.path.join(models_dir, f'{target}_arimax_model.joblib')
    joblib.dump(model, model_path)

    # Save metrics
    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- ARIMAX ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best ARIMA Order: {best_order}\n')
        f.write(f'ARIMAX MAE: {mae:.2f}\n')
        f.write(f'ARIMAX R²: {r2:.3f}\n')

    return results

run_arimax_model()