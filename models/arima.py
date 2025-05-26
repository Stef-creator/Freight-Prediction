import sys
import os
import datetime
import joblib

# Add project root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


def run_auto_arima_model(filepath='data/processed/processed.csv', target='Gulf'):
    """
    Train and evaluate an Auto ARIMA model for 1-step ahead forecasting of a target time series.

    Steps:
    1. Load the processed time series dataset.
    2. Perform time-aware train/test split (80/20).
    3. Automatically select ARIMA model order using pmdarima's auto_arima on training data.
    4. Forecast the test period.
    5. Evaluate forecast using MAE and R² metrics.
    6. Save prediction plot and metrics to reports directory.

    Args:
        filepath (str): Path to the processed dataset CSV.
        target (str): Target column name (default 'Gulf').

    Returns:
        model: Fitted pmdarima Auto ARIMA model.
    """

    #  Load and prepare data 
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', target]].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    #  Train/Test Split (80/20) 
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    #  Fit Auto ARIMA model 
    model = auto_arima(
        train['y'],
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action='ignore'
    )

    #  Forecast the test period 
    forecast = model.predict(n_periods=len(test))

    #  Evaluate forecast 
    df_compare = test.copy()
    df_compare['predicted'] = forecast
    mae = mean_absolute_error(df_compare['y'], df_compare['predicted'])
    r2 = r2_score(df_compare['y'], df_compare['predicted'])

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
    plt.plot(df_compare['ds'], df_compare['predicted'], label='Predicted', linestyle='--')
    plt.axvline(df['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    plt.title(f'Auto ARIMA: Actual vs Predicted {target}')
    plt.xlabel('Date')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f'{target}_auto_arima_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Save the trained model 
    model_path = os.path.join(models_dir, f'{target}_arima_model.joblib')
    joblib.dump(model, model_path)

    # Save metrics
    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Auto ARIMA ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best ARIMA Order: {model.order}, Seasonal Order: {model.seasonal_order}\n')
        f.write(f'Auto ARIMA MAE: {mae:.2f}\n')
        f.write(f'Auto ARIMA R²: {r2:.3f}\n')

    return model
