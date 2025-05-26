import sys
import os
import datetime

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
    Run an Auto ARIMA model on the specified target variable.

    This function performs the following steps:
    1. Loads a processed time series dataset
    2. Splits data into training and test sets
    3. Automatically selects ARIMA hyperparameters using pmdarima
    4. Forecasts the test period
    5. Evaluates the forecast using MAE and R² metrics
    6. Saves the prediction plot and metrics to the reports directory

    Args:
        filepath (str): Path to the processed dataset CSV file.
        target (str): Column name to be predicted (e.g., 'Gulf').

    Returns:
        model: Fitted pmdarima ARIMA model
    """
    # === Load and prepare data ===
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', target]].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    # === Train/Test Split ===
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    # === Fit Auto ARIMA ===
    model = auto_arima(
        train['y'],
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action='ignore'
    )
    print(f'Best ARIMA Order: {model.order}, Seasonal Order: {model.seasonal_order}')

    # === Forecast ===
    forecast = model.predict(n_periods=len(test))

    # === Evaluation ===
    df_compare = test.copy()
    df_compare['predicted'] = forecast
    mae = mean_absolute_error(df_compare['y'], df_compare['predicted'])
    r2 = r2_score(df_compare['y'], df_compare['predicted'])

    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'R² Score: {r2:.3f}')

    # === Plotting ===
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

    # === Save Outputs ===
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    plot_path = os.path.join(results_dir, f'{target}_auto_arima_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Auto ARIMA ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best ARIMA Order: {model.order}, Seasonal Order: {model.seasonal_order}\n')
        f.write(f'Auto ARIMA MAE: {mae:.2f}\n')
        f.write(f'Auto ARIMA R²: {r2:.3f}\n')

    return model
