import sys
import os
import datetime

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
    Run an ARIMAX model using contemporaneous exogenous regressors to forecast the target time series.

    This function performs the following:
    1. Loads a preprocessed dataset with the specified target and exogenous variables
    2. Splits the dataset into training and test sets (time-aware)
    3. Tunes ARIMA hyperparameters using auto_arima on the training set
    4. Fits a SARIMAX model using the optimal order
    5. Predicts across the full time range
    6. Evaluates prediction performance on the test set using MAE and R²
    7. Saves plot and metrics to the reports folder

    Args:
        filepath (str): Path to the processed dataset CSV file.
        target (str): Column name of the target variable (e.g., 'Gulf').

    Returns:
        results: Fitted SARIMAX model results object
    """
    # === Load and prepare data ===
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    exog_vars = ['bpi', 'PNW', 'brent_price', 'corn_price', 'ships_anchored']
    df = df[['date', target] + exog_vars].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    # === Train/Test Split ===
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    # === Tune ARIMA order ===
    print('Selecting best ARIMA order using auto_arima...')
    auto_model = auto_arima(
        train['y'],
        exogenous=train[exog_vars],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True
    )
    best_order = auto_model.order
    print(f'Best ARIMA Order: {best_order}')

    # === Fit SARIMAX on full dataset ===
    model = SARIMAX(df['y'], exog=df[exog_vars], order=best_order)
    results = model.fit(disp=False)

    # === Predict full range ===
    df['predicted'] = results.predict(start=0, end=len(df) - 1, exog=df[exog_vars])

    # === Evaluate on test set ===
    test_compare = df.iloc[split_idx:]
    mae = mean_absolute_error(test_compare['y'], test_compare['predicted'])
    r2 = r2_score(test_compare['y'], test_compare['predicted'])

    print(f'\n📊 Evaluation on Test Set:')
    print(f'MAE: {mae:.2f}')
    print(f'R² Score: {r2:.3f}')

    # === Plot ===
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

    # === Save outputs ===
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    plot_path = os.path.join(results_dir, f'{target}_arimax_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- ARIMAX ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best ARIMA Order: {best_order}\n')
        f.write(f'ARIMAX MAE: {mae:.2f}\n')
        f.write(f'ARIMAX R²: {r2:.3f}\n')

    return results
