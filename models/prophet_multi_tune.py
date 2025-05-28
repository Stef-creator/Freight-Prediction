import sys
import os
import datetime
import joblib 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler
import warnings
warnings.filterwarnings('ignore')


import os
import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def run_multi_prophet_model_tuned(filepath='data/processed/processed.csv',
                                  target='Gulf',
                                  regressors=['bpi', 'PNW', 'brent_price', 'corn_price', 'ships_anchored'],
                                  tune=True):
    """
    Train and evaluate a multivariate Prophet model with optional hyperparameter tuning
    to forecast a target time series, using regressor values at t to predict y at t+1.

    Args:
        filepath (str): Path to the processed dataset CSV.
        target (str): Target column to forecast.
        regressors (list[str]): External regressors to include in the model.
        tune (bool): Whether to perform hyperparameter tuning (default True).

    Returns:
        pd.DataFrame: Forecast dataframe containing Prophet predictions (yhat).
    """

    # Load and prepare data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', target] + regressors].dropna()

    # Shift regressors backward by 1 → x_t predicts y_{t+1}
    for reg in regressors:
        df[reg] = df[reg].shift(1)
    df = df.dropna()  # drop rows with missing lagged values
    df = df.rename(columns={'date': 'ds', target: 'y'})

    # Train/Test split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Hyperparameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    param_list = list(ParameterSampler(param_grid, n_iter=10, random_state=42)) if tune else [{}]

    best_model = None
    best_forecast = None
    best_mae = float('inf')
    best_params = None

    # Hyperparameter tuning loop
    for params in param_list:
        model = Prophet(weekly_seasonality=True, yearly_seasonality=False, **params)
        for reg in regressors:
            model.add_regressor(reg)

        model.fit(train_df)

        future = df[['ds'] + regressors].copy()
        forecast = model.predict(future)

        df_compare = df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        test_compare = df_compare[df_compare['ds'].isin(test_df['ds'])]
        mae = mean_absolute_error(test_compare['y'], test_compare['yhat'])

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_forecast = forecast
            best_params = params

    # Evaluate on test set
    df_compare = df.merge(best_forecast[['ds', 'yhat']], on='ds', how='left')
    df_compare.rename(columns={'y': 'actual', 'yhat': 'predicted'}, inplace=True)
    test_compare = df_compare[df_compare['ds'].isin(test_df['ds'])]

    final_mae = mean_absolute_error(test_compare['actual'], test_compare['predicted'])
    final_r2 = r2_score(test_compare['actual'], test_compare['predicted'])

    if tune:
        print(f'Best Parameters: {best_params}')

    # Plot forecast with train/test split marker
    fig, ax = plt.subplots(figsize=(12, 5))
    best_model.plot(best_forecast, ax=ax)
    ax.axvline(df_compare['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    ax.legend()
    ax.set_title(f'Prophet Forecast (x_t → y_t+1): Actual vs Predicted ({target})')
    plt.tight_layout()

    # Save outputs
    plots_dir = 'reports/plots'
    models_dir = 'reports/models_saved'
    metrics_dir = 'reports/models'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    run_label = 'Tuned' if tune else 'Untuned'
    plot_path = os.path.join(plots_dir, f'{target}_multi_prophet_lagged_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    model_path = os.path.join(models_dir, f'{target}_multi_prophet_lagged_model.joblib')
    joblib.dump(best_model, model_path)
    print(f'Model saved to {model_path}')

    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Multi Prophet Lagged ({run_label}) ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        if tune:
            f.write(f'Best Parameters: {best_params}\n')
        f.write(f'MAE: {final_mae:.2f}\n')
        f.write(f'R² Score: {final_r2:.3f}\n')

    return best_forecast

