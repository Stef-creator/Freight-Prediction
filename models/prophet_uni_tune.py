import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler
import warnings
warnings.filterwarnings('ignore')


def run_prophet_model_tuned(filepath='data/processed/processed.csv', target='Gulf'):
    # === Load and prepare data ===
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').resample('W-MON').mean().dropna().reset_index()
    df = df[['date', target]].rename(columns={'date': 'ds', target: 'y'})

    # === Train/Test Split ===
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    # === Parameter grid for tuning ===
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.3],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    param_list = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

    best_model = None
    best_forecast = None
    best_params = None
    best_mae = float('inf')

    # === Tuning loop ===
    for params in param_list:
        model = Prophet(weekly_seasonality=True, yearly_seasonality=False, **params)
        model.fit(train_df)

        future = model.make_future_dataframe(periods=len(test_df), freq='W-MON')
        forecast = model.predict(future)

        df_compare = df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        test_compare = df_compare[df_compare['ds'].isin(test_df['ds'])]
        mae = mean_absolute_error(test_compare['y'], test_compare['yhat'])

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_forecast = forecast
            best_params = params

    # === Final evaluation ===
    df_compare = df.merge(best_forecast[['ds', 'yhat']], on='ds', how='left')
    df_compare.rename(columns={'y': 'actual', 'yhat': 'predicted'}, inplace=True)
    test_compare = df_compare[df_compare['ds'].isin(test_df['ds'])]

    final_mae = mean_absolute_error(test_compare['actual'], test_compare['predicted'])
    final_r2 = r2_score(test_compare['actual'], test_compare['predicted'])

    print(f'Uni Prophet (Tuned) - MAE: {final_mae:.2f}')
    print(f'Uni Prophet (Tuned) - R² Score: {final_r2:.3f}')
    print(f'Best Parameters: {best_params}')

    # === Plot forecast ===
    fig = best_model.plot(best_forecast)
    plt.axvline(df_compare['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    plt.legend()
    plt.title(f'Prophet Forecast: Actual vs Predicted ({target})')
    plt.tight_layout()

    # === Save outputs ===
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Uni Prophet Regression (Tuned) ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Uni Prophet MAE: {final_mae:.2f}\n')
        f.write(f'Uni Prophet R² Score: {final_r2:.3f}\n')
        f.write(f'Best Parameters: {best_params}\n')

    plot_path = os.path.join(results_dir, f'{target}_uni_prophet_tuned_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return best_forecast
