import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler

def run_multi_prophet_model_tuned(filepath='data/processed/processed.csv', target='Gulf', regressors=['bpi', 'PNW', 'brent_price', 'corn_price', 'ships_anchored'], tune=True):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Prepare data for Prophet
    df = df[['date', target] + regressors].dropna()
    df = df.rename(columns={'date': 'ds', target: 'y'})

    # --- Split into train/test ---
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # --- Set up parameter space ---
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

    # --- Try all parameter combinations ---
    for params in param_list:
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=False,
            **params
        )
        for reg in regressors:
            model.add_regressor(reg)

        model.fit(train_df)

        future = df[['ds'] + regressors].copy()
        future = pd.concat([future, test_df[['ds'] + regressors]], ignore_index=True)
        future = future.drop_duplicates(subset='ds').sort_values('ds').reset_index(drop=True)

        forecast = model.predict(future)

        df_compare = df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        test_compare = df_compare[df_compare['ds'].isin(test_df['ds'])]
        mae = mean_absolute_error(test_compare['y'], test_compare['yhat'])

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_forecast = forecast
            best_params = params

    # --- Final evaluation ---
    df_compare = df.merge(best_forecast[['ds', 'yhat']], on='ds', how='left')
    df_compare.rename(columns={'y': 'actual', 'yhat': 'predicted'}, inplace=True)
    test_compare = df_compare[df_compare['ds'].isin(test_df['ds'])]

    final_mae = mean_absolute_error(test_compare['actual'], test_compare['predicted'])
    final_r2 = r2_score(test_compare['actual'], test_compare['predicted'])

    print(f'Multi Prophet ({"Tuned" if tune else "Default"}) - MAE: {final_mae:.2f}')
    print(f'Multi Prophet ({"Tuned" if tune else "Default"}) - R² Score: {final_r2:.3f}')
    if tune:
        print(f'Best Parameters: {best_params}\n')

    # --- Prophet built-in forecast plot ---
    fig = best_model.plot(best_forecast)
    plt.axvline(df_compare['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    plt.legend()
    plt.title('Prophet Forecast with Uncertainty')
    plt.tight_layout()

    # --- Save results ---
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Multi Prophet (Tuned) ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Multi Prophet MAE: {final_mae:.2f}\n')
        f.write(f'Multi Prophet R² Score: {final_r2:.3f}\n')
        if tune:
            f.write(f'Best Parameters: {best_params}\n')

    plot_path = os.path.join(results_dir, f'{target}_Multi_Prophet_prediction_plot_(Tuned).png')
    plt.savefig(plot_path)
    plt.close()

    return best_forecast

