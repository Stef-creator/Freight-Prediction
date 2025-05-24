import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score

def run_prophet_model(filepath='data/processed/processed.csv', target='Gulf'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Resample to weekly Monday and drop missing
    df = df.set_index('date').resample('W-MON').mean().dropna().reset_index()

    # Prepare data for Prophet
    df = df[['date', target]].rename(columns={'date': 'ds', target: 'y'})

    # --- Split into train/test ---
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # --- Fit model on training ---
    model = Prophet(weekly_seasonality=True, yearly_seasonality=False)
    model.fit(train_df)

    # --- Forecast into the test period ---
    future = model.make_future_dataframe(periods=len(test_df), freq='W-MON')
    forecast = model.predict(future)

    # --- Merge actuals and predictions ---
    df_compare = df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    df_compare.rename(columns={'y': 'actual', 'yhat': 'predicted'}, inplace=True)

    # --- Prophet built-in forecast plot ---
    fig = model.plot(forecast)
    plt.axvline(df_compare['ds'].iloc[split_idx], color='red', linestyle=':', label='Train/Test Split')
    plt.legend()
    plt.title('Prophet Forecast with Uncertainty')
    plt.tight_layout()

    # --- Evaluate only on test period ---
    test_compare = df_compare[df_compare['ds'].isin(test_df['ds'])]
    mae = mean_absolute_error(test_compare['actual'], test_compare['predicted'])
    r2 = r2_score(test_compare['actual'], test_compare['predicted'])

    print(f'Uni Prophet: Mean Absolute Error (MAE) on Test Set: {mae:.2f}')
    print(f'Uni Prophet: R² Score on Test Set: {r2:.3f}\n')

    # --- Save results ---
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Uni Prophet Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Uni Prophet MAE: {mae:.2f}\n')
        f.write(f'Uni Prophet R² Score: {r2:.3f}\n')

    # Save plot
    plot_path = os.path.join(results_dir, f'{target}_uni_prophet_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return forecast

run_prophet_model()
