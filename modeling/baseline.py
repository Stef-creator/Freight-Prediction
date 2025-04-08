import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import datetime

def evaluate_mean_baseline(filepath='data/processed/processed.csv', target='Gulf'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Shift target one week ahead
    df[f'{target}_target'] = df[target].shift(-1)
    df = df.dropna(subset=[target, f'{target}_target'])

    # Use mean of training data as naive forecast
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    mean_train = train_df[target].mean()
    y_test = test_df[f'{target}_target']
    y_pred = np.full_like(y_test, fill_value=mean_train, dtype=float)

    mae = mean_absolute_error(y_test, y_pred)

    print(f"🔍 Baseline Mean Prediction")
    print(f"Train Mean {target}: {mean_train:.2f}")
    print(f"MAE on Test Set: {mae:.2f}")

    # Save result
    results_dir = "reports/models"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Mean Baseline ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Baseline Mean MAE: {mae:.2f}\n')

