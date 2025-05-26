import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def run_ridge_regression(filepath='data/processed/processed.csv', target='Gulf'):
    # === Load and prepare data ===
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    df[f'{target}_target'] = df[target].shift(-1)
    df = df.drop(columns=['date', target]).dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    # === Time-aware train/test split ===
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # === Scale ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Ridge Regression with Cross-Validation ===
    alphas = np.logspace(-4, 4, 50)
    model = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Best alpha: {model.alpha_:.5f}')
    print(f'Test MAE: {mae:.2f}')
    print(f'R² Score: {r2:.3f}')

    # === Feature importance ===
    feature_importance = pd.Series(model.coef_, index=X.columns)
    feature_importance = feature_importance.sort_values(key=np.abs, ascending=False)

    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    feature_importance.head(20).plot(kind='bar')
    plt.title(f'Ridge Regression: Top Features for {target}')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'Ridge_Coefficients_{target}.png'))
    plt.close()

    # === Prediction plot ===
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'Ridge Regression: 1-Week Ahead {target} Prediction')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{target}_ridge_prediction_plot.png'))
    plt.close()

    # === Save metrics ===
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Ridge Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best Alpha: {model.alpha_:.5f}\n')
        f.write(f'Ridge MAE: {mae:.2f}\n')
        f.write(f'Ridge R² Score: {r2:.3f}\n')

    return model


