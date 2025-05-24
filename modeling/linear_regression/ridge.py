import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

def run_ridge_regression(filepath='data/processed/processed.csv', target='Gulf'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Shift target 1 week ahead
    df[f'{target}_target'] = df[target].shift(-1)
    df = df.dropna(subset=[f'{target}_target'])
    df = df.drop(columns=['date', target])
    df = df.dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
    model.fit(X_train_scaled, y_train)

    print(f'Best alpha: {model.alpha_:.5f}')

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Test MAE: {mae:.2f}')
    print(f'R² Score: {r2:.2f}')

    # --- Save results ---
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    # Plot feature importances
    feature_importance = pd.Series(model.coef_, index=X.columns)
    feature_importance = feature_importance.sort_values(key=np.abs, ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title(f'Ridge Regression Feature Importance for {target}')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'Ridge_Coefficients_{target}.png'))
    plt.close()

    # Plot predictions
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

    # Save metrics
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Ridge Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Ridge MAE: {mae:.2f}\n')
        f.write(f'Ridge R² Score: {r2:.3f}\n')

    return model

run_ridge_regression()