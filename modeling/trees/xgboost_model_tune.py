import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def run_xgboost_model(filepath='data/processed/processed.csv', target='Gulf'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    df[f'{target}_target'] = df[target].shift(-1)
    df = df.dropna(subset=[f'{target}_target'])
    df = df.drop(columns=['date', target, 'ship_cap', 'gscpi', 'trade_vol', 'ships_waiting', 
                          'bpi_volatility', 'wheat_price', 'brent_price_trend', 
                          'brent_price_seasonal', 'bpi_trend', 'bpi_seasonal'])
    df = df.dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    # --- Hyperparameter Tuning ---
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0]
    }

    xgb = XGBRegressor(objective='reg:squarederror', verbosity=0)
    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=30,
                                 cv=3, scoring='neg_mean_absolute_error', random_state=42,
                                 verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'XGBoost Test MAE: {mae:.2f}')
    print(f'R² Score: {r2:.3f}')
    print("Best Parameters:", search.best_params_)

    # --- Plot predictions ---
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.to_numpy(), label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'XGBoost - Actual vs Predicted {target} Prices')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    plot_path = os.path.join(results_dir, f'{target}_xgboost_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # --- Save metrics ---
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- XGBoost Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'XGBoost MAE: {mae:.2f}\n')
        f.write(f'XGBoost R² Score: {r2:.3f}\n')
        f.write(f'Best Parameters: {search.best_params_}\n')

    return model

run_xgboost_model()