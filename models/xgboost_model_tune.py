import sys
import os
import datetime
import joblib  

# Add root directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def run_xgboost_model_tuned(filepath='data/processed/processed.csv', target='Gulf'):
    """
    Train and evaluate a tuned XGBoost regression model for 1-week-ahead freight rate forecasting.

    Steps:
    1. Load processed dataset and shift the target variable 1 week forward
    2. Drop redundant or highly collinear columns
    3. Perform time-aware train/test split (80/20)
    4. Scale features using StandardScaler
    5. Tune hyperparameters via RandomizedSearchCV
    6. Train final model and evaluate with MAE and R² metrics
    7. Save results, prediction plot, and trained model to disk

    Args:
        filepath (str): Path to the input CSV file
        target (str): Name of the target column to forecast (default='Gulf')

    Returns:
        XGBRegressor: Trained and tuned XGBoost model
    """
    #  Load and prepare data 
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    df[f'{target}_target'] = df[target].shift(-1)

    drop_cols = ['date', target, 'ship_cap', 'gscpi', 'trade_vol', 'ships_waiting',
                 'bpi_volatility', 'wheat_price', 'brent_price_trend',
                 'brent_price_seasonal', 'bpi_trend', 'bpi_seasonal']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df = df.dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    #  Train/Test split (80/20) 
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    #  Scale 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  Hyperparameter tuning 
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

    model = XGBRegressor(objective='reg:squarederror', verbosity=0)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_absolute_error',
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train_scaled, y_train)
    best_model = search.best_estimator_

    #  Prediction and Evaluation 
    y_pred = best_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'XGBoost Test MAE: {mae:.2f}')
    print(f'XGBoost R² Score: {r2:.3f}')
    print('Best Parameters:', search.best_params_)

    # Define directories
    plots_dir = 'reports/plots'
    metrics_dir = 'reports/models'
    models_dir = 'reports/models_saved'

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save prediction plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'XGBoost Regression: 1-Week Ahead {target} Prediction')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, f'{target}_xgboost_prediction_plot_tuned.png'))
    plt.close()

    # Save the trained model 
    model_path = os.path.join(models_dir, f'{target}_xgboost_model.joblib')
    joblib.dump(best_model, model_path)
    print(f'Model saved to {model_path}')


    # Save evaluation metrics
    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- XGBoost Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best Parameters: {search.best_params_}\n')
        f.write(f'XGBoost MAE: {mae:.2f}\n')
        f.write(f'XGBoost R² Score: {r2:.3f}\n')
        

    return best_model
