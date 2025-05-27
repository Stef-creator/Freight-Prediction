import sys
import os
import datetime
import joblib  

# Ensure project root is in sys.path
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
    """
    Run Ridge Regression to forecast 1-week-ahead freight price for the specified target.

    This model:
    1. Loads and preprocesses time series data
    2. Shifts target variable by one step forward (y_{t+1})
    3. Perform time-aware train/test split (80/20)
    4. Scales input features
    5. Fits Ridge regression with cross-validation to select best regularization strength
    6. Evaluates prediction performance using MAE and R²
    7. Saves plots, model file, and evaluation metrics to disk

    Args:
        filepath (str): Path to processed dataset.
        target (str): Column name of the target variable (default='Gulf').

    Returns:
        sklearn.linear_model.RidgeCV: Trained Ridge regression model.
    """

    #  Load and prepare data 
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Shift target one week ahead (forecast y_{t+1})
    df[f'{target}_target'] = df[target].shift(-1)
    df = df.drop(columns=['date', target]).dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    #  Train/Test split (80/20) 
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    #  Scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  Fit RidgeCV model with TimeSeriesSplit 
    alphas = np.logspace(-4, 4, 50)
    model = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=5))
    model.fit(X_train_scaled, y_train)

    #  Predict and evaluate 
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #  Feature importance 
    feature_importance = pd.Series(model.coef_, index=X.columns)
    feature_importance = feature_importance.sort_values(key=np.abs, ascending=False)

    #  Define directories 
    plots_dir = 'reports/plots'
    models_dir = 'reports/models_saved'
    metrics_dir = 'reports/models'

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    #  Save plots 
    plt.figure(figsize=(10, 6))
    feature_importance.head(20).plot(kind='bar')
    plt.title(f'Ridge Regression: Top Features for {target}')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'Ridge_Coefficients_{target}.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'Ridge Regression: 1-Week Ahead {target} Prediction')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{target}_ridge_prediction_plot.png'))
    plt.close()

    #  Save the trained model 
    model_path = os.path.join(models_dir, f'{target}_ridge_model.joblib')
    joblib.dump(model, model_path)

    #  Save metrics 
    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Ridge Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best Alpha: {model.alpha_:.5f}\n')
        f.write(f'Ridge MAE: {mae:.2f}\n')
        f.write(f'Ridge R² Score: {r2:.3f}\n')

    return model
