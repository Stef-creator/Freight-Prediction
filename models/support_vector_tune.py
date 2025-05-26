import sys
import os
import datetime
import joblib 

# Add root directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import uniform
import warnings
warnings.filterwarnings('ignore')


def run_svm_regression_tuned(filepath='data/processed/processed.csv', target='Gulf'):
    """
    Train and evaluate a Support Vector Regression (SVR) model with randomized hyperparameter tuning
    for 1-week-ahead freight price prediction.

    Steps:
    1. Load preprocessed dataset and shift target column by one step forward
    2. Drop selected features not useful for SVM or collinear
    3. Perform time-aware train/test split (80/20)
    4. Scale features using StandardScaler
    5. Tune SVR using randomized search over hyperparameters
    6. Evaluate model performance on the test set using MAE and R²
    7. Save plots, metrics, and trained model to disk

    Args:
        filepath (str): Path to processed input dataset (CSV)
        target (str): Name of the target column (default='Gulf')

    Returns:
        sklearn.svm.SVR: Trained SVR model with best parameters
    """

    # Load and preprocess data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Shift target forward 1 week to predict next week
    df[f'{target}_target'] = df[target].shift(-1)

    # Drop columns redundant or highly collinear with SVR
    drop_cols = [
        'date', target, 'ship_cap', 'gscpi', 'trade_vol', 'ships_waiting',
        'bpi_volatility', 'wheat_price', 'brent_price_trend',
        'brent_price_seasonal', 'bpi_trend', 'bpi_seasonal'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df = df.dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    # Time-series aware train/test split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale features to zero mean, unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter search space for SVR
    param_dist = {
        'C': uniform(0.1, 100),         # Regularization strength
        'epsilon': uniform(0.01, 1.0),  # Tube width
        'gamma': ['scale', 'auto']      # Kernel coefficient
    }

    svr = SVR(kernel='linear')
    search = RandomizedSearchCV(
        svr,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_mean_absolute_error',
        cv=TimeSeriesSplit(n_splits=5),
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train_scaled, y_train)
    model = search.best_estimator_

    # Model evaluation
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'SVM Test MAE: {mae:.2f}')
    print(f'SVM R² Score: {r2:.3f}')
    print(f'Best SVM Params: {search.best_params_}')

    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'SVM Regression: 1-Week Ahead {target} Prediction')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Define directories
    plots_dir = 'reports/plots'
    models_dir = 'reports/models_saved'
    metrics_dir = 'reports/models'

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Save prediction plot
    plot_path = os.path.join(plots_dir, f'{target}_svm_prediction_plot_tuned.png')
    plt.savefig(plot_path)
    plt.close()

    # Save evaluation metrics
    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Support Vector Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Best SVM Params: {search.best_params_}\n')
        f.write(f'SVM MAE: {mae:.2f}\n')
        f.write(f'SVM R² Score: {r2:.3f}\n')
        

    # Save trained model for reuse
    model_path = os.path.join(models_dir, f'{target}_svm_model.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

    return model
