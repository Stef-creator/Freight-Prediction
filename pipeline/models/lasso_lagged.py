import sys
import os
import datetime
import joblib

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def run_lasso_with_lags(filepath='data/processed/processed.csv', target='Gulf', max_lag=5):
    """
    Train and evaluate a Lasso regression model using lagged features for 1-week-ahead forecasting.

    This function:
    1. Loads processed data and generates lagged features for all predictors.
    2. Perform time-aware train/test split (80/20)
    3. Scales features using StandardScaler.
    4. Trains LassoCV with TimeSeriesSplit cross-validation.
    5. Evaluates model on test set with MAE and R².
    6. Saves coefficient summary, prediction plots, and metrics.

    Args:
        filepath (str): Path to processed dataset CSV.
        target (str): Target variable column name.
        max_lag (int): Number of lag periods to create for all predictors.
        showplot (bool): Whether to generate and save plots.

    Returns:
        lasso (LassoCV): Trained LassoCV model object.
        coef_df (pd.DataFrame): DataFrame of feature names and coefficients sorted by absolute value.
    """

    #  Load and preprocess data 
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Generate lagged features
    lag_features = df.columns.difference(['date'])
    lagged_dfs = []
    for lag in range(1, max_lag + 1):
        lagged = df[lag_features].shift(lag).add_suffix(f'_lag{lag}')
        lagged_dfs.append(lagged)

    # Combine original and lagged features
    df = pd.concat([df] + lagged_dfs, axis=1)
    df[f'{target}_target'] = df[target]
    df = df.dropna()

    # Split into predictors and target
    X = df[[col for col in df.columns if '_lag' in col]]
    y = df[f'{target}_target']

    #  Train/Test split (80/20) 
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    #  Scale predictors 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  Train LassoCV model 
    model = LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=42)
    model.fit(X_train_scaled, y_train)

    #  Predict full range and Evaluate on test set 
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #  Create coefficient summary DataFrame 
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    #  Define directories 
    plots_dir = 'reports/plots'
    models_dir = 'reports/models_saved'
    metrics_dir = 'reports/models'

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    
    # Save prediction plot and top coefficients
    plt.figure(figsize=(10, 5))
    sns.barplot(data=coef_df.head(20), x='Coefficient', y='Feature', orient='h')
    plt.title(f'Top Lasso Coefficients for {target}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'Lasso_Coefficients_{target}.png'))
    plt.close()

    
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'Lasso Regression: Actual vs Predicted {target}')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{target} Freight Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{target}_lasso_prediction_lagged_plot.png'))
    plt.close()
 
    #  Save the trained model 
    model_path = os.path.join(models_dir, f'{target}_lasso_lagged_model.joblib')
    joblib.dump(model, model_path)

    # Save metrics
    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Lasso Regression with Lags ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Selected Alpha: {model.alpha_:.4f}\n')
        f.write(f'Lasso MAE: {mae:.2f}\n')
        f.write(f'Lasso R² Score: {r2:.3f}\n')


    return model, coef_df
