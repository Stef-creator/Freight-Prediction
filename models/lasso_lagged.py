import sys
import os
import datetime

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


def run_lasso_with_lags(filepath='data/processed/processed.csv', target='Gulf', max_lag=5, showplot=True):
    """
    Run Lasso Regression with lagged features to forecast a time series target.

    This model applies L1-regularized regression (Lasso) using lagged versions
    of all predictors in the dataset. It performs time-aware cross-validation,
    selects the optimal regularization parameter, and evaluates model performance
    on a holdout test set.

    Args:
        filepath (str): Path to the input processed dataset.
        target (str): Name of the target column to forecast.
        max_lag (int): Number of lags to generate for all non-date columns.
        showplot (bool): Whether to generate and save coefficient and prediction plots.

    Returns:
        lasso (LassoCV): Trained LassoCV model.
        coef_df (pd.DataFrame): DataFrame of feature names and their corresponding coefficients.
    """
    # === Load and preprocess ===
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Generate lag features for all variables except date
    lag_features = df.columns.difference(['date'])
    lagged_dfs = []
    for lag in range(1, max_lag + 1):
        lagged = df[lag_features].shift(lag).add_suffix(f'_lag{lag}')
        lagged_dfs.append(lagged)

    # Combine original with lagged data
    df = pd.concat([df] + lagged_dfs, axis=1)
    df[f'{target}_target'] = df[target]
    df = df.dropna()

    # Feature/target split
    X = df[[col for col in df.columns if '_lag' in col]]
    y = df[f'{target}_target']

    # === Time-series train/test split ===
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # === Scale predictors ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Fit Lasso with TimeSeriesSplit CV ===
    lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=42)
    lasso.fit(X_train_scaled, y_train)

    # === Evaluate ===
    y_pred = lasso.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Lasso Test MAE: {mae:.2f}')
    print(f'R² Score: {r2:.3f}')
    print(f'Selected Alpha: {lasso.alpha_:.4f}')

    # === Coefficient summary ===
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print('\nTop Lasso Coefficients:')
    print(coef_df.head(20))

    # === Save outputs ===
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    if showplot:
        # Coefficient plot
        plt.figure(figsize=(10, 5))
        sns.barplot(data=coef_df.head(20), x='Coefficient', y='Feature', orient='h')
        plt.title(f'Top Lasso Coefficients for {target}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'Lasso_Coefficients_{target}.png'))
        plt.close()

        # Prediction plot
        plt.figure(figsize=(12, 4))
        plt.plot(y_test.values, label='Actual', linewidth=2)
        plt.plot(y_pred, label='Predicted', linestyle='--')
        plt.title(f'Lasso Regression: Actual vs Predicted {target}')
        plt.xlabel('Test Sample Index')
        plt.ylabel(f'{target} Freight Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{target}_lasso_prediction_plot.png'))
        plt.close()

    # === Save evaluation metrics ===
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Lasso Regression with Lags ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Lasso MAE: {mae:.2f}\n')
        f.write(f'Lasso R² Score: {r2:.3f}\n')
        f.write(f'Selected Alpha: {lasso.alpha_:.4f}\n')

    return lasso, coef_df
