import sys
import os
import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def run_lasso_regression(filepath='data/processed/processed.csv', target='Gulf', showplot=True):
    """
    Run standard (non-lagged) Lasso Regression to predict one-step-ahead values of a target time series.

    This function performs the following:
    1. Loads and shifts the target column one step ahead (y_t+1)
    2. Splits the dataset into train/test sets (time-aware)
    3. Scales the features
    4. Trains a LassoCV model with TimeSeriesSplit
    5. Evaluates model performance on the test set
    6. Saves coefficients, plots, and metrics to the reports folder

    Args:
        filepath (str): Path to the preprocessed CSV file.
        target (str): Name of the target column to forecast.
        showplot (bool): If True, generates and saves plots.

    Returns:
        lasso (LassoCV): Trained LassoCV model object.
        coef_df (pd.DataFrame): DataFrame of feature names and learned coefficients.
    """
    # === Load data ===
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # === Shift target 1 week forward to create y_{t+1} ===
    df[f'{target}_target'] = df[target].shift(-1)
    df = df.drop(columns=['date', target]).dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    # === Time-aware train/test split ===
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # === Feature scaling ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Fit LassoCV with TimeSeriesSplit ===
    lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=42)
    lasso.fit(X_train_scaled, y_train)

    # === Predictions and evaluation ===
    y_pred = lasso.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Lasso Test MAE: {mae:.2f}')
    print(f'R² Score: {r2:.3f}')
    print(f'Selected Alpha: {lasso.alpha_:.4f}')

    # === Create coefficient DataFrame ===
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    # === Output folder ===
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    # === Plot coefficients and predictions ===
    if showplot:
        # Top coefficients
        plt.figure(figsize=(10, 5))
        sns.barplot(data=coef_df.head(20), x='Coefficient', y='Feature', orient='h')
        plt.title(f'Top Lasso Coefficients for {target}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'Lasso_Coefficients_{target}_nonlagged.png'))
        plt.close()

        # Prediction performance
        plt.figure(figsize=(12, 4))
        plt.plot(y_test.values, label='Actual', linewidth=2)
        plt.plot(y_pred, label='Predicted', linestyle='--')
        plt.title(f'Lasso Regression: Actual vs Predicted {target}')
        plt.xlabel('Test Sample Index')
        plt.ylabel(f'{target} Freight Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{target}_lasso_prediction_plot_nonlagged.png'))
        plt.close()

    # === Save results ===
    coef_df.to_csv(os.path.join(results_dir, f'{target}_lasso_coefficients_nonlagged.csv'), index=False)

    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Lasso Regression (Non-lagged) ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Lasso MAE: {mae:.2f}\n')
        f.write(f'Lasso R² Score: {r2:.3f}\n')
        f.write(f'Selected Alpha: {lasso.alpha_:.4f}\n')

    return lasso, coef_df
