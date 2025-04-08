import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

def run_lasso_with_lags(filepath='data/processed/processed.csv', target='Gulf', showplot=True):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Create lag features for all columns
    lag_features = df.columns.difference(['date'])
    max_lag = 5
    lagged_dfs = []
    for lag in range(1, max_lag + 1):
        lagged = df[lag_features].shift(lag).add_suffix(f'_lag{lag}')
        lagged_dfs.append(lagged)

    df = pd.concat([df] + lagged_dfs, axis=1)

    # Predict y_t using y_(t-1...t-5) and x_(t-1...t-5)
    df[f'{target}_target'] = df[target]
    df = df.dropna()

    X = df[[col for col in df.columns if col.startswith(tuple(lag_features)) and ('_lag' in col)]]
    y = df[f'{target}_target']

    # Train/test split while maintaining time order
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit Lasso model
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    y_pred = lasso.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Lasso Test MAE: {mae:.2f}')
    print(f'Average Actual {target}: {np.mean(y_test):.2f}')
    print(f'Selected Alpha: {lasso.alpha_:.4f}')

    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print('\nTop Lasso Coefficients:')
    print(coef_df.head(20))
    
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    if showplot:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=coef_df, x='Coefficient', y='Feature', orient='h')
        plt.title(f'Lasso Regression Coefficients for {target}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'Lasso_Coefficients_{target}.png'))
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
        plt.savefig(os.path.join(results_dir, f'{target}_lasso_prediction_plot.png'))
        plt.close()

    # Save metrics
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Lasso Regression with Lags ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Lasso MAE: {mae:.2f}\n')
        f.write(f'Lasso R² Score: {r2:.3f}\n')

    return lasso, coef_df


run_lasso_with_lags()