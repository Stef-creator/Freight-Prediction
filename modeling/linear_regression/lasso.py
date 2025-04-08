import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns


def run_lasso_regression(filepath='data/processed/processed.csv', target='Gulf', showplot=True):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Shift target 1 week ahead
    df[f'{target}_target'] = df[f'{target}'].shift(-1)

    df = df.drop(columns=['date', target])
    df = df.dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LassoCV automatically selects the best alpha
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    y_pred = lasso.predict(X_test_scaled)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = lasso.score(X_test_scaled, y_test)

    print(f'Lasso Test MAE: {mae:.2f}')
    print(f'Average Actual {target}: {np.mean(y_test):.2f}')
    print(f'Selected Alpha: {lasso.alpha_:.4f}')

    # Show coefficients
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lasso.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', orient='h')
    plt.title(f'Lasso Regression Coefficients for {target}')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'Lasso Regression Coefficients for {target}.png')
    plt.savefig(plot_path)
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
    plot_path = os.path.join(results_dir, f'{target}_lasso_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Save coefficients as CSV
    coef_df.to_csv(os.path.join(results_dir, f'{target}_lasso_coefficients.csv'), index=False)

    # Save metrics
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Lasso Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Lasso MAE: {mae:.2f}\n')
        f.write(f'Lasso R² Score: {r2:.3f}\n')

    return lasso, coef_df

