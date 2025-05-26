import sys
import os
import datetime
import joblib

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


def run_lasso_regression(filepath='data/processed/processed.csv', target='Gulf'):
    """
    Train and evaluate a standard (non-lagged) Lasso regression model for 1-week-ahead forecasting
    of a specified target time series.

    Steps:
    1. Load processed dataset and shift target variable one step forward (y_{t+1})
    2. Perform time-aware train/test split (80/20)
    3. Scale features using StandardScaler
    4. Fit LassoCV model with TimeSeriesSplit cross-validation
    5. Evaluate test set performance using MAE and R² metrics
    6. Save coefficients, prediction plots, and metrics to disk

    Args:
        filepath (str): Path to processed dataset CSV.
        target (str): Target variable column name.
        showplot (bool): Whether to generate and save plots.

    Returns:
        lasso (LassoCV): Trained LassoCV model object.
        coef_df (pd.DataFrame): DataFrame of feature names and coefficients sorted by absolute value.
    """

    #  Load data 
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    #  Shift target one week ahead (forecast y_{t+1}) 
    df[f'{target}_target'] = df[target].shift(-1)
    df = df.drop(columns=['date', target]).dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    #  Time-aware train/test split 
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    #  Scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  Train LassoCV model 
    model = LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=42)
    model.fit(X_train_scaled, y_train)

    #  Predict and evaluate 
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Lasso Test MAE: {mae:.2f}')
    print(f'R² Score: {r2:.3f}')
    print(f'Selected Alpha: {model.alpha_:.4f}')

    #  Create DataFrame of coefficients 
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

    # Save prediction plot and 20 coefficients barplot
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=coef_df.head(20), x='Coefficient', y='Feature', orient='h')
    plt.title(f'Top Lasso Coefficients for {target}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'Lasso_Coefficients_{target}_nonlagged.png'))
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
    plt.savefig(os.path.join(plots_dir, f'{target}_lasso_prediction_nonlagged_plot.png'))
    plt.close()

    #  Save the trained model 
    model_path = os.path.join(models_dir, f'{target}_lasso_nonlagged_model.joblib')
    joblib.dump(model, model_path)


    #  Save coefficients and metrics 
    coef_df.to_csv(os.path.join(metrics_dir, f'{target}_lasso_coefficients_nonlagged.csv'), index=False)

    with open(os.path.join(metrics_dir, 'model_results.txt'), 'a') as f:
        f.write(f'Selected Alpha: {model.alpha_:.4f}\n')
        f.write(f'\n--- Lasso Regression (Non-lagged) ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'Lasso MAE: {mae:.2f}\n')
        f.write(f'Lasso R² Score: {r2:.3f}\n')
        

    return model, coef_df
