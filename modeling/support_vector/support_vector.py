import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

def run_svm_regression(filepath='data/processed/processed.csv', target='Gulf'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Shift target 1 week ahead
    df[f'{target}_target'] = df[target].shift(-1)
    df = df.dropna(subset=[f'{target}_target'])
    df = df.drop(columns=['date', target, 'ship_cap', 'gscpi', 'trade_vol', 'ships_waiting', 'bpi_volatility', 'wheat_price', 'brent_price_trend', 'brent_price_seasonal', 'bpi_trend', 'bpi_seasonal'] )
    df = df.dropna()

    X = df.drop(columns=[f'{target}_target'])
    y = df[f'{target}_target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'SVM Test MAE: {mae:.2f}')
    print(f'SVM R² Score: {r2:.3f}\n')
    
    # Plot predictions
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'SVM Regression: 1-Week Ahead {target} Prediction')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{target} Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    
    results_dir = 'reports/models'
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics — append to the same file used by Prophet model
    with open(os.path.join(results_dir, 'model_results.txt'), 'a') as f:
        f.write(f'\n--- Support Vector Regression ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}) ---\n')
        f.write(f'SVM MAE: {mae:.2f}\n')
        f.write(f'SVM R² Score: {r2:.3f}\n')

    # Save plot
    plot_path = os.path.join(results_dir, f'{target}_svm_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return model
