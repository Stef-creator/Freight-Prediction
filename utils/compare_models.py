import os
import pandas as pd
import matplotlib.pyplot as plt
import re


def parse_model_results(filepath='reports/models/model_results.txt'):
    """
    Parses the model_results.txt file to extract MAE and R^2 for all models.

    Returns:
        pd.DataFrame: Table with columns: ['Model', 'MAE', 'R2']
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    results = []
    model_name = None
    mae = None
    r2 = None

    for line in lines:
        if line.startswith('---'):
            if model_name and mae is not None and r2 is not None:
                results.append({'Model': model_name, 'MAE': mae, 'R2': r2})
            model_name = re.sub(r'---\s*', '', line).strip().split('(')[0].strip()
            mae = None
            r2 = None

        if 'MAE' in line and 'Auto ARIMA' not in line:
            try:
                mae = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)[0])
            except:
                continue
        if 'R²' in line or 'R2' in line:
            try:
                r2 = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)[0])
            except:
                continue

    if model_name and mae is not None and r2 is not None:
        results.append({'Model': model_name, 'MAE': mae, 'R2': r2})

    df = pd.DataFrame(results)
    df = df.sort_values(by='MAE')
    return df


def plot_comparison(df):
    """
    Plot MAE and R2 score comparisons for all models
    """
    plt.figure(figsize=(10, 6))
    plt.barh(df['Model'], df['MAE'], color='skyblue')
    plt.xlabel('MAE')
    plt.title('Model Comparison - Mean Absolute Error')
    plt.tight_layout()
    plt.savefig('reports/plots/comparison_mae.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(df['Model'], df['R2'], color='lightgreen')
    plt.xlabel('R² Score')
    plt.title('Model Comparison - R² Score')
    plt.tight_layout()
    plt.savefig('reports/plots/comparison_r2.png')
    plt.close()
