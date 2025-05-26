import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.dates as mdates



def plot_autocorrelations_for_all(filepath='data/processed/processed.csv', lags=30):
    """
    Generate and save ACF/PACF plots for all numeric columns in a dataset.

    Args:
        filepath (str): Path to the CSV file with time series data.
        lags (int): Number of lags to show in ACF/PACF plots.

    Returns:
        None. Saves PNG plots to 'reports/autocorrelations/'.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.drop(columns=['date'], errors='ignore')
    df = df.select_dtypes(include='number').dropna()

    os.makedirs('reports/autocorrelations', exist_ok=True)

    for column in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(df[column], lags=lags, alpha=0.05, ax=axes[0])
        axes[0].set_title(f'Autocorrelation (ACF) for {column}')

        plot_pacf(df[column], lags=lags, alpha=0.05, method='ywm', ax=axes[1])
        axes[1].set_title(f'Partial Autocorrelation (PACF) for {column}')

        plt.tight_layout()
        plt.savefig(f'reports/autocorrelations/acf_pacf_{column}.png')
        plt.close()

    print("✅ ACF and PACF plots saved to 'reports/autocorrelations/'.")

def visualize_missing_data(filepath='data/processed/processed.csv'):
    """
    Visualize missing values in a dataset using a heatmap.

    Args:
        filepath (str): Path to the CSV file to visualize.

    Returns:
        None
    """
    df = pd.read_csv(filepath)
    plt.figure(figsize=(15, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Data Heatmap (Yellow = Missing)")
    plt.tight_layout()
    plt.show()

    return None

def check_multicollinearity(df, threshold=0.8, plot=True):
    """
    Identify pairs of numeric features with high pairwise correlation.

    Args:
        df (pd.DataFrame): Input DataFrame (all or mixed types).
        threshold (float): Correlation threshold above which pairs are flagged.
        plot (bool): Whether to show a heatmap of correlations.

    Returns:
        list of tuples: (feature1, feature2, correlation) above threshold.
    """
    df = df.drop(columns=['date'], inplace=True) if 'date' in df.columns else df
    
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr().round(2)  # round for cleaner labels
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = [
        (col1, col2, corr_matrix.loc[col1, col2])
        for col1 in upper.columns
        for col2 in upper.index
        if upper.loc[col1, col2] > threshold
    ]

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",           # show values
            cmap='coolwarm',
            square=True,
            cbar=True
        )
        plt.title("Correlation Matrix (annotated)")
        plt.tight_layout()
        plt.show()
        return high_corr_pairs



def compute_mutual_information(df, target='Gulf', feature_cols=None, plot=True):
    """
    Compute mutual information between a target variable and each feature.

    Args:
        df (pd.DataFrame): DataFrame containing all features and target.
        target (str): Name of the target variable.
        feature_cols (list or None): Columns to consider as features. If None, use all except target.
        plot (bool): Whether to plot the ranked MI scores as a bar chart.

    Returns:
        pd.DataFrame: A DataFrame with feature names and their mutual information scores,
                      sorted in descending order.
    """
    if feature_cols is None:
        feature_cols = df.columns.drop(target).tolist()

    X = df[feature_cols]
    y = df[target]

    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({'Feature': feature_cols, 'Mutual Information': mi_scores})
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=mi_df, x='Mutual Information', y='Feature', palette='viridis')
        plt.title(f'Mutual Information with Target: {target}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return mi_df
