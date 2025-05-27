import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor



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

def correlation_plot(df):
    """
    Plot the correlation matrix heatmap for all numeric columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        None. Displays a heatmap of the correlation matrix.
    """
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr().round(2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        cbar=True
    )
    plt.title("Correlation Matrix (annotated)")
    plt.tight_layout()
    plt.show()
    return

def vif(df, threshold=5.0, drop=True, verbose=True):
    """
    Calculate Variance Inflation Factor (VIF) for each numeric feature in the DataFrame.
    Optionally drop the feature with the highest VIF above the threshold iteratively.

    Args:
        df (pd.DataFrame): Input DataFrame (should not include target variable).
        threshold (float): VIF threshold above which to flag multicollinearity.
        drop (bool): Whether to iteratively drop features with highest VIF above threshold.
        verbose (bool): Whether to print dropped features and VIFs.

    Returns:
        pd.DataFrame: DataFrame with features and their VIF values, sorted descending.
    """
    X = df.select_dtypes(include=[np.number]).dropna().copy()
    dropped = []

    while True:
        vif_data = []
        columns = X.columns
        for i in range(len(columns)):
            vif_value = variance_inflation_factor(X.values, i)
            vif_data.append({'Feature': columns[i], 'VIF': vif_value})
        vif_df = pd.DataFrame(vif_data).sort_values(by='VIF', ascending=False)
        max_vif = vif_df.iloc[0]
        if max_vif['VIF'] > threshold and drop:
            dropped.append((max_vif['Feature'], max_vif['VIF']))
            if verbose:
                print(f"Dropping '{max_vif['Feature']}' with VIF={max_vif['VIF']:.2f}")
            X = X.drop(columns=[max_vif['Feature']])
            if X.shape[1] == 1:
                break
        else:
            break

    if verbose:
        if dropped:
            print("Dropped features due to high VIF:")
            for feat, v in dropped:
                print(f"  {feat}: VIF={v:.2f}")
        print(f"Final VIFs (threshold={threshold}):")
        print(vif_df)

    return vif_df


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
        plt.figure(figsize=(6, 5))
        sns.barplot(data=mi_df, x='Mutual Information', y='Feature', palette='viridis')
        plt.title(f'Mutual Information with Target: {target}', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

    return mi_df
