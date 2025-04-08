import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os


def plot_autocorrelations_for_all(filepath='data/processed/processed.csv', lags=30):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['date'])
    df = df.dropna()

    for column in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(df[column], lags=lags, alpha=0.05, ax=axes[0])
        axes[0].set_title(f'Autocorrelation (ACF) for {column}')
        plot_pacf(df[column], lags=lags, alpha=0.05, method='ywm', ax=axes[1])
        axes[1].set_title(f'Partial Autocorrelation (PACF) for {column}')
        plt.tight_layout()

        os.makedirs('reports/autocorrelations', exist_ok=True)
        plt.savefig(f'reports/autocorrelations/acf_pacf_{column}.png')
        plt.close()

    print("✅ Autocorrelation plots saved for all features.")


if __name__ == '__main__':
    plot_autocorrelations_for_all()