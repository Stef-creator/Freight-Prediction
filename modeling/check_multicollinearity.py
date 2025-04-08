import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def check_multicollinearity(filepath='data/processed/processed.csv', threshold=0.8, showplot=True):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['date'], errors='ignore')
    df = df.dropna()
    #cols_to_drop = ['brent_seasonal','bpi_seasonal','trade_vol','ship_cap', 'PNW', 'bpi_trend', 'brent_trend', 'gscpi', 'ships_anchored', 'wheat_price']
    #df = df.drop(columns=cols_to_drop, errors='ignore')

    corr_matrix = df.corr()

    # --- Plot heatmap ---
    if showplot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=True,          # Show correlation numbers
            fmt=".2f",           # Format to 2 decimal places
            cmap='coolwarm',     # Color scheme
            square=True, 
            linewidths=0.5, 
            annot_kws={"size": 10}  # Font size for annotations
        )
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    # --- Detect pairs with high collinearity ---
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((col1, col2, corr_value))

    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return high_corr_pairs


if __name__ == '__main__':
    high_corr = check_multicollinearity()
    print("\nHighly correlated feature pairs:")
    for col1, col2, val in high_corr:
        print(f"{col1} & {col2} => Correlation: {val:.2f}")
