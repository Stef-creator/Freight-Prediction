import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def visualize_missing_data(filepath = 'data/processed/processed.csv'):
    df = pd.read_csv(filepath)
    plt.figure(figsize=(15, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Data Heatmap (Yellow = Missing)")
    plt.tight_layout()
    plt.show()
