from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd 

filepath='data/processed/processed.csv'
df = pd.read_csv(filepath)

print(df.info())
names = df.columns.tolist()
#print(names)

# Example data (X = features, y = target)
X = df[['ships_anchored', 'brent_price', 'bpi', 'ship_cap', 'corn_price', 'gscpi', 'PNW', 'trade_vol', 'ships_waiting', 'wheat_price', 'bpi_volatility', 'brent_price_trend', 'brent_price_seasonal', 'bpi_trend', 'bpi_seasonal']]
y = df['Gulf']

# Calculate Mutual Information
mi = mutual_info_regression(X, y)

# Create a DataFrame to display feature names and their corresponding MI values
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})

# Sort the MI values in descending order to identify the most important features
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Display the result
print(mi_df)
