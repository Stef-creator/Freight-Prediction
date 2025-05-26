import sys
import os

# Ensure root project folder is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data_pipeline.run_fetch_all import fetch_all
from data_pipeline.load_all import load_all_data
from feature_engineering.run_feature_engineering import run_all_feature_engineering


def whole_pipeline():
    """
    Executes the full data pipeline:
    1. Fetch raw datasets from local or remote sources
    2. Load and align all datasets to weekly format
    3. Perform feature engineering and save final processed dataset

    Final output:
    - `data/processed/all_data.csv`: unified raw input
    - `data/processed/processed.csv`: cleaned and feature-rich dataset

    Returns:
        None
    """
    print("\n🚀 Starting full data pipeline...\n")

    fetch_all()
    load_all_data()
    run_all_feature_engineering()

    print("\n✅ Data Pipeline executed successfully")
    print("✅ Processed dataset ready for modeling at: data/processed/processed.csv")


if __name__ == '__main__':
    whole_pipeline()
