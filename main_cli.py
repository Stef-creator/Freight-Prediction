"""
Freight Forecasting CLI Pipeline

This script provides a command-line interface to control the execution of the full
freight forecasting pipeline. It allows selective execution of data fetching, 
preprocessing, feature engineering, model training, and results summarization.

Usage (from terminal):
----------------------
python freight_cli.py --fetch       # Fetch all raw data from local files
python freight_cli.py --prepare     # Load and merge datasets into a weekly-aligned CSV
python freight_cli.py --features    # Interpolate and engineer final features
python freight_cli.py --train       # Train and benchmark all predictive models
python freight_cli.py --report      # Generate model comparison dashboard

You can combine steps, e.g.:
python freight_cli.py --fetch --prepare --features --train --report

Command-line Arguments:
-----------------------
--fetch      : Run all raw data fetching scripts from the /data_pipeline module
--prepare    : Merge and align data into a single dataset (weekly resampled)
--features   : Run feature engineering (volatility, seasonality, interpolation)
--train      : Run all tuned models and save evaluation metrics/plots
--report     : Generate and save a summary dashboard for model comparisons

Modules Used:
-------------
- data_pipeline.run_fetch_all         → Fetches raw data and saves to /data/processed/
- data_pipeline.load_all              → Aligns and merges datasets into all_data.csv
- feature_engineering.run_feature_engineering → Builds processed.csv with full features
- models.run_all_models               → Trains and evaluates all ML/time series models
- reports.compare_models              → Compiles evaluation metrics into a visual report

Requirements:
-------------
This script assumes the following folder structure:
- /data/raw/          → contains original Excel/CSV files
- /data/processed/    → receives intermediate and final data files
- /models/            → contains model scripts (e.g., ARIMA, XGBoost)
- /reports/models/    → contains plots and performance logs
- /utils/             → holds utility scripts (e.g., diagnostics, preprocessing)

Author: Stefan Pilegaard Pedersen  
Last updated: May 2025
"""

import argparse

from scripts.run_fetch_all import fetch_all
from pipeline.fetch.load_all import load_all_data
from scripts.run_feature_engineering import run_all_feature_engineering
from pipeline.models.run_all_models import run_all_models
from utils.compare_models import parse_model_results, plot_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run the freight forecasting pipeline from raw data to model results."
    )

    parser.add_argument('--fetch', action='store_true', help='Fetch raw data')
    parser.add_argument('--prepare', action='store_true', help='Merge and align datasets')
    parser.add_argument('--features', action='store_true', help='Run feature engineering')
    parser.add_argument('--train', action='store_true', help='Train and evaluate all models')
    parser.add_argument('--report', action='store_true', help='Generate model comparison dashboard')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    print("\n🚀 Freight Prediction CLI Starting...\n")

    if args.fetch:
        print("📥 Fetching raw data...")
        fetch_all()

    if args.prepare:
        print("🧱 Loading and merging datasets...")
        load_all_data()

    if args.features:
        print("🧪 Running feature engineering...")
        run_all_feature_engineering()

    if args.train:
        print("📈 Training and benchmarking models...")
        run_all_models()

    if args.report:
        print("📊 Generating model comparison dashboard...")
        df = parse_model_results()
        print("\n📊 Model Performance Summary:\n")
        print(df.to_string(index=False))
        plot_comparison(df)
        print("\n✅ Comparison plots saved to 'reports/models/'.")

    print("\n✅ Pipeline complete.")


if __name__ == '__main__':
    main()