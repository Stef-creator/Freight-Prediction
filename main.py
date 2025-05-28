import sys
import os

# Add root path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts.run_fetch_all import fetch_all
from scripts.run_feature_engineering import run_all_feature_engineering
from pipeline.fetch.load_all import load_all_data
from pipeline.models.run_all_models import run_all_models


def run_program():
    """
    Run the full freight prediction pipeline from raw data to model results.

    Steps:
    1. Fetch raw data from source files
    2. Merge and align datasets to weekly format
    3. Interpolate, compute volatility, and extract trends/seasonality
    4. Benchmark all predictive models and save results to reports/

    Returns:
        None
    """
    print("\n🚀 Starting full freight pipeline...\n")

    fetch_all()
    load_all_data()
    run_all_feature_engineering()
    run_all_models()

    print("\n✅ All stages complete!")
    print("📦 Final dataset:      data/processed/processed.csv")
    print("📊 Model results:      reports/models/")
    print("📈 Ready for review.\n")


if __name__ == '__main__':
    run_program()
