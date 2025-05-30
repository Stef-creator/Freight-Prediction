# Freight Forecasting Pipeline

## Overview

This project provides a comprehensive pipeline for forecasting freight prices using multiple datasets, advanced feature engineering, and a suite of machine learning and time series models. It covers the full workflow from raw data ingestion through preprocessing, feature extraction, model training, and results reporting.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Pipeline Stages](#pipeline-stages)  
- [Output](#output)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [Author](#author)  

---

## Project Structure

/data/raw/ # Raw data files (Excel, CSV)
/data/processed/ # Cleaned and feature-engineered datasets
/models/ # Model scripts (ARIMA, XGBoost, Lasso, Prophet, etc.)
/reports/models/ # Model evaluation metrics and plots
/utils/ # Utility functions (preprocessing, diagnostics)
/data_pipeline/ # Scripts to fetch and load raw data
/feature_engineering/ # Feature engineering pipeline scripts
/notebooks/ # Jupyter notebooks and demo


---
## 

## Installation

### Prerequisites

- Python 3.10 or newer
- Recommended: Virtual environment (venv or conda)

### Setup

Clone the repository:

### bash

cd freight-forecasting

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

## Usage

### CLI Interface

Run any part or all of the pipeline using the CLI interface main_cli.py.

### Command-line Arguments

- `--fetch` Fetch all raw data from source files  
- `--prepare` Merge and align raw data into a single weekly dataset  
- `--features` Perform feature engineering (interpolation, volatility, seasonality)  
- `--train` Train and benchmark all predictive models with hyperparameter tuning  
- `--report` Generate model comparison dashboards and summary tables  



## Pipeline Stages
1. Data Fetching
    Scripts in data_pipeline/ load raw data from Excel/CSV files, perform initial cleaning, and save intermediate processed CSVs.

2. Data Preparation
    Loading and merging datasets, resampling to a consistent weekly Monday frequency, and aligning time series.

3. Feature Engineering
    Interpolation of missing values, computation of volatility indicators, and extraction of seasonal/trend components for key variables.

4. Model Training
    Multiple models trained and benchmarked, including:

    - Auto ARIMA
    - SARIMAX with exogenous variables
    - Lasso (with and without lags)
    - Ridge Regression
    - Support Vector Regression (with hyperparameter tuning)
    - XGBoost Regression (with hyperparameter tuning)
    - Prophet (uni- and multivariate, tuned)

6. Results Reporting
    Generation of performance summaries, metrics logs, and comparison plots saved in reports/models/.

## Output
    Processed Data: data/processed/processed.csv (final merged and feature-engineered dataset)

    Model Performance Plots: Saved in reports/models/ as PNG files

    Model Comparison Dashboard: Summary plots comparing MAE and R² scores across models

## Troubleshooting
Ensure all dependencies in requirements.txt are installed.

Confirm that raw data files exist in /data/raw/ before fetching or preprocessing.

Check that data/processed/ contains necessary intermediate files before training models.

## Contributing

Contributions are welcome! Please feel free to write to me to open an issue and discuss your ideas.

## Author
Stefan Pilegaard Pedersen
May 2025

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

