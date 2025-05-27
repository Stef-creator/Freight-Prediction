import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.linear_model import LassoCV, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

# === Path config ===
MODEL_DIR = "reports/models_saved"
DATA_PATH = "data/processed/processed.csv"
PLOTS_DIR = "reports/interpretability_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model(model_name):
    """
    Loads a machine learning model from the specified model directory.

    Args:
        model_name (str): The filename of the model to load.

    Returns:
        object: The loaded model object.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        Exception: If there is an error during model loading.

    Example:
        model = load_model("my_model.pkl")
    """
    path = os.path.join(MODEL_DIR, model_name)
    return joblib.load(path)


def interpret_linear_model(model, feature_names, model_name):
    """
    Analyzes and visualizes the coefficients of a linear model.

    Parameters:
        model: A fitted linear model object with a `coef_` attribute (e.g., from scikit-learn).
        feature_names (list of str): List of feature names corresponding to the model's coefficients.
        model_name (str): Name of the model, used for plot title and filename.

    Returns:
        pandas.DataFrame: DataFrame containing features, their coefficients, and absolute coefficient values,
                          sorted by absolute value in descending order.

    Side Effects:
        Saves a horizontal bar plot of the top 20 coefficients to the directory specified by `PLOTS_DIR`,
        with the filename format "{model_name}_coefficients.png".
    """
    coefs = model.coef_
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False)

    # Save plot
    plt.figure(figsize=(8, 6))
    coef_df.head(20).plot(kind="barh", x="Feature", y="Coefficient", legend=False)
    plt.title(f"Top Coefficients: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_coefficients.png"))
    plt.close()
    return coef_df


def interpret_model_shap(model, X, model_name):
    """
    Generates and saves a SHAP summary plot for a given model and dataset.

    This function uses the SHAP library to compute SHAP values for the provided model and input data,
    then creates a summary plot visualizing feature importance. The plot is saved to disk with a filename
    based on the model name.

    Args:
        model: The trained machine learning model to interpret.
        X (pd.DataFrame or np.ndarray): The input features used for SHAP value computation.
        model_name (str): The name of the model, used for plot titling and filename.

    Side Effects:
        Saves a SHAP summary plot as a PNG file in the directory specified by PLOTS_DIR.

    Raises:
        Any exceptions raised by SHAP, matplotlib, or file I/O operations.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary Plot: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_shap_summary.png"))
    plt.close()


def interpret_arimax(model_path):
    """
    Loads an ARIMAX model from the specified path, prepares the relevant dataset, and generates diagnostic plots.

    Args:
        model_path (str): The file path to the saved ARIMAX model (joblib format).

    Returns:
        pandas.DataFrame: A DataFrame containing the model's coefficients with their names as index and "Coefficient" as the column.

    Side Effects:
        - Reads the dataset from DATA_PATH.
        - Saves diagnostic plots to the PLOTS_DIR as "ARIMAX_diagnostics.png".

    Notes:
        - Assumes DATA_PATH and PLOTS_DIR are defined elsewhere in the code.
        - The input CSV must contain the columns: 'date', 'Gulf', 'bpi', 'PNW', 'brent_price', 'corn_price', 'ships_anchored'.
        - The function renames 'date' to 'ds' and 'Gulf' to 'y' for compatibility.
    """
    model = joblib.load(model_path)
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'Gulf', 'bpi', 'PNW', 'brent_price', 'corn_price', 'ships_anchored']].dropna()
    df = df.rename(columns={'date': 'ds', 'Gulf': 'y'})

    rebuild = SARIMAXResults(model=model)
    results = rebuild

    fig = results.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ARIMAX_diagnostics.png"))
    plt.close()
    return results.params.to_frame(name="Coefficient")


