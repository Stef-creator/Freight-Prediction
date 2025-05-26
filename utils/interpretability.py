import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

def interpret_xgboost_model(model_path, data_path, target, plot_dir='reports/interpretability'):
    """
    Load saved XGBoost model and data, compute SHAP values, plot global and dependence plots.

    Args:
        model_path (str): Path to saved XGBoost model (.joblib).
        data_path (str): Path to processed dataset CSV.
        target (str): Target column name.
        plot_dir (str): Directory to save interpretability plots.

    Returns:
        None
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    drop_cols = [
        'date', target, 'ship_cap', 'gscpi', 'trade_vol', 'ships_waiting',
        'bpi_volatility', 'wheat_price', 'brent_price_trend',
        'brent_price_seasonal', 'bpi_trend', 'bpi_seasonal'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df = df.dropna()
    X = df.drop(columns=[target, f'{target}_target'], errors='ignore')

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP summary plot (global feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'xgboost_shap_summary.png'))
    plt.close()

    # SHAP dependence plots for top 5 features
    top_features = X.columns[:5]
    for feature in top_features:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.title(f'SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'xgboost_shap_dependence_{feature}.png'))
        plt.close()

    print(f'XGBoost SHAP plots saved to {plot_dir}')


def interpret_svr_model(model_path, data_path, target, plot_dir='reports/interpretability'):
    """
    Load saved SVR model and data, compute SHAP values using KernelExplainer, plot global explanations.

    Args:
        model_path (str): Path to saved SVR model (.joblib).
        data_path (str): Path to processed dataset CSV.
        target (str): Target column name.
        plot_dir (str): Directory to save interpretability plots.

    Returns:
        None
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    drop_cols = [
        'date', target, 'ship_cap', 'gscpi', 'trade_vol', 'ships_waiting',
        'bpi_volatility', 'wheat_price', 'brent_price_trend',
        'brent_price_seasonal', 'bpi_trend', 'bpi_seasonal'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df = df.dropna()
    X = df.drop(columns=[target, f'{target}_target'], errors='ignore')

    # Sample background for KernelExplainer to speed up
    background = shap.sample(X, 100, random_state=42)

    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X)

    # SHAP summary plot (global feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot (SVR)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'svr_shap_summary.png'))
    plt.close()

    print(f'SVR SHAP plots saved to {plot_dir}')

interpret_svr_model('reports/models_saved/Gulf_svm_model.joblib','data/processed/processed.csv', 'Gulf')
interpret_xgboost_model('reports/models_saved/Gulf_xgboost_model.joblib','data/processed/processed.csv', 'Gulf')
