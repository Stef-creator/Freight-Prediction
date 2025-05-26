import sys
import os

# Add root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import all model wrappers
from models.arima import run_auto_arima_model
from models.arimax import run_arimax_model
from models.arimax_lagged import run_arimax_lagged_exog
from models.lasso import run_lasso_regression
from models.lasso_lagged import run_lasso_with_lags
from models.ridge import run_ridge_regression
from models.support_vector_tune import run_svm_regression_tuned
from models.xgboost_model_tune import run_xgboost_model_tuned
from models.prophet_multi_tune import run_multi_prophet_model_tuned
from models.prophet_uni_tune import run_prophet_model_tuned


def run_all_models():
    """
    Run all predictive models for the specified target variable.

    This function calls each model's dedicated script and executes the prediction
    pipeline using 'Gulf' as the target freight rate. Model outputs are saved to
    the `reports/models/` folder, including:
    - Prediction plots (actual vs. predicted)
    - Model evaluation metrics (MAE, R², best hyperparameters if tuned)
    - Coefficients or feature importances when applicable

    Models run:
    - ARIMA
    - ARIMAX
    - ARIMAX with lagged exogenous variables
    - Lasso Regression
    - Lasso Regression with lags
    - Ridge Regression
    - Prophet (univariate tuned)
    - Prophet (multivariate tuned)
    - Support Vector Regression (tuned)
    - XGBoost Regression (tuned)

    Returns:
        None
    """
    print("\n🚀 Running all models for target: Gulf\n")

    run_auto_arima_model(target='Gulf')
    run_arimax_model(target='Gulf')
    run_arimax_lagged_exog(target='Gulf')

    run_lasso_regression(target='Gulf')
    run_lasso_with_lags(target='Gulf')
    run_ridge_regression(target='Gulf')

    run_prophet_model_tuned(target='Gulf')
    run_multi_prophet_model_tuned(target='Gulf')

    run_svm_regression_tuned(target='Gulf')
    run_xgboost_model_tuned(target='Gulf')

    print("\n✅ All models executed and results saved to reports/models/\n")


if __name__ == '__main__':
    run_all_models()
