import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from modeling.arma.arima import run_auto_arima_model
from modeling.arma.arimax import run_arimax_model
from modeling.arma.arimax_lagged import run_arimax_lagged_exog
from modeling.linear_regression.lasso import run_lasso_regression
from modeling.linear_regression.lasso_lagged import run_lasso_with_lags
from modeling.linear_regression.ridge import run_ridge_regression
from modeling.prop.prophet_uni import run_prophet_model
from modeling.prop.prophet_multi import run_multi_prophet_model
from modeling.support_vector.support_vector import run_svm_regression
from modeling.support_vector.support_vector_tune import run_svm_regression_tuned
from modeling.trees.xgboost_model import run_xgboost_model
from modeling.trees.xgboost_model_tune import run_xgboost_model_tuned
from modeling.prop.prophet_multi_tune import run_multi_prophet_model_tuned
from modeling.prop.prophet_uni_tune import run_prophet_model_tuned

def run_all_models():
    print("\nRunning all models for target: Gulf\n")

    run_auto_arima_model(target='Gulf')
    run_arimax_model(target='Gulf')
    run_arimax_lagged_exog(target='Gulf')
    run_lasso_regression(target='Gulf')
    run_lasso_with_lags(target='Gulf')
    run_ridge_regression(target='Gulf')
    run_prophet_model(target='Gulf')
    run_prophet_model_tuned(target='Gulf')
    run_multi_prophet_model(target='Gulf')
    run_multi_prophet_model_tuned(target='Gulf')
    run_svm_regression(target='Gulf')
    run_svm_regression_tuned(target='Gulf')
    run_xgboost_model(target='Gulf')
    run_xgboost_model_tuned(target='Gulf')
    


if __name__ == '__main__':
    run_all_models()
