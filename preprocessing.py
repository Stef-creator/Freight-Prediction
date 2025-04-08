import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data_pipeline.run_fetch_all import fetch_all
from data_pipeline.load_all import load_all_data
from feature_engineering.run_feature_engineering import run_all_feature_engineering


def whole_pipeline():
         
    fetch_all()
    load_all_data()
    run_all_feature_engineering()

    return

if __name__ == '__main__':
    whole_pipeline()
    print('✅ Data Pipeline executed \n✅ Ready to model')
