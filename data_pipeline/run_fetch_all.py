import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline.fetch_bpi import fetch_bpi
from data_pipeline.fetch_brent import fetch_brent
from data_pipeline.fetch_commodities import fetch_wheat, fetch_corn
from data_pipeline.fetch_congestion_metrics import fetch_anchored, fetch_awaiting, fetch_capacity
from data_pipeline.fetch_gscpi import fetch_gscpi
from data_pipeline.fetch_targets import fetch_targets
from data_pipeline.fetch_trade_vol import fetch_tvol


def fetch_all():
    
    fetch_bpi()
    print('Fetching BPI...')

    fetch_brent()
    print('Fetching Brent...')
    
    fetch_wheat(), fetch_corn()
    print('Fetching Commodities...')

    fetch_anchored(), fetch_awaiting(), fetch_capacity() 
    print('Fetching Congestion measures...')

    fetch_gscpi()
    print('Fetching GSCPI...')

    fetch_targets()
    print('Fetching targets...')

    fetch_tvol()
    print('Fetching Trade Volume...')

    print('✅ All datasets fetched and saved!')
    
    return None

if __name__ == '__main__':
    fetch_all()