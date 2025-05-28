import sys
import os

# Add parent directory to sys.path so imports work when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Individual fetch functions
from pipeline.fetch.fetch_bpi import fetch_bpi
from pipeline.fetch.fetch_brent import fetch_brent
from pipeline.fetch.fetch_commodities import fetch_wheat, fetch_corn
from pipeline.fetch.fetch_congestion_metrics import fetch_anchored, fetch_awaiting, fetch_capacity
from pipeline.fetch.fetch_gscpi import fetch_gscpi
from pipeline.fetch.fetch_targets import fetch_targets
from pipeline.fetch.fetch_trade_vol import fetch_tvol


def fetch_all():
    """
    Run all data fetch scripts and save cleaned outputs to `data/processed/`.

    This function orchestrates the loading, cleaning, and saving of all relevant
    data sources needed for the freight prediction pipeline. Each source is read
    from its raw format (CSV, Excel, UTF-16, etc.) and exported as a clean, 
    standardized `.csv` file for downstream processing.

    Datasets fetched:
    - BPI (Baltic Panamax Index)
    - Brent crude oil prices
    - U.S. agricultural commodities (corn, wheat)
    - Port congestion metrics (anchored ships, waiting ships, containership capacity)
    - Global Supply Chain Pressure Index (GSCPI)
    - U.S. Gulf and Pacific Northwest freight targets
    - Global trade volume (CPB)

    Returns:
        None
    """
    print('📦 Fetching all raw data sources...\n')

    print('🔹 Fetching BPI...')
    fetch_bpi()

    print('🔹 Fetching Brent...')
    fetch_brent()

    print('🔹 Fetching Commodities...')
    fetch_wheat()
    fetch_corn()

    print('🔹 Fetching Port Congestion Metrics...')
    fetch_anchored()
    fetch_awaiting()
    fetch_capacity()

    print('🔹 Fetching GSCPI...')
    fetch_gscpi()

    print('🔹 Fetching Targets (Gulf & PNW)...')
    fetch_targets()

    print('🔹 Fetching Trade Volume...')
    fetch_tvol()

    print('\n✅ All datasets fetched and saved to data/processed/')
