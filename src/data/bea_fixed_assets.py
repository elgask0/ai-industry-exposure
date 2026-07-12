"""
BEA Fixed Assets Data - Download and Process

Downloads data from the Bureau of Economic Analysis Fixed Assets tables:

1. FAAt301I:   IPP net stock by industry (2021)
2. FAAt301ESI: Total private fixed assets net stock by industry (2021)

Data source: BEA API (https://apps.bea.gov/api/)
Dataset: FixedAssets (uses TableName, not TableID; no Frequency/Industry params)
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from src.data.bea import load_api_key, get_project_root, load_config, BEA_API_URL


# =============================================================================
# CONFIGURATION
# =============================================================================

FIXED_ASSETS_TABLES = {
    "FAAt301I": {"years": "2021", "desc": "IPP net stock by industry"},
    "FAAt301ESI": {"years": "2021", "desc": "Total private fixed assets net stock by industry"},
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_fixed_assets_table(
    api_key: str, table_name: str, years: str, raw_dir: Path,
) -> pd.DataFrame:
    """
    Download one FixedAssets table from BEA API with CSV caching.

    Key difference from bea.py download_bea_api_data():
      FixedAssets uses TableName (not TableID) and has no Frequency/Industry params.

    Returns DataFrame of downloaded data.
    """
    cache_csv = raw_dir / f"bea_fixedassets_{table_name}_{years.replace(',', '_')}.csv"
    if cache_csv.exists() and cache_csv.stat().st_size > 0:
        print(f"    [CACHE] {table_name} years={years}")
        return pd.read_csv(cache_csv)

    print(f"    [API]   {table_name} years={years}...")
    params = {
        "method": "GetData",
        "datasetname": "FixedAssets",
        "TableName": table_name,
        "Year": years,
        "UserID": api_key,
        "ResultFormat": "JSON",
    }
    resp = requests.get(BEA_API_URL, params=params, timeout=120)
    resp.raise_for_status()
    js = resp.json()

    # Save raw JSON
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"bea_fixedassets_{table_name}_{years.replace(',', '_')}.json"
    with open(json_path, "w") as f:
        json.dump(js, f, indent=2)

    # Extract Data block (same pattern as bea.py:175-196)
    results = js.get("BEAAPI", {}).get("Results", None)
    if results is None:
        raise RuntimeError(f"No BEAAPI.Results for {table_name}")

    data_rows = None
    if isinstance(results, dict) and "Data" in results:
        data_rows = results["Data"]
    elif isinstance(results, list):
        for obj in results:
            if isinstance(obj, dict) and "Data" in obj:
                data_rows = obj["Data"]
                break
    if data_rows is None:
        raise RuntimeError(f"No Data block for {table_name}")

    df = pd.DataFrame(data_rows)
    df.to_csv(cache_csv, index=False)
    print(f"            -> {len(df)} rows cached")
    return df


def download_all_fixed_assets(config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Download Fixed Assets stock tables.

    Returns dict mapping table name to DataFrame.
    """
    root = get_project_root()
    if config is not None:
        raw_dir = root / config["paths"]["raw_data"] / "bea" / "fixed_assets"
    else:
        raw_dir = root / "data" / "raw" / "bea" / "fixed_assets"
    raw_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()

    tables = {}
    for name, info in FIXED_ASSETS_TABLES.items():
        df = download_fixed_assets_table(api_key, name, info["years"], raw_dir)
        if df.empty:
            raise RuntimeError(f"Empty data for {name}")
        tables[name] = df

    return tables


def download_and_process_fixed_assets(config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Main entry point: download all Fixed Assets tables.

    Matches bea.py download_and_process_bea_data() pattern.
    Returns dict mapping table name to DataFrame.
    """
    if config is None:
        config = load_config()

    print("\n" + "-" * 50)
    print("BEA FIXED ASSETS DATA")
    print("-" * 50)

    tables = download_all_fixed_assets(config)

    print(f"\n  Summary:")
    for name, df in tables.items():
        print(f"    {name}: {len(df):,} rows")

    return tables
