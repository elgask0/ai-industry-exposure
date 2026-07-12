"""
BEA Data - Download and Process

This module handles downloading data from the Bureau of Economic Analysis:

1. GDP by Industry - Value Added (Table 1)
   - Annual data for 2021
   - All industries
   - Via BEA API

2. BEA Industry and Commodity Codes NAICS Concordance
   - Maps BEA industry codes to NAICS 2017
   - Direct Excel download

Data sources:
- BEA API: https://apps.bea.gov/api/
- Concordance: https://apps.bea.gov/industry/xls/io-annual/
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            config_file = parent / "config.yaml"
            if config_file.exists():
                config_path = str(config_file)
                break

    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError("config.yaml not found")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no config.yaml found)")


def load_api_key() -> str:
    """Load BEA API key from .env file."""
    root = get_project_root()
    env_path = root / '.env'

    if not env_path.exists():
        raise FileNotFoundError(
            ".env file not found. Create one with BEA_API_KEY=your_key"
        )

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEA_API_KEY='):
                return line.split('=', 1)[1].strip()

    raise ValueError("BEA_API_KEY not found in .env file")


# =============================================================================
# DATA SOURCES
# =============================================================================

BEA_API_URL = "https://apps.bea.gov/api/data"

BEA_SOURCES = {
    # GDP by Industry - Value Added (Table 1)
    'gdp_by_industry_va': {
        'dataset': 'GDPbyIndustry',
        'table_id': '1',
        'frequency': 'A',
        'year': '2021',
        'industry': 'ALL',
        'description': 'GDP by Industry - Value Added (Table 1) for 2021'
    },
    # BEA Industry/Commodity NAICS Concordance
    'bea_naics_concordance': {
        'url': 'https://www.bea.gov/sites/default/files/2023-10/BEA-Industry-and-Commodity-Codes-and-NAICS-Concordance.xlsx',
        'description': 'BEA Industry and Commodity Codes NAICS Concordance',
        'is_file': True
    }
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """
    Download a file from URL to destination path.

    Returns True if successful, False otherwise.
    """
    try:
        print(f"    Downloading: {url.split('/')[-1]}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
        }

        response = requests.get(url, headers=headers, timeout=timeout,
                                allow_redirects=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, 'wb') as f:
            f.write(response.content)

        file_size = os.path.getsize(dest_path)
        print(f"    Downloaded: {file_size:,} bytes")
        return True

    except requests.exceptions.RequestException as e:
        print(f"    ERROR: {e}")
        return False


def download_bea_api_data(api_key: str, dataset: str, table_id: str,
                          frequency: str, year: str, industry: str,
                          raw_dir: Path) -> Optional[str]:
    """
    Download data from BEA API.

    Returns path to extracted CSV file, or None on failure.
    """
    print(f"    Calling BEA API: {dataset} Table {table_id}, Year {year}...")

    params = {
        "method": "GetData",
        "datasetname": dataset,
        "TableID": table_id,
        "Frequency": frequency,
        "Year": year,
        "Industry": industry,
        "UserID": api_key,
        "ResultFormat": "JSON",
    }

    try:
        response = requests.get(BEA_API_URL, params=params, timeout=60)
        response.raise_for_status()
        js = response.json()

        # Save raw JSON
        json_path = raw_dir / f"bea_{dataset.lower()}_table{table_id}_year{year}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(js, f, ensure_ascii=False, indent=2)
        print(f"    Saved raw JSON: {json_path.name}")

        # Extract Data block robustly
        results = js.get("BEAAPI", {}).get("Results", None)
        if results is None:
            print(f"    ERROR: No BEAAPI.Results in response")
            return None

        data_rows = None
        if isinstance(results, dict) and "Data" in results:
            data_rows = results["Data"]
        elif isinstance(results, list):
            for obj in results:
                if isinstance(obj, dict) and "Data" in obj:
                    data_rows = obj["Data"]
                    break

        if data_rows is None:
            print("    ERROR: No 'Data' block found in BEAAPI.Results")
            return None

        # Save as CSV
        csv_path = raw_dir / f"bea_{dataset.lower()}_table{table_id}_year{year}.csv"
        df = pd.DataFrame(data_rows)
        df.to_csv(csv_path, index=False)
        print(f"    Saved raw CSV: {csv_path.name} ({len(df):,} rows)")

        return str(csv_path)

    except requests.exceptions.RequestException as e:
        print(f"    ERROR: API request failed: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    ERROR: Failed to parse response: {e}")
        return None


def download_all_bea_data(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Download all BEA data files.

    Returns dict mapping names to downloaded file paths.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'bea'
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load API key
    try:
        api_key = load_api_key()
        print(f"    Loaded BEA API key from .env")
    except (FileNotFoundError, ValueError) as e:
        print(f"    ERROR: {e}")
        return {}

    downloaded = {}

    print("\n  Downloading BEA data...")

    for name, info in BEA_SOURCES.items():
        # Create folder for each dataset
        dataset_dir = raw_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if info.get('is_file'):
            # Direct file download
            url = info['url']
            filename = url.split('/')[-1]
            dest_path = dataset_dir / filename

            if download_file(url, str(dest_path)):
                downloaded[name] = str(dest_path)
            else:
                print(f"    WARNING: Failed to download {name}")
        else:
            # API download
            csv_path = download_bea_api_data(
                api_key=api_key,
                dataset=info['dataset'],
                table_id=info['table_id'],
                frequency=info['frequency'],
                year=info['year'],
                industry=info['industry'],
                raw_dir=dataset_dir
            )
            if csv_path:
                downloaded[name] = csv_path
            else:
                print(f"    WARNING: Failed to download {name}")

    return downloaded


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_gdp_by_industry(raw_path: str, processed_dir: Path) -> str:
    """
    Process GDP by Industry value added data.

    Keeps all columns, no calculations.
    """
    print("    Processing GDP by Industry value added...")

    df = pd.read_csv(raw_path)

    # Clean column names (uppercase, strip whitespace)
    df.columns = df.columns.str.strip().str.upper()

    # Remove completely empty rows
    df = df.dropna(how='all')

    output_path = processed_dir / 'gdp_by_industry_va_2021.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")

    return str(output_path)


def process_bea_naics_concordance(raw_path: str, processed_dir: Path) -> str:
    """
    Process BEA Industry/Commodity NAICS Concordance.

    Keeps all columns, no calculations.
    """
    print("    Processing BEA NAICS Concordance...")

    # Read the Excel file - NAICS Codes sheet
    xl = pd.ExcelFile(raw_path)
    print(f"    Sheets found: {xl.sheet_names}")

    # Read with header rows at 3-4 (0-indexed), data starting at row 5
    # Row 3 has: BEA Industry Code, nan, nan, nan, Industry Title, nan, nan
    # Row 4 has: Sector, Summary, U.Summary, Detail, nan, Notes, Related 2017 NAICS Codes
    df = pd.read_excel(raw_path, sheet_name='NAICS Codes', skiprows=4, header=0)

    # Set proper column names
    df.columns = ['SECTOR', 'SUMMARY', 'U_SUMMARY', 'DETAIL', 'INDUSTRY_TITLE', 'NOTES', 'RELATED_NAICS_2017']

    # Remove completely empty rows
    df = df.dropna(how='all')

    output_path = processed_dir / 'bea_naics_concordance.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")

    return str(output_path)


def process_all_bea_data(downloaded_files: Dict[str, str],
                         config: Optional[dict] = None) -> Dict[str, str]:
    """
    Process all downloaded BEA files.

    Returns dict mapping names to processed file paths.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config['paths']['processed_data'] / 'bea'
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed = {}

    print("\n  Processing BEA files...")

    # GDP by Industry
    if 'gdp_by_industry_va' in downloaded_files:
        path = process_gdp_by_industry(downloaded_files['gdp_by_industry_va'], processed_dir)
        processed['gdp_by_industry_va'] = path

    # BEA NAICS Concordance
    if 'bea_naics_concordance' in downloaded_files:
        path = process_bea_naics_concordance(downloaded_files['bea_naics_concordance'], processed_dir)
        processed['bea_naics_concordance'] = path

    return processed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def download_and_process_bea_data(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Main entry point: download and process all BEA data.

    Returns dict with all processed file paths.
    """
    if config is None:
        config = load_config()

    print("\n" + "-" * 50)
    print("BEA DATA")
    print("-" * 50)

    # Download
    downloaded = download_all_bea_data(config)

    if not downloaded:
        print("  ERROR: No files downloaded")
        return {}

    # Process
    processed = process_all_bea_data(downloaded, config)

    # Summary
    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'bea'
    processed_dir = root / config['paths']['processed_data'] / 'bea'

    print("\n  Summary:")
    print(f"    Raw files: {raw_dir}")
    print(f"    Processed files: {processed_dir}")
    print(f"    Downloaded: {len(downloaded)} datasets")
    print(f"    Processed: {len(processed)} CSV files")

    return {
        'downloaded': downloaded,
        'processed': processed
    }


def print_validation_summary(output: Dict, config: dict):
    """Print validation summary for BEA data."""
    print("\n" + "-" * 50)
    print("BEA VALIDATION")
    print("-" * 50)

    processed = output.get('processed', {})

    for name, path in processed.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n  {name}:")
            print(f"    Rows: {len(df):,}")
            print(f"    Columns: {len(df.columns)}")
            print(f"    Column names: {', '.join(df.columns[:6])}...")


if __name__ == "__main__":
    output = download_and_process_bea_data()
    if output:
        print_validation_summary(output, load_config())
