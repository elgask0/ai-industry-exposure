"""
BLS Data - Download and Process

This module handles downloading and processing BLS data:

1. OEWS May 2021 - National estimates by 4-digit NAICS industry
   - OCC_CODE: SOC 2018 or OEWS hybrid occupation code
   - O_GROUP: Occupation group label
   - TOT_EMP: Employment in the cell
   - A_MEAN: Mean annual wage
   - NAICS industry identifier

2. OEWS May 2018 - National estimates by occupation
   - SOC 2010 occupation codes
   - National employment

3. OEWS Hybrid Structure 2019
   - Maps OEWS hybrid codes to SOC 2018 components

4. QCEW 2021 Annual by Industry
   - Annual wages by NAICS 6-digit
   - Used to weight NAICS6 composition within NAICS4

Data sources:
- OEWS tables: https://www.bls.gov/oes/tables.htm
- Hybrid structure: https://www.bls.gov/oes/oes_2019_hybrid_structure.xlsx
- QCEW: https://www.bls.gov/cew/downloadable-data-files.htm
"""

import os
import zipfile
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


# =============================================================================
# DATA SOURCES
# =============================================================================

BLS_SOURCES = {
    # OEWS May 2021 - National by 4-digit NAICS industry
    'oews_may2021_nat4d': {
        'url': 'https://www.bls.gov/oes/special-requests/oesm21in4.zip',
        'description': 'OEWS May 2021 national estimates by 4-digit NAICS',
        'year': '2021',
        'extract_file': 'nat4d_M2021_dl.xlsx'
    },
    # OEWS May 2018 - National estimates (uses SOC 2010)
    'oews_may2018_nat': {
        'url': 'https://www.bls.gov/oes/special-requests/oesm18nat.zip',
        'description': 'OEWS May 2018 national estimates',
        'year': '2018',
        'extract_file': 'national_M2018_dl.xlsx'
    },
    # OEWS Hybrid Structure 2019
    'oews_hybrid_structure': {
        'url': 'https://www.bls.gov/oes/oes_2019_hybrid_structure.xlsx',
        'description': 'OEWS 2019 hybrid occupation structure',
        'is_xlsx': True
    },
    # QCEW 2021 Annual by Industry
    'qcew_2021_annual': {
        'url': 'https://data.bls.gov/cew/data/files/2021/csv/2021_annual_by_industry.zip',
        'description': 'QCEW 2021 annual averages by industry (NAICS 6-digit)',
        'year': '2021',
        'is_qcew': True
    }
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url: str, dest_path: str, timeout: int = 120) -> bool:
    """
    Download a file from URL to destination path.

    Returns True if successful, False otherwise.
    """
    try:
        print(f"    Downloading: {url.split('/')[-1]}")

        # Add headers to avoid 403 errors from BLS
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;'
                      'q=0.9,*/*;q=0.8',
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


def extract_zip(zip_path: str, extract_dir: str, target_file: Optional[str] = None) -> str:
    """
    Extract ZIP file to directory.

    If target_file specified, extracts only that file to the extract_dir directly
    (flattening any subdirectory structure in the ZIP).
    Returns path to extracted file or directory.
    """
    print(f"    Extracting: {os.path.basename(zip_path)}")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        if target_file:
            # Find the file in the archive (may be in a subdirectory)
            for name in zf.namelist():
                if name.endswith(target_file):
                    # Extract to a flat structure (no subdirectories)
                    target_path = os.path.join(extract_dir, target_file)
                    with zf.open(name) as src, open(target_path, 'wb') as dst:
                        dst.write(src.read())
                    print(f"    Extracted: {target_file}")
                    return target_path
            # If not found, extract all
            print(f"    Target file {target_file} not found, extracting all...")

        zf.extractall(extract_dir)
        print(f"    Extracted all files to {extract_dir}")

    return extract_dir


def download_all_bls_data(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Download all BLS OEWS data files.

    Returns dict mapping names to downloaded/extracted file paths.
    Structure: data/raw/bls/{dataset_name}/{zip + extracted files}
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'bls'
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    print("\n  Downloading BLS OEWS files...")

    for name, info in BLS_SOURCES.items():
        url = info['url']
        filename = url.split('/')[-1]

        # Each dataset gets its own folder
        dataset_dir = raw_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dataset_dir / filename

        if download_file(url, str(dest_path)):
            # If ZIP, extract files
            if filename.endswith('.zip'):
                target_file = info.get('extract_file')
                is_qcew = info.get('is_qcew', False)

                if is_qcew:
                    # QCEW: extract all files, return directory path
                    extract_zip(str(dest_path), str(dataset_dir), None)
                    downloaded[name] = str(dataset_dir)
                else:
                    # OEWS: extract single target file
                    extracted = extract_zip(str(dest_path), str(dataset_dir), target_file)
                    downloaded[name] = extracted
            else:
                downloaded[name] = str(dest_path)
        else:
            print(f"    WARNING: Failed to download {name}")

    return downloaded


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_oews_may2021(raw_path: str, processed_dir: Path) -> str:
    """
    Process OEWS May 2021 national by 4-digit NAICS.

    Keeps all columns, no calculations.
    """
    print("    Processing OEWS May 2021 (4-digit NAICS)...")

    df = pd.read_excel(raw_path)

    # Clean column names (uppercase, strip whitespace)
    df.columns = df.columns.str.strip().str.upper()

    # Remove completely empty rows
    df = df.dropna(how='all')

    output_path = processed_dir / 'oews_may2021_nat4d.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")

    return str(output_path)


def process_oews_may2018(raw_path: str, processed_dir: Path) -> str:
    """
    Process OEWS May 2018 national estimates.

    Keeps all columns, no calculations.
    """
    print("    Processing OEWS May 2018 (national)...")

    df = pd.read_excel(raw_path)

    # Clean column names (uppercase, strip whitespace)
    df.columns = df.columns.str.strip().str.upper()

    # Remove completely empty rows
    df = df.dropna(how='all')

    output_path = processed_dir / 'oews_may2018_nat.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")

    return str(output_path)


def process_oews_hybrid_structure(raw_path: str, processed_dir: Path) -> str:
    """
    Process OEWS 2019 hybrid structure.

    Keeps all columns, no calculations.
    """
    print("    Processing OEWS hybrid structure...")

    # Skip header rows - actual headers are at row 5 (0-indexed), data starts at row 6
    df = pd.read_excel(raw_path, skiprows=5, header=0)

    # Clean column names (uppercase, strip whitespace, replace spaces)
    df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')

    # Remove completely empty rows
    df = df.dropna(how='all')

    output_path = processed_dir / 'oews_hybrid_structure.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")

    return str(output_path)


def process_qcew_2021(raw_dir: str, processed_dir: Path) -> str:
    """
    Process QCEW 2021 annual by industry.

    Concatenates all CSV files from the extracted ZIP.
    Keeps all columns, no calculations.
    """
    print("    Processing QCEW 2021 annual by industry...")

    # Find all CSV files in the raw directory (may be in subfolder)
    raw_path = Path(raw_dir)
    csv_files = list(raw_path.glob('*.csv'))

    # If no files found, check subfolders
    if not csv_files:
        csv_files = list(raw_path.glob('**/*.csv'))

    if not csv_files:
        print(f"    ERROR: No CSV files found in {raw_dir}")
        return None

    print(f"    Found {len(csv_files)} CSV files")

    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, dtype=str, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not read {csv_file.name}: {e}")

    if not dfs:
        print("    ERROR: Could not read any CSV files")
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Clean column names (uppercase, strip whitespace)
    df.columns = df.columns.str.strip().str.upper()

    # Remove completely empty rows
    df = df.dropna(how='all')

    output_path = processed_dir / 'qcew_2021_annual.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")

    return str(output_path)


def process_all_bls_data(downloaded_files: Dict[str, str],
                         config: Optional[dict] = None) -> Dict[str, str]:
    """
    Process all downloaded BLS files.

    Returns dict mapping names to processed file paths.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config['paths']['processed_data'] / 'bls'
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed = {}

    print("\n  Processing BLS files...")

    # OEWS May 2021
    if 'oews_may2021_nat4d' in downloaded_files:
        path = process_oews_may2021(downloaded_files['oews_may2021_nat4d'], processed_dir)
        processed['oews_may2021_nat4d'] = path

    # OEWS May 2018
    if 'oews_may2018_nat' in downloaded_files:
        path = process_oews_may2018(downloaded_files['oews_may2018_nat'], processed_dir)
        processed['oews_may2018_nat'] = path

    # OEWS Hybrid Structure
    if 'oews_hybrid_structure' in downloaded_files:
        path = process_oews_hybrid_structure(downloaded_files['oews_hybrid_structure'], processed_dir)
        processed['oews_hybrid_structure'] = path

    # QCEW 2021
    if 'qcew_2021_annual' in downloaded_files:
        path = process_qcew_2021(downloaded_files['qcew_2021_annual'], processed_dir)
        if path:
            processed['qcew_2021_annual'] = path

    return processed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def download_and_process_bls_data(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Main entry point: download and process all BLS data (OEWS + QCEW).

    Returns dict with all processed file paths.
    """
    if config is None:
        config = load_config()

    print("\n" + "-" * 50)
    print("BLS DATA (OEWS + QCEW)")
    print("-" * 50)

    # Download
    downloaded = download_all_bls_data(config)

    if not downloaded:
        print("  ERROR: No files downloaded")
        return {}

    # Process
    processed = process_all_bls_data(downloaded, config)

    # Summary
    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'bls'
    processed_dir = root / config['paths']['processed_data'] / 'bls'

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
    """Print validation summary for BLS data."""
    print("\n" + "-" * 50)
    print("BLS VALIDATION")
    print("-" * 50)

    processed = output.get('processed', {})

    for name, path in processed.items():
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=1000, low_memory=False)  # Sample for large files
            total_rows = sum(1 for _ in open(path)) - 1  # Count actual rows

            print(f"\n  {name}:")
            print(f"    Rows: {total_rows:,}")
            print(f"    Columns: {len(df.columns)}")
            print(f"    Column names: {', '.join(df.columns[:6])}...")

            # Show key columns if present
            if 'qcew' in name:
                key_cols = ['AREA_FIPS', 'INDUSTRY_CODE', 'OWN_CODE', 'ANNUAL_AVG_EMPLVL', 'TOTAL_ANNUAL_WAGES']
            else:
                key_cols = ['OCC_CODE', 'O_GROUP', 'TOT_EMP', 'A_MEAN', 'NAICS', 'NAICS_TITLE']

            present = [c for c in key_cols if c in df.columns]
            if present:
                print(f"    Key columns present: {', '.join(present)}")


if __name__ == "__main__":
    output = download_and_process_bls_data()
    if output:
        print_validation_summary(output, load_config())
