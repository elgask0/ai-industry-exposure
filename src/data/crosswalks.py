"""
Crosswalk Data - Download and Process

This module handles downloading and processing crosswalk files:
- NAICS crosswalks (Census Bureau): 1987 SIC to 2002 NAICS, and NAICS vintage updates
- SOC crosswalks (BLS): SOC 2010 to 2018

Data sources:
- Census NAICS: https://www.census.gov/naics/concordances/
- BLS SOC: https://www.bls.gov/soc/2018/

Output files (processed):
- soc_2010_to_2018.csv: SOC occupation code crosswalk
- sic87_to_naics02.csv: SIC 1987 to NAICS 2002
- naics02_to_naics07.csv: NAICS 2002 to 2007
- naics07_to_naics12.csv: NAICS 2007 to 2012
- naics12_to_naics17.csv: NAICS 2012 to 2017
"""

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


# =============================================================================
# DATA SOURCES
# =============================================================================

CROSSWALK_SOURCES = {
    # NAICS crosswalks from Census Bureau
    'sic87_to_naics02': {
        'url': 'https://www.census.gov/naics/concordances/1987_SIC_to_2002_NAICS.xls',
        'description': 'SIC 1987 to NAICS 2002 crosswalk',
        'source': 'Census Bureau'
    },
    'naics02_to_naics07': {
        'url': 'https://www.census.gov/naics/concordances/2002_to_2007_NAICS.xls',
        'description': 'NAICS 2002 to 2007 crosswalk',
        'source': 'Census Bureau'
    },
    'naics07_to_naics12': {
        'url': 'https://www.census.gov/naics/concordances/2007_to_2012_NAICS.xls',
        'description': 'NAICS 2007 to 2012 crosswalk',
        'source': 'Census Bureau'
    },
    'naics12_to_naics17': {
        'url': 'https://www.census.gov/naics/concordances/2012_to_2017_NAICS.xlsx',
        'description': 'NAICS 2012 to 2017 crosswalk',
        'source': 'Census Bureau'
    },
    # SOC crosswalk from BLS
    'soc_2010_to_2018': {
        'url': 'https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx',
        'description': 'SOC 2010 to 2018 occupation crosswalk',
        'source': 'BLS'
    },
    # SOC crosswalk documentation (PDF)
    'soc_crosswalk_notes': {
        'url': 'https://www.bls.gov/soc/2018/soc_note_2010_to_2018_crosswalk.pdf',
        'description': 'SOC 2010 to 2018 crosswalk documentation',
        'source': 'BLS',
        'is_pdf': True
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

        # Add headers to avoid 403 errors from BLS and other government sites
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


def download_all_crosswalks(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Download all crosswalk files.

    Returns dict mapping names to downloaded file paths.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'crosswalks'
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    print("\n  Downloading crosswalk files...")

    for name, info in CROSSWALK_SOURCES.items():
        url = info['url']
        ext = url.split('.')[-1]
        filename = f"{name}.{ext}"
        dest_path = raw_dir / filename

        if download_file(url, str(dest_path)):
            downloaded[name] = str(dest_path)
        else:
            print(f"    WARNING: Failed to download {name}")

    return downloaded


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_soc_crosswalk(raw_path: str, processed_dir: Path) -> str:
    """Process SOC 2010 to 2018 crosswalk."""
    print("    Processing SOC 2010 to 2018 crosswalk...")

    # Skip initial rows and use row with actual column names as header
    df = pd.read_excel(raw_path, skiprows=8, header=0)

    # If first row looks like headers, use it
    if df.iloc[0, 0] == '2010 SOC Code':
        df.columns = df.iloc[0]
        df = df.iloc[1:]

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

    # Standardize column names
    rename_map = {}
    for col in df.columns:
        if '2010' in col and 'code' in col:
            rename_map[col] = 'soc_2010_code'
        elif '2010' in col and 'title' in col:
            rename_map[col] = 'soc_2010_title'
        elif '2018' in col and 'code' in col:
            rename_map[col] = 'soc_2018_code'
        elif '2018' in col and 'title' in col:
            rename_map[col] = 'soc_2018_title'

    if rename_map:
        df = df.rename(columns=rename_map)

    # Keep only relevant columns
    cols_to_keep = [c for c in df.columns if 'soc_' in c]
    if cols_to_keep:
        df = df[cols_to_keep]

    # Remove empty rows
    df = df.dropna(how='all')

    # Clean codes
    for col in df.columns:
        if 'code' in col:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    output_path = processed_dir / 'soc_2010_to_2018.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows)")

    return str(output_path)


def process_sic_to_naics02(raw_path: str, processed_dir: Path) -> str:
    """Process SIC 1987 to NAICS 2002 crosswalk."""
    print("    Processing SIC 1987 to NAICS 2002 crosswalk...")

    df = pd.read_excel(raw_path)

    # Rename columns directly based on expected structure
    # Columns: SIC, SIC Title (and note), 2002 NAICS, 2002 NAICS Title
    df.columns = ['sic_code', 'sic_title', 'naics_2002_code', 'naics_2002_title']

    df = df.dropna(how='all')

    # Clean codes
    df['sic_code'] = df['sic_code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['naics_2002_code'] = df['naics_2002_code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df['sic_title'] = df['sic_title'].astype(str).str.strip()
    df['naics_2002_title'] = df['naics_2002_title'].astype(str).str.strip()

    # Remove rows where codes are 'nan' or empty
    df = df[~df['sic_code'].isin(['nan', '', 'None'])]
    df = df[~df['naics_2002_code'].isin(['nan', '', 'None'])]

    output_path = processed_dir / 'sic87_to_naics02.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows)")

    return str(output_path)


def process_naics_crosswalk(raw_path: str, processed_dir: Path,
                            from_year: str, to_year: str) -> str:
    """Process NAICS vintage crosswalk."""
    print(f"    Processing NAICS {from_year} to {to_year} crosswalk...")

    # Read raw to find header row
    df_raw = pd.read_excel(raw_path, header=None)

    # Find the header row (contains "Code")
    header_row = 0
    for i in range(min(10, len(df_raw))):
        row_str = ' '.join(str(x) for x in df_raw.iloc[i].values)
        if 'Code' in row_str and 'NAICS' in row_str:
            header_row = i
            break

    # Read data starting after header row
    df = pd.read_excel(raw_path, skiprows=header_row + 1, header=None)

    # Take only first 4 columns (codes and titles)
    df = df.iloc[:, :4]

    # Set standard column names
    df.columns = [f'naics_{from_year}_code', f'naics_{from_year}_title',
                  f'naics_{to_year}_code', f'naics_{to_year}_title']

    df = df.dropna(how='all')

    # Clean codes
    df[f'naics_{from_year}_code'] = df[f'naics_{from_year}_code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df[f'naics_{to_year}_code'] = df[f'naics_{to_year}_code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    df[f'naics_{from_year}_title'] = df[f'naics_{from_year}_title'].astype(str).str.strip()
    df[f'naics_{to_year}_title'] = df[f'naics_{to_year}_title'].astype(str).str.strip()

    # Remove rows where codes are 'nan' or empty
    df = df[~df[f'naics_{from_year}_code'].isin(['nan', '', 'None'])]
    df = df[~df[f'naics_{to_year}_code'].isin(['nan', '', 'None'])]

    output_path = processed_dir / f'naics{from_year}_to_naics{to_year}.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path.name} ({len(df):,} rows)")

    return str(output_path)


def process_all_crosswalks(downloaded_files: Dict[str, str],
                           config: Optional[dict] = None) -> Dict[str, str]:
    """
    Process all downloaded crosswalk files.

    Returns dict mapping names to processed file paths.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config['paths']['processed_data'] / 'crosswalks'
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed = {}

    print("\n  Processing crosswalk files...")

    # SOC crosswalk
    if 'soc_2010_to_2018' in downloaded_files:
        path = process_soc_crosswalk(downloaded_files['soc_2010_to_2018'], processed_dir)
        processed['soc_2010_to_2018'] = path

    # SIC to NAICS 2002
    if 'sic87_to_naics02' in downloaded_files:
        path = process_sic_to_naics02(downloaded_files['sic87_to_naics02'], processed_dir)
        processed['sic87_to_naics02'] = path

    # NAICS crosswalks
    naics_pairs = [
        ('naics02_to_naics07', '2002', '2007'),
        ('naics07_to_naics12', '2007', '2012'),
        ('naics12_to_naics17', '2012', '2017'),
    ]

    for name, from_yr, to_yr in naics_pairs:
        if name in downloaded_files:
            path = process_naics_crosswalk(downloaded_files[name], processed_dir, from_yr, to_yr)
            processed[name] = path

    return processed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def download_and_process_crosswalks(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Main entry point: download and process all crosswalk data.

    Returns dict with all processed file paths.
    """
    if config is None:
        config = load_config()

    print("\n" + "-" * 50)
    print("CROSSWALK DATA")
    print("-" * 50)

    # Download
    downloaded = download_all_crosswalks(config)

    if not downloaded:
        print("  ERROR: No files downloaded")
        return {}

    # Process (skip PDFs)
    downloaded_non_pdf = {k: v for k, v in downloaded.items()
                          if not CROSSWALK_SOURCES.get(k, {}).get('is_pdf', False)}
    processed = process_all_crosswalks(downloaded_non_pdf, config)

    # Summary
    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'crosswalks'
    processed_dir = root / config['paths']['processed_data'] / 'crosswalks'

    print("\n  Summary:")
    print(f"    Raw files: {raw_dir}")
    print(f"    Processed files: {processed_dir}")
    print(f"    Downloaded: {len(downloaded)} files")
    print(f"    Processed: {len(processed)} CSV files")

    return {
        'downloaded': downloaded,
        'processed': processed
    }


def print_validation_summary(output: Dict, config: dict):
    """Print validation summary for crosswalk data."""
    print("\n" + "-" * 50)
    print("CROSSWALK VALIDATION")
    print("-" * 50)

    processed = output.get('processed', {})

    for name, path in processed.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  {name}: {len(df):,} rows, {len(df.columns)} columns")
            print(f"    Columns: {', '.join(df.columns[:4])}...")


if __name__ == "__main__":
    output = download_and_process_crosswalks()
    if output:
        print_validation_summary(output, load_config())
