"""
French Data Library - Download and Process All French Data

This module handles downloading and processing data from Ken French's Data Library:
- 49 Industry Portfolios (monthly and daily)
- Fama-French 3 Factors (monthly and daily)
- Fama-French 5 Factors (monthly and daily)
- Momentum Factor (monthly and daily)
- Industry Definitions (SIC codes to FF49 mapping)

Raw data structure (each dataset in its own folder with ZIP + extracted file):
    data/raw/french/
    ├── 49_industry_portfolios_monthly/
    │   ├── 49_Industry_Portfolios_CSV.zip
    │   └── 49_Industry_Portfolios.CSV
    ├── ff3_factors_monthly/
    │   ├── F-F_Research_Data_Factors_CSV.zip
    │   └── F-F_Research_Data_Factors.CSV
    └── ...

Processed data structure:
    data/processed/french/
    ├── 49_industry_portfolios_monthly/
    │   ├── metadata.txt
    │   ├── value_weighted_returns.csv
    │   └── ...
    └── ...
"""

import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """
    Download a file from URL to destination path.

    Returns True if successful, False otherwise.
    """
    try:
        print(f"    Downloading: {url.split('/')[-1]}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, 'wb') as f:
            f.write(response.content)

        return True

    except requests.exceptions.RequestException as e:
        print(f"    ERROR: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str) -> Optional[str]:
    """
    Extract contents of ZIP file to directory.

    Returns path to extracted file, or None if failed.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get the main file (CSV or TXT)
            files = zf.namelist()
            main_file = None
            for f in files:
                if f.lower().endswith(('.csv', '.txt')):
                    main_file = f
                    break

            if main_file is None:
                print(f"    ERROR: No CSV/TXT file found in {zip_path}")
                return None

            # Extract to directory
            zf.extract(main_file, extract_dir)
            extracted_path = os.path.join(extract_dir, main_file)
            return extracted_path

    except zipfile.BadZipFile as e:
        print(f"    ERROR: Bad ZIP file {zip_path}: {e}")
        return None


def download_all_french_data(config: Optional[dict] = None) -> Dict[str, Dict[str, str]]:
    """
    Download all French data files, each to its own folder.

    Returns dict mapping dataset names to {'zip': path, 'extracted': path}
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'french'

    base_url = config['data']['french']['base_url']
    datasets = config['data']['french']['datasets']

    downloaded = {}

    for dataset_name, zip_filename in datasets.items():
        print(f"\n  [{dataset_name}]")

        # Create folder for this dataset
        dataset_dir = raw_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download ZIP
        zip_url = f"{base_url}/{zip_filename}"
        zip_path = dataset_dir / zip_filename

        if not download_file(zip_url, str(zip_path)):
            print(f"    FAILED to download")
            continue

        # Extract
        extracted_path = extract_zip(str(zip_path), str(dataset_dir))

        if extracted_path:
            downloaded[dataset_name] = {
                'zip': str(zip_path),
                'extracted': extracted_path
            }
            print(f"    Extracted: {os.path.basename(extracted_path)}")
        else:
            print(f"    FAILED to extract")

    return downloaded


# =============================================================================
# DATE PARSING UTILITIES
# =============================================================================

def _parse_date(date_val, frequency: str = 'auto') -> str:
    """
    Parse date values from French data.

    Handles:
    - YYYYMMDD (8 digits) -> YYYY-MM-DD (daily)
    - YYYYMM (6 digits) -> YYYY-MM-01 (monthly)
    - YYYY (4 digits) -> YYYY-12-31 (annual)
    """
    try:
        date_str = str(int(float(date_val)))
    except (ValueError, TypeError):
        return str(date_val)

    if frequency == 'auto':
        if len(date_str) == 8:
            frequency = 'daily'
        elif len(date_str) == 6:
            frequency = 'monthly'
        elif len(date_str) == 4:
            frequency = 'annual'

    if frequency == 'daily' and len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    elif frequency == 'monthly' and len(date_str) == 6:
        return f"{date_str[:4]}-{date_str[4:6]}-01"
    elif frequency == 'annual' and len(date_str) == 4:
        return f"{date_str}-12-31"
    else:
        return date_str


# =============================================================================
# PORTFOLIO PROCESSING
# =============================================================================

def _identify_portfolio_tables(lines: List[str]) -> List[Tuple[str, int, int]]:
    """
    Identify table sections in portfolio files.

    Returns list of (table_name, start_line, end_line) tuples.
    """
    sections = []
    current_table = None
    current_start = None

    # Table headers start with 2 spaces then a letter
    header_pattern = re.compile(r'^  [A-Za-z]')

    for i, line in enumerate(lines):
        if header_pattern.match(line):
            if current_table is not None:
                sections.append((current_table, current_start, i - 1))
            current_table = line.strip()
            current_start = i + 1

    if current_table is not None:
        sections.append((current_table, current_start, len(lines) - 1))

    return sections


def _clean_portfolio_table_name(raw_name: str) -> str:
    """Convert portfolio table name to clean filename."""
    name_map = {
        "Average Value Weighted Returns -- Monthly": "value_weighted_returns",
        "Average Equal Weighted Returns -- Monthly": "equal_weighted_returns",
        "Average Value Weighted Returns -- Annual": "value_weighted_returns_annual",
        "Average Equal Weighted Returns -- Annual": "equal_weighted_returns_annual",
        "Average Value Weighted Returns -- Daily": "value_weighted_returns",
        "Average Equal Weighted Returns -- Daily": "equal_weighted_returns",
        "Number of Firms in Portfolios": "number_of_firms",
        "Average Firm Size": "average_firm_size",
        "Sum of BE / Sum of ME": "sum_be_sum_me",
        "Value-Weighted Average of BE/ME": "vw_average_be_me",
    }
    return name_map.get(raw_name.strip(), re.sub(r'\W+', '_', raw_name.strip().lower()))


def _parse_csv_table(lines: List[str], start: int, end: int) -> pd.DataFrame:
    """
    Parse a CSV-style table from lines.
    Auto-detects date format from data.
    """
    # Skip blank lines to find header
    data_start = start
    while data_start <= end and lines[data_start].strip() == '':
        data_start += 1

    if data_start > end:
        return pd.DataFrame()

    # Collect non-blank lines
    data_lines = []
    for i in range(data_start, end + 1):
        line = lines[i].strip()
        if line:
            data_lines.append(line)

    if not data_lines:
        return pd.DataFrame()

    # Parse header
    header = [h.strip() for h in data_lines[0].split(',')]
    if header[0] == '':
        header[0] = 'date'

    # Parse data rows
    rows = []
    for line in data_lines[1:]:
        if not re.match(r'^\s*\d', line):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= len(header):
            rows.append(parts[:len(header)])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)

    # Convert date column (auto-detect format)
    if 'date' in df.columns:
        df['date'] = df['date'].apply(lambda x: _parse_date(x, 'auto'))

    # Convert numeric columns
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace missing value indicators
    df = df.replace(-99.99, pd.NA)
    df = df.replace(-999, pd.NA)

    return df


def process_portfolios(extracted_path: str, output_dir: str) -> Dict[str, str]:
    """
    Process 49 Industry Portfolios file.

    Returns dict mapping table names to output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(extracted_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Source: Ken French Data Library\n")
        f.write(f"URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html\n")
        f.write(f"File: {os.path.basename(extracted_path)}\n")
        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- Original Header ---\n")
        f.write('\n'.join(lines[:10]))

    sections = _identify_portfolio_tables(lines)
    print(f"    Found {len(sections)} tables")

    output_files = {'metadata': metadata_path}

    for table_name, start, end in sections:
        clean_name = _clean_portfolio_table_name(table_name)
        df = _parse_csv_table(lines, start, end)

        if df.empty:
            continue

        output_path = os.path.join(output_dir, f"{clean_name}.csv")
        df.to_csv(output_path, index=False)
        output_files[clean_name] = output_path

        date_info = f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A"
        print(f"      {clean_name}: {len(df)} rows, {date_info}")

    return output_files


# =============================================================================
# FACTOR PROCESSING
# =============================================================================

def _find_factor_sections(lines: List[str]) -> List[Tuple[str, int, int]]:
    """
    Find sections in factor files (may have monthly + annual).

    Returns list of (section_name, start, end) tuples.
    """
    sections = []

    # Find header line (contains factor names)
    header_idx = None
    for i, line in enumerate(lines):
        if 'Mkt-RF' in line or ',Mom' in line or line.strip().startswith(',Mom'):
            header_idx = i
            break

    if header_idx is None:
        return sections

    # Find annual section (if exists)
    annual_idx = None
    for i, line in enumerate(lines[header_idx + 1:], start=header_idx + 1):
        if 'Annual' in line and ('Factor' in line or 'Return' in line):
            annual_idx = i
            break

    if annual_idx:
        sections.append(('monthly', header_idx, annual_idx - 1))
        sections.append(('annual', annual_idx, len(lines) - 1))
    else:
        # Single section - could be monthly or daily
        sections.append(('factors', header_idx, len(lines) - 1))

    return sections


def _parse_factor_table(lines: List[str], start: int, end: int,
                        frequency: str = 'auto') -> pd.DataFrame:
    """
    Parse a factor table.
    """
    # Find header in range
    header_idx = None
    header = None
    for i in range(start, min(end + 1, len(lines))):
        line = lines[i]
        if 'Mkt-RF' in line or ',Mom' in line or line.strip().startswith(',Mom'):
            header = [p.strip() for p in line.split(',')]
            if header[0] == '':
                header[0] = 'date'
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    # Parse data rows
    rows = []
    for i in range(header_idx + 1, end + 1):
        line = lines[i].strip()
        if not line or not re.match(r'^\d', line):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= len(header):
            rows.append(parts[:len(header)])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)

    # Convert date column
    if 'date' in df.columns:
        df['date'] = df['date'].apply(lambda x: _parse_date(x, frequency))

    # Convert numeric columns
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.replace(-99.99, pd.NA)
    df = df.replace(-999, pd.NA)

    return df


def process_factors(extracted_path: str, output_dir: str,
                    is_daily: bool = False) -> Dict[str, str]:
    """
    Process factor files (FF3, FF5, or Momentum).

    Returns dict mapping output names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(extracted_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Source: Ken French Data Library\n")
        f.write(f"URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html\n")
        f.write(f"File: {os.path.basename(extracted_path)}\n")
        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- Original Header ---\n")
        f.write('\n'.join(lines[:15]))

    output_files = {'metadata': metadata_path}

    if is_daily:
        # Daily files have single table
        df = _parse_factor_table(lines, 0, len(lines) - 1, 'daily')

        if not df.empty:
            output_path = os.path.join(output_dir, 'factors.csv')
            df.to_csv(output_path, index=False)
            output_files['factors'] = output_path
            print(f"      factors: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
    else:
        # Monthly files may have monthly + annual sections
        sections = _find_factor_sections(lines)

        for section_name, start, end in sections:
            freq = 'annual' if section_name == 'annual' else 'monthly'
            df = _parse_factor_table(lines, start, end, freq)

            if df.empty:
                continue

            output_path = os.path.join(output_dir, f'{section_name}_factors.csv')
            df.to_csv(output_path, index=False)
            output_files[f'{section_name}_factors'] = output_path
            print(f"      {section_name}: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")

    return output_files


# =============================================================================
# INDUSTRY DEFINITIONS PROCESSING
# =============================================================================

def process_industry_definitions(extracted_path: str, output_dir: str) -> Dict[str, str]:
    """
    Process Siccodes49.txt into clean CSV.

    Returns dict with output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(extracted_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Source: Ken French Data Library\n")
        f.write(f"URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html\n")
        f.write(f"File: {os.path.basename(extracted_path)}\n")
        f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nDescription: SIC code to FF49 industry mappings\n")

    # Parse industry definitions
    records = []
    current_code = None
    current_name = None
    current_abbrev = None

    header_pattern = re.compile(r'^\s*(\d+)\s+(\w+)\s+(.+)$')
    sic_pattern = re.compile(r'^\s+(\d{4})-(\d{4})\s+(.*)$')

    for line in lines:
        header_match = header_pattern.match(line)
        if header_match:
            current_code = int(header_match.group(1))
            current_abbrev = header_match.group(2).strip()
            current_name = header_match.group(3).strip()
            continue

        sic_match = sic_pattern.match(line)
        if sic_match and current_code is not None:
            records.append({
                'ff49_code': current_code,
                'ff49_abbrev': current_abbrev,
                'ff49_name': current_name,
                'sic_start': int(sic_match.group(1)),
                'sic_end': int(sic_match.group(2)),
                'sic_description': sic_match.group(3).strip()
            })

    df = pd.DataFrame(records)

    output_path = os.path.join(output_dir, 'sic_to_ff49.csv')
    df.to_csv(output_path, index=False)

    print(f"      {len(df)} SIC ranges across {df['ff49_code'].nunique()} industries")

    return {
        'metadata': metadata_path,
        'sic_to_ff49': output_path
    }


# =============================================================================
# MAIN PROCESSING ORCHESTRATION
# =============================================================================

def process_all_french_data(downloaded: Dict[str, Dict[str, str]],
                            config: Optional[dict] = None) -> Dict[str, Dict[str, str]]:
    """
    Process all downloaded French data files.

    Parameters
    ----------
    downloaded : dict
        Output from download_all_french_data()
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    dict
        Nested dictionary with processed file paths
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config['paths']['processed_data'] / 'french'

    output_files = {}

    for dataset_name, paths in downloaded.items():
        extracted_path = paths['extracted']
        output_dir = processed_dir / dataset_name

        print(f"\n  [{dataset_name}]")

        try:
            if 'industry_portfolios' in dataset_name:
                files = process_portfolios(extracted_path, str(output_dir))
            elif 'industry_definitions' in dataset_name:
                files = process_industry_definitions(extracted_path, str(output_dir))
            else:
                # Factor files
                is_daily = 'daily' in dataset_name
                files = process_factors(extracted_path, str(output_dir), is_daily)

            output_files[dataset_name] = files

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return output_files


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def download_and_process_french_data(config: Optional[dict] = None) -> Dict[str, Dict[str, str]]:
    """
    Download and process all French data in one step.

    Returns nested dictionary with processed file paths.
    """
    if config is None:
        config = load_config()

    print("=" * 70)
    print("DOWNLOADING FRENCH DATA")
    print("=" * 70)

    downloaded = download_all_french_data(config)

    if not downloaded:
        print("\nERROR: No files were downloaded")
        return {}

    print(f"\n\nDownloaded {len(downloaded)} datasets")

    print("\n" + "=" * 70)
    print("PROCESSING FRENCH DATA")
    print("=" * 70)

    processed = process_all_french_data(downloaded, config)

    return processed


if __name__ == "__main__":
    download_and_process_french_data()
