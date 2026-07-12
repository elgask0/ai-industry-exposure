"""
Felten AIOE/AIIE Data - Download and Process

This module handles downloading and processing AI exposure data from Felten et al.:
- AIOE (AI Occupational Exposure) - occupation-level exposure to AI
- AIIE (AI Industry Exposure) - industry-level exposure from employment-weighted AIOE
- AIGE (AI Geographic Exposure) - county-level exposure

Data sources:
- AIOE 2021 (SMJ paper): Narrow AI exposure (10 applications)
- Language Modeling 2023: LLM-specific exposure
- Image Generation 2023: Image AI-specific exposure

Classification systems (from original papers):
- Occupations: SOC 2010, 6-digit codes (collapsed from O*NET 8-digit)
- Industries: NAICS 2017, 4-digit codes (based on 2019 BLS employment data)

GitHub repository: https://github.com/AIOE-Data/AIOE
"""

import os
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

def download_felten_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """
    Download a file from URL to destination path.

    Returns True if successful, False otherwise.
    """
    try:
        print(f"    Downloading: {url.split('/')[-1]}")
        response = requests.get(url, timeout=timeout, allow_redirects=True)
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


def download_all_felten_data(config: Optional[dict] = None) -> Dict[str, str]:
    """
    Download all Felten data files from GitHub.

    Returns dict mapping dataset names to downloaded file paths.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config['paths']['raw_data'] / 'felten'
    raw_dir.mkdir(parents=True, exist_ok=True)

    # File mappings: (config_key, local_filename)
    files_to_download = [
        ('aioe_2021_file', 'AIOE_DataAppendix.xlsx'),
        ('aiie_lm_2023_file', 'Language_Modeling_AIOE_and_AIIE.xlsx'),
        ('aiie_ig_2023_file', 'Image_Generation_AIOE_and_AIIE.xlsx'),
    ]

    downloaded = {}

    for config_key, local_name in files_to_download:
        remote_name = config['data']['felten'].get(config_key)
        if not remote_name:
            print(f"  WARNING: {config_key} not found in config")
            continue

        # GitHub raw URLs need URL encoding for spaces
        base_url = "https://github.com/AIOE-Data/AIOE/raw/main"
        url = f"{base_url}/{remote_name.replace(' ', '%20')}"

        dest_path = raw_dir / local_name

        print(f"\n  [{local_name}]")
        if download_felten_file(url, str(dest_path)):
            downloaded[local_name] = str(dest_path)

    return downloaded


# =============================================================================
# EXCEL INSPECTION
# =============================================================================

def inspect_xlsx_structure(filepath: str) -> Dict:
    """
    Inspect Excel file structure.

    Returns dict with sheet names, columns, and sample data.
    """
    xl = pd.ExcelFile(filepath)

    structure = {
        'filename': os.path.basename(filepath),
        'sheets': {}
    }

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name)

        structure['sheets'][sheet_name] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_rows': df.head(3).to_dict('records')
        }

    return structure


# =============================================================================
# SOC/NAICS VALIDATION
# =============================================================================

def validate_soc_codes(df: pd.DataFrame, code_column: str = 'soc_code') -> Dict:
    """
    Validate SOC codes and determine version (SOC 2010 vs 2018).

    Key difference: Computer Programmers is 15-1131 in SOC 2010, 15-1251 in SOC 2018.
    """
    validation = {
        'total_codes': len(df),
        'unique_codes': df[code_column].nunique(),
        'sample_codes': df[code_column].head(10).tolist(),
        'soc_version': 'unknown',
        'validation_checks': {}
    }

    # Check for SOC 2010 indicator (15-1131 = Computer Programmers)
    has_15_1131 = df[df[code_column] == '15-1131'].shape[0] > 0
    has_15_1251 = df[df[code_column] == '15-1251'].shape[0] > 0

    validation['validation_checks']['has_15_1131'] = has_15_1131
    validation['validation_checks']['has_15_1251'] = has_15_1251

    if has_15_1131 and not has_15_1251:
        validation['soc_version'] = 'SOC 2010'
    elif has_15_1251 and not has_15_1131:
        validation['soc_version'] = 'SOC 2018'
    else:
        validation['soc_version'] = 'indeterminate'

    # Check format (should be XX-XXXX)
    format_check = df[code_column].str.match(r'^\d{2}-\d{4}$')
    validation['format_valid'] = format_check.all()
    validation['invalid_format_count'] = (~format_check).sum()

    return validation


def validate_naics_codes(df: pd.DataFrame, code_column: str = 'naics_code') -> Dict:
    """
    Validate NAICS codes.
    """
    # Convert to string for validation
    codes = df[code_column].astype(str)

    validation = {
        'total_codes': len(df),
        'unique_codes': df[code_column].nunique(),
        'sample_codes': df[code_column].head(10).tolist(),
        'min_value': df[code_column].min(),
        'max_value': df[code_column].max(),
    }

    # Check digit lengths
    lengths = codes.str.len()
    validation['digit_lengths'] = lengths.value_counts().to_dict()

    # All should be 4 digits (or can start with leading zeros represented as 3)
    validation['all_4_digit'] = (lengths == 4).all()

    return validation


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_aioe_2021(filepath: str, output_dir: str) -> Dict[str, str]:
    """
    Process AIOE_DataAppendix.xlsx into clean CSVs.

    Creates:
    - aioe_by_occupation.csv
    - aiie_by_industry.csv
    - aige_by_geography.csv
    - application_ability_matrix.csv
    - ability_level_exposure.csv
    - metadata.txt
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}

    xl = pd.ExcelFile(filepath)

    # Process Appendix A: AIOE by Occupation
    print("    Processing: AIOE by Occupation (Appendix A)")
    df_aioe = pd.read_excel(xl, sheet_name='Appendix A')
    df_aioe.columns = ['soc_code', 'occupation_title', 'aioe']
    df_aioe['soc_code'] = df_aioe['soc_code'].astype(str)

    output_path = os.path.join(output_dir, 'aioe_by_occupation.csv')
    df_aioe.to_csv(output_path, index=False)
    output_files['aioe_by_occupation'] = output_path
    print(f"      {len(df_aioe)} occupations, AIOE range: {df_aioe['aioe'].min():.4f} to {df_aioe['aioe'].max():.4f}")

    # Process Appendix B: AIIE by Industry
    print("    Processing: AIIE by Industry (Appendix B)")
    df_aiie = pd.read_excel(xl, sheet_name='Appendix B')
    df_aiie.columns = ['naics_code', 'industry_title', 'aiie']
    df_aiie['naics_code'] = df_aiie['naics_code'].astype(str).str.zfill(4)

    output_path = os.path.join(output_dir, 'aiie_by_industry.csv')
    df_aiie.to_csv(output_path, index=False)
    output_files['aiie_by_industry'] = output_path
    print(f"      {len(df_aiie)} industries, AIIE range: {df_aiie['aiie'].min():.4f} to {df_aiie['aiie'].max():.4f}")

    # Process Appendix C: AIGE by Geography
    print("    Processing: AIGE by Geography (Appendix C)")
    df_aige = pd.read_excel(xl, sheet_name='Appendix C')
    df_aige.columns = ['fips_code', 'geographic_area', 'aige']
    df_aige['fips_code'] = df_aige['fips_code'].astype(str).str.zfill(5)

    output_path = os.path.join(output_dir, 'aige_by_geography.csv')
    df_aige.to_csv(output_path, index=False)
    output_files['aige_by_geography'] = output_path
    print(f"      {len(df_aige)} geographic areas, AIGE range: {df_aige['aige'].min():.4f} to {df_aige['aige'].max():.4f}")

    # Process Appendix D: Application-Ability Matrix
    print("    Processing: Application-Ability Matrix (Appendix D)")
    df_matrix = pd.read_excel(xl, sheet_name='Appendix D')
    df_matrix = df_matrix.rename(columns={df_matrix.columns[0]: 'ability'})

    output_path = os.path.join(output_dir, 'application_ability_matrix.csv')
    df_matrix.to_csv(output_path, index=False)
    output_files['application_ability_matrix'] = output_path
    print(f"      {len(df_matrix)} abilities × {len(df_matrix.columns)-1} applications")

    # Process Appendix E: Ability-Level Exposure
    print("    Processing: Ability-Level Exposure (Appendix E)")
    df_ability = pd.read_excel(xl, sheet_name='Appendix E')
    df_ability.columns = ['ability', 'ability_level_ai_exposure']

    output_path = os.path.join(output_dir, 'ability_level_exposure.csv')
    df_ability.to_csv(output_path, index=False)
    output_files['ability_level_exposure'] = output_path
    print(f"      {len(df_ability)} abilities")

    # Generate metadata
    metadata = [
        f"Source: Felten, Raj, Seamans (2021) Strategic Management Journal",
        f"File: {os.path.basename(filepath)}",
        f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Classification Systems:",
        f"  Occupations: SOC 2010, 6-digit (e.g., 15-1131)",
        f"  Industries: NAICS 2017 (inferred), 4-digit (e.g., 5112)",
        f"",
        f"Sheets Processed:",
        f"  Appendix A: AIOE by Occupation ({len(df_aioe)} rows)",
        f"  Appendix B: AIIE by Industry ({len(df_aiie)} rows)",
        f"  Appendix C: AIGE by Geography ({len(df_aige)} rows)",
        f"  Appendix D: Application-Ability Matrix ({len(df_matrix)} rows × {len(df_matrix.columns)-1} cols)",
        f"  Appendix E: Ability-Level Exposure ({len(df_ability)} rows)",
        f"",
        f"Score Ranges:",
        f"  AIOE: {df_aioe['aioe'].min():.4f} to {df_aioe['aioe'].max():.4f} (mean: {df_aioe['aioe'].mean():.4f})",
        f"  AIIE: {df_aiie['aiie'].min():.4f} to {df_aiie['aiie'].max():.4f} (mean: {df_aiie['aiie'].mean():.4f})",
        f"  AIGE: {df_aige['aige'].min():.4f} to {df_aige['aige'].max():.4f} (mean: {df_aige['aige'].mean():.4f})",
    ]

    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata))
    output_files['metadata'] = metadata_path

    return output_files


def process_language_modeling(filepath: str, output_dir: str) -> Dict[str, str]:
    """
    Process Language Modeling AIOE and AIIE.xlsx into clean CSVs.

    Creates:
    - lm_aioe_by_occupation.csv
    - lm_aiie_by_industry.csv
    - metadata.txt
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}

    xl = pd.ExcelFile(filepath)

    # Process LM AIOE
    print("    Processing: LM AIOE by Occupation")
    df_aioe = pd.read_excel(xl, sheet_name='LM AIOE')
    df_aioe.columns = ['soc_code', 'occupation_title', 'lm_aioe']
    df_aioe['soc_code'] = df_aioe['soc_code'].astype(str)

    output_path = os.path.join(output_dir, 'lm_aioe_by_occupation.csv')
    df_aioe.to_csv(output_path, index=False)
    output_files['lm_aioe_by_occupation'] = output_path
    print(f"      {len(df_aioe)} occupations, LM-AIOE range: {df_aioe['lm_aioe'].min():.4f} to {df_aioe['lm_aioe'].max():.4f}")

    # Process LM AIIE
    print("    Processing: LM AIIE by Industry")
    df_aiie = pd.read_excel(xl, sheet_name='LM AIIE')
    df_aiie.columns = ['naics_code', 'industry_title', 'lm_aiie']
    df_aiie['naics_code'] = df_aiie['naics_code'].astype(str).str.zfill(4)

    output_path = os.path.join(output_dir, 'lm_aiie_by_industry.csv')
    df_aiie.to_csv(output_path, index=False)
    output_files['lm_aiie_by_industry'] = output_path
    print(f"      {len(df_aiie)} industries, LM-AIIE range: {df_aiie['lm_aiie'].min():.4f} to {df_aiie['lm_aiie'].max():.4f}")

    # Generate metadata
    metadata = [
        f"Source: Felten, Raj, Seamans (2023) arXiv:2303.01157 - Language Modeling",
        f"File: {os.path.basename(filepath)}",
        f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Classification Systems:",
        f"  Occupations: SOC 2010, 6-digit (same base as AIOE 2021)",
        f"  Industries: NAICS 2017 (inferred), 4-digit",
        f"",
        f"Sheets Processed:",
        f"  LM AIOE: Language Modeling AIOE by Occupation ({len(df_aioe)} rows)",
        f"  LM AIIE: Language Modeling AIIE by Industry ({len(df_aiie)} rows)",
        f"",
        f"Score Ranges:",
        f"  LM-AIOE: {df_aioe['lm_aioe'].min():.4f} to {df_aioe['lm_aioe'].max():.4f} (mean: {df_aioe['lm_aioe'].mean():.4f})",
        f"  LM-AIIE: {df_aiie['lm_aiie'].min():.4f} to {df_aiie['lm_aiie'].max():.4f} (mean: {df_aiie['lm_aiie'].mean():.4f})",
    ]

    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata))
    output_files['metadata'] = metadata_path

    return output_files


def process_image_generation(filepath: str, output_dir: str) -> Dict[str, str]:
    """
    Process Image Generation AIOE and AIIE.xlsx into clean CSVs.

    Creates:
    - ig_aioe_by_occupation.csv
    - ig_aiie_by_industry.csv
    - metadata.txt
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}

    xl = pd.ExcelFile(filepath)

    # Process IG AIOE
    print("    Processing: IG AIOE by Occupation")
    df_aioe = pd.read_excel(xl, sheet_name='IG AIOE')
    df_aioe.columns = ['soc_code', 'occupation_title', 'ig_aioe']
    df_aioe['soc_code'] = df_aioe['soc_code'].astype(str)

    output_path = os.path.join(output_dir, 'ig_aioe_by_occupation.csv')
    df_aioe.to_csv(output_path, index=False)
    output_files['ig_aioe_by_occupation'] = output_path
    print(f"      {len(df_aioe)} occupations, IG-AIOE range: {df_aioe['ig_aioe'].min():.4f} to {df_aioe['ig_aioe'].max():.4f}")

    # Process IG AIIE
    print("    Processing: IG AIIE by Industry")
    df_aiie = pd.read_excel(xl, sheet_name='IG AIIE')
    df_aiie.columns = ['naics_code', 'industry_title', 'ig_aiie']
    df_aiie['naics_code'] = df_aiie['naics_code'].astype(str).str.zfill(4)

    output_path = os.path.join(output_dir, 'ig_aiie_by_industry.csv')
    df_aiie.to_csv(output_path, index=False)
    output_files['ig_aiie_by_industry'] = output_path
    print(f"      {len(df_aiie)} industries, IG-AIIE range: {df_aiie['ig_aiie'].min():.4f} to {df_aiie['ig_aiie'].max():.4f}")

    # Generate metadata
    metadata = [
        f"Source: Felten, Raj, Seamans - Image Generation (from GitHub)",
        f"File: {os.path.basename(filepath)}",
        f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Classification Systems:",
        f"  Occupations: SOC 2010, 6-digit (same base as AIOE 2021)",
        f"  Industries: NAICS 2017 (inferred), 4-digit",
        f"",
        f"Sheets Processed:",
        f"  IG AIOE: Image Generation AIOE by Occupation ({len(df_aioe)} rows)",
        f"  IG AIIE: Image Generation AIIE by Industry ({len(df_aiie)} rows)",
        f"",
        f"Score Ranges:",
        f"  IG-AIOE: {df_aioe['ig_aioe'].min():.4f} to {df_aioe['ig_aioe'].max():.4f} (mean: {df_aioe['ig_aioe'].mean():.4f})",
        f"  IG-AIIE: {df_aiie['ig_aiie'].min():.4f} to {df_aiie['ig_aiie'].max():.4f} (mean: {df_aiie['ig_aiie'].mean():.4f})",
    ]

    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata))
    output_files['metadata'] = metadata_path

    return output_files


# =============================================================================
# MAIN PROCESSING ORCHESTRATION
# =============================================================================

def process_all_felten_data(downloaded: Dict[str, str],
                            config: Optional[dict] = None) -> Dict[str, Dict[str, str]]:
    """
    Process all downloaded Felten data files.

    Parameters
    ----------
    downloaded : dict
        Output from download_all_felten_data() - maps filename to path
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
    processed_dir = root / config['paths']['processed_data'] / 'felten'

    output_files = {}

    # Process AIOE 2021
    if 'AIOE_DataAppendix.xlsx' in downloaded:
        print("\n  [aioe_2021]")
        filepath = downloaded['AIOE_DataAppendix.xlsx']
        output_dir = processed_dir / 'aioe_2021'
        output_files['aioe_2021'] = process_aioe_2021(filepath, str(output_dir))

    # Process Language Modeling
    if 'Language_Modeling_AIOE_and_AIIE.xlsx' in downloaded:
        print("\n  [language_modeling]")
        filepath = downloaded['Language_Modeling_AIOE_and_AIIE.xlsx']
        output_dir = processed_dir / 'language_modeling'
        output_files['language_modeling'] = process_language_modeling(filepath, str(output_dir))

    # Process Image Generation
    if 'Image_Generation_AIOE_and_AIIE.xlsx' in downloaded:
        print("\n  [image_generation]")
        filepath = downloaded['Image_Generation_AIOE_and_AIIE.xlsx']
        output_dir = processed_dir / 'image_generation'
        output_files['image_generation'] = process_image_generation(filepath, str(output_dir))

    return output_files


def download_and_process_felten_data(config: Optional[dict] = None) -> Dict[str, Dict[str, str]]:
    """
    Download and process all Felten data in one step.

    Returns nested dictionary with processed file paths.
    """
    if config is None:
        config = load_config()

    print("=" * 70)
    print("DOWNLOADING FELTEN DATA")
    print("=" * 70)

    downloaded = download_all_felten_data(config)

    if not downloaded:
        print("\nERROR: No files were downloaded")
        return {}

    print(f"\n\nDownloaded {len(downloaded)} files")

    print("\n" + "=" * 70)
    print("PROCESSING FELTEN DATA")
    print("=" * 70)

    processed = process_all_felten_data(downloaded, config)

    return processed


# =============================================================================
# VALIDATION
# =============================================================================

def print_validation_summary(processed_files: Dict[str, Dict[str, str]],
                             config: Optional[dict] = None) -> None:
    """Print comprehensive validation summary for Felten data."""

    if config is None:
        config = load_config()

    root = get_project_root()
    felten_dir = root / config['paths']['processed_data'] / 'felten'

    print("\n" + "=" * 70)
    print("FELTEN DATA VALIDATION SUMMARY")
    print("=" * 70)

    # SOC Code Validation
    print("\n" + "-" * 70)
    print("SOC CODE VALIDATION (Confirming SOC 2010)")
    print("-" * 70)

    aioe_path = felten_dir / 'aioe_2021' / 'aioe_by_occupation.csv'
    if aioe_path.exists():
        df = pd.read_csv(aioe_path, dtype={'soc_code': str})
        validation = validate_soc_codes(df)

        print(f"  Total occupations: {validation['total_codes']}")
        print(f"  SOC Version: {validation['soc_version']}")
        print(f"  Has 15-1131 (SOC 2010 Computer Programmers): {validation['validation_checks']['has_15_1131']}")
        print(f"  Has 15-1251 (SOC 2018 Computer Programmers): {validation['validation_checks']['has_15_1251']}")
        print(f"  Format valid (XX-XXXX): {validation['format_valid']}")

    # NAICS Code Validation
    print("\n" + "-" * 70)
    print("NAICS CODE VALIDATION")
    print("-" * 70)

    aiie_path = felten_dir / 'aioe_2021' / 'aiie_by_industry.csv'
    if aiie_path.exists():
        df = pd.read_csv(aiie_path, dtype={'naics_code': str})
        validation = validate_naics_codes(df)

        print(f"  Total industries: {validation['total_codes']}")
        print(f"  Unique industries: {validation['unique_codes']}")
        print(f"  NAICS range: {validation['min_value']} to {validation['max_value']}")
        print(f"  Digit lengths: {validation['digit_lengths']}")
        print(f"  All 4-digit: {validation['all_4_digit']}")

    # Dataset Summary
    print("\n" + "-" * 70)
    print("PROCESSED DATASETS")
    print("-" * 70)

    datasets = [
        ('aioe_2021', [
            ('aioe_by_occupation.csv', 'AIOE (Occupation)', 'aioe'),
            ('aiie_by_industry.csv', 'AIIE (Industry)', 'aiie'),
            ('aige_by_geography.csv', 'AIGE (Geography)', 'aige'),
        ]),
        ('language_modeling', [
            ('lm_aioe_by_occupation.csv', 'LM-AIOE (Occupation)', 'lm_aioe'),
            ('lm_aiie_by_industry.csv', 'LM-AIIE (Industry)', 'lm_aiie'),
        ]),
        ('image_generation', [
            ('ig_aioe_by_occupation.csv', 'IG-AIOE (Occupation)', 'ig_aioe'),
            ('ig_aiie_by_industry.csv', 'IG-AIIE (Industry)', 'ig_aiie'),
        ]),
    ]

    print(f"\n{'Dataset':<30} {'Rows':>8} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 78)

    for folder, files in datasets:
        for filename, display_name, score_col in files:
            filepath = felten_dir / folder / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                if score_col in df.columns:
                    print(f"{display_name:<30} {len(df):>8} {df[score_col].min():>10.4f} "
                          f"{df[score_col].max():>10.4f} {df[score_col].mean():>10.4f} "
                          f"{df[score_col].std():>10.4f}")
                else:
                    print(f"{display_name:<30} {len(df):>8} {'N/A':>10}")
            else:
                print(f"{display_name:<30} {'MISSING':>8}")

    # Missing value check
    print("\n" + "-" * 70)
    print("MISSING VALUES CHECK")
    print("-" * 70)

    all_ok = True
    for folder, files in datasets:
        for filename, display_name, score_col in files:
            filepath = felten_dir / folder / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                missing = df.isna().sum().sum()
                if missing > 0:
                    print(f"  {display_name}: {missing} missing values")
                    all_ok = False

    if all_ok:
        print("  All datasets complete - no missing values")


if __name__ == "__main__":
    processed = download_and_process_felten_data()
    print_validation_summary(processed)
