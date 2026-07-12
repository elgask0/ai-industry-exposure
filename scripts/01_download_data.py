#!/usr/bin/env python3
"""
01_download_data.py - Download and process all data files

Downloads and processes data from:

1. Ken French Data Library:
   - 49 Industry Portfolios (monthly and daily)
   - Fama-French 3 Factors (monthly and daily)
   - Fama-French 5 Factors (monthly and daily)
   - Momentum Factor (monthly and daily)
   - Industry Definitions (SIC code mappings)

2. Felten AIOE/AIIE Data (GitHub):
   - AIOE 2021 (occupation-level AI exposure)
   - AIIE 2021 (industry-level AI exposure)
   - Language Modeling AIOE/AIIE 2023
   - Image Generation AIOE/AIIE 2023

3. Crosswalk Data (Census/BLS):
   - SIC 1987 to NAICS 2002
   - NAICS vintage updates (2002->2007->2012->2017)
   - SOC 2010 to 2018

4. BLS OEWS Data:
   - OEWS May 2021 national by 4-digit NAICS
   - OEWS May 2018 national estimates
   - OEWS hybrid structure 2019

5. BEA Data:
   - GDP by Industry Value Added (Table 1) for 2021
   - BEA Industry/Commodity NAICS Concordance

Usage:
    python scripts/01_download_data.py [--french-only] [--felten-only] [--crosswalks-only] [--bls-only] [--bea-only]
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data.french import download_and_process_french_data, load_config
from src.data.felten import (
    download_and_process_felten_data,
    print_validation_summary as print_felten_validation
)
from src.data.crosswalks import (
    download_and_process_crosswalks,
    print_validation_summary as print_crosswalks_validation
)
from src.data.bls import (
    download_and_process_bls_data,
    print_validation_summary as print_bls_validation
)
from src.data.bea import (
    download_and_process_bea_data,
    print_validation_summary as print_bea_validation
)


def print_validation_summary(processed_files: dict, config: dict):
    """Print comprehensive validation summary."""

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    french_dir = project_root / 'data/processed/french'

    estimation_start = config['estimation']['start']
    estimation_end = config['estimation']['end']
    event_start = config['event']['pre_window_start']
    event_end = config['event']['post_window_end']

    print(f"\nRequired coverage:")
    print(f"  Estimation window: {estimation_start} to {estimation_end}")
    print(f"  Event window:      {event_start} to {event_end}")

    # Dataset validation
    datasets_to_check = [
        ('49_industry_portfolios_monthly', 'value_weighted_returns.csv', 'Portfolios Monthly'),
        ('49_industry_portfolios_daily', 'value_weighted_returns.csv', 'Portfolios Daily'),
        ('ff3_factors_monthly', 'monthly_factors.csv', 'FF3 Monthly'),
        ('ff3_factors_daily', 'factors.csv', 'FF3 Daily'),
        ('ff5_factors_monthly', 'monthly_factors.csv', 'FF5 Monthly'),
        ('ff5_factors_daily', 'factors.csv', 'FF5 Daily'),
        ('momentum_monthly', 'monthly_factors.csv', 'Momentum Monthly'),
        ('momentum_daily', 'factors.csv', 'Momentum Daily'),
        ('industry_definitions', 'sic_to_ff49.csv', 'Industry Definitions'),
    ]

    print("\n" + "-" * 70)
    print(f"{'Dataset':<22} {'Rows':>10} {'Date Range':<27} {'Status'}")
    print("-" * 70)

    for folder, filename, display_name in datasets_to_check:
        file_path = french_dir / folder / filename

        if not file_path.exists():
            print(f"{display_name:<22} {'MISSING':>10}")
            continue

        df = pd.read_csv(file_path)

        if 'date' in df.columns:
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            df_est = df[(df['date'] >= estimation_start) & (df['date'] <= estimation_end)]
            df_evt = df[(df['date'] >= event_start) & (df['date'] <= event_end)]
            status = "OK" if len(df_est) > 0 and len(df_evt) > 0 else "PARTIAL"
        else:
            date_range = "N/A (static)"
            status = "OK" if len(df) > 0 else "EMPTY"

        print(f"{display_name:<22} {len(df):>10,} {date_range:<27} {status}")

    # Factor column validation
    print("\n" + "-" * 70)
    print("FACTOR COLUMNS")
    print("-" * 70)

    factor_checks = [
        ('ff3_factors_monthly/monthly_factors.csv', ['Mkt-RF', 'SMB', 'HML', 'RF'], 'FF3'),
        ('ff5_factors_monthly/monthly_factors.csv', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'], 'FF5'),
        ('momentum_monthly/monthly_factors.csv', ['Mom'], 'Momentum'),
    ]

    for rel_path, expected_cols, name in factor_checks:
        file_path = french_dir / rel_path
        if file_path.exists():
            df = pd.read_csv(file_path)
            present = [c for c in expected_cols if c in df.columns]
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"  {name}: MISSING {missing}")
            else:
                print(f"  {name}: OK - {present}")
        else:
            print(f"  {name}: FILE NOT FOUND")

    # Portfolio validation
    print("\n" + "-" * 70)
    print("PORTFOLIO STRUCTURE")
    print("-" * 70)

    portfolio_path = french_dir / '49_industry_portfolios_monthly/value_weighted_returns.csv'
    if portfolio_path.exists():
        df = pd.read_csv(portfolio_path)
        industries = [c for c in df.columns if c != 'date']
        print(f"  Industries: {len(industries)} (expected: 49)")
        print(f"  Sample: {', '.join(industries[:5])}...")

    definitions_path = french_dir / 'industry_definitions/sic_to_ff49.csv'
    if definitions_path.exists():
        df = pd.read_csv(definitions_path)
        print(f"  SIC ranges: {len(df)}")
        print(f"  Unique industries: {df['ff49_code'].nunique()}")

    # Event window sample
    print("\n" + "-" * 70)
    print("EVENT WINDOW SAMPLE (FF5 Monthly)")
    print("-" * 70)

    ff5_path = french_dir / 'ff5_factors_monthly/monthly_factors.csv'
    if ff5_path.exists():
        df = pd.read_csv(ff5_path)
        sample = df[(df['date'] >= '2022-09-01') & (df['date'] <= '2023-02-01')]
        if not sample.empty:
            print(sample.to_string(index=False))


def generate_data_dictionary(processed_files: dict, config: dict):
    """Generate DATA_DICTIONARY.md file."""

    french_dir = project_root / 'data/processed/french'
    output_path = project_root / 'data/processed/DATA_DICTIONARY.md'

    lines = [
        "# Data Dictionary - French Data Library",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "All datasets from Ken French Data Library for the ChatGPT-Shock event study.",
        "",
        "**Source:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "",
        "---",
        "",
        "## Data Sources",
        "",
        "| Dataset | Frequency | Tables | Description |",
        "|---------|-----------|--------|-------------|",
    ]

    # Add source info
    source_info = [
        ("49_industry_portfolios_monthly", "Monthly", "8", "Returns + characteristics for 49 industries"),
        ("49_industry_portfolios_daily", "Daily", "2", "Daily returns only (no characteristics)"),
        ("ff3_factors_monthly", "Monthly", "2", "Market, Size, Value factors + annual"),
        ("ff3_factors_daily", "Daily", "1", "Daily FF3 factors"),
        ("ff5_factors_monthly", "Monthly", "2", "FF3 + Profitability, Investment + annual"),
        ("ff5_factors_daily", "Daily", "1", "Daily FF5 factors"),
        ("momentum_monthly", "Monthly", "2", "Momentum factor + annual"),
        ("momentum_daily", "Daily", "1", "Daily momentum factor"),
        ("industry_definitions", "Static", "1", "SIC to FF49 industry mapping"),
    ]

    for name, freq, tables, desc in source_info:
        lines.append(f"| {name} | {freq} | {tables} | {desc} |")

    # Portfolio section
    lines.extend([
        "",
        "---",
        "",
        "## 49 Industry Portfolios",
        "",
        "### Monthly (8 tables)",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| value_weighted_returns.csv | Value-weighted monthly returns (%) |",
        "| equal_weighted_returns.csv | Equal-weighted monthly returns (%) |",
        "| value_weighted_returns_annual.csv | Value-weighted annual returns (%) |",
        "| equal_weighted_returns_annual.csv | Equal-weighted annual returns (%) |",
        "| number_of_firms.csv | Count of firms per industry |",
        "| average_firm_size.csv | Average market cap (millions USD) |",
        "| sum_be_sum_me.csv | Aggregate book-to-market |",
        "| vw_average_be_me.csv | Value-weighted average B/M |",
        "",
        "### Daily (2 tables)",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| value_weighted_returns.csv | Value-weighted daily returns (%) |",
        "| equal_weighted_returns.csv | Equal-weighted daily returns (%) |",
        "",
        "**Note:** Daily data has fewer tables because firm characteristics are calculated monthly.",
        "",
    ])

    # Add statistics
    vw_path = french_dir / '49_industry_portfolios_monthly/value_weighted_returns.csv'
    if vw_path.exists():
        df = pd.read_csv(vw_path)
        lines.extend([
            "### Statistics",
            "",
            f"- **Monthly rows:** {len(df):,}",
            f"- **Industries:** {len(df.columns) - 1}",
            f"- **Date range:** {df['date'].min()} to {df['date'].max()}",
        ])

    vw_daily_path = french_dir / '49_industry_portfolios_daily/value_weighted_returns.csv'
    if vw_daily_path.exists():
        df = pd.read_csv(vw_daily_path)
        lines.append(f"- **Daily rows:** {len(df):,}")
        lines.append(f"- **Daily range:** {df['date'].min()} to {df['date'].max()}")

    # Factor sections
    lines.extend([
        "",
        "---",
        "",
        "## Fama-French Factors",
        "",
        "### FF3 Factors",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| Mkt-RF | Market excess return (%) |",
        "| SMB | Size factor - Small Minus Big (%) |",
        "| HML | Value factor - High Minus Low B/M (%) |",
        "| RF | Risk-free rate (%) |",
        "",
        "### FF5 Factors (adds to FF3)",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| RMW | Profitability - Robust Minus Weak (%) |",
        "| CMA | Investment - Conservative Minus Aggressive (%) |",
        "",
        "### Momentum Factor",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| Mom | Winners Minus Losers (%) |",
        "",
    ])

    # Add factor stats
    for name, path in [
        ("FF3 Monthly", "ff3_factors_monthly/monthly_factors.csv"),
        ("FF3 Daily", "ff3_factors_daily/factors.csv"),
        ("FF5 Monthly", "ff5_factors_monthly/monthly_factors.csv"),
        ("FF5 Daily", "ff5_factors_daily/factors.csv"),
        ("Momentum Monthly", "momentum_monthly/monthly_factors.csv"),
        ("Momentum Daily", "momentum_daily/factors.csv"),
    ]:
        file_path = french_dir / path
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                lines.append(f"- **{name}:** {len(df):,} rows, {df['date'].min()} to {df['date'].max()}")

    # Industry definitions
    lines.extend([
        "",
        "---",
        "",
        "## Industry Definitions",
        "",
        "**File:** `industry_definitions/sic_to_ff49.csv`",
        "",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| ff49_code | int | Industry number (1-49) |",
        "| ff49_abbrev | str | Short code (e.g., 'Softw') |",
        "| ff49_name | str | Full name |",
        "| sic_start | int | SIC range start |",
        "| sic_end | int | SIC range end |",
        "| sic_description | str | Description |",
        "",
    ])

    definitions_path = french_dir / 'industry_definitions/sic_to_ff49.csv'
    if definitions_path.exists():
        df = pd.read_csv(definitions_path)
        lines.append(f"**Total SIC ranges:** {len(df)}")
        lines.append(f"**Industries:** {df['ff49_code'].nunique()}")

        lines.extend([
            "",
            "### Industry List",
            "",
            "| Code | Abbrev | Name |",
            "|------|--------|------|",
        ])

        for _, row in df.drop_duplicates('ff49_code').sort_values('ff49_code').iterrows():
            lines.append(f"| {row['ff49_code']} | {row['ff49_abbrev']} | {row['ff49_name']} |")

    # Event dates
    lines.extend([
        "",
        "---",
        "",
        "## Event Study Dates",
        "",
        "| Period | Start | End |",
        "|--------|-------|-----|",
        f"| Estimation | {config['estimation']['start']} | {config['estimation']['end']} |",
        f"| Pre-event | {config['event']['pre_window_start']} | {config['event']['pre_window_end']} |",
        f"| Post-event | {config['event']['post_window_start']} | {config['event']['post_window_end']} |",
        f"| Event | {config['event']['event_date']} | ChatGPT release |",
        "",
        "---",
        "",
        "## Data Notes",
        "",
        "1. All returns are in **percentage points** (2.5 = 2.5%)",
        "2. Missing values (-99.99, -999) converted to NA",
        "3. Portfolios rebalanced end of June each year",
        "4. Daily files have no characteristics (monthly only)",
        "",
    ])

    # Directory structure
    lines.extend([
        "---",
        "",
        "## Directory Structure",
        "",
        "```",
        "data/",
        "├── raw/french/                    # Downloaded data",
        "│   ├── 49_industry_portfolios_monthly/",
        "│   │   ├── *.zip                  # Original ZIP",
        "│   │   └── *.CSV                  # Extracted file",
        "│   └── .../",
        "│",
        "└── processed/french/              # Clean CSVs",
        "    ├── 49_industry_portfolios_monthly/",
        "    │   ├── metadata.txt",
        "    │   ├── value_weighted_returns.csv",
        "    │   └── ...",
        "    └── .../",
        "```",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nGenerated: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download and process data for ChatGPT-Shock event study'
    )
    parser.add_argument('--french-only', action='store_true',
                        help='Only download French Data Library')
    parser.add_argument('--felten-only', action='store_true',
                        help='Only download Felten AIOE/AIIE data')
    parser.add_argument('--crosswalks-only', action='store_true',
                        help='Only download crosswalk data')
    parser.add_argument('--bls-only', action='store_true',
                        help='Only download BLS OEWS data')
    parser.add_argument('--bea-only', action='store_true',
                        help='Only download BEA data')
    args = parser.parse_args()

    print("=" * 70)
    print("CHATGPT-SHOCK DATA PIPELINE - STEP 1: DOWNLOAD DATA")
    print("=" * 70)
    print()

    try:
        config = load_config()
        print("Loaded configuration from config.yaml")
    except FileNotFoundError:
        print("ERROR: config.yaml not found")
        sys.exit(1)

    french_output = {}
    felten_output = {}
    crosswalks_output = {}
    bls_output = {}
    bea_output = {}

    # Determine what to download
    download_all = not (args.french_only or args.felten_only or args.crosswalks_only or args.bls_only or args.bea_only)

    # Download French data
    if download_all or args.french_only:
        print("\n" + "=" * 70)
        print("SECTION 1: FRENCH DATA LIBRARY")
        print("=" * 70)

        try:
            french_output = download_and_process_french_data(config)
        except Exception as e:
            print(f"ERROR downloading French data: {e}")
            import traceback
            traceback.print_exc()
            if not args.french_only:
                print("Continuing with Felten data...")
            else:
                sys.exit(1)

        if french_output:
            print_validation_summary(french_output, config)

    # Download Felten data
    if download_all or args.felten_only:
        print("\n" + "=" * 70)
        print("SECTION 2: FELTEN AIOE/AIIE DATA")
        print("=" * 70)

        try:
            felten_output = download_and_process_felten_data(config)
        except Exception as e:
            print(f"ERROR downloading Felten data: {e}")
            import traceback
            traceback.print_exc()
            if not args.felten_only:
                print("Continuing with other data...")
            else:
                sys.exit(1)

        if felten_output:
            print_felten_validation(felten_output, config)

    # Download Crosswalk data
    if download_all or args.crosswalks_only:
        print("\n" + "=" * 70)
        print("SECTION 3: CROSSWALK DATA")
        print("=" * 70)

        try:
            crosswalks_output = download_and_process_crosswalks(config)
        except Exception as e:
            print(f"ERROR downloading crosswalk data: {e}")
            import traceback
            traceback.print_exc()
            if not args.crosswalks_only:
                print("Continuing...")
            else:
                sys.exit(1)

        if crosswalks_output:
            print_crosswalks_validation(crosswalks_output, config)

    # Download BLS data
    if download_all or args.bls_only:
        print("\n" + "=" * 70)
        print("SECTION 4: BLS OEWS DATA")
        print("=" * 70)

        try:
            bls_output = download_and_process_bls_data(config)
        except Exception as e:
            print(f"ERROR downloading BLS data: {e}")
            import traceback
            traceback.print_exc()
            if not args.bls_only:
                print("Continuing...")
            else:
                sys.exit(1)

        if bls_output:
            print_bls_validation(bls_output, config)

    # Download BEA data
    if download_all or args.bea_only:
        print("\n" + "=" * 70)
        print("SECTION 5: BEA DATA")
        print("=" * 70)

        try:
            bea_output = download_and_process_bea_data(config)
        except Exception as e:
            print(f"ERROR downloading BEA data: {e}")
            import traceback
            traceback.print_exc()
            if not args.bea_only:
                print("Continuing...")
            else:
                sys.exit(1)

        if bea_output:
            print_bea_validation(bea_output, config)

    # Generate/update data dictionary if French data was processed
    if french_output:
        print("\n" + "=" * 70)
        print("GENERATING DATA DICTIONARY")
        print("=" * 70)

        generate_data_dictionary(french_output, config)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    if french_output:
        print("\nFrench Data:")
        print("  Raw: data/raw/french/[dataset]/")
        print("  Processed: data/processed/french/[dataset]/")

    if felten_output:
        print("\nFelten Data:")
        print("  Raw: data/raw/felten/*.xlsx")
        print("  Processed: data/processed/felten/[aioe_2021|language_modeling|image_generation]/")

    if crosswalks_output:
        print("\nCrosswalk Data:")
        print("  Raw: data/raw/crosswalks/")
        print("  Processed: data/processed/crosswalks/")

    if bls_output:
        print("\nBLS OEWS Data:")
        print("  Raw: data/raw/bls/")
        print("  Processed: data/processed/bls/")

    if bea_output:
        print("\nBEA Data:")
        print("  Raw: data/raw/bea/")
        print("  Processed: data/processed/bea/")

    print("\nDocumentation:")
    print("  data/processed/DATA_DICTIONARY.md")
    print("  data/processed/felten/CLASSIFICATION_NOTES.md")


if __name__ == "__main__":
    main()
