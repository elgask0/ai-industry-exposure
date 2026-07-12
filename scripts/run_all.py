#!/usr/bin/env python3
"""
Run the full ChatGPT shock analysis pipeline.

Steps:
  01  Download raw data (French, Felten, BLS, BEA, crosswalks)
  02  Build exposure measures (AIIE + ExpWB at FF49 level)
  02b Build IPP intensity measures (BEA Fixed Assets → FF49)
  03  Build panel dataset (returns + factors + exposures)
  04  Run analysis (DiD regressions + pre-trends + reports)
  05  Produce publication tables/figures (Table 1-3, 6, 8, Figure 2)

Usage:
    python scripts/run_all.py                  # Full pipeline
    python scripts/run_all.py --skip-download  # Skip step 01 (data already downloaded)
    python scripts/run_all.py --skip-ipp       # Skip step 02b (IPP already built)
    python scripts/run_all.py --skip-outputs    # Skip step 05 (outputs already produced)
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = [
    ('01_download_data.py', 'Download raw data'),
    ('02_build_exposures.py', 'Build exposure measures'),
    ('02b_build_ipp.py', 'Build IPP intensity measures'),
    ('03_build_panel.py', 'Build panel dataset'),
    ('04_run_analysis.py', 'Run analysis'),
    ('05_produce_outputs.py', 'Produce publication tables/figures'),
]


def main():
    parser = argparse.ArgumentParser(description='Run full analysis pipeline')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip step 01 (data already downloaded)')
    parser.add_argument('--skip-ipp', action='store_true',
                        help='Skip step 02b (IPP measures already built)')
    parser.add_argument('--skip-outputs', action='store_true',
                        help='Skip step 05 (outputs already produced)')
    args = parser.parse_args()

    print("=" * 70)
    print("CHATGPT SHOCK — FULL PIPELINE")
    print("=" * 70)

    for script, description in STEPS:
        if args.skip_download and script == '01_download_data.py':
            print(f"\n  Skipping: {description}")
            continue
        if args.skip_ipp and script == '02b_build_ipp.py':
            print(f"\n  Skipping: {description}")
            continue
        if args.skip_outputs and script == '05_produce_outputs.py':
            print(f"\n  Skipping: {description}")
            continue

        print(f"\n{'=' * 70}")
        print(f"STEP: {description}")
        print(f"  Script: {script}")
        print(f"{'=' * 70}\n")

        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            check=False,
        )
        if result.returncode != 0:
            print(f"\nERROR: {script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"  Regression results: outputs/analysis/")
    print(f"  Publication tables/figures: outputs/analysis/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
