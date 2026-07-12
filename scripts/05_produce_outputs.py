#!/usr/bin/env python3
"""
Produce Publication-Ready Tables and Figures

Generates all publication-ready outputs for the ChatGPT shock analysis:
- Table 1: Exposure measures and pre-period characteristics
- Table 2: Multiple risk-adjustment models (horizontal format)
- Table 3: High-Low portfolio + alphas
- Table 6: Composition check (excluding upstream)
- Figure 2: Placebo p-value distribution
- Table 8: Heterogeneity analysis

All outputs are saved in both .txt (human-readable) and .tex (LaTeX) formats.

Usage:
    python scripts/05_produce_outputs.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import subprocess

SCRIPTS_DIR = PROJECT_ROOT / 'scripts' / 'test_outputs'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SCRIPTS = [
    ('06_table1_sample_construct.py', 'Table 1: Sample descriptives'),
    ('07_table2_risk_models.py', 'Table 2: Risk adjustment models (5 columns with FF5+MOM)'),
    ('16_table3_eventstudy.py', 'Table 3: Event-study coefficients'),
    ('10_figure2_placebo_dist.py', 'Table 4: Placebo test summary (2 dates with different signs)'),
    ('08_table5_highlow_alpha.py', 'Table 5: High-Low portfolio alphas'),
    ('09_table6_upstream_exclusion.py', 'Table 6: Upstream exclusion'),
    ('11_table8_heterogeneity.py', 'Table 7: Heterogeneity (panels A-C only)'),
]


def main():
    print("=" * 70)
    print("PRODUCING PUBLICATION-READY OUTPUTS")
    print("=" * 70)

    for script, description in OUTPUT_SCRIPTS:
        print(f"\n{'=' * 70}")
        print(f"GENERATING: {description}")
        print(f"  Script: {script}")
        print(f"{'=' * 70}\n")

        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            check=False,
        )

        if result.returncode != 0:
            print(f"\nWARNING: {script} failed with exit code {result.returncode}")
            print("  Continuing with other outputs...")
        else:
            print(f"  ✓ {description} generated successfully")

    print(f"\n{'=' * 70}")
    print("OUTPUT PRODUCTION COMPLETE")
    print(f"  Results in: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
