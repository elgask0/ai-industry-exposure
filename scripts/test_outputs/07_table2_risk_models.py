#!/usr/bin/env python3
"""
Table 2: Multiple Risk-Adjustment Models (Horizontal Format)

Generates a single table with 4 columns showing DiD coefficient stability
across different risk models:
- (1) Raw excess returns (no factor controls)
- (2) CAPM alpha (MktRF control only)
- (3) FF5 alpha (baseline - 5 factors)
- (4) FF5 alpha excluding upstream producers (re-ranked)

Outputs:
- outputs/test/table2.txt (human-readable text format)
- outputs/test/table2.tex (LaTeX booktabs + siunitx format)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_prep import create_binary_treatment, load_panel
from scripts.lib.estimator import run_single_regression

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Upstream portfolios to exclude
UPSTREAM_PORTFOLIOS = ['Chips', 'Softw', 'ElcEq', 'BusEq']


def run_baseline_specification(df, return_col='ret_ew_ex',
                                factor_set='ff5',
                                fe_type='portfolio_only',
                                se_type='normal'):
    """Run a single DiD specification using binary treatment."""
    # Create post2 × treated interaction
    df['post2_x_treated'] = df['post2'] * df['treated']

    # Define specification
    x_vars = ['post2_x_treated']

    # Run regression
    result = run_single_regression(
        df,
        return_col,
        x_vars,
        fe_type=fe_type,
        factor_set=factor_set,
        se_type=se_type,
    )

    return result


def spec_raw_returns(treated_df):
    """Column (1): Raw excess returns (no factor controls)."""
    result = run_baseline_specification(
        treated_df,
        factor_set='none',
        fe_type='portfolio_only',
        se_type='normal'
    )

    coef_data = result['coefficients'].get('post2_x_treated', {})

    return {
        'name': 'Raw',
        'coef': coef_data.get('coef'),
        'se': coef_data.get('se'),
        't': coef_data.get('t'),
        'p': coef_data.get('p'),
        'nobs': result.get('nobs'),
        'n_treated': treated_df[treated_df['treated'] == 1]['ff49'].nunique(),
        'n_control': treated_df[treated_df['treated'] == 0]['ff49'].nunique(),
        'r2': result.get('r2'),
    }


def spec_capm(treated_df):
    """Column (2): CAPM alpha (MktRF control only)."""
    result = run_baseline_specification(
        treated_df,
        factor_set='mktrf',
        fe_type='portfolio_only',
        se_type='normal'
    )

    coef_data = result['coefficients'].get('post2_x_treated', {})

    return {
        'name': 'CAPM',
        'factors': 'MKTRF',
        'coef': coef_data.get('coef'),
        'se': coef_data.get('se'),
        't': coef_data.get('t'),
        'p': coef_data.get('p'),
        'nobs': result.get('nobs'),
        'n_treated': treated_df[treated_df['treated'] == 1]['ff49'].nunique(),
        'n_control': treated_df[treated_df['treated'] == 0]['ff49'].nunique(),
        'r2': result.get('r2'),
    }


def spec_ff5(treated_df):
    """Column (3): FF5 alpha (baseline)."""
    result = run_baseline_specification(
        treated_df,
        factor_set='ff5',
        fe_type='portfolio_only',
        se_type='normal'
    )

    coef_data = result['coefficients'].get('post2_x_treated', {})

    return {
        'name': 'FF5',
        'factors': 'FF5',
        'coef': coef_data.get('coef'),
        'se': coef_data.get('se'),
        't': coef_data.get('t'),
        'p': coef_data.get('p'),
        'nobs': result.get('nobs'),
        'n_treated': treated_df[treated_df['treated'] == 1]['ff49'].nunique(),
        'n_control': treated_df[treated_df['treated'] == 0]['ff49'].nunique(),
        'r2': result.get('r2'),
    }


def spec_ff5_mom(treated_df):
    """Column (5): FF5+MOM alpha (R7 specification)."""
    result = run_baseline_specification(
        treated_df,
        factor_set='ff5_mom',  # FF5 + Momentum
        fe_type='portfolio_only',
        se_type='normal'
    )

    coef_data = result['coefficients'].get('post2_x_treated', {})

    return {
        'name': 'FF5+MOM',
        'factors': 'FF5+MOM',
        'coef': coef_data.get('coef'),
        'se': coef_data.get('se'),
        't': coef_data.get('t'),
        'p': coef_data.get('p'),
        'nobs': result.get('nobs'),
        'n_treated': treated_df[treated_df['treated'] == 1]['ff49'].nunique(),
        'n_control': treated_df[treated_df['treated'] == 0]['ff49'].nunique(),
        'r2': result.get('r2'),
    }


def spec_ff5_no_upstream(df):
    """Column (4): FF5 alpha excluding upstream producers."""
    # Exclude upstream portfolios
    df_filtered = df[~df['ff49_abbrev'].isin(UPSTREAM_PORTFOLIOS)].copy()

    # Re-rank treatment within restricted sample
    treated_restricted = create_binary_treatment(df_filtered, 'aiie_23_ew_z', 'binary_q1_q4')

    # Filter to sample_60m
    treated_restricted = treated_restricted[treated_restricted['sample_60m'] == 1].copy()

    # Run regression
    result = run_baseline_specification(
        treated_restricted,
        factor_set='ff5',
        fe_type='portfolio_only',
        se_type='normal'
    )

    coef_data = result['coefficients'].get('post2_x_treated', {})

    return {
        'name': 'FF5 (no upstream)',
        'factors': 'FF5',
        'coef': coef_data.get('coef'),
        'se': coef_data.get('se'),
        't': coef_data.get('t'),
        'p': coef_data.get('p'),
        'nobs': result.get('nobs'),
        'n_treated': treated_restricted[treated_restricted['treated'] == 1]['ff49'].nunique(),
        'n_control': treated_restricted[treated_restricted['treated'] == 0]['ff49'].nunique(),
        'r2': result.get('r2'),
    }


def format_text_table(spec1, spec2, spec3, spec4, spec5):
    """Format table as human-readable text (horizontal)."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 2. PORTFOLIO RETURNS BY AIIE EXPOSURE: MULTIPLE RISK-ADJUSTMENT MODELS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Specification: r_p,t = α_p + β(Post_t × Treated_p) + Θ'Model_t + ε_p,t")
    lines.append("")
    lines.append("| | (1) Raw | (2) CAPM | (3) FF5 | (4) FF5 (no upstream) | (5) FF5+MOM |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    # Post × Treated row
    def get_stars(p):
        return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    coef1 = f"{spec1['coef']:+.2f}{get_stars(spec1['p'])}" if spec1['coef'] else ""
    coef2 = f"{spec2['coef']:+.2f}{get_stars(spec2['p'])}" if spec2['coef'] else ""
    coef3 = f"{spec3['coef']:+.2f}{get_stars(spec3['p'])}" if spec3['coef'] else ""
    coef4 = f"{spec4['coef']:+.2f}{get_stars(spec4['p'])}" if spec4['coef'] else ""
    coef5 = f"{spec5['coef']:+.2f}{get_stars(spec5['p'])}" if spec5['coef'] else ""

    lines.append(f"| **Post × Treated** | {coef1} | {coef2} | {coef3} | {coef4} | {coef5} |")

    # SE row
    se1 = f"({spec1['se']:.2f})" if spec1['se'] else ""
    se2 = f"({spec2['se']:.2f})" if spec2['se'] else ""
    se3 = f"({spec3['se']:.2f})" if spec3['se'] else ""
    se4 = f"({spec4['se']:.2f})" if spec4['se'] else ""
    se5 = f"({spec5['se']:.2f})" if spec5['se'] else ""

    lines.append(f"|  | {se1} | {se2} | {se3} | {se4} | {se5} |")

    # t-stat row
    t1 = f"[{spec1['t']:.2f}]" if spec1['t'] else ""
    t2 = f"[{spec2['t']:.2f}]" if spec2['t'] else ""
    t3 = f"[{spec3['t']:.2f}]" if spec3['t'] else ""
    t4 = f"[{spec4['t']:.2f}]" if spec4['t'] else ""
    t5 = f"[{spec5['t']:.2f}]" if spec5['t'] else ""

    lines.append(f"|  | {t1} | {t2} | {t3} | {t4} | {t5} |")

    # Specification details
    lines.append("| **Portfolio FE** | Yes | Yes | Yes | Yes | Yes |")
    lines.append(f"| **Factors** | None | MKTRF | FF5 | FF5 | FF5+MOM |")
    lines.append("| **SE clustering** | OLS | OLS | OLS | OLS | OLS |")
    lines.append(f"| **Observations** | {spec1['nobs']:,} | {spec2['nobs']:,} | {spec3['nobs']:,} | {spec4['nobs']:,} | {spec5['nobs']:,} |")
    lines.append(f"| **N (treated)** | {spec1['n_treated']} | {spec2['n_treated']} | {spec3['n_treated']} | {spec4['n_treated']} | {spec5['n_treated']} |")
    lines.append(f"| **N (control)** | {spec1['n_control']} | {spec2['n_control']} | {spec3['n_control']} | {spec4['n_control']} | {spec5['n_control']} |")

    lines.append("")
    lines.append("*Note: Coefficients in % per month. Standard errors in parentheses. t-statistics in brackets.")
    lines.append("*** p<0.01, ** p<0.05, * p<0.10. Column 4 excludes FF49 portfolios: Computers (Chips),")
    lines.append("Software (Softw), Electronic Equipment (ElcEq), Business Equipment (BusEq); treatment")
    lines.append("re-ranked within subsample. Column 5 adds Momentum factor to FF5. Post period = Dec 2022 + Jan 2023.")
    lines.append("Pre period = Dec 2017 to Nov 2022.*")

    return '\n'.join(lines)


def format_latex_table(spec1, spec2, spec3, spec4, spec5):
    """Format table as LaTeX with booktabs + siunitx (horizontal)."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Portfolio returns by AIIE exposure: Multiple risk-adjustment models}")
    lines.append("\\label{tab:risk_models}")
    lines.append("")
    lines.append("\\begin{threeparttable}")
    lines.append("")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l S S S S S}")
    lines.append("\\toprule")
    lines.append(" & {(1) Raw} & {(2) CAPM} & {(3) FF5} & {(4) FF5 (no upstream)} & {(5) FF5+MOM} \\\\")
    lines.append("\\midrule")

    # Helper for stars
    def get_stars(p):
        return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    # Post × Treated row
    coef1 = f"{spec1['coef']:+.2f}" if spec1['coef'] else ""
    coef2 = f"{spec2['coef']:+.2f}" if spec2['coef'] else ""
    coef3 = f"{spec3['coef']:+.2f}" if spec3['coef'] else ""
    coef4 = f"{spec4['coef']:+.2f}" if spec4['coef'] else ""
    coef5 = f"{spec5['coef']:+.2f}" if spec5['coef'] else ""

    lines.append(f"Post $\\times$ Treated & "
                 f"{coef1}{get_stars(spec1['p'])} & "
                 f"{coef2}{get_stars(spec2['p'])} & "
                 f"{coef3}{get_stars(spec3['p'])} & "
                 f"{coef4}{get_stars(spec4['p'])} & "
                 f"{coef5}{get_stars(spec5['p'])} \\\\")

    # SE row
    se1 = f"({spec1['se']:.2f})" if spec1['se'] else ""
    se2 = f"({spec2['se']:.2f})" if spec2['se'] else ""
    se3 = f"({spec3['se']:.2f})" if spec3['se'] else ""
    se4 = f"({spec4['se']:.2f})" if spec4['se'] else ""
    se5 = f"({spec5['se']:.2f})" if spec5['se'] else ""

    lines.append(f" & {se1} & {se2} & {se3} & {se4} & {se5} \\\\")

    # t-stat row
    t1 = f"[{spec1['t']:.2f}]" if spec1['t'] else ""
    t2 = f"[{spec2['t']:.2f}]" if spec2['t'] else ""
    t3 = f"[{spec3['t']:.2f}]" if spec3['t'] else ""
    t4 = f"[{spec4['t']:.2f}]" if spec4['t'] else ""
    t5 = f"[{spec5['t']:.2f}]" if spec5['t'] else ""

    lines.append(f" & {t1} & {t2} & {t3} & {t4} & {t5} \\\\")
    lines.append("\\addlinespace")
    lines.append("Portfolio FE & Yes & Yes & Yes & Yes & Yes \\\\")
    lines.append("Factors & None & MKTRF & FF5 & FF5 & FF5+MOM \\\\")
    lines.append("SE clustering & OLS & OLS & OLS & OLS & OLS \\\\")
    lines.append(f"Observations & {spec1['nobs']:,} & {spec2['nobs']:,} & {spec3['nobs']:,} & {spec4['nobs']:,} & {spec5['nobs']:,} \\\\")
    lines.append(f"N (treated) & {spec1['n_treated']} & {spec2['n_treated']} & {spec3['n_treated']} & {spec4['n_treated']} & {spec5['n_treated']} \\\\")
    lines.append(f"N (control) & {spec1['n_control']} & {spec2['n_control']} & {spec3['n_control']} & {spec4['n_control']} & {spec5['n_control']} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")
    lines.append("\\end{threeparttable}")
    lines.append("")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Notes:} Coefficients in \\% per month. Standard errors in parentheses. t-statistics in brackets. *** p<0.01, ** p<0.05, * p<0.10. Column 4 excludes FF49 portfolios: Computers (Chips), Software (Softw), Electronic Equipment (ElcEq), Business Equipment (BusEq); treatment re-ranked within subsample. Column 5 adds Momentum factor to FF5. Post period = Dec 2022 + Jan 2023. Pre period = Dec 2017 to Nov 2022.")
    lines.append("\\end{tablenotes}")
    lines.append("")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("TABLE 2: MULTIPLE RISK-ADJUSTMENT MODELS")
    print("=" * 70)

    # Load data and create treatment
    print("\n[1/6] Loading data and creating treatment...")
    df = load_panel()
    treated_df = create_binary_treatment(df, 'aiie_23_ew_z', 'binary_q1_q4')

    # Filter to sample_60m
    treated_df = treated_df[treated_df['sample_60m'] == 1].copy()

    print(f"  Sample: {len(treated_df):,} observations")

    # Run specifications
    print("[2/6] Spec (1): Raw returns...")
    spec1 = spec_raw_returns(treated_df)

    print("[3/6] Spec (2): CAPM...")
    spec2 = spec_capm(treated_df)

    print("[4/6] Spec (3): FF5 (baseline)...")
    spec3 = spec_ff5(treated_df)

    print("[5/6] Spec (4): FF5 excluding upstream...")
    spec4 = spec_ff5_no_upstream(df)

    print("[6/6] Spec (5): FF5+MOM...")
    spec5 = spec_ff5_mom(treated_df)

    # Generate outputs
    print("\n[7/7] Saving outputs...")

    # Save text version
    txt_content = format_text_table(spec1, spec2, spec3, spec4, spec5)
    txt_path = OUTPUT_DIR / 'txt' / 'table2.txt'
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(txt_content)
    print(f"  Saved: txt/{txt_path.name}")

    # Save LaTeX version
    latex_content = format_latex_table(spec1, spec2, spec3, spec4, spec5)
    latex_path = OUTPUT_DIR / 'tex' / 'table2.tex'
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: tex/{latex_path.name}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"(1) Raw:       γ = {spec1['coef']:+.2f} (t = {spec1['t']:.2f})")
    print(f"(2) CAPM:      γ = {spec2['coef']:+.2f} (t = {spec2['t']:.2f})")
    print(f"(3) FF5:       γ = {spec3['coef']:+.2f} (t = {spec3['t']:.2f})")
    print(f"(4) No upstream: γ = {spec4['coef']:+.2f} (t = {spec4['t']:.2f})")
    print(f"(5) FF5+MOM:   γ = {spec5['coef']:+.2f} (t = {spec5['t']:.2f})")
    print(f"  N(treated) = {spec5['n_treated']}, N(control) = {spec5['n_control']}")
    print("\nDone.")


if __name__ == '__main__':
    main()
