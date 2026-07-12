#!/usr/bin/env python3
"""
Table 6: Composition Check - Excluding Upstream Producers

Compares baseline DiD results with a specification that excludes
obvious AI-producing tech portfolios (upstream providers).

Outputs:
- outputs/test/table6.txt (human-readable text format)
- outputs/test/table6.tex (LaTeX booktabs + siunitx format)
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


def spec_baseline(treated_df):
    """Column (1): Baseline FF5 specification."""
    result = run_baseline_specification(
        treated_df,
        factor_set='ff5',
        fe_type='portfolio_only',
        se_type='normal'
    )

    coef_data = result['coefficients'].get('post2_x_treated', {})

    return {
        'name': 'Baseline',
        'coef': coef_data.get('coef'),
        'se': coef_data.get('se'),
        't': coef_data.get('t'),
        'p': coef_data.get('p'),
        'nobs': result.get('nobs'),
        'n_treated': treated_df[treated_df['treated'] == 1]['ff49'].nunique(),
        'n_control': treated_df[treated_df['treated'] == 0]['ff49'].nunique(),
        'r2': result.get('r2'),
    }


def spec_excluding_upstream(df):
    """Column (2): FF5 excluding upstream producers (re-ranked)."""
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
        'name': 'Excluding upstream',
        'coef': coef_data.get('coef'),
        'se': coef_data.get('se'),
        't': coef_data.get('t'),
        'p': coef_data.get('p'),
        'nobs': result.get('nobs'),
        'n_treated': treated_restricted[treated_restricted['treated'] == 1]['ff49'].nunique(),
        'n_control': treated_restricted[treated_restricted['treated'] == 0]['ff49'].nunique(),
        'r2': result.get('r2'),
    }


def format_text_table(spec1, spec2):
    """Format table as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 6. COMPOSITION CHECK: EXCLUDING UPSTREAM PRODUCER PORTFOLIOS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Specification: r_p,t = α_p + β(Post_t × Treated_p) + Θ'FF5_t + ε_p,t")
    lines.append("")
    lines.append("| | Baseline | Excluding upstream producers |")
    lines.append("|---|---:|---:|")

    # Post × Treated row
    def get_stars(p):
        return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    coef1 = f"{spec1['coef']:+.2f}{get_stars(spec1['p'])}" if spec1['coef'] else ""
    coef2 = f"{spec2['coef']:+.2f}{get_stars(spec2['p'])}" if spec2['coef'] else ""

    lines.append(f"| **Post × Treated** | {coef1} | {coef2} |")

    # SE row
    se1 = f"({spec1['se']:.2f})" if spec1['se'] else ""
    se2 = f"({spec2['se']:.2f})" if spec2['se'] else ""

    lines.append(f"|  | {se1} | {se2} |")

    # t-stat row
    t1 = f"[{spec1['t']:.2f}]" if spec1['t'] else ""
    t2 = f"[{spec2['t']:.2f}]" if spec2['t'] else ""

    lines.append(f"| **t-stat** | {t1} | {t2} |")

    # Specification details
    lines.append("| **Portfolio FE** | Yes | Yes |")
    lines.append("| **FF5 factors** | Yes | Yes |")
    lines.append("| **SE clustering** | OLS | OLS |")
    lines.append(f"| **N (treated)** | {spec1['n_treated']} | {spec2['n_treated']} |")
    lines.append(f"| **N (control)** | {spec1['n_control']} | {spec2['n_control']} |")
    lines.append(f"| **Observations** | {spec1['nobs']:,} | {spec2['nobs']:,} |")
    lines.append(f"| **R²** | {spec1['r2']:.4f} | {spec2['r2']:.4f} |")

    lines.append("")
    lines.append("*Note: Coefficients in % per month. Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("Excluding upstream producers drops FF49 portfolios: Computers (Chips), Software (Softw),")
    lines.append("Electronic Equipment (ElcEq), Business Equipment (BusEq). Treatment groups are")
    lines.append("re-ranked within the restricted sample. Post period = Dec 2022 + Jan 2023.*")

    return '\n'.join(lines)


def format_latex_table(spec1, spec2):
    """Format table as LaTeX with booktabs + siunitx."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Composition check: Excluding upstream producer portfolios}")
    lines.append("\\label{tab:upstream_exclusion}")
    lines.append("")
    lines.append("\\begin{threeparttable}")
    lines.append("")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l S S}")
    lines.append("\\toprule")
    lines.append(" & {Baseline} & {Excluding upstream producers} \\\\")
    lines.append("\\midrule")

    # Helper for stars
    def get_stars(p):
        return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    # Post × Treated row
    coef1 = f"{spec1['coef']:+.2f}" if spec1['coef'] else ""
    coef2 = f"{spec2['coef']:+.2f}" if spec2['coef'] else ""

    lines.append(f"Post $\\times$ Treated & "
                 f"{coef1}{get_stars(spec1['p'])} & "
                 f"{coef2}{get_stars(spec2['p'])} \\\\")

    # SE row
    se1 = f"({spec1['se']:.2f})" if spec1['se'] else ""
    se2 = f"({spec2['se']:.2f})" if spec2['se'] else ""

    lines.append(f" & {se1} & {se2} \\\\")

    # t-stat row
    t1 = f"[{spec1['t']:.2f}]" if spec1['t'] else ""
    t2 = f"[{spec2['t']:.2f}]" if spec2['t'] else ""

    lines.append(f"t-stat & {t1} & {t2} \\\\")
    lines.append("\\addlinespace")
    lines.append("Portfolio FE & Yes & Yes \\\\")
    lines.append("FF5 factors & Yes & Yes \\\\")
    lines.append("SE clustering & OLS & OLS \\\\")
    lines.append(f"N (treated) & {spec1['n_treated']} & {spec2['n_treated']} \\\\")
    lines.append(f"N (control) & {spec1['n_control']} & {spec2['n_control']} \\\\")
    lines.append(f"Observations & {spec1['nobs']:,} & {spec2['nobs']:,} \\\\")
    lines.append(f"$R^2$ & {spec1['r2']:.4f} & {spec2['r2']:.4f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")
    lines.append("\\end{threeparttable}")
    lines.append("")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Notes:} Coefficients in \\% per month. Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10. Excluding upstream producers drops FF49 portfolios: Computers (Chips), Software (Softw), Electronic Equipment (ElcEq), Business Equipment (BusEq). Treatment groups are re-ranked within the restricted sample. Post period = Dec 2022 + Jan 2023.")
    lines.append("\\end{tablenotes}")
    lines.append("")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("TABLE 6: COMPOSITION CHECK - EXCLUDING UPSTREAM PRODUCERS")
    print("=" * 70)

    # Load data and create treatment
    print("\n[1/3] Loading data and creating treatment...")
    df = load_panel()
    treated_df = create_binary_treatment(df, 'aiie_23_ew_z', 'binary_q1_q4')

    # Filter to sample_60m
    treated_df = treated_df[treated_df['sample_60m'] == 1].copy()
    print(f"  Sample: {len(treated_df):,} observations")

    # Run specifications
    print("[2/3] Running specifications...")
    spec1 = spec_baseline(treated_df)
    print(f"  Baseline: γ = {spec1['coef']:+.2f} (t = {spec1['t']:.2f})")

    spec2 = spec_excluding_upstream(df)
    print(f"  No upstream: γ = {spec2['coef']:+.2f} (t = {spec2['t']:.2f})")

    # Generate outputs
    print("\n[3/3] Saving outputs...")

    # Save text version
    txt_content = format_text_table(spec1, spec2)
    txt_path = OUTPUT_DIR / 'txt' / 'table6.txt'
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(txt_content)
    print(f"  Saved: txt/table6.txt")

    # Save LaTeX version
    latex_content = format_latex_table(spec1, spec2)
    latex_path = OUTPUT_DIR / 'tex' / 'table6.tex'
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: tex/table6.tex")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline:       γ = {spec1['coef']:+.2f}% (t = {spec1['t']:.2f}, p = {spec1['p']:.4f})")
    print(f"Excl. upstream: γ = {spec2['coef']:+.2f}% (t = {spec2['t']:.2f}, p = {spec2['p']:.4f})")
    print(f"\nUpstream excluded: {', '.join(UPSTREAM_PORTFOLIOS)}")
    print(f"N(treated) changes: {spec1['n_treated']} → {spec2['n_treated']}")
    print("\nDone.")


if __name__ == '__main__':
    main()
