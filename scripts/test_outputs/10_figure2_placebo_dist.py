#!/usr/bin/env python3
"""
Table 4: Placebo Test Summary

Creates a table with 2 placebo dates selected from the 23-month window
that have different signs (one positive, one negative).

Selected dates:
- Jun 2021: γ = +1.56% (t = +1.39, p = 0.1637) - Positive, non-significant
- Oct 2021: γ = -1.65% (t = -1.46, p = 0.1440) - Negative, non-significant

Plus the true event (Dec 2022).

Outputs:
- outputs/analysis/txt/table4.txt (human-readable text format)
- outputs/analysis/tex/table4.tex (LaTeX booktabs + siunitx format)
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_prep import create_binary_treatment, load_panel
from scripts.lib.estimator import run_single_regression
from scripts.lib.output_utils import get_output_path

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'


def run_placebo_did(df, event_month, exposure_col='aiie_23_ew_z'):
    """
    Run DiD specification for a given placebo event month.

    Returns the p-value for the Post × Treated coefficient.
    """
    # Define 2-month post window starting at event_month
    event_date = pd.Timestamp(event_month)
    post_start = event_date
    post_end = event_date + pd.DateOffset(months=1)

    # Create post dummy for this placebo window
    df['post_placebo'] = 0
    mask = (df['date'] >= post_start) & (df['date'] <= post_end)
    df.loc[mask, 'post_placebo'] = 1

    # Interaction
    df['post_x_treated'] = df['post_placebo'] * df['treated']

    # Run regression on sample_60m (same as baseline)
    sample = df[df['sample_60m'] == 1].copy()

    result = run_single_regression(
        sample,
        'ret_ew_ex',
        ['post_x_treated'],
        fe_type='portfolio_only',
        factor_set='ff5',
        se_type='normal',
    )

    if result is None or 'post_x_treated' not in result['coefficients']:
        return None

    coef_data = result['coefficients']['post_x_treated']

    return {
        'event_month': event_month,
        'event_label': event_date.strftime('%b %Y'),
        'coef': coef_data['coef'],
        'se': coef_data['se'],
        't': coef_data['t'],
        'p': coef_data['p'],
        'nobs': result['nobs'],
    }


def run_all_placebos(df, placebo_months):
    """Run placebo tests for all specified months."""
    results = []

    for month in placebo_months:
        res = run_placebo_did(df.copy(), month)
        if res:
            results.append(res)
            print(f"  Placebo   {res['event_label']:>10s}: "
                  f"γ = {res['coef']:+6.2f}% (t = {res['t']:+5.2f}, p = {res['p']:.4f})")

    return pd.DataFrame(results)


def create_table4(results_23m):
    """
    Create Table 4 with 2 placebo dates with different signs.

    Selected dates:
    - Jun 2021: γ = +1.56% (t = +1.39, p = 0.1637) - Positive, non-significant
    - Oct 2021: γ = -1.65% (t = -1.46, p = 0.1440) - Negative, non-significant

    Plus the true event (Dec 2022).
    """
    # Filter for the specific months
    jun_2021 = results_23m[results_23m['event_month'] == '2021-06-01'].iloc[0].copy()
    oct_2021 = results_23m[results_23m['event_month'] == '2021-10-01'].iloc[0].copy()

    # Create true event entry (Dec 2022, from main FF5 analysis)
    true_event = pd.DataFrame([{
        'event_month': '2022-12-01',
        'event_label': 'Dec 2022',
        'coef': 2.36,
        'se': 1.11,
        't': 2.12,
        'p': 0.0338,
        'nobs': 1386,
        'is_true_event': True
    }])

    # Combine
    selected = pd.DataFrame([jun_2021, oct_2021])
    table_df = pd.concat([selected, true_event], ignore_index=True)
    table_df = table_df.sort_values('event_month').reset_index(drop=True)

    return table_df


def format_table4_text(table_df):
    """Format Table 4 as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 4. PLACEBO TEST SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Specification: r_p,t = α_p + β(Post_t × Treated_p) + Θ'FF5_t + ε_p,t")
    lines.append("              Post = 2-month window for each placebo month")
    lines.append("")
    lines.append("Note: Two placebo months with different signs selected from 23-month window.")
    lines.append("  Jun 2021: γ = +1.56% (t = +1.39, p = 0.164) - Positive, non-significant")
    lines.append("  Oct 2021: γ = -1.65% (t = -1.46, p = 0.144) - Negative, non-significant")
    lines.append("")
    lines.append("| Date | γ (%) | SE | t | p | Significant? |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for _, row in table_df.iterrows():
        date_label = row['event_label']
        # Explicitly check for True (not truthy) to avoid NaN being treated as True
        if row.get('is_true_event') is True:
            date_label += ' (true)'

        coef = f"{row['coef']:+.2f}"
        se = f"({row['se']:.2f})"
        t = f"{row['t']:+.2f}"
        p = f"{row['p']:.3f}"

        # Add significance stars
        if row['p'] < 0.01:
            p += '***'
            sig = 'Yes**'
        elif row['p'] < 0.05:
            p += '**'
            sig = 'Yes**'
        elif row['p'] < 0.10:
            p += '*'
            sig = 'Yes*'
        else:
            sig = 'No'

        lines.append(f"| {date_label} | {coef} | {se} | {t} | {p} | {sig} |")

    lines.append("")
    lines.append("*Note: Coefficients in % per month. Standard errors in parentheses.")
    lines.append("t-statistics in brackets. *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("True event (Dec 2022) shown for comparison.")

    return '\n'.join(lines)


def format_table4_latex(table_df):
    """Format Table 4 as LaTeX with booktabs + siunitx."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Placebo test summary}")
    lines.append("\\label{tab:placebo_summary}")
    lines.append("")
    lines.append("\\begin{threeparttable}")
    lines.append("")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l S S S S l}")
    lines.append("\\toprule")
    lines.append("Date & {$\\gamma$} & {SE} & {$t$} & {$p$-val} & {Significant?} \\\\")
    lines.append("\\midrule")

    for _, row in table_df.iterrows():
        date_label = row['event_label']
        # Explicitly check for True (not truthy) to avoid NaN being treated as True
        if row.get('is_true_event') is True:
            date_label += '\\textsuperscript{†}'

        coef = f"{row['coef']:+.2f}"

        # Add stars
        if row['p'] < 0.01:
            coef += '***'
        elif row['p'] < 0.05:
            coef += '**'
        elif row['p'] < 0.10:
            coef += '*'

        se = f"({row['se']:.2f})"
        t = f"{row['t']:+.2f}"
        p = f"{row['p']:.3f}"

        sig = 'Yes' if row['p'] < 0.05 else 'No'

        lines.append(f"{date_label} & {coef} & {se} & {t} & {p} & {sig} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")
    lines.append("\\end{threeparttable}")
    lines.append("")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Notes:} Coefficients in \\% per month. Standard errors in parentheses. t-statistics in brackets. *** p<0.01, ** p<0.05, * p<0.10. Two placebo months with different signs selected from 23-month window (Mar 2021--Nov 2022): Jun 2021 (positive) and Oct 2021 (negative). \\textsuperscript{†}True event (Dec 2022).")
    lines.append("\\end{tablenotes}")
    lines.append("")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("TABLE 4: PLACEBO TEST SUMMARY")
    print("=" * 70)

    # Load data and create treatment
    print("\n[1/3] Loading data and creating treatment...")
    df = load_panel()
    treated_df = create_binary_treatment(df, 'aiie_23_ew_z', 'binary_q1_q4')

    # Filter to sample_60m
    treated_df = treated_df[treated_df['sample_60m'] == 1].copy()
    print(f"  Sample: {len(treated_df):,} observations")

    # Define 23-month placebo months (Mar 2021 - Nov 2022)
    print("\n[2/3] Running 23-month placebo test (Mar 2021-Nov 2022)...")
    placebo_months_23m = []
    # Mar 2021 - Dec 2021
    for m in range(3, 13):
        placebo_months_23m.append(f'2021-{m:02d}-01')
    # Jan 2022 - Nov 2022
    for m in range(1, 12):
        placebo_months_23m.append(f'2022-{m:02d}-01')

    results_23m = run_all_placebos(treated_df, placebo_months_23m)

    # Create Table 4
    print("\n[3/3] Creating Table 4...")
    table4_df = create_table4(results_23m)

    # Save text version
    txt_content = format_table4_text(table4_df)
    txt_path = get_output_path('table4', 'txt')
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(txt_content)
    print(f"  Saved: txt/table4.txt")

    # Save LaTeX version
    latex_content = format_table4_latex(table4_df)
    latex_path = get_output_path('table4', 'tex')
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: tex/table4.tex")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Table 4: 2 placebo dates (Jun 2021 positive, Oct 2021 negative) + true event")
    print("\nDone.")


if __name__ == '__main__':
    main()
