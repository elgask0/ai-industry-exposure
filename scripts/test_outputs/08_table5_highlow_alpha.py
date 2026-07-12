#!/usr/bin/env python3
"""
Table 5: High-Low Portfolio + Alphas

Constructs a long-short portfolio (Q4 minus Q1) and estimates alphas
using time-series regressions with Newey-West HAC standard errors.

This was previously labeled as Table 3 in earlier versions.

Outputs:
- outputs/analysis/txt/table5.txt (human-readable text format)
- outputs/analysis/tex/table5.tex (LaTeX booktabs + siunitx format)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_prep import create_binary_treatment, load_panel

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def construct_ls_portfolio(treated_df):
    """
    Construct long-short portfolio returns.

    LS_t = (1/N_Q4) * Σ(ret_ew_ex for Q4) - (1/N_Q1) * Σ(ret_ew_ex for Q1)

    Returns DataFrame with monthly LS returns and FF5 factors.
    """
    # Get monthly returns by treatment group
    monthly_data = []

    for ym in treated_df['ym'].unique():
        month_df = treated_df[treated_df['ym'] == ym]

        # Q4 (treated) average return
        q4_returns = month_df[month_df['treated'] == 1]['ret_ew_ex']
        q4_avg = q4_returns.mean() if len(q4_returns) > 0 else np.nan

        # Q1 (control) average return
        q1_returns = month_df[month_df['treated'] == 0]['ret_ew_ex']
        q1_avg = q1_returns.mean() if len(q1_returns) > 0 else np.nan

        # LS portfolio
        ls_return = q4_avg - q1_avg

        # Get FF5 factors (same for all portfolios in a month)
        row = month_df.iloc[0]
        monthly_data.append({
            'ym': ym,
            'date': row['date'],
            'ls_ret': ls_return,
            'mktrf': row['mktrf'],
            'smb': row['smb'],
            'hml': row['hml'],
            'rmw': row['rmw'],
            'cma': row['cma'],
        })

    return pd.DataFrame(monthly_data).dropna()


def run_timeseries_regression(ls_data, factors):
    """
    Run time-series regression: LS_t = α + β' * Factors_t + ε_t

    Uses Newey-West HAC standard errors with 3 lags.
    """
    # Prepare data
    y = ls_data['ls_ret']
    X = ls_data[list(factors)].copy()
    X = sm.add_constant(X)

    # OLS regression
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})

    return results


def compute_summary_statistics(ls_data, period='all'):
    """Compute mean return and t-stat for LS portfolio."""
    if period == 'all':
        returns = ls_data['ls_ret']
    elif period == 'post':
        # Post period: Dec 2022 + Jan 2023
        post_dates = ls_data[(ls_data['ym'] == 202212) | (ls_data['ym'] == 202301)]
        returns = post_dates['ls_ret']
    else:
        returns = ls_data['ls_ret']

    mean_return = returns.mean()
    std_return = returns.std()
    n = len(returns)
    t_stat = mean_return / (std_return / np.sqrt(n)) if std_return > 0 else 0

    return {
        'mean': mean_return,
        'std': std_return,
        'n': n,
        't_stat': t_stat,
    }


def format_table5_text(capm_results, ff5_results, stats_all, stats_post):
    """Format Table 5 as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 3. HIGH-MINUS-LOW EXPOSURE PORTFOLIO: PERFORMANCE AND FACTOR-ADJUSTED ALPHAS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Portfolios are equal-weighted averages of FF49 industry portfolios within each")
    lines.append("exposure quartile. The long-short spread is LS_t = r̄_High,t − r̄_Low,t.")
    lines.append("Alphas are intercepts from time-series regressions of LS_t on factor models.")
    lines.append("Newey–West standard errors with 3 lags in parentheses.")
    lines.append("")
    lines.append("| | Low (Q1) | High (Q4) | High–Low Spread |")
    lines.append("|---|---:|---:|---:|")

    # Mean excess return (Post)
    lines.append(f"| **Mean excess return** (Post, % per month) | — | — | {stats_post['mean']:+.2f} |")
    lines.append(f"|  | t-stat | — | — | ({stats_post['t_stat']:.2f}) |")

    # CAPM alpha
    capm_alpha = capm_results.params['const']
    capm_alpha_se = capm_results.bse['const']
    capm_alpha_t = capm_results.tvalues['const']
    capm_p = capm_results.pvalues['const']
    stars_capm = '***' if capm_p < 0.01 else '**' if capm_p < 0.05 else '*' if capm_p < 0.10 else ''
    lines.append(f"| **CAPM alpha** (% per month) | — | — | {capm_alpha:+.2f}{stars_capm} |")
    lines.append(f"|  | (NW t) | — | — | ({capm_alpha_t:.2f}) |")

    # FF5 alpha
    ff5_alpha = ff5_results.params['const']
    ff5_alpha_se = ff5_results.bse['const']
    ff5_alpha_t = ff5_results.tvalues['const']
    ff5_p = ff5_results.pvalues['const']
    stars_ff5 = '***' if ff5_p < 0.01 else '**' if ff5_p < 0.05 else '*' if ff5_p < 0.10 else ''
    lines.append(f"| **FF5 alpha** (% per month) | — | — | {ff5_alpha:+.2f}{stars_ff5} |")
    lines.append(f"|  | (NW t) | — | — | ({ff5_alpha_t:.2f}) |")

    # Market beta (CAPM)
    capm_beta = capm_results.params['mktrf']
    capm_beta_t = capm_results.tvalues['mktrf']
    lines.append(f"| **Market beta** (CAPM) | — | — | {capm_beta:.2f} |")

    # Observations
    lines.append(f"| **Observations** (months) | — | — | {stats_all['n']} |")

    lines.append("")
    lines.append("*Post period: Dec 2022 + Jan 2023 (2 months). Pre period: Dec 2017 to Nov 2022 (60 months).")
    lines.append("Newey-West standard errors with 3 lags. Portfolios are equal-weighted averages of")
    lines.append("FF49 industry portfolios within each exposure quartile.")

    return '\n'.join(lines)


def format_table5_latex(capm_results, ff5_results, stats_all, stats_post):
    """Format Table 5 as LaTeX with booktabs + siunitx."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{High-minus-low exposure portfolio: Performance and factor-adjusted alphas}")
    lines.append("\\label{tab:highlow_alpha}")
    lines.append("")
    lines.append("\\begin{threeparttable}")
    lines.append("")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l l S}")
    lines.append("\\toprule")
    lines.append(" & & {High--Low Spread} \\\\")
    lines.append(" & & {LS$_t$} \\\\")
    lines.append("\\midrule")

    # Mean excess return (Post)
    mean = stats_post['mean']
    t_stat = stats_post['t_stat']
    stars_mean = '***' if abs(t_stat) > 2.576 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.645 else ''
    lines.append(f"Mean excess return (Post, \\% per month) & & {mean:+.2f}{stars_mean} \\\\")
    lines.append(f" & t-stat & ({t_stat:.2f}) \\\\")

    # CAPM alpha
    capm_alpha = capm_results.params['const']
    capm_alpha_t = capm_results.tvalues['const']
    capm_p = capm_results.pvalues['const']
    stars_capm = '***' if capm_p < 0.01 else '**' if capm_p < 0.05 else '*' if capm_p < 0.10 else ''
    lines.append(f"CAPM alpha (\\% per month) & & {capm_alpha:+.2f}{stars_capm} \\\\")
    lines.append(f" & (NW t) & ({capm_alpha_t:.2f}) \\\\")

    # FF5 alpha
    ff5_alpha = ff5_results.params['const']
    ff5_alpha_t = ff5_results.tvalues['const']
    ff5_p = ff5_results.pvalues['const']
    stars_ff5 = '***' if ff5_p < 0.01 else '**' if ff5_p < 0.05 else '*' if ff5_p < 0.10 else ''
    lines.append(f"FF5 alpha (\\% per month) & & {ff5_alpha:+.2f}{stars_ff5} \\\\")
    lines.append(f" & (NW t) & ({ff5_alpha_t:.2f}) \\\\")

    # Market beta
    capm_beta = capm_results.params['mktrf']
    lines.append(f"Market beta (CAPM) & & {capm_beta:.2f} \\\\")

    # Observations
    lines.append(f"Observations (months) & & {stats_all['n']} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")
    lines.append("\\end{threeparttable}")
    lines.append("")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Notes:} High (Low) portfolio is the equal-weighted average of monthly excess returns across all Treated (Control) portfolios. The long-short spread is LS$_t$ = $\\bar{r}$$_{High,t}$ $-$ $\\bar{r}$$_{Low,t}$. Alphas are intercepts from time-series regressions of LS$_t$ on the indicated factor model. Newey--West standard errors with 3 lags. Post period: Dec 2022 + Jan 2023 (2 months). Pre period: Dec 2017 to Nov 2022 (60 months). *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("\\end{tablenotes}")
    lines.append("")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("TABLE 3: HIGH-LOW PORTFOLIO + ALPHAS")
    print("=" * 70)

    # Load data and create treatment
    print("\n[1/5] Loading data and creating treatment...")
    df = load_panel()
    treated_df = create_binary_treatment(df, 'aiie_23_ew_z', 'binary_q1_q4')

    # Filter to sample_60m
    treated_df = treated_df[treated_df['sample_60m'] == 1].copy()
    print(f"  Sample: {len(treated_df):,} observations")

    # Construct LS portfolio
    print("[2/5] Constructing long-short portfolio...")
    ls_data = construct_ls_portfolio(treated_df)
    print(f"  LS portfolio: {len(ls_data)} months")

    # Compute summary statistics
    print("[3/5] Computing summary statistics...")
    stats_all = compute_summary_statistics(ls_data, period='all')
    stats_post = compute_summary_statistics(ls_data, period='post')
    print(f"  Post-period mean: {stats_post['mean']:+.2f}% (t = {stats_post['t_stat']:.2f})")

    # Run CAPM regression
    print("[4/5] Running CAPM regression...")
    capm_results = run_timeseries_regression(ls_data, ['mktrf'])

    # Run FF5 regression
    print("[5/5] Running FF5 regression...")
    ff5_results = run_timeseries_regression(ls_data, ['mktrf', 'smb', 'hml', 'rmw', 'cma'])

    # Save outputs
    print("\n[6/6] Saving outputs...")

    # Text version
    txt_content = format_table5_text(capm_results, ff5_results, stats_all, stats_post)
    txt_path = OUTPUT_DIR / 'txt' / 'table5.txt'
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(txt_content)
    print(f"  Saved: txt/table5.txt")

    # LaTeX version
    latex_content = format_table5_latex(capm_results, ff5_results, stats_all, stats_post)
    latex_path = OUTPUT_DIR / 'tex' / 'table5.tex'
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: tex/table5.tex")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"LS Portfolio (Post): {stats_post['mean']:+.2f}% per month (t = {stats_post['t_stat']:.2f})")
    print(f"CAPM Alpha: {capm_results.params['const']:+.2f}% (t = {capm_results.tvalues['const']:.2f})")
    print(f"FF5 Alpha: {ff5_results.params['const']:+.2f}% (t = {ff5_results.tvalues['const']:.2f})")
    print(f"CAPM Beta: {capm_results.params['mktrf']:.2f}")
    print(f"Observations: {stats_all['n']} months")
    print("\nDone.")


if __name__ == '__main__':
    main()
