#!/usr/bin/env python3
"""
Table 1: Sample Construction, Exposure Measures, and Pre-Period Characteristics

Generates 4 panels:
- Panel A: Sample Construction (portfolio counts, months, observations)
- Panel B: Exposure Separation (Q1 vs Q4 exposure stats)
- Panel C: Pre-Period Returns (return statistics by treatment group)
- Panel D: Exposure Measure Correlations

Outputs:
- outputs/test/table1.txt (human-readable text format)
- outputs/test/table1.tex (LaTeX booktabs + siunitx format)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_prep import create_binary_treatment, load_panel

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_data():
    """Load all data needed for Table 1."""
    # Load main panel
    df = load_panel()

    # Load exposure measures
    exposure = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'exposure' / 'ff49_exposure_measures.csv'
    )

    # Load IPP data
    ipp = pd.read_csv(
        PROJECT_ROOT / 'data' / 'processed' / 'ipp' / 'panelA_ipp_ff49.csv'
    )

    return df, exposure, ipp


def create_treatment_sample(df):
    """Create treatment assignment using baseline AIIE 2023 EW."""
    treated_df = create_binary_treatment(df, 'aiie_23_ew_z', 'binary_q1_q4')
    return treated_df


# Panel A removed - sample construction explained in footnote instead


def panel_b_exposure_separation(treated_df, exposure):
    """Panel B: Exposure separation by treatment group."""
    # Get treatment assignment per portfolio
    portfolio_treatment = treated_df.groupby('ff49')[['treated', 'ff49_abbrev', 'ff49_name']].first()

    # Merge with exposure measures
    exp_with_treatment = exposure.merge(
        portfolio_treatment[['treated']],
        left_on='ff49',
        right_index=True,
        how='inner'
    )

    # Separate Q1 (control) and Q4 (treated)
    q1_portfolios = exp_with_treatment[exp_with_treatment['treated'] == 0]
    q4_portfolios = exp_with_treatment[exp_with_treatment['treated'] == 1]

    # Calculate statistics
    def exposure_stats(portfolio_df, col):
        return {
            'n': len(portfolio_df),
            'mean': portfolio_df[col].mean(),
            'sd': portfolio_df[col].std(),
        }

    q1_stats = exposure_stats(q1_portfolios, 'aiie_23_ew')
    q4_stats = exposure_stats(q4_portfolios, 'aiie_23_ew')

    # Z-score stats (should be -0.5ish for Q1 and +0.5ish for Q4 due to quartile split)
    q1_z_stats = exposure_stats(q1_portfolios, 'aiie_23_ew_z')
    q4_z_stats = exposure_stats(q4_portfolios, 'aiie_23_ew_z')

    # Q4 - Q1 difference
    diff_mean = q4_stats['mean'] - q1_stats['mean']
    diff_z_mean = q4_z_stats['mean'] - q1_z_stats['mean']

    lines = []
    lines.append("")
    lines.append("Panel B: Exposure Separation (EW)")
    lines.append("")
    lines.append("| | Control (Q1) | Treated (Q4) | Q4–Q1 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| N (portfolios) | {q1_stats['n']} | {q4_stats['n']} | — |")
    lines.append(f"| AIIE^EW mean | {q1_stats['mean']:.4f} | {q4_stats['mean']:.4f} | {diff_mean:.4f} |")
    lines.append(f"| AIIE^EW sd | {q1_stats['sd']:.4f} | {q4_stats['sd']:.4f} | — |")
    lines.append(f"| AIIE^EW,z mean | {q1_z_stats['mean']:.4f} | {q4_z_stats['mean']:.4f} | {diff_z_mean:.4f} |")

    return {
        'q1': {'n': q1_stats['n'], 'mean': q1_stats['mean'], 'sd': q1_stats['sd'], 'z_mean': q1_z_stats['mean']},
        'q4': {'n': q4_stats['n'], 'mean': q4_stats['mean'], 'sd': q4_stats['sd'], 'z_mean': q4_z_stats['mean']},
        'diff': {'mean': diff_mean, 'z_mean': diff_z_mean},
        'text': '\n'.join(lines)
    }


def panel_c_preperiod_returns(treated_df):
    """Panel C: Pre-period return statistics (EW)."""
    # Pre-period = sample_60m and before post period (Dec 2017 - Nov 2022)
    pre_period = treated_df[
        (treated_df['sample_60m'] == 1) &
        (treated_df['post2'] == 0)  # Exclude post period
    ].copy()

    # Separate by treatment
    q1_pre = pre_period[pre_period['treated'] == 0]['ret_ew_ex']
    q4_pre = pre_period[pre_period['treated'] == 1]['ret_ew_ex']

    def return_stats(returns):
        return {
            'mean': returns.mean(),
            'sd': returns.std(),
        }

    q1_stats = return_stats(q1_pre)
    q4_stats = return_stats(q4_pre)
    diff_mean = q4_stats['mean'] - q1_stats['mean']

    lines = []
    lines.append("")
    lines.append("Panel C: Pre-Period Returns (EW)")
    lines.append("")
    lines.append("| | Control (Q1) | Treated (Q4) | Q4–Q1 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Mean r_p,t (pre, % per month) | {q1_stats['mean']:.4f} | {q4_stats['mean']:.4f} | {diff_mean:.4f} |")
    lines.append(f"| SD r_p,t (pre, % per month) | {q1_stats['sd']:.4f} | {q4_stats['sd']:.4f} | — |")

    lines.append("")
    lines.append("*Pre-period = December 2017 to November 2022 (60 months).")

    return {
        'q1': q1_stats,
        'q4': q4_stats,
        'diff': {'mean': diff_mean},
        'text': '\n'.join(lines)
    }


def panel_d_correlations(df, exposure, ipp):
    """Panel D: Correlation matrix of exposure measures."""
    # Get portfolio-level exposure measures
    exp_cols = ['aiie_23_ew', 'aiie_23_vw', 'expwb_23_ew']

    # Merge IPP data
    merged = exposure.merge(
        ipp[['ff49', 'Z_IPP_p_EW']],
        on='ff49',
        how='inner'
    )

    # Calculate correlations
    corr_data = merged[exp_cols + ['Z_IPP_p_EW']].corr()

    lines = []
    lines.append("")
    lines.append("Panel D: Exposure Measure Correlations")
    lines.append("")
    lines.append("| | AIIE^EW | AIIE^VW | ExpWB | IPP intensity |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append("| AIIE^EW | 1.00 | | | |")

    # AIIE^VW row
    lines.append(f"| AIIE^VW | {corr_data.loc['aiie_23_ew', 'aiie_23_vw']:.2f} | 1.00 | | |")

    # ExpWB row
    lines.append(f"| ExpWB | {corr_data.loc['aiie_23_ew', 'expwb_23_ew']:.2f} | {corr_data.loc['aiie_23_vw', 'expwb_23_ew']:.2f} | 1.00 | |")

    # IPP row
    lines.append(f"| IPP intensity | {corr_data.loc['aiie_23_ew', 'Z_IPP_p_EW']:.2f} | {corr_data.loc['aiie_23_vw', 'Z_IPP_p_EW']:.2f} | {corr_data.loc['expwb_23_ew', 'Z_IPP_p_EW']:.2f} | 1.00 |")

    return {
        'correlations': corr_data,
        'text': '\n'.join(lines)
    }


def format_latex_table(panel_b_data, panel_c_data, panel_d_data):
    """Format table as LaTeX with booktabs + siunitx."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Exposure measures and pre-period characteristics}")
    lines.append("\\label{tab:sample_descriptives}")
    lines.append("")
    lines.append("\\begin{threeparttable}")
    lines.append("")
    lines.append("\\setlength{\\tabcolsep}{3pt}")

    # Panel A: Exposure Separation (was Panel B)
    lines.append("\\textbf{Panel A: Exposure Separation (EW)} \\\\")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l S S S}")
    lines.append("\\toprule")
    lines.append(" & {Control (Q1)} & {Treated (Q4)} & {Q4--Q1} \\\\")
    lines.append("\\midrule")
    lines.append(f"N (portfolios) & {panel_b_data['q1']['n']} & {panel_b_data['q4']['n']} & -- \\\\")
    lines.append(f"AIIE$^{{EW}}$ mean & {panel_b_data['q1']['mean']:.4f} & {panel_b_data['q4']['mean']:.4f} & {panel_b_data['diff']['mean']:.4f} \\\\")
    lines.append(f"AIIE$^{{EW}}$ sd & {panel_b_data['q1']['sd']:.4f} & {panel_b_data['q4']['sd']:.4f} & -- \\\\")
    lines.append(f"AIIE$^{{EW,z}}$ mean & {panel_b_data['q1']['z_mean']:.4f} & {panel_b_data['q4']['z_mean']:.4f} & {panel_b_data['diff']['z_mean']:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")

    # Panel B: Pre-Period Returns (was Panel C)
    lines.append("\\vspace{0.2cm}")
    lines.append("\\textbf{Panel B: Pre-Period Returns (EW)} \\\\")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l S S S}")
    lines.append("\\toprule")
    lines.append(" & {Control (Q1)} & {Treated (Q4)} & {Q4--Q1} \\\\")
    lines.append("\\midrule")
    lines.append(f"Mean $r_{{p,t}}$ (pre, \\% per month) & {panel_c_data['q1']['mean']:.4f} & {panel_c_data['q4']['mean']:.4f} & {panel_c_data['diff']['mean']:.4f} \\\\")
    lines.append(f"SD $r_{{p,t}}$ (pre, \\% per month) & {panel_c_data['q1']['sd']:.4f} & {panel_c_data['q4']['sd']:.4f} & -- \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")

    # Panel C: Correlations (was Panel D)
    corr = panel_d_data['correlations']
    lines.append("\\vspace{0.2cm}")
    lines.append("\\textbf{Panel C: Exposure Measure Correlations} \\\\")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l S S S S}")
    lines.append("\\toprule")
    lines.append(" & {AIIE$^{{EW}}$} & {AIIE$^{{VW}}$} & {ExpWB} & {IPP intensity} \\\\")
    lines.append("\\midrule")
    lines.append(f"AIIE$^{{EW}}$ & 1.00 & & & \\\\")
    lines.append(f"AIIE$^{{VW}}$ & {corr.loc['aiie_23_ew', 'aiie_23_vw']:.2f} & 1.00 & & \\\\")
    lines.append(f"ExpWB & {corr.loc['aiie_23_ew', 'expwb_23_ew']:.2f} & {corr.loc['aiie_23_vw', 'expwb_23_ew']:.2f} & 1.00 & \\\\")
    lines.append(f"IPP intensity & {corr.loc['aiie_23_ew', 'Z_IPP_p_EW']:.2f} & {corr.loc['aiie_23_vw', 'Z_IPP_p_EW']:.2f} & {corr.loc['expwb_23_ew', 'Z_IPP_p_EW']:.2f} & 1.00 \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")

    lines.append("\\end{threeparttable}")
    lines.append("")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Notes:} Sample: 22 industry portfolios (11 Q4 $+$ 11 Q1 by AIIE$^{EW}$ exposure) $\\times$ 63 months (Dec 2017--Feb 2023) = 1,386 observations. Pre-period = Dec 2017--Nov 2022 (60 months). 6 portfolios lack NAICS mapping. Quartile cutpoints: q25 = -0.94, q75 = 0.63. AIIE = AI Industry Exposure, ExpWB = Exposure-Weighted by Wage Bill, IPP = Intellectual Property Products intensity.")
    lines.append("\\end{tablenotes}")
    lines.append("")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("TABLE 1: EXPOSURE MEASURES AND PRE-PERIOD CHARACTERISTICS")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df, exposure, ipp = load_all_data()

    # Create treatment
    print("[2/4] Creating treatment assignment...")
    treated_df = create_treatment_sample(df)

    # Generate panels
    print("[3/4] Generating panels...")
    panel_a = panel_b_exposure_separation(treated_df, exposure)  # Now Panel A
    panel_b = panel_c_preperiod_returns(treated_df)  # Now Panel B
    panel_c = panel_d_correlations(df, exposure, ipp)  # Now Panel C

    # Combine text output
    print("[4/4] Saving outputs...")
    full_text = (
        panel_a['text'] + '\n' +
        panel_b['text'] + '\n' +
        panel_c['text']
    )

    # Save text version
    txt_path = OUTPUT_DIR / 'txt' / 'table1.txt'
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(full_text)
    print(f"  Saved: txt/table1.txt")

    # Save LaTeX version
    latex_path = OUTPUT_DIR / 'tex' / 'table1.tex'
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_content = format_latex_table(panel_a, panel_b, panel_c)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: tex/table1.tex")
    print(f"  Saved: {latex_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Q1 portfolios: {panel_a['q1']['n']}")
    print(f"Q4 portfolios: {panel_a['q4']['n']}")
    print(f"Q4-Q1 exposure gap: {panel_a['diff']['mean']:.4f}")
    print("\nDone.")


if __name__ == '__main__':
    main()
