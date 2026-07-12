#!/usr/bin/env python3
"""
Table 3: Dynamic Effects - Event-Study Coefficients

Extracts event-time DiD coefficients γᵗ from pretrends_results.txt
and formats them as a publication table showing τ = -12 to +2.

Outputs:
- outputs/analysis/txt/table3.txt (human-readable text format)
- outputs/analysis/tex/table3.tex (LaTeX booktabs + siunitx format)
"""

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.output_utils import get_output_path

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'


def parse_pretrends_results(file_path: Path) -> pd.DataFrame:
    """
    Parse pretrends_results.txt into a DataFrame.

    Extracts the coefficient table from lines 19-35.

    Returns:
        DataFrame with columns: tau, month, coef, se, t, p, significant
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Use regex to find all coefficient lines
    # Pattern matches: ±DD  Month YYYY  ±number  number  ±number  number
    pattern = r'\s*([+-]?\d+)\s+(\w+\s+\d{4})\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)'

    matches = re.findall(pattern, content)

    if not matches:
        raise ValueError("Could not find coefficient table in pretrends_results.txt")

    data = []
    for match in matches:
        tau = int(match[0])
        month = match[1]
        coef = float(match[2].replace('+', '').replace(',', ''))
        se = float(match[3].replace('+', '').replace(',', ''))
        t = float(match[4].replace('+', ''))
        p = float(match[5].replace('+', ''))

        data.append({
            'tau': tau,
            'month': month,
            'coef': coef,
            'se': se,
            't': t,
            'p': p,
            'significant': p < 0.10
        })

    df = pd.DataFrame(data)

    # Sort by tau
    df = df.sort_values('tau').reset_index(drop=True)

    return df


def format_text_table(df: pd.DataFrame) -> str:
    """Format table as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 3. DYNAMIC EFFECTS: RISK-ADJUSTED EVENT-TIME DID COEFFICIENTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Specification: r_p,t = α_p + Σ_τ γᵗ(τ_t × Treated_p) + Θ'FF5_t + ε_p,t")
    lines.append("")
    lines.append("| τ | γ (%) | SE | t | p |")
    lines.append("|---:|---:|---:|---:|---:|")

    for _, row in df.iterrows():
        tau_label = f"{row['tau']:+d}"
        coef_label = f"{row['coef']:+.2f}"
        se_label = f"{row['se']:.2f}"
        t_label = f"{row['t']:+.2f}"
        p_label = f"{row['p']:.3f}"

        # Add significance stars
        if row['p'] < 0.01:
            p_label += '***'
        elif row['p'] < 0.05:
            p_label += '**'
        elif row['p'] < 0.10:
            p_label += '*'

        lines.append(f"| {tau_label} | {coef_label} | {se_label} | {t_label} | {p_label} |")

    lines.append("")
    lines.append("*Note: Coefficients in % per month. τ is months relative to Dec 2022.")
    lines.append("Reference τ = -1 (Nov 2022) is omitted. *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("Standard errors are OLS (non-robust). Factors: FF5 (MKTRF, SMB, HML, RMW, CMA).")

    return '\n'.join(lines)


def format_latex_table(df: pd.DataFrame) -> str:
    """Format table as LaTeX with booktabs + siunitx."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Dynamic effects: Risk-adjusted event-time DiD coefficients}")
    lines.append("\\label{tab:event_study}")
    lines.append("")
    lines.append("\\begin{threeparttable}")
    lines.append("")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l S S S S}")
    lines.append("\\toprule")
    lines.append("τ & {Coef} & {SE} & {$t$-stat} & {$p$-val} \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        tau_label = f"{row['tau']:+d}"
        coef_label = f"{row['coef']:+.2f}"
        se_label = f"{row['se']:.2f}"
        t_label = f"{row['t']:+.2f}"
        p_label = f"{row['p']:.3f}"

        # Add significance stars
        if row['p'] < 0.01:
            coef_label += '***'
        elif row['p'] < 0.05:
            coef_label += '**'
        elif row['p'] < 0.10:
            coef_label += '*'

        lines.append(f"{tau_label} & {coef_label} & {se_label} & {t_label} & {p_label} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\normalsize")
    lines.append("")
    lines.append("\\end{threeparttable}")
    lines.append("")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Notes:} Coefficients in \\% per month. $\\tau$ is months relative to Dec 2022. Reference $\\tau = -1$ (Nov 2022) is omitted. Standard errors are OLS (non-robust). Factors: FF5 (MKTRF, SMB, HML, RMW, CMA). *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("\\end{tablenotes}")
    lines.append("")
    lines.append("\\end{table}")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("TABLE 3: EVENT-STUDY COEFFICIENTS")
    print("=" * 70)

    # Parse pretrends results
    print("\n[1/3] Parsing pretrends_results.txt...")
    pretrends_file = OUTPUT_DIR / 'pretrends_results.txt'

    if not pretrends_file.exists():
        print(f"  ERROR: {pretrends_file} not found.")
        print("  Run scripts/04_run_analysis.py first to generate pretrends results.")
        return

    df = parse_pretrends_results(pretrends_file)

    print(f"  Parsed {len(df)} event-time coefficients")
    print(f"  τ range: {df['tau'].min():+d} to {df['tau'].max():+d}")

    # Print summary stats
    print("\n[2/3] Summary statistics:")
    print(f"  Significant at 5%: {(df['p'] < 0.05).sum()} / {len(df)}")
    print(f"  τ=+1 coefficient: γ = {df[df['tau']==1]['coef'].values[0]:+.2f} (t={df[df['tau']==1]['t'].values[0]:+.2f})")

    # Generate outputs
    print("\n[3/3] Saving outputs...")

    # Save text version
    txt_content = format_text_table(df)
    txt_path = get_output_path('table3', 'txt')
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(txt_content)
    print(f"  Saved: txt/table3.txt")

    # Save LaTeX version
    latex_content = format_latex_table(df)
    latex_path = get_output_path('table3', 'tex')
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: tex/table3.tex")

    print("\nDone.")


if __name__ == '__main__':
    main()
