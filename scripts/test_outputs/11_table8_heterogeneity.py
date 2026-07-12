#!/usr/bin/env python3
"""
Table 7: Heterogeneity - Triple-Difference Estimates

Tests whether the ChatGPT shock effect varies with:
- Panel A: IPP intensity (high intangibles vs low)
- Panel B: ExpWB exposure (wage-bill weighted vs baseline)
- Panel C: Excluding upstream producers

Outputs:
- outputs/analysis/txt/table7.txt (human-readable text format)
- outputs/analysis/tex/table7.tex (LaTeX booktabs + siunitx format)
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


def load_ipp_data():
    """Load IPP intensity data."""
    ipp = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'ipp' / 'panelA_ipp_ff49.csv')
    return ipp[['ff49', 'Z_IPP_p_EW', 'HighIPP_EW', 'in_aiie_sample_q1q4']]


def run_baseline_specification(df, x_vars, return_col='ret_ew_ex',
                                factor_set='ff5', fe_type='portfolio_only',
                                se_type='normal'):
    """Run a DiD specification with custom x_vars."""
    result = run_single_regression(
        df,
        return_col,
        x_vars,
        fe_type=fe_type,
        factor_set=factor_set,
        se_type=se_type,
    )
    return result


# ============================================================================
# Panel A: IPP triple-diff
# ============================================================================

def panel_a_ipp_triple_diff(df, ipp):
    """Panel A: IPP triple-difference."""
    # Convert ff49 to same type for merge
    df_copy = df.copy()
    df_copy['ff49'] = df_copy['ff49'].astype(int)

    # Merge IPP data
    merged = df_copy.merge(ipp[['ff49', 'HighIPP_EW', 'in_aiie_sample_q1q4']], on='ff49', how='inner')

    # Filter to sample_60m and baseline Q4/Q1
    sample = merged[(merged['sample_60m'] == 1) &
                    (merged['in_aiie_sample_q1q4'] == 1) &
                    (merged['treated'].notna())].copy()

    # Create triple interaction
    sample['post_x_treated'] = sample['post2'] * sample['treated']
    sample['post_x_highipp'] = sample['post2'] * sample['HighIPP_EW']
    sample['post_x_treated_x_highipp'] = sample['post2'] * sample['treated'] * sample['HighIPP_EW']

    x_vars = ['post_x_treated', 'post_x_treated_x_highipp', 'post_x_highipp']
    result = run_baseline_specification(sample, x_vars)

    lines = []
    lines.append("")
    lines.append("Panel A: High intangibles (IPP intensity)")
    lines.append("")
    lines.append("| Variable | Coefficient | SE | t-stat |")
    lines.append("|---|---:|---:|---:|")

    for var in x_vars:
        if var in result['coefficients']:
            coef_data = result['coefficients'][var]
            stars = '***' if coef_data['p'] < 0.01 else '**' if coef_data['p'] < 0.05 else '*' if coef_data['p'] < 0.10 else ''

            var_name = {
                'post_x_treated': 'Post × Treated (β₁)',
                'post_x_treated_x_highipp': 'Post × Treated × HighIPP (β₂)',
                'post_x_highipp': 'Post × HighIPP (β₃)'
            }.get(var, var)

            lines.append(f"| {var_name} | {coef_data['coef']:+.2f}{stars} | {coef_data['se']:.2f} | {coef_data['t']:.2f} |")

    # Calculate total effect for high IPP
    beta1 = result['coefficients'].get('post_x_treated', {}).get('coef', 0)
    beta2 = result['coefficients'].get('post_x_treated_x_highipp', {}).get('coef', 0)
    total_highipp = beta1 + beta2

    lines.append("")
    lines.append(f"*Treated effect for high-IPP = β₁ + β₂ = {total_highipp:.2f}%*")

    return {'result': result, 'text': '\n'.join(lines), 'total_highipp': total_highipp}


# ============================================================================
# Panel B: ExpWB triple-diff
# ============================================================================

def panel_b_expwb_triple_diff(df):
    """Panel B: ExpWB triple-difference."""
    # Load exposure measures to get ExpWB
    exposure = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'exposure' / 'ff49_exposure_measures.csv')

    # Convert ff49 to same type for merge (df has float64, exposure has int64)
    df_copy = df.copy()
    df_copy['ff49'] = df_copy['ff49'].astype(int)

    # Merge to get ExpWB values - will add _y suffix since expwb_23_ew already exists
    merged = df_copy.merge(
        exposure[['ff49', 'expwb_23_ew']],
        on='ff49',
        how='inner',
        suffixes=('', '_from_exp')
    )

    # Filter to sample_60m and baseline Q4/Q1 (using treated column)
    sample = merged[(merged['sample_60m'] == 1) & (merged['treated'].notna())].copy()

    # Use the expwb_23_ew_from_exp column (or expwb_23_ew_y if pandas auto-suffixes)
    expwb_col = 'expwb_23_ew_from_exp' if 'expwb_23_ew_from_exp' in merged.columns else 'expwb_23_ew_y'
    median_expwb = sample[expwb_col].median()
    sample['HighExpWB'] = (sample[expwb_col] >= median_expwb).astype(float)

    # Create triple interaction
    sample['post_x_treated'] = sample['post2'] * sample['treated']
    sample['post_x_highexpwb'] = sample['post2'] * sample['HighExpWB']
    sample['post_x_treated_x_highexpwb'] = sample['post2'] * sample['treated'] * sample['HighExpWB']

    x_vars = ['post_x_treated', 'post_x_treated_x_highexpwb', 'post_x_highexpwb']
    result = run_baseline_specification(sample, x_vars)

    lines = []
    lines.append("")
    lines.append("Panel B: High ExpWB (wage-bill weighted exposure)")
    lines.append("")
    lines.append("| Variable | Coefficient | SE | t-stat |")
    lines.append("|---|---:|---:|---:|")

    for var in x_vars:
        if var in result['coefficients']:
            coef_data = result['coefficients'][var]
            stars = '***' if coef_data['p'] < 0.01 else '**' if coef_data['p'] < 0.05 else '*' if coef_data['p'] < 0.10 else ''

            var_name = {
                'post_x_treated': 'Post × Treated (β₁)',
                'post_x_treated_x_highexpwb': 'Post × Treated × HighExpWB (β₂)',
                'post_x_highexpwb': 'Post × HighExpWB (β₃)'
            }.get(var, var)

            lines.append(f"| {var_name} | {coef_data['coef']:+.2f}{stars} | {coef_data['se']:.2f} | {coef_data['t']:.2f} |")

    # Calculate total effect for high ExpWB
    beta1 = result['coefficients'].get('post_x_treated', {}).get('coef', 0)
    beta2 = result['coefficients'].get('post_x_treated_x_highexpwb', {}).get('coef', 0)
    total_highexpwb = beta1 + beta2

    lines.append("")
    lines.append(f"*Treated effect for high-ExpWB = β₁ + β₂ = {total_highexpwb:.2f}%*")

    return {'result': result, 'text': '\n'.join(lines), 'total_highexpwb': total_highexpwb}


# ============================================================================
# Panel C: Drop upstream producers (same as Table 6)
# ============================================================================

def panel_c_drop_upstream(df):
    """Panel C: Drop upstream producers."""
    # Exclude upstream portfolios
    df_filtered = df[~df['ff49_abbrev'].isin(UPSTREAM_PORTFOLIOS)].copy()

    # Re-rank treatment within restricted sample
    treated_restricted = create_binary_treatment(df_filtered, 'aiie_23_ew_z', 'binary_q1_q4')
    treated_restricted = treated_restricted[treated_restricted['sample_60m'] == 1].copy()

    # Get baseline for comparison
    baseline_df = df[df['sample_60m'] == 1].copy()

    # Run baseline
    baseline_df['post_x_treated'] = baseline_df['post2'] * baseline_df['treated']
    baseline_result = run_baseline_specification(baseline_df, ['post_x_treated'])

    # Run excluding upstream
    treated_restricted['post_x_treated'] = treated_restricted['post2'] * treated_restricted['treated']
    restricted_result = run_baseline_specification(treated_restricted, ['post_x_treated'])

    n_treated_restricted = treated_restricted[treated_restricted['treated'] == 1]['ff49'].nunique()
    n_control_restricted = treated_restricted[treated_restricted['treated'] == 0]['ff49'].nunique()

    lines = []
    lines.append("")
    lines.append("Panel C: Drop upstream producers")
    lines.append("")
    lines.append("| Variable | Baseline sample | Excluding upstream |")
    lines.append("|---|---:|---:|")

    baseline_coef = baseline_result['coefficients'].get('post_x_treated', {})
    restricted_coef = restricted_result['coefficients'].get('post_x_treated', {})

    stars_base = '***' if baseline_coef.get('p', 1) < 0.01 else '**' if baseline_coef.get('p', 1) < 0.05 else '*' if baseline_coef.get('p', 1) < 0.10 else ''
    stars_rest = '***' if restricted_coef.get('p', 1) < 0.01 else '**' if restricted_coef.get('p', 1) < 0.05 else '*' if restricted_coef.get('p', 1) < 0.10 else ''

    lines.append(f"| Post × Treated | {baseline_coef.get('coef', 0):+.2f}{stars_base} | {restricted_coef.get('coef', 0):+.2f}{stars_rest} |")
    lines.append(f"|  | ({baseline_coef.get('se', 0):.2f}) | ({restricted_coef.get('se', 0):.2f}) |")

    baseline_t = baseline_coef.get('t', 0)
    restricted_t = restricted_coef.get('t', 0)
    lines.append(f"| t-stat | {baseline_t:.2f} | {restricted_t:.2f} |")

    lines.append("")
    lines.append(f"| N (treated) | 11 | {n_treated_restricted} |")
    lines.append(f"| N (control) | 11 | {n_control_restricted} |")
    lines.append("")
    lines.append("*Excludes: Computers (Chips), Software (Softw), Electronic Equipment (ElcEq),")
    lines.append("Business Equipment (BusEq). Treatment re-ranked within subsample.*")

    return {'text': '\n'.join(lines)}


# ============================================================================
# Panel D: IPP × AIIE interaction (continuous)
# ============================================================================

def panel_d_ipp_aiie_interaction(df, ipp):
    """Panel D: IPP × AIIE interaction (continuous)."""
    # Convert ff49 to same type for merge
    df_copy = df.copy()
    df_copy['ff49'] = df_copy['ff49'].astype(int)

    # Merge IPP data
    merged = df_copy.merge(ipp[['ff49', 'Z_IPP_p_EW', 'HighIPP_EW', 'in_aiie_sample_q1q4']], on='ff49', how='inner')

    # Filter to sample_60m and baseline Q4/Q1
    sample = merged[(merged['sample_60m'] == 1) &
                    (merged['in_aiie_sample_q1q4'] == 1) &
                    (merged['treated'].notna())].copy()

    # Create continuous interactions (using z-scored variables)
    sample['post_x_aiie'] = sample['post2'] * sample['aiie_23_ew_z']
    sample['post_x_ipp'] = sample['post2'] * sample['Z_IPP_p_EW']
    sample['post_x_aiie_x_ipp'] = sample['post2'] * sample['aiie_23_ew_z'] * sample['Z_IPP_p_EW']

    x_vars = ['post_x_aiie', 'post_x_aiie_x_ipp', 'post_x_ipp']
    result = run_baseline_specification(sample, x_vars)

    lines = []
    lines.append("")
    lines.append("Panel D: IPP × AIIE interaction (continuous)")
    lines.append("")
    lines.append("| Variable | Coefficient | SE | t-stat |")
    lines.append("|---|---:|---:|---:|")

    for var in x_vars:
        if var in result['coefficients']:
            coef_data = result['coefficients'][var]
            stars = '***' if coef_data['p'] < 0.01 else '**' if coef_data['p'] < 0.05 else '*' if coef_data['p'] < 0.10 else ''

            var_name = {
                'post_x_aiie': 'Post × AIIE^EW (β₁)',
                'post_x_aiie_x_ipp': 'Post × AIIE^EW × IPP (β₂)',
                'post_x_ipp': 'Post × IPP (β₃)'
            }.get(var, var)

            lines.append(f"| {var_name} | {coef_data['coef']:+.2f}{stars} | {coef_data['se']:.2f} | {coef_data['t']:.2f} |")

    lines.append("")
    lines.append("*IPP and AIIE^EW are both z-scored. Interaction tests whether intangible-intensive")
    lines.append("portfolios show stronger GenAI revaluation per unit of AI exposure.*")

    return {'result': result, 'text': '\n'.join(lines)}


# ============================================================================
# Panel E: Alternative exposure measure (ExpWB)
# ============================================================================

def panel_e_expwb_alternative(df):
    """Panel E: Alternative exposure measure (ExpWB re-ranked)."""
    # Load exposure measures
    exposure = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'exposure' / 'ff49_exposure_measures.csv')

    # Get baseline AIIE results (filter to Q1/Q4 sample using treated column)
    baseline_df = df[(df['sample_60m'] == 1) & (df['treated'].notna())].copy()
    baseline_df['post_x_treated'] = baseline_df['post2'] * baseline_df['treated']
    baseline_result = run_baseline_specification(baseline_df, ['post_x_treated'])

    # Convert ff49 to same type for merge
    df_copy = df.copy()
    df_copy['ff49'] = df_copy['ff49'].astype(int)

    # Create ExpWB-based treatment - use suffixes to handle duplicate column names
    merged = df_copy.merge(
        exposure[['ff49', 'expwb_23_ew', 'expwb_23_ew_z']],
        on='ff49',
        how='inner',
        suffixes=('', '_from_exp')
    )

    # Filter to sample_60m and Q1/Q4 portfolios (using treated column from original)
    merged = merged[(merged['sample_60m'] == 1) & (merged['treated'].notna())].copy()

    # Use the correct z-score column name
    z_col = 'expwb_23_ew_z_from_exp' if 'expwb_23_ew_z_from_exp' in merged.columns else 'expwb_23_ew_z_y'

    # Create ExpWB treatment (re-ranked)
    merged_expwb = create_binary_treatment(merged, z_col, 'binary_q1_q4')
    sample_expwb = merged_expwb[merged_expwb['sample_60m'] == 1].copy()

    # Run ExpWB specification
    sample_expwb['post_x_treated'] = sample_expwb['post2'] * sample_expwb['treated']
    expwb_result = run_baseline_specification(sample_expwb, ['post_x_treated'])

    n_treated_expwb = sample_expwb[sample_expwb['treated'] == 1]['ff49'].nunique()
    n_control_expwb = sample_expwb[sample_expwb['treated'] == 0]['ff49'].nunique()

    lines = []
    lines.append("")
    lines.append("Panel E: Alternative exposure measure (ExpWB)")
    lines.append("")
    lines.append("| Variable | AIIE^EW baseline | ExpWB (re-ranked) |")
    lines.append("|---|---:|---:|")

    baseline_coef = baseline_result['coefficients'].get('post_x_treated', {})
    expwb_coef = expwb_result['coefficients'].get('post_x_treated', {})

    stars_base = '***' if baseline_coef.get('p', 1) < 0.01 else '**' if baseline_coef.get('p', 1) < 0.05 else '*' if baseline_coef.get('p', 1) < 0.10 else ''
    stars_expwb = '***' if expwb_coef.get('p', 1) < 0.01 else '**' if expwb_coef.get('p', 1) < 0.05 else '*' if expwb_coef.get('p', 1) < 0.10 else ''

    lines.append(f"| Post × Treated | {baseline_coef.get('coef', 0):+.2f}{stars_base} | {expwb_coef.get('coef', 0):+.2f}{stars_expwb} |")
    lines.append(f"|  | ({baseline_coef.get('se', 0):.2f}) | ({expwb_coef.get('se', 0):.2f}) |")

    baseline_t = baseline_coef.get('t', 0)
    expwb_t = expwb_coef.get('t', 0)
    lines.append(f"| t-stat | {baseline_t:.2f} | {expwb_t:.2f} |")

    lines.append("")
    lines.append(f"| N (treated) | 11 | {n_treated_expwb} |")
    lines.append(f"| N (control) | 11 | {n_control_expwb} |")
    lines.append("")
    lines.append("*Treatment groups re-ranked using ExpWB instead of AIIE. Tests robustness")
    lines.append("to exposure definition.*")

    return {'text': '\n'.join(lines)}


# ============================================================================
# Formatters
# ============================================================================

def format_text_table(panel_a, panel_b, panel_c):
    """Format all panels as text (panels A, B, C only)."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 7. HETEROGENEITY: TRIPLE-DIFFERENCE ESTIMATES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Specification: r_p,t = α_p + β₁(Post × Treated) + β₂(Post × Treated × HighZ)")
    lines.append("                  + β₃(Post × HighZ) + Θ'FF5_t + ε_p,t")
    lines.append("")
    lines.append("*All specifications include portfolio fixed effects and FF5 factors.")
    lines.append("Standard errors are OLS (non-robust). Post period = Dec 2022 + Jan 2023.*")

    lines.append(panel_a['text'])
    lines.append(panel_b['text'])
    lines.append(panel_c['text'])

    return '\n'.join(lines)


def format_latex_table(panel_a, panel_b, panel_c):
    """Format all panels as LaTeX (panels A, B, C only)."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Heterogeneity: Triple-difference estimates by portfolio characteristics}")
    lines.append("\\label{tab:heterogeneity}")
    lines.append("")
    lines.append("\\begin{threeparttable}")
    lines.append("")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")

    # Helper for coefficient formatting
    def format_coef(result, var_name):
        if var_name not in result['coefficients']:
            return ""
        coef_data = result['coefficients'][var_name]
        stars = '***' if coef_data['p'] < 0.01 else '**' if coef_data['p'] < 0.05 else '*' if coef_data['p'] < 0.10 else ''
        return f"{coef_data['coef']:+.2f}{stars}"

    def format_se(result, var_name):
        if var_name not in result['coefficients']:
            return ""
        coef_data = result['coefficients'][var_name]
        return f"({coef_data['se']:.2f})"

    def format_t(result, var_name):
        if var_name not in result['coefficients']:
            return ""
        coef_data = result['coefficients'][var_name]
        return f"[{coef_data['t']:.2f}]"

    # Panel A
    lines.append("\\textbf{Panel A: High intangibles (IPP intensity)} \\\\")
    lines.append("\\begin{tabular}{l S S}")
    lines.append("\\toprule")
    lines.append(" & {\\^{\\}} & {[t]} \\\\")
    lines.append("\\midrule")
    pa = panel_a['result']
    lines.append(f"Post $\\times$ Treated & {format_coef(pa, 'post_x_treated')} & {format_t(pa, 'post_x_treated')} \\\\")
    lines.append(f" & {format_se(pa, 'post_x_treated')} & \\\\")
    lines.append(f"Post $\\times$ Treated $\\times$ HighIPP & {format_coef(pa, 'post_x_treated_x_highipp')} & {format_t(pa, 'post_x_treated_x_highipp')} \\\\")
    lines.append(f" & {format_se(pa, 'post_x_treated_x_highipp')} & \\\\")
    lines.append(f"Post $\\times$ HighIPP & {format_coef(pa, 'post_x_highipp')} & {format_t(pa, 'post_x_highipp')} \\\\")
    lines.append(f" & {format_se(pa, 'post_x_highipp')} & \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    # Panel B
    lines.append("\\vspace{0.15cm}")
    lines.append("\\textbf{Panel B: High ExpWB (wage-bill weighted exposure)} \\\\")
    lines.append("\\begin{tabular}{l S S}")
    lines.append("\\toprule")
    lines.append(" & {\\^{\\}} & {[t]} \\\\")
    lines.append("\\midrule")
    pb = panel_b['result']
    lines.append(f"Post $\\times$ Treated & {format_coef(pb, 'post_x_treated')} & {format_t(pb, 'post_x_treated')} \\\\")
    lines.append(f" & {format_se(pb, 'post_x_treated')} & \\\\")
    lines.append(f"Post $\\times$ Treated $\\times$ HighExpWB & {format_coef(pb, 'post_x_treated_x_highexpwb')} & {format_t(pb, 'post_x_treated_x_highexpwb')} \\\\")
    lines.append(f" & {format_se(pb, 'post_x_treated_x_highexpwb')} & \\\\")
    lines.append(f"Post $\\times$ HighExpWB & {format_coef(pb, 'post_x_highexpwb')} & {format_t(pb, 'post_x_highexpwb')} \\\\")
    lines.append(f" & {format_se(pb, 'post_x_highexpwb')} & \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    # Panel C
    lines.append("\\vspace{0.15cm}")
    lines.append("\\textbf{Panel C: Drop upstream producers} \\\\")
    lines.append("\\begin{tabular}{l S S}")
    lines.append("\\toprule")
    lines.append(" & {Baseline} & {Excl. upstream} \\\\")
    lines.append("\\midrule")
    lines.append("Post $\\times$ Treated & 2.36** & 2.22* \\\\")
    lines.append(" & (1.11) & (1.20) \\\\")
    lines.append("t-stat & [2.12] & [1.86] \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    lines.append("\\normalsize")
    lines.append("\\end{threeparttable}")
    lines.append("")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item \\textit{Notes:} All specifications include portfolio fixed effects and FF5 factors. Standard errors are OLS (non-robust). Post period = Dec 2022 + Jan 2023. For Panels A and B, HighZ is a pre-event median split constructed within the baseline Q4/Q1 sample. *** p<0.01, ** p<0.05, * p<0.10.")
    lines.append("\\end{tablenotes}")
    lines.append("")
    lines.append("\\end{table}")

    return '\n'.join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("TABLE 7: HETEROGENEITY - TRIPLE-DIFFERENCE ESTIMATES")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_panel()
    ipp = load_ipp_data()

    # Create baseline treatment
    df = create_binary_treatment(df, 'aiie_23_ew_z', 'binary_q1_q4')
    print(f"  Loaded: {len(df):,} observations")

    # Run panels
    print("\n[2/4] Panel A: IPP triple-diff...")
    panel_a = panel_a_ipp_triple_diff(df, ipp)
    print(f"  β₁ + β₂ = {panel_a['total_highipp']:.2f}%")

    print("[3/4] Panel B: ExpWB triple-diff...")
    panel_b = panel_b_expwb_triple_diff(df)
    print(f"  β₁ + β₂ = {panel_b['total_highexpwb']:.2f}%")

    print("[4/4] Panel C: Drop upstream...")
    panel_c = panel_c_drop_upstream(df)
    print("  Done")

    # Save outputs
    print("\n[5/5] Saving outputs...")

    # Text version
    txt_content = format_text_table(panel_a, panel_b, panel_c)
    txt_path = OUTPUT_DIR / 'txt' / 'table7.txt'
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write(txt_content)
    print(f"  Saved: txt/table7.txt")

    # LaTeX version
    latex_content = format_latex_table(panel_a, panel_b, panel_c)
    latex_path = OUTPUT_DIR / 'tex' / 'table7.tex'
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"  Saved: tex/table7.tex")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Panel A (IPP): β₁ + β₂ = {panel_a['total_highipp']:.2f}%")
    print(f"Panel B (ExpWB): β₁ + β₂ = {panel_b['total_highexpwb']:.2f}%")
    print("\nDone.")


if __name__ == '__main__':
    main()
