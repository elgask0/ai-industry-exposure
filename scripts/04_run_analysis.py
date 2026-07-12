#!/usr/bin/env python3
"""
ChatGPT Shock Event Study — DiD Analysis.

Single baseline (AIIE 2023 EW) + 7 robustness checks (including ExpWB 2023 EW).
Full regression detail for each specification + compact summary table.
Pre-trends event study for the AIIE baseline only.

Baseline specification (informed by specification search):
  portfolio_only + ff5 + aiie_23_ew_z + normal SE + post2

Post-period rationale:
  - ChatGPT released Nov 30, 2022
  - Dec 2022 (τ=0): broad market decline (~-7%), AI signal drowned by macro noise
  - Jan 2023 (τ=+1): broad rally (+14%), AI differential emerges (t=2.35**)
  - Feb 2023 (τ=+2): essentially zero (t=0.13) → post3 would dilute
  - Post2 (Dec + Jan) captures the full two-month reaction window

Usage:
    python scripts/04_run_analysis.py
"""

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.data_prep import (
    create_binary_treatment,
    create_sample_window,
    load_panel,
)
from scripts.lib.estimator import get_return_col, run_single_regression
from scripts.lib.pretrends import evaluate_pretrends
from scripts.lib.config import FACTOR_GROUPS
from scripts.lib.output_utils import get_output_path

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'analysis'


# ============================================================================
# Specification data structure
# ============================================================================

@dataclass
class FinalSpec:
    """One DiD regression specification."""
    spec_id: int            # e.g. 0=Baseline, 1=R1, ..., 10=R10
    label: str              # e.g. "Baseline", "R1: 2021 version"
    exposure_measure: str   # e.g. 'aiie_23_ew_z'
    post_type: str          # 'post1' or 'post2'
    pre_window: int         # 60 or 120
    fe_type: str            # 'portfolio_only'
    factor_set: str         # 'ff3', 'ff5', etc.
    se_type: str            # 'normal' or 'cluster_month'
    is_baseline: bool


# ============================================================================
# Formatting helpers
# ============================================================================

def _stars(p: float) -> str:
    if pd.isna(p):
        return ''
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def _friendly_exposure(exp: str) -> str:
    """Human-readable exposure name."""
    mapping = {
        'aiie_23_ew_z': 'AIIE 2023 EW (z-scored)',
        'aiie_21_ew_z': 'AIIE 2021 EW (z-scored)',
        'aiie_23_vw_z': 'AIIE 2023 VW (z-scored)',
        'expwb_23_ew_z': 'ExpWB 2023 EW (z-scored)',
    }
    return mapping.get(exp, exp)


def _friendly_factors(factor_set: str) -> str:
    """Human-readable factor set name."""
    cols = FACTOR_GROUPS.get(factor_set, [])
    if not cols:
        return 'None'
    return ', '.join(c.upper() for c in cols)


def _friendly_fe(fe_type: str) -> str:
    mapping = {
        'portfolio_only': 'Portfolio',
        'twoway': 'Portfolio + Month',
        'month_only': 'Month',
        'none': 'None',
    }
    return mapping.get(fe_type, fe_type)


def _friendly_se(se_type: str) -> str:
    mapping = {
        'normal': 'OLS (non-robust)',
        'hc1': 'Heteroskedasticity-robust (HC1)',
        'hc3': 'Heteroskedasticity-robust (HC3)',
        'cluster_month': 'Clustered by month',
        'cluster_portfolio': 'Clustered by portfolio',
    }
    return mapping.get(se_type, se_type)


def _friendly_post(post_type: str) -> str:
    mapping = {
        'post1': 'Dec 2022 only (1 month)',
        'post2': 'Dec 2022 + Jan 2023 (2 months)',
        'post3': 'Dec 2022 – Feb 2023 (3 months)',
    }
    return mapping.get(post_type, post_type)


def _friendly_sample(pre_window: int) -> str:
    event = pd.Timestamp('2022-12-01')
    start = event - pd.DateOffset(months=pre_window)
    return f"{start.strftime('%b %Y')} – Feb 2023 ({pre_window} months pre-event)"


def _friendly_return(exp: str) -> str:
    if '_ew_' in exp:
        return 'ret_ew_ex (EW excess return)'
    if '_vw_' in exp:
        return 'ret_vw_ex (VW excess return)'
    return '?'


# ============================================================================
# Baseline and robustness configuration
# ============================================================================

BASELINE = {
    'exposure_measure': 'aiie_23_ew_z',
    'post_type': 'post2',
    'pre_window': 60,
    'fe_type': 'portfolio_only',
    'factor_set': 'ff3',  # FF3 baseline (matches CLAUDE.md)
    'se_type': 'normal',
}

# Each robustness check overrides ONE parameter from the baseline.
ROBUSTNESS_CHECKS = [
    {'id': 'R1',  'label': '2021 version',
     'changes': {'exposure_measure': 'aiie_21_ew_z'}},
    {'id': 'R2',  'label': 'VW weighting',
     'changes': {'exposure_measure': 'aiie_23_vw_z'}},
    {'id': 'R3',  'label': 'Post 1m (Dec only)',
     'changes': {'post_type': 'post1'}},
    {'id': 'R4',  'label': '120m pre-window',
     'changes': {'pre_window': 120}},
    {'id': 'R5',  'label': 'FF5 factors',
     'changes': {'factor_set': 'ff5'}},
    {'id': 'R6',  'label': 'FF3 + MOM',
     'changes': {'factor_set': 'ff3_mom'}},
    {'id': 'R7',  'label': 'FF5 + MOM',
     'changes': {'factor_set': 'ff5_mom'}},
    {'id': 'R8',  'label': 'CAPM only',
     'changes': {'factor_set': 'capm'}},
    {'id': 'R9',  'label': 'Clustered SE (month)',
     'changes': {'se_type': 'cluster_month'}},
    {'id': 'R10', 'label': 'ExpWB 2023 EW',
     'changes': {'exposure_measure': 'expwb_23_ew_z'}},
]


def build_all_specs() -> List[FinalSpec]:
    """Build 11 FinalSpec objects: 1 baseline + 10 robustness checks."""
    specs = []

    # Baseline
    specs.append(FinalSpec(
        spec_id=0,
        label='Baseline',
        is_baseline=True,
        **BASELINE,
    ))

    # Robustness checks
    for i, rob in enumerate(ROBUSTNESS_CHECKS, start=1):
        merged = dict(BASELINE)
        merged.update(rob['changes'])
        specs.append(FinalSpec(
            spec_id=i,
            label=f"{rob['id']}: {rob['label']}",
            is_baseline=False,
            **merged,
        ))

    return specs


# ============================================================================
# Sample preparation
# ============================================================================

def prepare_samples(df: pd.DataFrame, specs: List[FinalSpec]) -> Dict:
    """Cache binary Q1/Q4 subsets for each unique (exposure, pre_window)."""
    samples = {}
    for spec in specs:
        key = (spec.exposure_measure, spec.pre_window)
        if key not in samples:
            sample = create_sample_window(df, spec.pre_window)
            sample = create_binary_treatment(
                sample, spec.exposure_measure, 'binary_q1_q4',
            )
            samples[key] = sample
    return samples


# ============================================================================
# DiD regressions
# ============================================================================

def run_did(sample: pd.DataFrame, spec: FinalSpec) -> Optional[Dict]:
    """Run single-coefficient DiD regression for one spec.

    Returns the full regression result including all non-FE coefficients
    (treatment + factors) and model diagnostics.
    """
    ret_col = get_return_col(spec.exposure_measure)

    work = sample.copy()
    work['post_treated'] = work[spec.post_type].astype(float) * work['treated']

    x_vars = ['post_treated']
    # Include treated main effect when portfolio FE is absent
    if spec.fe_type in ('none', 'month_only'):
        work['treated_ctrl'] = work['treated']
        x_vars.append('treated_ctrl')

    result = run_single_regression(
        work, ret_col, x_vars,
        fe_type=spec.fe_type,
        factor_set=spec.factor_set,
        se_type=spec.se_type,
    )
    if result is None:
        return None

    stats = result['coefficients'].get('post_treated')
    if stats is None:
        return None

    # Return full result: key treatment stats at top level for convenience,
    # plus all coefficients and model diagnostics
    return {
        'coef': stats['coef'],
        'se': stats['se'],
        't': stats['t'],
        'p': stats['p'],
        'nobs': result['nobs'],
        'r2': result['r2'],
        'coefficients': result['coefficients'],
        'f_stat': result.get('f_stat'),
        'f_pval': result.get('f_pval'),
        'aic': result.get('aic'),
        'bic': result.get('bic'),
        'n_fe_dummies': result.get('n_fe_dummies', 0),
    }


def run_all_did(
    specs: List[FinalSpec],
    samples: Dict,
) -> List[Tuple[FinalSpec, Optional[Dict]]]:
    """Run DiD for all specs. Returns list of (spec, result_dict) tuples."""
    results = []
    for spec in specs:
        sample = samples[(spec.exposure_measure, spec.pre_window)]
        did = run_did(sample, spec)
        results.append((spec, did))
    return results


# ============================================================================
# Pre-trends event study
# ============================================================================

def run_pretrends_analysis(df: pd.DataFrame) -> Optional[Dict]:
    """Run dynamic event study for the AIIE baseline with 12m pre-window."""
    exposure = BASELINE['exposure_measure']
    ret_col = get_return_col(exposure)
    fe_type = BASELINE['fe_type']
    factor_set = BASELINE['factor_set']
    se_type = BASELINE['se_type']

    # 12-month pre-window for pre-trends
    sample = create_sample_window(df, 12)
    sample = create_binary_treatment(sample, exposure, 'binary_q1_q4')

    # Event times: -12 to +2, omitting -1 (reference)
    event_times = list(range(-12, 0)) + list(range(0, 3))
    event_times.remove(-1)

    work = sample.copy()
    x_vars = []
    for tau in event_times:
        vname = f'evt_{tau}'
        work[vname] = (work['event_time'] == tau).astype(float) * work['treated']
        x_vars.append(vname)

    result = run_single_regression(
        work, ret_col, x_vars,
        fe_type=fe_type,
        factor_set=factor_set,
        se_type=se_type,
    )
    if result is None:
        return None

    pre_results: Dict[int, dict] = {}
    post_results: Dict[int, dict] = {}
    for tau in event_times:
        vname = f'evt_{tau}'
        if vname in result['coefficients']:
            if tau < 0:
                pre_results[tau] = result['coefficients'][vname]
            else:
                post_results[tau] = result['coefficients'][vname]

    pt = evaluate_pretrends(pre_results, post_results)

    return {
        'result': result,
        'pretrends': pt,
        'pre_results': pre_results,
        'post_results': post_results,
        'event_times': event_times,
    }


# ============================================================================
# Formatting: full regression blocks
# ============================================================================

def _friendly_varname(var: str) -> str:
    """Human-readable variable name for coefficient table."""
    mapping = {
        'post_treated': 'Post × Treated',
        'treated_ctrl': 'Treated',
        'mktrf': 'MktRF',
        'smb': 'SMB',
        'hml': 'HML',
        'rmw': 'RMW',
        'cma': 'CMA',
        'mom': 'MOM',
    }
    return mapping.get(var, var)


# Preferred variable display order: treatment first, then factors
_VAR_ORDER = [
    'post_treated', 'treated_ctrl',
    'mktrf', 'smb', 'hml', 'rmw', 'cma', 'mom',
]


def format_regression_block(
    spec: FinalSpec,
    did: Optional[Dict],
    block_number: int,
) -> str:
    """
    Produce a detailed regression block with full coefficient table.

    Shows all substantive coefficients (treatment + factors). Portfolio FE
    dummies are noted as suppressed.
    """
    sep = "─" * 68
    tbl_sep = "─" * 65

    lines = []
    lines.append(sep)
    lines.append(f"  ({block_number}) {spec.label}")
    lines.append(sep)
    lines.append(f"    Dep. var:       {_friendly_return(spec.exposure_measure)}")
    lines.append(f"    Exposure:       {_friendly_exposure(spec.exposure_measure)}")
    lines.append(f"    FE:             {_friendly_fe(spec.fe_type)}")
    lines.append(f"    Factors:        {_friendly_factors(spec.factor_set)}")
    lines.append(f"    SE:             {_friendly_se(spec.se_type)}")
    lines.append(f"    Post period:    {_friendly_post(spec.post_type)}")
    lines.append(f"    Sample:         {_friendly_sample(spec.pre_window)}")
    lines.append(f"    Treatment:      Binary Q1 vs Q4 (top/bottom quartile by exposure)")
    lines.append("")

    if did is None:
        lines.append(f"    {tbl_sep}")
        lines.append(f"    REGRESSION FAILED")
        lines.append(f"    {tbl_sep}")
    else:
        coefficients = did.get('coefficients', {})
        nobs = did['nobs']
        r2 = did['r2']
        f_stat = did.get('f_stat')
        f_pval = did.get('f_pval')
        n_fe = did.get('n_fe_dummies', 0)

        # Sort variables: known order first, then any remaining alphabetically
        known = [v for v in _VAR_ORDER if v in coefficients]
        remaining = sorted([v for v in coefficients if v not in _VAR_ORDER])
        ordered_vars = known + remaining

        # Coefficient table header
        lines.append(f"    {tbl_sep}")
        lines.append(
            f"    {'Variable':<20s}  {'Coef':>10s}  {'Std.Err.':>10s}  "
            f"{'t':>8s}  {'P>|t|':>8s}  {'':>5s}"
        )
        lines.append(f"    {tbl_sep}")

        for var in ordered_vars:
            d = coefficients[var]
            name = _friendly_varname(var)
            sig = _stars(d['p'])
            lines.append(
                f"    {name:<20s}  {d['coef']:+10.4f}  {d['se']:10.4f}  "
                f"{d['t']:+8.3f}  {d['p']:8.4f}  {sig:>5s}"
            )

        lines.append(f"    {tbl_sep}")

        # Model diagnostics
        if n_fe > 0:
            lines.append(f"    Portfolio FE:       {n_fe} dummies (suppressed)")
        lines.append(f"    Observations:      {nobs:,}")
        lines.append(f"    R²:                {r2:.4f}")
        lines.append(f"    {tbl_sep}")

    return "\n".join(lines)


# ============================================================================
# Formatting: compact summary table
# ============================================================================

def format_summary_table(results: List[Tuple[FinalSpec, Optional[Dict]]]) -> str:
    """
    Produce a compact single-column summary table of all 11 regressions.

    Spec                           γ        (SE)      t       p      N      R²
    ─────────────────────────────────────────────────────────────────────────────
    Baseline                    +2.45**   (1.111)  +2.204  0.028   1386  0.6458
    R1: 2021 version            +2.19**   (1.073)  +2.045  0.041   1386  0.6780
    ...
    R10: ExpWB 2023 EW          +2.52**   (1.071)  +2.354  0.019   1386  0.6697
    """
    lines = []
    hdr = (
        f"  {'Spec':<30s}  {'γ':>12s}  {'(SE)':>10s}  "
        f"{'t':>8s}  {'p':>7s}  {'N':>6s}  {'R²':>7s}"
    )
    sep = "  " + "─" * 85

    lines.append(hdr)
    lines.append(sep)

    for spec, did in results:
        label = spec.label
        if did is None:
            lines.append(f"  {label:<30s}  {'FAIL':>12s}")
            continue

        coef = did['coef']
        se = did['se']
        t = did['t']
        p = did['p']
        nobs = did['nobs']
        r2 = did['r2']
        stars = _stars(p)

        coef_str = f"{coef:+.4f}{stars}"
        se_str = f"({se:.4f})"
        t_str = f"{t:+.3f}"
        p_str = f"{p:.4f}"
        n_str = f"{int(nobs)}"
        r2_str = f"{r2:.4f}"

        lines.append(
            f"  {label:<30s}  {coef_str:>12s}  {se_str:>10s}  "
            f"{t_str:>8s}  {p_str:>7s}  {n_str:>6s}  {r2_str:>7s}"
        )

    lines.append(sep)
    lines.append("")
    lines.append("  Stars: *** p<0.01, ** p<0.05, * p<0.10")
    lines.append("  γ = coefficient on Post × Treated interaction")

    return "\n".join(lines)


# ============================================================================
# Formatting: full DiD report (blocks + summary)
# ============================================================================

def format_did_report(results: List[Tuple[FinalSpec, Optional[Dict]]]) -> str:
    """Combine individual regression blocks + compact summary table."""
    lines = []
    lines.append("=" * 88)
    lines.append("DiD REGRESSION RESULTS — DETAILED")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Baseline: AIIE 2023 EW, portfolio FE, FF3 factors, OLS SE,")
    lines.append("          post2 (Dec 2022 + Jan 2023), 60m pre-window, binary Q1/Q4")
    lines.append("Each robustness check changes ONE parameter from the baseline.")
    lines.append("")

    # Individual regression blocks
    for i, (spec, did) in enumerate(results, start=1):
        lines.append(format_regression_block(spec, did, i))
        lines.append("")

    # Summary table
    lines.append("")
    lines.append("=" * 88)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 88)
    lines.append("")
    lines.append(format_summary_table(results))

    return "\n".join(lines)


# ============================================================================
# Formatting: pre-trends text report
# ============================================================================

def format_pretrends_report(pt_output: Dict) -> str:
    """Format pre-trends results as readable text."""
    pt = pt_output['pretrends']
    result = pt_output['result']
    pre_results = pt_output['pre_results']
    post_results = pt_output['post_results']

    lines = []
    lines.append("=" * 88)
    lines.append("PRE-TRENDS EVENT STUDY — AIIE 2023 EW")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Specification:")
    lines.append(f"  Exposure:    {_friendly_exposure(BASELINE['exposure_measure'])}")
    lines.append(f"  Dep. var:    {_friendly_return(BASELINE['exposure_measure'])}")
    lines.append(f"  FE:          {_friendly_fe(BASELINE['fe_type'])}")
    lines.append(f"  Factors:     {_friendly_factors(BASELINE['factor_set'])}")
    lines.append(f"  SE:          {_friendly_se(BASELINE['se_type'])}")
    lines.append(f"  Treatment:   Binary Q1 vs Q4")
    lines.append(f"  Pre-window:  12 months (Dec 2021 – Nov 2022)")
    lines.append(f"  Reference:   τ = -1 (Nov 2022, omitted)")
    lines.append(f"  Post:        τ ∈ {{0, 1, 2}} (Dec 2022, Jan 2023, Feb 2023)")
    lines.append("")
    lines.append(f"  N = {result['nobs']:,}    R² = {result['r2']:.4f}")
    lines.append("")

    # Coefficient table
    col_hdr = (
        f"  {'τ':>5s}  {'Month':>10s}  {'Coef':>10s}  "
        f"{'SE':>10s}  {'t-stat':>8s}  {'p-val':>8s}  {'':>5s}"
    )
    col_sep = f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*5}"

    lines.append(col_hdr)
    lines.append(col_sep)

    # Month labels
    event_date = pd.Timestamp('2022-12-01')
    def tau_to_month(tau):
        return (event_date + pd.DateOffset(months=tau)).strftime('%b %Y')

    all_taus = sorted(list(pre_results.keys()) + list(post_results.keys()))

    for tau in all_taus:
        d = pre_results.get(tau) or post_results.get(tau)
        sig = _stars(d['p'])
        tag = '  [POST]' if tau >= 0 else ''
        month = tau_to_month(tau)
        lines.append(
            f"  {tau:+5d}  {month:>10s}  {d['coef']:+10.4f}  "
            f"{d['se']:10.4f}  {d['t']:+8.3f}  {d['p']:8.4f}  {sig:>5s}{tag}"
        )

    # Reference period
    lines.append(
        f"    -1  {tau_to_month(-1):>10s}       0.0000"
        f"                                   (reference)"
    )

    # Diagnostics
    lines.append("")
    lines.append("  " + "─" * 50)
    lines.append("  PRE-TRENDS DIAGNOSTICS")
    lines.append("  " + "─" * 50)
    lines.append(f"    Pre-period coefficients significant at 10%:  {pt.n_pre_significant_10} / 11")
    lines.append(f"    Pre-period coefficients significant at  5%:  {pt.n_pre_significant_05} / 11")
    lines.append(f"    Maximum |t-statistic| among pre-period:      {pt.max_pre_abs_t:.3f}")
    lines.append(f"    Joint F-statistic (all pre-period = 0):      {pt.joint_f_stat:.3f}")
    lines.append(f"    Joint F p-value:                             {pt.joint_f_pval:.4f}")
    lines.append("")
    lines.append(f"    Strict criteria:    {'✓ PASS' if pt.passes_strict else '✗ FAIL'}")
    lines.append(f"    Moderate criteria:  {'✓ PASS' if pt.passes_moderate else '✗ FAIL'}")
    lines.append("")

    # Interpretation
    if pt.passes_strict:
        lines.append("  Interpretation: No evidence of differential pre-trends. The parallel")
        lines.append("  trends assumption is well supported. All pre-event coefficients are")
        lines.append("  individually and jointly insignificant.")
    elif pt.passes_moderate:
        lines.append("  Interpretation: Pre-trends pass moderate but not strict criteria.")
        lines.append("  There is limited evidence of differential pre-trends, but the")
        lines.append("  joint F-test does not reject the null of parallel trends.")
    else:
        lines.append("  Interpretation: Pre-trends fail both strict and moderate criteria.")
        lines.append("  The parallel trends assumption may be violated.")

    return "\n".join(lines)


# ============================================================================
# Pre-trends plotting
# ============================================================================

def plot_pretrends(pt_output: Dict, output_path: Path):
    """
    Generate professional event study plot using seaborn.

    Key design changes:
    - Uses seaborn for scientific styling
    - No yellow annotation box
    - No orange arrow annotations
    - τ=-1 and τ=0 points are joined (continuous line)
    - Professional color scheme
    """
    pre_results = pt_output['pre_results']
    post_results = pt_output['post_results']
    pt = pt_output['pretrends']

    # Collect taus, coefs, SEs
    taus = sorted(list(pre_results.keys()) + list(post_results.keys()))

    # Add τ=-1 (reference period) if not already present
    if -1 not in taus:
        taus = sorted(taus + [-1])

    coefs = []
    ses = []
    p_values = []
    for tau in taus:
        if tau == -1:
            # Reference period (omitted from regression)
            coefs.append(0.0)
            ses.append(0.0)
            p_values.append(1.0)
        else:
            d = pre_results.get(tau) or post_results.get(tau)
            coefs.append(d['coef'])
            ses.append(d['se'])
            p_values.append(d.get('p', d.get('p_value', 1.0)))

    ci_95 = [1.96 * s if s > 0 else 0 for s in ses]  # Handle se=0 for τ=-1
    lower = [c - ci for c, ci in zip(coefs, ci_95)]
    upper = [c + ci for c, ci in zip(coefs, ci_95)]

    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'tau': taus,
        'coef': coefs,
        'se': ses,
        'lower': lower,
        'upper': upper,
        'period': ['Pre' if t < 0 else 'Post' for t in taus],
        'p_value': p_values
    })

    # --- Plotting with seaborn ---
    # Set seaborn style
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.1)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Colors
    pre_color = '#1f77b4'  # Professional blue
    post_color = '#d62728'  # Professional red

    # Fill CI bands (separate for pre/post for clarity)
    pre_mask = df['tau'] < 0
    post_mask = df['tau'] >= 0

    # Plot pre-period points (τ < 0) in blue
    ax.plot(df.loc[pre_mask, 'tau'], df.loc[pre_mask, 'coef'], 'o-',
            color=pre_color, linewidth=1.8, markersize=6, label='Pre-period')

    # Plot post-period points (τ ≥ 0) in red
    ax.plot(df.loc[post_mask, 'tau'], df.loc[post_mask, 'coef'], 'o-',
            color=post_color, linewidth=1.8, markersize=6, label='Post-period')

    # Add error bars (separate for pre/post)
    ax.errorbar(df.loc[pre_mask, 'tau'], df.loc[pre_mask, 'coef'],
                yerr=df.loc[pre_mask, 'se'], fmt='none', ecolor=pre_color,
                alpha=0.5, capsize=3, linewidth=1.2)

    ax.errorbar(df.loc[post_mask, 'tau'], df.loc[post_mask, 'coef'],
                yerr=df.loc[post_mask, 'se'], fmt='none', ecolor=post_color,
                alpha=0.5, capsize=3, linewidth=1.2)

    # Fill CI bands
    ax.fill_between(df.loc[pre_mask, 'tau'],
                    df.loc[pre_mask, 'lower'],
                    df.loc[pre_mask, 'upper'],
                    alpha=0.15, color=pre_color, label='95% CI')

    ax.fill_between(df.loc[post_mask, 'tau'],
                    df.loc[post_mask, 'lower'],
                    df.loc[post_mask, 'upper'],
                    alpha=0.15, color=post_color, label='95% CI')

    # Highlight significant points
    sig_mask = df['p_value'] < 0.05
    ax.scatter(df.loc[sig_mask, 'tau'], df.loc[sig_mask, 'coef'],
               s=100, facecolors='none', edgecolors='red', linewidth=2,
               label='Significant (p<0.05)', zorder=5)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)

    # Event boundary line
    ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Event boundary (Nov 30, 2022)')

    # Shaded post-period background
    ax.axvspan(-0.5, max(df['tau']) + 0.5, alpha=0.05, color='gray')

    # Labels and title
    ax.set_xlabel('Event Month (τ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient γᵗ (%)', fontsize=12, fontweight='bold')
    factor_set_label = BASELINE['factor_set'].upper()
    ax.set_title(f'Event-Study Coefficients: AIIE 2023 EW ({factor_set_label} Factors)',
                 fontsize=13, fontweight='bold', pad=12)

    # Grid (subtle)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)

    # X-axis ticks
    ax.set_xticks(df['tau'])
    ax.tick_params(axis='both', labelsize=10)

    # Legend
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95,
             fancybox=True, shadow=False, frameon=True)

    # Add month labels on secondary x-axis
    event_date = pd.Timestamp('2022-12-01')
    month_labels = [(event_date + pd.DateOffset(months=t)).strftime('%b %y') for t in df['tau']]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(df['tau'])
    ax2.set_xticklabels(month_labels, fontsize=8, alpha=0.7, rotation=45, ha='left')
    ax2.tick_params(length=0, axis='x')
    ax2.set_xlabel('Calendar Month', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Full report
# ============================================================================

def assemble_full_report(did_report: str, pt_report: str) -> str:
    """Combine DiD report and pre-trends into one document."""
    lines = []
    lines.append("=" * 88)
    lines.append("ANALYSIS — ChatGPT Shock Event Study")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Event:     ChatGPT release (November 30, 2022)")
    lines.append("Method:    Difference-in-Differences with binary treatment")
    lines.append("Treatment: Top quartile (Q4) vs bottom quartile (Q1) of AI exposure")
    lines.append("Baseline:  AIIE 2023 EW (AI Industry Exposure, 2023 Language Model version)")
    lines.append("Post:      Two-month event window (Dec 2022 + Jan 2023)")
    lines.append("")
    lines.append("─" * 88)
    lines.append("")
    lines.append("")
    lines.append("PART 1: DiD REGRESSIONS (1 BASELINE + 10 ROBUSTNESS)")
    lines.append("=" * 88)
    lines.append("")
    lines.append(did_report)
    lines.append("")
    lines.append("")
    lines.append("PART 2: PRE-TRENDS EVENT STUDY")
    lines.append("=" * 88)
    lines.append("")
    lines.append(pt_report)
    return "\n".join(lines)


# ============================================================================
# CSV export
# ============================================================================

def export_csv(results: List[Tuple[FinalSpec, Optional[Dict]]]) -> pd.DataFrame:
    """Export all regression results to a flat CSV."""
    rows = []
    for spec, did in results:
        row = {
            'spec_id': spec.spec_id,
            'label': spec.label,
            'is_baseline': spec.is_baseline,
            'exposure_measure': spec.exposure_measure,
            'post_type': spec.post_type,
            'pre_window': spec.pre_window,
            'fe_type': spec.fe_type,
            'factor_set': spec.factor_set,
            'se_type': spec.se_type,
        }
        if did:
            row.update(did)
        else:
            row['error'] = 'regression_failed'
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def run_full_analysis(factor_set='ff3'):
    """Run complete analysis with specified factor set (FF3 or FF5)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Update BASELINE with the specified factor_set
    BASELINE['factor_set'] = factor_set

    print("=" * 70)
    print(f"ANALYSIS: ChatGPT Shock Event Study ({factor_set.upper()})")
    print(f"  Baseline: AIIE 2023 EW ({factor_set.upper()}) + 10 robustness checks")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading panel data...")
    df = load_panel()
    print(f"  {len(df):,} rows, {df['ff49'].nunique()} portfolios")

    # 2. Build specs
    specs = build_all_specs()
    print(f"\n[2/5] {len(specs)} DiD specifications defined:")
    for s in specs:
        tag = "★ BASELINE" if s.is_baseline else f"  {s.label.split(':')[0]}"
        print(f"  {tag:>12s}  {s.label}")

    # 3. Prepare samples
    print(f"\n[3/5] Preparing binary Q1/Q4 samples...")
    samples = prepare_samples(df, specs)
    for key, sample in samples.items():
        print(f"  ({key[0]}, {key[1]}m): {len(sample):,} obs, "
              f"{sample['treated'].sum():.0f} treated, "
              f"{(~sample['treated'].astype(bool)).sum():.0f} control")

    # 4. Run DiD regressions
    print(f"\n[4/5] Running DiD regressions ({len(specs)} total)...")
    print("─" * 70)
    t0 = time.time()
    did_results = run_all_did(specs, samples)
    elapsed = time.time() - t0

    # Print quick results
    for spec, did in did_results:
        if did:
            sig = _stars(did['p'])
            print(f"  {'★' if spec.is_baseline else ' '} {spec.label:<30s}  "
                  f"γ={did['coef']:+.4f}{sig:<3s}  t={did['t']:+.3f}  "
                  f"p={did['p']:.4f}  N={int(did['nobs'])}")
        else:
            print(f"    {spec.label:<30s}  FAILED")
    print(f"\n  Completed in {elapsed:.1f}s")

    # Save DiD outputs
    did_report = format_did_report(did_results)
    csv_df = export_csv(did_results)

    csv_df.to_csv(OUTPUT_DIR / 'did_table.csv', index=False)
    with open(OUTPUT_DIR / 'did_report.txt', 'w') as f:
        f.write(did_report)
    print(f"\n  Saved: did_table.csv, did_report.txt")

    # 5. Pre-trends event study (AIIE only)
    print(f"\n[5/5] Pre-trends event study (AIIE 2023 EW, 12m window)...")
    print("─" * 70)

    pt_output = run_pretrends_analysis(df)
    if pt_output:
        pt = pt_output['pretrends']
        print(f"  N = {pt_output['result']['nobs']:,}")
        print(f"  Pre-period coefficients sig at 10%: {pt.n_pre_significant_10}/11")
        print(f"  Max |t|: {pt.max_pre_abs_t:.3f}")
        print(f"  Joint F p-val: {pt.joint_f_pval:.4f}")
        print(f"  Strict:   {'✓ PASS' if pt.passes_strict else '✗ FAIL'}")
        print(f"  Moderate: {'✓ PASS' if pt.passes_moderate else '✗ FAIL'}")

        # Plot (save to figures/ subfolder)
        plot_filename = f'pretrends_{factor_set}.png'
        plot_path = OUTPUT_DIR / 'figures' / plot_filename
        plot_pretrends(pt_output, plot_path)
        print(f"  Saved: figures/{plot_filename}")

        # Text report
        pt_report = format_pretrends_report(pt_output)
        txt_filename = f'pretrends_results_{factor_set}.txt'
        with open(OUTPUT_DIR / txt_filename, 'w') as f:
            f.write(pt_report)
        print(f"  Saved: {txt_filename}")
    else:
        print("  FAILED — regression returned None")
        pt_report = "PRE-TRENDS ANALYSIS FAILED"

    # 6. Full combined report
    full_report = assemble_full_report(did_report, pt_report)
    with open(OUTPUT_DIR / 'full_report.txt', 'w') as f:
        f.write(full_report)
    print(f"\n  Saved: full_report.txt")

    # Summary
    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        size = f.stat().st_size
        print(f"  {f.name:<35s}  {size:>8,} bytes")
    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print("=" * 70)


def main():
    """Run both FF3 and FF5 analyses."""
    # Run FF3 analysis
    run_full_analysis(factor_set='ff3')

    # Run FF5 analysis
    print("\n\n" + "=" * 70)
    print("SWITCHING TO FF5 FACTOR MODEL")
    print("=" * 70)
    run_full_analysis(factor_set='ff5')


if __name__ == '__main__':
    main()
