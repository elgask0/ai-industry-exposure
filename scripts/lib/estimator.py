"""
Core regression engine with proper collinearity handling.

CRITICAL FIX: When month FE is included, FF factors MUST be excluded.
FF factors vary only over time (constant within a month across portfolios),
so they are perfectly collinear with month dummies. The previous code
silently dropped them, giving the false impression they were controlled for.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from .config import FACTOR_GROUPS


def get_return_col(exposure_measure: str) -> str:
    """Match exposure type to return type. EW exposure -> EW returns."""
    if '_ew_' in exposure_measure:
        return 'ret_ew_ex'
    if '_vw_' in exposure_measure:
        return 'ret_vw_ex'
    raise ValueError(f"Cannot determine return type for {exposure_measure}")


def build_design_matrix(
    df: pd.DataFrame,
    x_vars: List[str],
    fe_type: str,
    factor_set: str,
    additional_controls: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the full design matrix with proper collinearity handling.

    When month FE is included, FF factors are EXCLUDED because they are
    perfectly collinear (both vary only over time).
    """
    parts = []

    # 1. Variables of interest (event-time interactions)
    parts.append(df[x_vars].astype(float))

    # 2. Factor controls (ONLY if no month FE)
    factor_cols = list(FACTOR_GROUPS.get(factor_set, []))
    if fe_type in ('twoway', 'month_only'):
        factor_cols = []

    if factor_cols:
        parts.append(df[factor_cols].astype(float))

    # 3. Additional controls (factor-loading interactions, etc.)
    if additional_controls:
        valid = [c for c in additional_controls if c in df.columns]
        if valid:
            parts.append(df[valid].astype(float))

    # 4. Fixed effects via dummies
    if fe_type in ('twoway', 'portfolio_only'):
        port_dummies = pd.get_dummies(
            df['ff49'], prefix='fe_p', drop_first=True, dtype=float,
        )
        parts.append(port_dummies)

    if fe_type in ('twoway', 'month_only'):
        time_dummies = pd.get_dummies(
            df['ym_str'], prefix='fe_t', drop_first=True, dtype=float,
        )
        parts.append(time_dummies)

    X = pd.concat(parts, axis=1)

    # Add constant only if no FE (with FE the constant is redundant)
    if fe_type == 'none':
        X = sm.add_constant(X)

    return X, x_vars


def run_single_regression(
    df: pd.DataFrame,
    y_col: str,
    x_vars: List[str],
    fe_type: str,
    factor_set: str,
    se_type: str,
    additional_controls: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Run OLS regression with specified FE, factors, and SE.

    Parameters
    ----------
    df : Panel data (already filtered to sample window)
    y_col : Dependent variable
    x_vars : Variables of interest (event-time interactions)
    fe_type : 'twoway', 'month_only', 'portfolio_only', 'none'
    factor_set : Key into FACTOR_GROUPS
    se_type : 'normal', 'hc1', 'hc3', 'cluster_month', 'cluster_portfolio'
    additional_controls : Extra control column names

    Returns
    -------
    Dict with nobs, r2, coefficients[var] = {coef, se, t, p}
    """
    # Determine all needed columns for dropna
    factor_cols = list(FACTOR_GROUPS.get(factor_set, []))
    if fe_type in ('twoway', 'month_only'):
        factor_cols = []

    cols_needed = list(set(
        [y_col] + x_vars + factor_cols + ['ff49', 'ym', 'ym_str']
        + (additional_controls or [])
    ))
    cols_present = [c for c in cols_needed if c in df.columns]
    sample = df[cols_present].dropna()

    if len(sample) < 50:
        return None

    y = sample[y_col].astype(float).values
    X, interest_vars = build_design_matrix(
        sample, x_vars, fe_type, factor_set, additional_controls,
    )
    X = X.astype(float)

    # Choose SE type
    if se_type == 'normal':
        cov_type = 'nonrobust'
        cov_kwds = {}
    elif se_type == 'hc1':
        cov_type = 'HC1'
        cov_kwds = {}
    elif se_type == 'hc3':
        cov_type = 'HC3'
        cov_kwds = {}
    elif se_type == 'cluster_month':
        cov_type = 'cluster'
        cov_kwds = {'groups': sample['ym'].values}
    elif se_type == 'cluster_portfolio':
        cov_type = 'cluster'
        cov_kwds = {'groups': sample['ff49'].values}
    else:
        raise ValueError(f"Unknown se_type: {se_type}")

    try:
        model = OLS(y, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    except Exception:
        return None

    results: Dict = {
        'nobs': int(model.nobs),
        'r2': model.rsquared,
        'coefficients': {},
        'f_stat': float(model.fvalue) if np.isscalar(model.fvalue) else float(model.fvalue[0]) if hasattr(model.fvalue, '__len__') else np.nan,
        'f_pval': float(model.f_pvalue) if np.isscalar(model.f_pvalue) else float(model.f_pvalue[0]) if hasattr(model.f_pvalue, '__len__') else np.nan,
        'aic': model.aic,
        'bic': model.bic,
        'n_fe_dummies': len([c for c in X.columns if c.startswith('fe_p_') or c.startswith('fe_t_')]),
    }

    # Return all non-FE, non-constant coefficients (treatment + factors)
    report_vars = [
        v for v in X.columns
        if not v.startswith('fe_p_')
        and not v.startswith('fe_t_')
        and v != 'const'
    ]
    for var in report_vars:
        results['coefficients'][var] = {
            'coef': model.params[var],
            'se': model.bse[var],
            't': model.tvalues[var],
            'p': model.pvalues[var],
        }

    return results
