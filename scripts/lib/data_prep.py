"""
Data loading, winsorization, and binary treatment construction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mstats


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_panel() -> pd.DataFrame:
    """Load the final dataset and parse dates."""
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'final_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['ym_str'] = df['date'].dt.to_period('M').astype(str)
    return df


def winsorize_returns(
    df: pd.DataFrame, pct: float, ret_cols: list,
) -> pd.DataFrame:
    """Winsorize return columns at pct / (1-pct) levels (cross-sectional, per month)."""
    if pct == 0.0:
        return df
    result = df.copy()
    for col in ret_cols:
        result[col] = mstats.winsorize(result[col].values, limits=[pct, pct])
    return result


def create_sample_window(df: pd.DataFrame, pre_months: int) -> pd.DataFrame:
    """Filter to [event - pre_months, Feb 2023]."""
    event_date = pd.Timestamp('2022-12-01')
    start_date = event_date - pd.DateOffset(months=pre_months)
    end_date = pd.Timestamp('2023-02-01')
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    return df.loc[mask].copy()


def create_binary_treatment(
    df: pd.DataFrame,
    exposure_col: str,
    treatment_type: str,
) -> pd.DataFrame:
    """
    Create binary treatment from continuous exposure.

    Ranks portfolios by exposure_col, keeps top and bottom groups,
    drops middle portfolios, and adds a 'treated' column.
    """
    # One exposure value per portfolio (time-invariant)
    port_exp = df.groupby('ff49')[exposure_col].first().dropna()

    if treatment_type == 'binary_top10_bot10':
        top = set(port_exp.nlargest(10).index)
        bot = set(port_exp.nsmallest(10).index)
    elif treatment_type == 'binary_q1_q4':
        q25 = port_exp.quantile(0.25)
        q75 = port_exp.quantile(0.75)
        top = set(port_exp[port_exp >= q75].index)
        bot = set(port_exp[port_exp <= q25].index)
    elif treatment_type == 'binary_t1_t3':
        t33 = port_exp.quantile(1 / 3)
        t67 = port_exp.quantile(2 / 3)
        top = set(port_exp[port_exp >= t67].index)
        bot = set(port_exp[port_exp <= t33].index)
    else:
        raise ValueError(f"Unknown treatment type: {treatment_type}")

    keep = top | bot
    result = df[df['ff49'].isin(keep)].copy()
    result['treated'] = result['ff49'].isin(top).astype(float)
    return result
