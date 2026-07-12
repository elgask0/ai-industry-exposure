"""
Build Final Dataset for Regressions

Merges:
- FF49 portfolio returns (VW and EW)
- Fama-French factors (FF3, FF5, MOM)
- AI exposure measures (AIIE and ExpWB)
- Event time variables

Output: data/processed/final_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def load_returns(root):
    """Load FF49 portfolio returns."""
    print("\n" + "="*60)
    print("Loading FF49 Returns")
    print("="*60)

    # Value-weighted returns
    vw = pd.read_csv(root / 'data' / 'processed' / 'french' / '49_industry_portfolios_monthly' / 'value_weighted_returns.csv')
    vw['date'] = pd.to_datetime(vw['date'])

    # Equal-weighted returns
    ew = pd.read_csv(root / 'data' / 'processed' / 'french' / '49_industry_portfolios_monthly' / 'equal_weighted_returns.csv')
    ew['date'] = pd.to_datetime(ew['date'])

    # Get portfolio columns (all except date)
    port_cols = [c for c in vw.columns if c != 'date']

    # Melt to long format
    vw_long = vw.melt(id_vars='date', var_name='ff49_abbrev', value_name='ret_vw')
    ew_long = ew.melt(id_vars='date', var_name='ff49_abbrev', value_name='ret_ew')

    # Merge
    returns = vw_long.merge(ew_long, on=['date', 'ff49_abbrev'])

    print(f"  VW returns: {len(vw)} months × {len(port_cols)} portfolios")
    print(f"  EW returns: {len(ew)} months × {len(port_cols)} portfolios")
    print(f"  Long format: {len(returns)} observations")

    return returns


def load_factors(root):
    """Load Fama-French factors."""
    print("\n" + "="*60)
    print("Loading Fama-French Factors")
    print("="*60)

    # FF5 factors
    ff5 = pd.read_csv(root / 'data' / 'processed' / 'french' / 'ff5_factors_monthly' / 'monthly_factors.csv')
    ff5['date'] = pd.to_datetime(ff5['date'])

    # Momentum
    mom = pd.read_csv(root / 'data' / 'processed' / 'french' / 'momentum_monthly' / 'monthly_factors.csv')
    mom['date'] = pd.to_datetime(mom['date'])

    # Merge
    factors = ff5.merge(mom[['date', 'Mom']], on='date', how='left')

    # Standardize column names
    factors = factors.rename(columns={
        'Mkt-RF': 'mktrf',
        'SMB': 'smb',
        'HML': 'hml',
        'RMW': 'rmw',
        'CMA': 'cma',
        'RF': 'rf',
        'Mom': 'mom'
    })

    print(f"  Factors loaded: {len(factors)} months")
    print(f"  Columns: {list(factors.columns)}")

    return factors


def load_exposures(root):
    """Load AI exposure measures."""
    print("\n" + "="*60)
    print("Loading Exposure Measures")
    print("="*60)

    exp = pd.read_csv(root / 'data' / 'processed' / 'exposure' / 'ff49_exposure_measures.csv')

    print(f"  Portfolios: {len(exp)}")
    print(f"  Exposure columns: {[c for c in exp.columns if 'aiie' in c or 'expwb' in c]}")

    return exp


def create_event_variables(panel):
    """Create event time and post indicators."""
    print("\n" + "="*60)
    print("Creating Event Variables")
    print("="*60)

    # Event date: ChatGPT released Nov 30, 2022 -> Event month is Dec 2022
    event_date = pd.Timestamp('2022-12-01')

    # Event time: months relative to Dec 2022
    panel['event_time'] = ((panel['date'].dt.year - event_date.year) * 12 +
                           (panel['date'].dt.month - event_date.month))

    # Post indicators
    panel['post1'] = (panel['date'] == '2022-12-01').astype(int)  # Dec 2022 only
    panel['post2'] = (panel['date'].isin(['2022-12-01', '2023-01-01'])).astype(int)  # Dec 2022 - Jan 2023

    # Sample indicators
    # 60-month pre: Dec 2017 to Nov 2022 (60 months), plus Dec 2022 - Feb 2023
    panel['sample_60m'] = (
        (panel['date'] >= '2017-12-01') &
        (panel['date'] <= '2023-02-01')
    ).astype(int)

    # 120-month pre: Dec 2012 to Nov 2022 (120 months), plus Dec 2022 - Feb 2023
    panel['sample_120m'] = (
        (panel['date'] >= '2012-12-01') &
        (panel['date'] <= '2023-02-01')
    ).astype(int)

    # Year and month for Stata
    panel['year'] = panel['date'].dt.year
    panel['month'] = panel['date'].dt.month
    panel['ym'] = panel['year'] * 100 + panel['month']

    print(f"  Event date: {event_date.date()}")
    print(f"  post1 observations: {panel['post1'].sum()}")
    print(f"  post2 observations: {panel['post2'].sum()}")
    print(f"  sample_60m observations: {panel['sample_60m'].sum()}")
    print(f"  sample_120m observations: {panel['sample_120m'].sum()}")

    return panel


def create_excess_returns(panel):
    """Create excess returns."""
    panel['ret_vw_ex'] = panel['ret_vw'] - panel['rf']
    panel['ret_ew_ex'] = panel['ret_ew'] - panel['rf']
    return panel


def create_interactions(panel):
    """Create interaction terms for baseline regressions."""
    print("\n" + "="*60)
    print("Creating Interaction Terms")
    print("="*60)

    # Main exposure columns for interactions
    exp_cols = ['aiie_21_ew_z', 'aiie_21_vw_z', 'aiie_23_ew_z', 'aiie_23_vw_z',
                'expwb_21_ew_z', 'expwb_21_vw_z', 'expwb_23_ew_z', 'expwb_23_vw_z']

    for exp_col in exp_cols:
        if exp_col in panel.columns:
            # Post1 interactions
            panel[f'post1_{exp_col}'] = panel['post1'] * panel[exp_col]
            # Post2 interactions
            panel[f'post2_{exp_col}'] = panel['post2'] * panel[exp_col]

    n_interactions = len([c for c in panel.columns if c.startswith('post')])
    print(f"  Created {n_interactions} interaction terms")

    return panel


def create_event_dummies(panel):
    """Create event time dummies for pre-trends analysis."""
    print("\n" + "="*60)
    print("Creating Event Time Dummies")
    print("="*60)

    # Create dummies for τ = -12 to τ = +2 (excluding τ = -1 as reference)
    for tau in range(-12, 3):
        if tau != -1:  # τ = -1 is reference
            if tau < 0:
                col_name = f'D_m{abs(tau)}'
            else:
                col_name = f'D_{tau}'
            panel[col_name] = (panel['event_time'] == tau).astype(int)

    n_dummies = len([c for c in panel.columns if c.startswith('D_')])
    print(f"  Created {n_dummies} event time dummies (τ=-12 to τ=+2, excl τ=-1)")

    return panel


def order_columns(panel):
    """Order columns logically."""
    # Define column order
    id_cols = ['ff49', 'ff49_abbrev', 'date', 'year', 'month', 'ym']
    return_cols = ['ret_vw', 'ret_ew', 'rf', 'ret_vw_ex', 'ret_ew_ex']
    factor_cols = ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'mom']
    exp_cols = [c for c in panel.columns if 'aiie' in c or 'expwb' in c]
    exp_cols = [c for c in exp_cols if not c.startswith('post')]
    event_cols = ['event_time', 'post1', 'post2', 'sample_60m', 'sample_120m']
    dummy_cols = sorted([c for c in panel.columns if c.startswith('D_')])
    interaction_cols = sorted([c for c in panel.columns if c.startswith('post1_') or c.startswith('post2_')])
    other_cols = ['portfolio_va', 'n_naics', 'ff49_name']

    # Get all columns in order
    ordered = []
    for col_list in [id_cols, return_cols, factor_cols, exp_cols, event_cols, dummy_cols, interaction_cols, other_cols]:
        ordered.extend([c for c in col_list if c in panel.columns])

    # Add any remaining columns
    remaining = [c for c in panel.columns if c not in ordered]
    ordered.extend(remaining)

    return panel[ordered]


def main():
    print("="*60)
    print("BUILDING FINAL DATASET FOR REGRESSIONS")
    print("="*60)

    root = get_project_root()

    # Step 1: Load returns
    returns = load_returns(root)

    # Step 2: Load factors
    factors = load_factors(root)

    # Step 3: Load exposures
    exposures = load_exposures(root)

    # Step 4: Merge returns with factors
    print("\n" + "="*60)
    print("Merging Data")
    print("="*60)

    panel = returns.merge(factors, on='date', how='left')
    print(f"  After factor merge: {len(panel)} observations")

    # Step 5: Merge with exposures
    panel = panel.merge(exposures, on='ff49_abbrev', how='left')
    print(f"  After exposure merge: {len(panel)} observations")
    print(f"  Portfolios with exposures: {panel['aiie_21_ew_z'].notna().sum() // panel['date'].nunique()}")

    # Step 6: Create event variables
    panel = create_event_variables(panel)

    # Step 7: Create excess returns
    panel = create_excess_returns(panel)

    # Step 8: Create interactions
    panel = create_interactions(panel)

    # Step 9: Create event time dummies
    panel = create_event_dummies(panel)

    # Step 10: Order columns
    panel = order_columns(panel)

    # Step 11: Save
    print("\n" + "="*60)
    print("Saving Final Dataset")
    print("="*60)

    output_path = root / 'data' / 'processed' / 'final_dataset.csv'
    panel.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(f"\nDimensions:")
    print(f"  Observations: {len(panel):,}")
    print(f"  Variables: {len(panel.columns)}")
    print(f"  Portfolios: {panel['ff49_abbrev'].nunique()}")
    print(f"  Months: {panel['date'].nunique()}")
    print(f"  Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")

    print(f"\nSample sizes:")
    print(f"  sample_60m: {panel['sample_60m'].sum():,}")
    print(f"  sample_120m: {panel['sample_120m'].sum():,}")

    print(f"\nEvent indicators:")
    print(f"  post1 (Dec 2022): {panel['post1'].sum()}")
    print(f"  post2 (Dec 2022 + Jan 2023): {panel['post2'].sum()}")

    print(f"\nExposure coverage:")
    for col in ['aiie_21_ew_z', 'expwb_21_ew_z']:
        valid = panel[col].notna().sum()
        print(f"  {col}: {valid:,} ({100*valid/len(panel):.1f}%)")

    print("\n" + "="*60)
    print("FINAL DATASET COMPLETE")
    print("="*60)

    return panel


if __name__ == "__main__":
    panel = main()
