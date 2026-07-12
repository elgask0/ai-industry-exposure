"""
Build AI Exposure Measures for FF49 Portfolios

This script constructs two types of exposure measures:
1. AIIE Simple: Felten's AI Industry Exposure mapped directly from NAICS to FF49
2. ExpWB: AI Industry Exposure weighted by occupation wage bill shares
   Formula: ExpWB_i = Σ(AIOE_o × (wagebill_io / Σ_o wagebill_io))
   This is an adimensional intensity index, NOT a dollar amount

Each measure has:
- 2021 and 2023 versions (different AIOE/AIIE data)
- EW (equal-weighted) and VW (value-added weighted) aggregations
- Raw and z-score transformations

Output: data/processed/exposure/ff49_exposure_measures.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_project_root():
    return Path(__file__).parent.parent


def standardize(x):
    """Standardize to z-score (mean=0, std=1)."""
    if x.std() == 0 or x.isna().all():
        return x * 0
    return (x - x.mean()) / x.std()


def load_sic_to_ff49(root):
    """Load SIC to FF49 mapping."""
    df = pd.read_csv(root / 'data' / 'processed' / 'french' / 'industry_definitions' / 'sic_to_ff49.csv')
    return df


def load_naics_to_sic_crosswalk(root):
    """Load NAICS to SIC crosswalk and create reverse mapping."""
    # This crosswalk goes SIC -> NAICS, we need the reverse
    df = pd.read_csv(root / 'data' / 'processed' / 'crosswalks' / 'sic87_to_naics02.csv')

    # Create NAICS -> SIC mapping (many-to-many)
    df['naics_4'] = df['naics_2002_code'].astype(str).str[:4]
    df['sic_4'] = df['sic_code'].astype(str).str.zfill(4)[:4]

    return df[['naics_4', 'sic_4', 'sic_code', 'naics_2002_code']].drop_duplicates()


def map_sic_to_ff49(sic_code, sic_to_ff49_df):
    """Map a SIC code to FF49 portfolio."""
    try:
        sic = int(sic_code)
    except (ValueError, TypeError):
        return None

    for _, row in sic_to_ff49_df.iterrows():
        if row['sic_start'] <= sic <= row['sic_end']:
            return int(row['ff49_code'])
    return None


def create_naics_to_ff49_mapping(root):
    """Create NAICS 4-digit to FF49 mapping using crosswalks."""
    print("\n" + "="*60)
    print("Creating NAICS to FF49 Mapping")
    print("="*60)

    sic_to_ff49 = load_sic_to_ff49(root)
    naics_sic = load_naics_to_sic_crosswalk(root)

    # Map each NAICS-SIC pair to FF49
    naics_sic['ff49'] = naics_sic['sic_code'].apply(lambda x: map_sic_to_ff49(x, sic_to_ff49))

    # For each NAICS 4-digit, find the modal FF49
    naics_ff49_counts = naics_sic.groupby(['naics_4', 'ff49']).size().reset_index(name='count')

    # Get the most common FF49 for each NAICS
    idx = naics_ff49_counts.groupby('naics_4')['count'].idxmax()
    naics_to_ff49 = naics_ff49_counts.loc[idx, ['naics_4', 'ff49']].copy()
    naics_to_ff49 = naics_to_ff49.dropna()
    naics_to_ff49['ff49'] = naics_to_ff49['ff49'].astype(int)

    # Add FF49 names
    ff49_names = sic_to_ff49[['ff49_code', 'ff49_abbrev', 'ff49_name']].drop_duplicates()
    ff49_names.columns = ['ff49', 'ff49_abbrev', 'ff49_name']
    naics_to_ff49 = naics_to_ff49.merge(ff49_names, on='ff49', how='left')

    print(f"  Mapped {len(naics_to_ff49)} NAICS-4 codes to FF49 portfolios")
    print(f"  FF49 coverage: {naics_to_ff49['ff49'].nunique()} portfolios")

    return naics_to_ff49


def load_aiie_data(root):
    """Load AIIE data (2021 and LM/2023 versions)."""
    print("\n" + "="*60)
    print("Loading AIIE Data")
    print("="*60)

    # AIIE 2021 (4-digit NAICS)
    aiie_2021 = pd.read_csv(root / 'data' / 'processed' / 'felten' / 'aioe_2021' / 'aiie_by_industry.csv')
    aiie_2021['naics_4'] = aiie_2021['naics_code'].astype(str).str[:4]
    aiie_2021 = aiie_2021.rename(columns={'aiie': 'aiie_21'})
    print(f"  AIIE 2021: {len(aiie_2021)} industries")

    # LM AIIE (mixed NAICS levels)
    lm_aiie = pd.read_csv(root / 'data' / 'processed' / 'felten' / 'language_modeling' / 'lm_aiie_by_industry.csv')
    lm_aiie['naics_4'] = lm_aiie['naics_code'].astype(str).str[:4]
    lm_aiie = lm_aiie.rename(columns={'lm_aiie': 'aiie_23'})
    print(f"  LM AIIE (2023): {len(lm_aiie)} industries")

    # Average within NAICS-4
    aiie_2021_4 = aiie_2021.groupby('naics_4')['aiie_21'].mean().reset_index()
    lm_aiie_4 = lm_aiie.groupby('naics_4')['aiie_23'].mean().reset_index()

    # Merge
    aiie_combined = aiie_2021_4.merge(lm_aiie_4, on='naics_4', how='outer')
    print(f"  Combined: {len(aiie_combined)} unique NAICS-4 codes")

    return aiie_combined


def load_aioe_data(root):
    """Load AIOE data (occupation-level) for ExpWB construction."""
    print("\n" + "="*60)
    print("Loading AIOE Data")
    print("="*60)

    # AIOE 2021
    aioe_2021 = pd.read_csv(root / 'data' / 'processed' / 'felten' / 'aioe_2021' / 'aioe_by_occupation.csv')
    aioe_2021 = aioe_2021.rename(columns={'aioe': 'aioe_21'})
    print(f"  AIOE 2021: {len(aioe_2021)} occupations")

    # LM AIOE
    lm_aioe = pd.read_csv(root / 'data' / 'processed' / 'felten' / 'language_modeling' / 'lm_aioe_by_occupation.csv')
    lm_aioe = lm_aioe.rename(columns={'lm_aioe': 'aioe_23'})
    print(f"  LM AIOE (2023): {len(lm_aioe)} occupations")

    # Merge
    aioe_combined = aioe_2021[['soc_code', 'aioe_21']].merge(
        lm_aioe[['soc_code', 'aioe_23']], on='soc_code', how='outer'
    )
    print(f"  Combined: {len(aioe_combined)} unique SOC codes")

    return aioe_combined


def load_oews_data(root):
    """Load OEWS wage and employment data."""
    print("\n" + "="*60)
    print("Loading OEWS Data")
    print("="*60)

    oews = pd.read_csv(root / 'data' / 'processed' / 'bls' / 'oews_may2021_nat4d.csv')

    # Keep only detailed occupations (6-digit SOC)
    oews = oews[oews['O_GROUP'] == 'detailed'].copy()

    # Clean SOC codes
    oews['soc_code'] = oews['OCC_CODE'].str.strip()

    # Clean NAICS codes - extract 4-digit
    oews['naics_4'] = oews['NAICS'].astype(str).str[:4]

    # Clean employment and wages
    oews['emp'] = pd.to_numeric(oews['TOT_EMP'].replace(['**', '*', '#'], np.nan), errors='coerce')
    oews['wage'] = pd.to_numeric(oews['A_MEAN'].replace(['**', '*', '#'], np.nan), errors='coerce')

    # Filter to valid records
    oews = oews[oews['emp'].notna() & oews['wage'].notna() & (oews['emp'] > 0)]

    print(f"  OEWS records: {len(oews)}")
    print(f"  Unique NAICS-4: {oews['naics_4'].nunique()}")
    print(f"  Unique SOC: {oews['soc_code'].nunique()}")

    return oews[['naics_4', 'soc_code', 'emp', 'wage']]


def load_value_added(root):
    """Load BEA value added data."""
    print("\n" + "="*60)
    print("Loading BEA Value Added Data")
    print("="*60)

    va = pd.read_csv(root / 'data' / 'processed' / 'bea' / 'gdp_by_industry_va_2021.csv')

    # BEA data has INDUSTRY column with codes like "211", "212", "311FT", etc.
    # DATAVALUE is in billions
    va = va.rename(columns={'INDUSTRY': 'naics', 'DATAVALUE': 'value_added'})

    # Convert value_added to actual dollars (from billions)
    va['value_added'] = va['value_added'] * 1e9

    # Clean NAICS codes - extract numeric prefix
    # BEA codes like "211", "311FT", "3364OT" -> extract leading digits
    va['naics_clean'] = va['naics'].astype(str).str.extract(r'^(\d+)')[0]
    va = va.dropna(subset=['naics_clean'])

    # Take first 3 digits to match with NAICS-4 at 3-digit level
    # This is a rough match - BEA uses industry codes, not pure NAICS
    va['naics_4'] = va['naics_clean'].str[:3]

    # Filter to detailed industry codes (avoid aggregates)
    # Keep codes that are 2-3 digits (sector level)
    va = va[va['naics_clean'].str.len() <= 3].copy()

    # Aggregate to first 3 digits (since BEA is at 2-3 digit level)
    va_3 = va.groupby('naics_4')['value_added'].sum().reset_index()

    print(f"  BEA industries: {len(va)}")
    print(f"  Unique NAICS prefixes: {len(va_3)}")
    print(f"  Total VA: ${va_3['value_added'].sum()/1e12:.2f}T")

    return va_3


def build_expwb(oews, aioe):
    """
    Build AIIE (AI Industry Exposure) at NAICS-4 level.

    Following the methodology from Felten et al.:
    AIIE_i = Σ(AIOE_o × (wagebill_io / Σ_o wagebill_io))

    This gives an adimensional intensity index, NOT a dollar amount.
    """
    print("\n" + "="*60)
    print("Building AIIE (AI Industry Exposure)")
    print("="*60)

    # Merge OEWS with AIOE
    merged = oews.merge(aioe, on='soc_code', how='left')

    # Check merge quality
    n_matched = merged['aioe_21'].notna().sum()
    print(f"  OEWS-AIOE match rate: {n_matched}/{len(merged)} ({100*n_matched/len(merged):.1f}%)")

    # Calculate wage bill for each occupation-industry cell
    merged['wagebill'] = merged['emp'] * merged['wage']

    # Compute total wage bill for each NAICS-4 industry
    industry_wagebill = merged.groupby('naics_4')['wagebill'].transform('sum')

    # Compute wage bill share for each occupation within its industry
    merged['wagebill_share'] = merged['wagebill'] / industry_wagebill

    # Calculate ExpWB as AIOE weighted by wage bill share
    # This gives an adimensional intensity index (NOT a dollar amount)
    merged['expwb_21'] = merged['aioe_21'] * merged['wagebill_share']
    merged['expwb_23'] = merged['aioe_23'] * merged['wagebill_share']

    # Aggregate to NAICS-4 level
    # Sum the weighted contributions across occupations
    expwb = merged.groupby('naics_4').agg({
        'emp': 'sum',
        'wagebill': 'sum',
        'expwb_21': 'sum',
        'expwb_23': 'sum'
    }).reset_index()

    print(f"  ExpWB computed for {len(expwb)} NAICS-4 industries")
    print(f"  Total wage bill: ${expwb['wagebill'].sum()/1e12:.2f}T")
    print(f"  ExpWB_21 range: [{expwb['expwb_21'].min():.4f}, {expwb['expwb_21'].max():.4f}]")

    return expwb


def aggregate_to_ff49(naics_measures, naics_to_ff49, va_data):
    """
    Aggregate NAICS-level measures to FF49 portfolios.

    EW: Simple mean across NAICS in portfolio
    VW: VA-weighted mean across NAICS in portfolio
    """
    print("\n" + "="*60)
    print("Aggregating to FF49 Portfolios")
    print("="*60)

    # Merge NAICS measures with FF49 mapping
    merged = naics_measures.merge(naics_to_ff49[['naics_4', 'ff49', 'ff49_abbrev', 'ff49_name']],
                                   on='naics_4', how='inner')

    # For VA merge, match on first 2-3 digits (BEA level)
    merged['naics_3'] = merged['naics_4'].str[:3]
    merged['naics_2'] = merged['naics_4'].str[:2]

    # Try to merge VA at 3-digit level first, then 2-digit
    va_data = va_data.rename(columns={'naics_4': 'naics_prefix'})

    # Merge at 3-digit level
    merged = merged.merge(va_data, left_on='naics_3', right_on='naics_prefix', how='left')
    merged = merged.drop(columns=['naics_prefix'], errors='ignore')

    # For missing, try 2-digit level
    va_2 = va_data.copy()
    va_2['naics_prefix'] = va_2['naics_prefix'].str[:2]
    va_2 = va_2.groupby('naics_prefix')['value_added'].sum().reset_index()
    missing_va = merged['value_added'].isna()
    if missing_va.any():
        merged_missing = merged[missing_va].drop(columns=['value_added'])
        merged_missing = merged_missing.merge(va_2, left_on='naics_2', right_on='naics_prefix', how='left')
        merged_missing = merged_missing.drop(columns=['naics_prefix'], errors='ignore')
        merged.loc[missing_va, 'value_added'] = merged_missing['value_added'].values

    merged['value_added'] = merged['value_added'].fillna(0)

    print(f"  NAICS matched to FF49: {len(merged)}")
    print(f"  FF49 portfolios covered: {merged['ff49'].nunique()}")

    # Identify measure columns
    # aiie_21, aiie_23: direct Felten industry exposure (AIIE Simple)
    # expwb_21, expwb_23: occupation-weighted by wage bill share (ExpWB)
    measure_cols = [c for c in merged.columns if c.startswith(('aiie_', 'expwb_'))]

    results = []
    for ff49, group in merged.groupby('ff49'):
        row = {
            'ff49': ff49,
            'ff49_abbrev': group['ff49_abbrev'].iloc[0],
            'ff49_name': group['ff49_name'].iloc[0],
            'n_naics': len(group),
            'portfolio_va': group['value_added'].sum()
        }

        for col in measure_cols:
            valid = group[group[col].notna()]
            if len(valid) > 0:
                # EW: Simple mean
                row[f'{col}_ew'] = valid[col].mean()

                # VW: VA-weighted mean
                if valid['value_added'].sum() > 0:
                    row[f'{col}_vw'] = np.average(valid[col], weights=valid['value_added'])
                else:
                    row[f'{col}_vw'] = valid[col].mean()  # Fallback to EW
            else:
                row[f'{col}_ew'] = np.nan
                row[f'{col}_vw'] = np.nan

        results.append(row)

    ff49_measures = pd.DataFrame(results)

    # Create z-score versions
    for col in [c for c in ff49_measures.columns if c.endswith(('_ew', '_vw'))]:
        ff49_measures[f'{col}_z'] = standardize(ff49_measures[col])

    print(f"  Final FF49 measures: {len(ff49_measures)} portfolios")

    return ff49_measures


def create_clean_output(ff49_measures):
    """Create clean output with standardized column names."""
    print("\n" + "="*60)
    print("Creating Clean Output")
    print("="*60)

    # Select and rename columns for final output
    output = pd.DataFrame()
    output['ff49'] = ff49_measures['ff49']
    output['ff49_abbrev'] = ff49_measures['ff49_abbrev']
    output['ff49_name'] = ff49_measures['ff49_name']
    output['portfolio_va'] = ff49_measures['portfolio_va']
    output['n_naics'] = ff49_measures['n_naics']

    # AIIE Simple measures (direct industry exposure from Felten)
    for year in ['21', '23']:
        for agg in ['ew', 'vw']:
            col_raw = f'aiie_{year}_{agg}'
            col_z = f'aiie_{year}_{agg}_z'
            if col_raw in ff49_measures.columns:
                output[f'aiie_{year}_{agg}'] = ff49_measures[col_raw]
                output[f'aiie_{year}_{agg}_z'] = ff49_measures[col_z]

    # ExpWB measures (occupation-weighted by wage bill share)
    # This is ExpWB - an adimensional intensity index, NOT a dollar amount
    for year in ['21', '23']:
        for agg in ['ew', 'vw']:
            col_raw = f'expwb_{year}_{agg}'
            col_z = f'expwb_{year}_{agg}_z'
            if col_raw in ff49_measures.columns:
                output[f'expwb_{year}_{agg}'] = ff49_measures[col_raw]
                output[f'expwb_{year}_{agg}_z'] = ff49_measures[col_z]

    # Sort by FF49 number
    output = output.sort_values('ff49').reset_index(drop=True)

    print(f"  Output columns: {list(output.columns)}")
    print(f"  Output rows: {len(output)}")

    return output


def save_outputs(output, naics_measures, naics_to_ff49, root):
    """Save all output files."""
    print("\n" + "="*60)
    print("Saving Output Files")
    print("="*60)

    output_dir = root / 'data' / 'processed' / 'exposure'

    # Main output
    output_path = output_dir / 'ff49_exposure_measures.csv'
    output.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Intermediate: NAICS-level measures
    naics_path = output_dir / 'naics_measures.csv'
    naics_measures.to_csv(naics_path, index=False)
    print(f"  Saved: {naics_path}")

    # Intermediate: NAICS to FF49 mapping
    mapping_path = output_dir / 'naics_to_ff49_mapping.csv'
    naics_to_ff49.to_csv(mapping_path, index=False)
    print(f"  Saved: {mapping_path}")


def print_summary(output):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print("\nPortfolio Coverage:")
    print(f"  Total portfolios: {len(output)}")
    print(f"  Portfolios with AIIE 2021: {output['aiie_21_ew'].notna().sum()}")
    print(f"  Portfolios with AIIE 2023: {output['aiie_23_ew'].notna().sum()}")
    print(f"  Portfolios with ExpWB 2021: {output['expwb_21_ew'].notna().sum()}")
    print(f"  Portfolios with ExpWB 2023: {output['expwb_23_ew'].notna().sum()}")

    print("\nTop 5 by AIIE 2021 (EW, z-score):")
    top5 = output.nlargest(5, 'aiie_21_ew_z')[['ff49_abbrev', 'aiie_21_ew_z', 'expwb_21_ew_z']]
    for _, row in top5.iterrows():
        print(f"  {row['ff49_abbrev']:8s}: AIIE_z={row['aiie_21_ew_z']:6.2f}, ExpWB_z={row['expwb_21_ew_z']:6.2f}")

    print("\nTop 5 by ExpWB 2021 (EW, z-score):")
    top5 = output.nlargest(5, 'expwb_21_ew_z')[['ff49_abbrev', 'aiie_21_ew_z', 'expwb_21_ew_z']]
    for _, row in top5.iterrows():
        print(f"  {row['ff49_abbrev']:8s}: AIIE_z={row['aiie_21_ew_z']:6.2f}, ExpWB_z={row['expwb_21_ew_z']:6.2f}")

    print("\nCorrelation: AIIE vs ExpWB (2021 EW z-scores):")
    corr = output['aiie_21_ew_z'].corr(output['expwb_21_ew_z'])
    print(f"  Pearson r = {corr:.3f}")


def main():
    print("="*60)
    print("BUILDING AI EXPOSURE MEASURES FOR FF49 PORTFOLIOS")
    print("="*60)

    root = get_project_root()

    # Step 1: Create NAICS to FF49 mapping
    naics_to_ff49 = create_naics_to_ff49_mapping(root)

    # Step 2: Load all input data
    aiie_data = load_aiie_data(root)
    aioe_data = load_aioe_data(root)
    oews_data = load_oews_data(root)
    va_data = load_value_added(root)

    # Step 3: Build ExpWB (wage bill weighted) at NAICS level
    expwb_data = build_expwb(oews_data, aioe_data)

    # Step 4: Combine all NAICS-level measures
    # aiie_data has: aiie_21, aiie_23 (direct Felten industry exposure)
    # expwb_data has: expwb_21, expwb_23 (occupation-weighted by wage bill share)
    naics_measures = aiie_data.merge(expwb_data, on='naics_4', how='outer')
    print(f"\nCombined NAICS measures: {len(naics_measures)} industries")

    # Step 5: Aggregate to FF49 portfolios
    ff49_measures = aggregate_to_ff49(naics_measures, naics_to_ff49, va_data)

    # Step 6: Create clean output
    output = create_clean_output(ff49_measures)

    # Step 7: Save outputs
    save_outputs(output, naics_measures, naics_to_ff49, root)

    # Step 8: Print summary
    print_summary(output)

    print("\n" + "="*60)
    print("EXPOSURE MEASURES COMPLETE")
    print("="*60)

    return output


if __name__ == "__main__":
    output = main()
