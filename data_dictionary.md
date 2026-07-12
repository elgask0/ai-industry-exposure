# Data Dictionary

**Project**: chatgpt-shock
**Last Updated**: 2026-02-07

Complete dictionary of all processed data files used in the ChatGPT shock event study analysis.

---

## Table of Contents

1. [Final Dataset](#final-dataset)
2. [Exposure Measures](#exposure-measures)
3. [French Data Library](#french-data-library)
4. [Felten AI Exposure Data](#felten-ai-exposure-data)
5. [BLS Data](#bls-data)
6. [BEA Data](#bea-data)
7. [Crosswalk Files](#crosswalk-files)

---

## Final Dataset

### `data/processed/final_dataset.csv`

**Purpose**: Main panel dataset for all regression analyses

**Dimensions**: 58,457 observations (43 portfolios with exposure × 1,361 months)

**Key Variables**:

#### Identifiers
| Variable | Type | Description |
|----------|------|-------------|
| ff49 | int | Portfolio number (1-49) |
| ff49_abbrev | str | Portfolio abbreviation (e.g., "Agric", "Softw") |
| ff49_name | str | Portfolio full name |
| date | date | Month-end date (YYYY-MM-DD) |
| year | int | Year |
| month | int | Month (1-12) |
| ym | int | Year-month integer (YYYYMM) |
| ym_str | period | Year-month period for FE |

#### Returns
All returns are in percentage points (e.g., 2.5 = 2.5%).

| Variable | Type | Description |
|----------|------|-------------|
| ret_vw | float | Value-weighted portfolio return |
| ret_ew | float | Equal-weighted portfolio return |
| rf | float | Risk-free rate (1-month T-bill) |
| ret_vw_ex | float | VW excess return (ret_vw - rf) |
| ret_ew_ex | float | EW excess return (ret_ew - rf) |

#### Fama-French Factors
All factors are in percentage points.

| Variable | Type | Description |
|----------|------|-------------|
| mktrf | float | Market excess return (Mkt-RF) |
| smb | float | Size factor (Small Minus Big) |
| hml | float | Value factor (High Minus Low) |
| rmw | float | Profitability factor (Robust Minus Weak) |
| cma | float | Investment factor (Conservative Minus Aggressive) |
| mom | float | Momentum factor |

#### Exposure Measures

**AIIE (AI Industry Exposure)** - Felten's AI Industry Exposure, direct mapping from NAICS to FF49

| Variable | Type | Description |
|----------|------|-------------|
| aiie_21_ew | float | AIIE 2021, EW aggregation (raw) |
| aiie_21_ew_z | float | AIIE 2021, EW (z-score, mean=0, std=1) |
| aiie_21_vw | float | AIIE 2021, VW aggregation (raw) |
| aiie_21_vw_z | float | AIIE 2021, VW (z-score) |
| aiie_23_ew | float | AIIE 2023 (LM), EW (raw) |
| aiie_23_ew_z | float | AIIE 2023 (LM), EW (z-score) |
| aiie_23_vw | float | AIIE 2023 (LM), VW (raw) |
| aiie_23_vw_z | float | AIIE 2023 (LM), VW (z-score) |

**ExpWB (Exposure-Weighted by Wage Bill)** - Occupation-weighted industry exposure

Formula: `ExpWB_i = Σ(AIOE_o × (wagebill_io / Σ_o wagebill_io))`

This is an adimensional intensity index, NOT a dollar amount.

| Variable | Type | Description |
|----------|------|-------------|
| expwb_21_ew | float | ExpWB 2021, EW aggregation (raw) |
| expwb_21_ew_z | float | ExpWB 2021, EW (z-score) |
| expwb_21_vw | float | ExpWB 2021, VW aggregation (raw) |
| expwb_21_vw_z | float | ExpWB 2021, VW (z-score) |
| expwb_23_ew | float | ExpWB 2023 (LM), EW (raw) |
| expwb_23_ew_z | float | ExpWB 2023 (LM), EW (z-score) |
| expwb_23_vw | float | ExpWB 2023 (LM), VW (raw) |
| expwb_23_vw_z | float | ExpWB 2023 (LM), VW (z-score) |

#### Event Variables
| Variable | Type | Description |
|----------|------|-------------|
| event_time | int | Months relative to Dec 2022 (τ=0 for Dec 2022, τ=-1 for Nov 2022) |
| post1 | int | 1 if Dec 2022 only, 0 otherwise |
| post2 | int | 1 if Dec 2022 or Jan 2023, 0 otherwise |

#### Sample Indicators
| Variable | Type | Description |
|----------|------|-------------|
| sample_60m | int | 1 if Dec 2017 to Feb 2023 (60-month window) |
| sample_120m | int | 1 if Dec 2012 to Feb 2023 (120-month window) |

#### Event Time Dummies
For pre-trends/dynamic event study. τ = -1 is reference (omitted).

| Variable | Description |
|----------|-------------|
| D_m12 to D_m2 | 1 if τ = -12 to -2 (pre-event months) |
| D_0 | 1 if τ = 0 (Dec 2022, event month) |
| D_1, D_2 | 1 if τ = +1, +2 (Jan 2023, Feb 2023) |

#### Interaction Terms
Pre-computed for baseline regressions.

| Variable | Description |
|----------|-------------|
| post1_aiie_21_ew_z | post1 × aiie_21_ew_z |
| post1_aiie_21_vw_z | post1 × aiie_21_vw_z |
| post1_expwb_21_ew_z | post1 × expwb_21_ew_z |
| post1_expwb_21_vw_z | post1 × expwb_21_vw_z |
| post2_* | Same as post1_* but for post2 |

#### Other Variables
| Variable | Type | Description |
|----------|------|-------------|
| portfolio_va | float | Value added allocated to portfolio (USD) |
| n_naics | int | Number of NAICS industries mapped to portfolio |
| ff49_name | str | Portfolio full name |

---

## Exposure Measures

### `data/processed/exposure/ff49_exposure_measures.csv`

**Purpose**: Cross-sectional exposure measures at FF49 portfolio level

**Dimensions**: 43 portfolios (6 portfolios missing: Chems, FabPr, Ships, Guns, Gold, Boxes)

**Columns**:
- Portfolio identifiers: ff49, ff49_abbrev, ff49_name
- Portfolio characteristics: portfolio_va, n_naics
- AIIE measures: aiie_21_ew, aiie_21_ew_z, aiie_21_vw, aiie_21_vw_z, aiie_23_*, aiie_23_*
- ExpWB measures: expwb_21_ew, expwb_21_ew_z, expwb_21_vw, expwb_21_vw_z, expwb_23_*, expwb_23_*

---

## IPP Intensity Data

### `data/processed/ipp/panelA_ipp_ff49.csv`

**Purpose**: IPP intensity and HighIPP moderator at FF49 portfolio level

**Dimensions**: 45 portfolios

**Columns**:
| Variable | Type | Description |
|----------|------|-------------|
| ff49 | int | Portfolio number (1-49) |
| ff49_abbrev | str | Portfolio abbreviation |
| ff49_name | str | Portfolio full name |
| n_naics_ipp | int | NAICS4 industries with IPP data in portfolio |
| n_naics_mapped | int | NAICS4 industries mapped to portfolio |
| Z_IPP_p_EW | float | IPP intensity, equal-weighted across NAICS4 |
| Z_IPP_p_VW | float | IPP intensity, VA-weighted across NAICS4 |
| portfolio_va | float | Total value added in portfolio (USD) |
| Z_IPP_p_stock_share | float | Literal stock-share: sum(IPP) / sum(Total) |
| in_aiie_sample_q1q4 | int | 1 if portfolio in baseline AIIE Q1 or Q4 |
| HighIPP_EW | float | 1 if Z_IPP_p_EW >= median within Q1∪Q4 (NaN outside) |
| HighIPP_VW | float | 1 if Z_IPP_p_VW >= median within Q1∪Q4 (NaN outside) |

**Notes**:
- IPP = Intellectual Property Products (software, R&D, entertainment originals)
- Z_IPP = IPP_netstock / TotalPrivateFixedAssets_netstock (BEA Fixed Assets)
- HighIPP = median split within baseline AIIE Q1/Q4 sample; NaN for portfolios outside
- Stock-share variant: sum of IPP stocks / sum of total stocks across NAICS4 in portfolio

---

### `data/processed/ipp/ipp_naics4_pre_event.csv`

**Purpose**: NAICS4-level IPP intensity (intermediate step)

**Dimensions**: 290 NAICS4 industries (61 distinct Z_IPP values — BEA data is at SUMMARY level, so NAICS4 within the same BEA industry share identical ratios)

**Columns**:
| Variable | Type | Description |
|----------|------|-------------|
| naics4 | str | NAICS 4-digit code |
| Z_IPP_pre | float | IPP/Total net stock ratio (year 2021) |
| IPP_stock_pre | float | IPP net stock (2021, millions USD) |
| Total_stock_pre | float | Total fixed assets net stock (2021, millions USD) |
| n_years | int | Number of years with valid data |
| bea_code_matched | str | BEA industry code used for match |
| match_level | str | Concordance hierarchy level (DETAIL/SUMMARY/etc.) |
| source_years | str | Year used: "2021" |
| VA_pre | float | Value added (USD, 2021) |

---

### `data/processed/ipp/ipp_mapping_audit.csv`

**Purpose**: Row-level audit of BEA concordance → NAICS4 mapping

**Columns**: bea_code, bea_level, industry_title, related_naics_2017_raw, extracted_naics4, n_naics4, parent_u_summary, parent_summary, parent_sector

---

### `data/processed/ipp/ipp_coverage_report.json`

**Purpose**: Coverage diagnostics (BEA table row counts, mapping stats, HighIPP split counts)

---

### `data/processed/exposure/naics_measures.csv`

**Purpose**: NAICS 4-digit level exposure measures (intermediate step)

**Columns**:
| Variable | Description |
|----------|-------------|
| naics_4 | NAICS 4-digit code |
| aiie_21 | AIIE score (2021 version) |
| aiie_23 | AIIE score (2023/LM version) |
| emp | Total employment in NAICS |
| wagebill | Total wage bill in NAICS (USD) |
| expwb_21 | ExpWB 2021 (raw, dimensionless index) |
| expwb_23 | ExpWB 2023/LM (raw, dimensionless index) |

---

### `data/processed/exposure/naics_to_ff49_mapping.csv`

**Purpose**: Mapping from NAICS 4-digit to FF49 for audit trail

**Columns**:
| Variable | Description |
|----------|-------------|
| naics_4 | NAICS 4-digit code |
| ff49 | Assigned FF49 portfolio number |
| ff49_abbrev | Portfolio abbreviation |
| ff49_name | Portfolio full name |

---

## French Data Library

### 49 Industry Portfolios

#### Monthly Returns

**`data/processed/french/49_industry_portfolios_monthly/value_weighted_returns.csv`**
- Value-weighted monthly returns (%)
- Columns: date, plus 49 industry columns (Agric, Food, Soda, Beer, Smoke, Toys, Fun, Books, Hshld, Clths, Hlth, MedEq, Drugs, Chems, Rubbr, Txtls, BldMt, Cnstr, Steel, FabPr, Mach, ElcEq, Autos, Aero, Ships, Guns, Gold, Mines, Coal, Oil, Util, Telcm, PerSv, BusSv, Hardw, Chips, LabEq, Paper, Boxes, Trans, Whlsl, Rtail, Meals, Fin, Other, Banks, Insur, RlEst, Fin)

**`data/processed/french/49_industry_portfolios_monthly/equal_weighted_returns.csv`**
- Equal-weighted monthly returns (%)
- Same structure as VW returns

**`data/processed/french/49_industry_portfolios_monthly/value_weighted_returns_annual.csv`**
- Value-weighted annual returns (%)

**`data/processed/french/49_industry_portfolios_monthly/equal_weighted_returns_annual.csv`**
- Equal-weighted annual returns (%)

#### Portfolio Characteristics

**`data/processed/french/49_industry_portfolios_monthly/number_of_firms.csv`**
- Number of firms in each portfolio each month

**`data/processed/french/49_industry_portfolios_monthly/average_firm_size.csv`**
- Average firm size (market cap in millions USD)

**`data/processed/french/49_industry_portfolios_monthly/sum_be_sum_me.csv`**
- Book equity (BE) and ME (market equity) aggregations

**`data/processed/french/49_industry_portfolios_monthly/vw_average_be_me.csv`**
- Value-weighted average BE/ME ratios

#### Daily Returns

**`data/processed/french/49_industry_portfolios_daily/value_weighted_returns.csv`**
**`data/processed/french/49_industry_portfolios_daily/equal_weighted_returns.csv`**
- Daily returns (same structure as monthly)

---

### Fama-French Factors

#### FF5 Factors (5-Factor Model)

**`data/processed/french/ff5_factors_monthly/monthly_factors.csv`**
- Columns: date, Mkt-RF, SMB, HML, RMW, CMA, RF
- All values in percentage points

**`data/processed/french/ff5_factors_monthly/annual_factors.csv`**
- Annual factors

**`data/processed/french/ff5_factors_daily/factors.csv`**
- Daily factors

#### FF3 Factors (3-Factor Model)

**`data/processed/french/ff3_factors_monthly/monthly_factors.csv`**
- Columns: date, Mkt-RF, SMB, HML, RF

**`data/processed/french/ff3_factors_monthly/annual_factors.csv`**
- Annual factors

**`data/processed/french/ff3_factors_daily/factors.csv`**
- Daily factors

---

### Momentum Factor

**`data/processed/french/momentum_monthly/monthly_factors.csv`**
- Columns: date, Mom
- Momentum factor (prior return winners minus losers)

**`data/processed/french/momentum_monthly/annual_factors.csv`**
- Annual momentum

**`data/processed/french/momentum_daily/factors.csv`**
- Daily momentum

---

### Industry Definitions

**`data/processed/french/industry_definitions/sic_to_ff49.csv`**
- Maps SIC codes to FF49 portfolios

**Columns**:
| Variable | Description |
|----------|-------------|
| ff49_code | Portfolio number (1-49) |
| ff49_abbrev | Short name (e.g., "Softw") |
| ff49_name | Full name (e.g., "Computer Software") |
| sic_start | SIC code range start |
| sic_end | SIC code range end |
| sic_description | SIC industry description |

---

## Felten AI Exposure Data

### 2021 AIOE (AI Occupational Exposure)

**`data/processed/felten/aioe_2021/aioe_by_occupation.csv`**
- AI occupational exposure scores by SOC code
- Columns: soc_code, occupation, aioe

**`data/processed/felten/aioe_2021/aiie_by_industry.csv`**
- AI industry exposure scores by NAICS code
- Columns: naics_code, industry, aiie

**`data/processed/felten/aioe_2021/ability_level_exposure.csv`**
- Exposure by AI ability level

**`data/processed/felten/aioe_2021/application_ability_matrix.csv`**
- Matrix of applications × abilities

**`data/processed/felten/aioe_2021/aiie_by_geography.csv`**
- AIIE by geographic region

---

### 2023 Language Modeling AIOE

**`data/processed/felten/language_modeling/lm_aioe_by_occupation.csv`**
- Language model AIOE by SOC 2010 code
- Columns: soc_code, occupation, lm_aioe

**`data/processed/felten/language_modeling/lm_aiie_by_industry.csv`**
- Language model AIIE by NAICS code
- Columns: naics_code, industry, lm_aiie

---

### Image Generation AIOE

**`data/processed/felten/image_generation/ig_aioe_by_occupation.csv`**
**`data/processed/felten/image_generation/ig_aiie_by_industry.csv`**
- Image generation exposure scores (not used in main analysis)

---

## BLS Data

### OEWS May 2021 (Primary)

**`data/processed/bls/oews_may2021_nat4d.csv`**

**Purpose**: Occupational employment and wage data by NAICS industry

**Dimensions**: 81,130 rows (occupation × NAICS combinations)

**Key Columns**:
| Column | Description |
|--------|-------------|
| OCC_CODE | SOC occupation code |
| NAICS | 4-digit NAICS industry code |
| TOT_EMP | Total employment |
| A_MEAN | Mean annual wage (USD) |
| O_GROUP | Occupation group ("detailed" = use these rows) |
| ANNUAL | Annual flag |

**Missing values**: Coded as `**`, `*`, or `#` → treat as NaN

---

### OEWS May 2018

**`data/processed/bls/oews_may2018_nat.csv`**
- Used for employment weights in SOC harmonization
- 1,379 rows

---

### OEWS Hybrid Structure

**`data/processed/bls/oews_hybrid_structure.csv`**
- Maps hybrid occupation codes to SOC 2010 components
- 868 rows

---

### QCEW 2021

**`data/processed/bls/qcew_2021_annual.csv`**

**Purpose**: Quarterly Census of Employment and Wages for NAICS6 weighting

**Key Filters**:
- AREA_FIPS = "US000" (national)
- OWN_CODE = "5" (private sector)
- QTR = "A" (annual)
- AGGLVL_CODE = "18" (NAICS 6-digit)

**Key Columns**:
- NAICS2017_CODE: 6-digit NAICS
- ANNUAL_TAXABLE_PAYROLL: Annual wages (USD)
- ANNUAL_EMPLOYMENT: Average annual employment

---

## BEA Data

### GDP by Industry Value Added

**`data/processed/bea/gdp_by_industry_va_2021.csv`**

**Purpose**: Value added by industry for VW aggregation weights

**Columns**:
| Column | Description |
|--------|-------------|
| INDUSTRY | BEA industry code (e.g., "211", "311FT") |
| INDUSTRYDESCRIPTION | Industry name |
| DATAVALUE | Value added in billions USD (multiply by 1e9 for dollars) |
| YEAR | Year (2021) |

**Notes**:
- DATAVALUE is in billions USD
- BEA codes are at 2-3 digit level, not pure NAICS
- Some codes are aggregated (e.g., "311FT" = Food, Beverage, Tobacco)

---

### BEA-NAICS Concordance

**`data/processed/bea/bea_naics_concordance.csv`**

**Purpose**: Maps BEA industry codes to NAICS codes

**Columns**:
| Column | Description |
|--------|-------------|
| DETAIL | BEA detail industry code |
| RELATED_NAICS_2017 | NAICS codes (may include ranges like "311-312") |

**Usage**: Helps match BEA value added to NAICS industries

---

## Crosswalk Files

### SOC Crosswalk

**`data/processed/crosswalks/soc_2010_to_2018.csv`**
- Maps SOC 2010 codes to SOC 2018 codes
- Used for harmonizing Felten AIOE (SOC 2010) to OEWS (SOC 2018)

---

### SIC-NAICS Crosswalk

**`data/processed/crosswalks/sic87_to_naics02.csv`**

**Purpose**: Maps SIC 1987 codes to NAICS 2002 codes

**Columns**:
| Column | Description |
|--------|-------------|
| sic_code | SIC 1987 code |
| naics_2002_code | NAICS 2002 code |
| sic_description | SIC industry description |

**Usage**: Critical for mapping FF49 portfolios (SIC-based) to exposure data (NAICS-based)

---

### NAICS Version Crosswalks

**`data/processed/crosswalks/naics2002_to_naics2007.csv`**
**`data/processed/crosswalks/naics2007_to_naics2012.csv`**
**`data/processed/crosswalks/naics2012_to_naics2017.csv`**

- Maps NAICS codes across different vintages
- Used when data sources use different NAICS versions

---

## Data Processing Notes

### Missing Portfolios

Six FF49 portfolios lack exposure data due to SIC→NAICS mapping gaps:
- **Chems** (Chemicals)
- **FabPr** (Fabricated Products)
- **Ships** (Shipbuilding, Railroad Equipment)
- **Guns** (Defense)
- **Gold** (Precious Metals)
- **Boxes** (Shipping Containers)

**Root cause**: These portfolios' SIC codes don't appear in the SIC→NAICS crosswalk.

**Impact**: Analysis uses 43/49 portfolios (87.8% coverage). Sensitivity analysis shows imputing these portfolios doesn't change coefficients.

---

### Value-Weighted Aggregation

When aggregating NAICS measures to FF49 portfolios:
- Uses BEA value added as weights
- BEA codes mapped to NAICS at 2-3 digit level
- Fallback to EW aggregation if VA missing

---

### Z-Score Standardization

All `_z` variables (e.g., `aiie_21_ew_z`, `expwb_21_ew_z`) are standardized:
- Mean = 0
- Standard deviation = 1
- Computed across the 43 portfolios with valid data

---

### Returns-Exposure Matching Rule

Critical for regression specification:
- **EW exposure** (e.g., `aiie_21_ew_z`) → **EW returns** (`ret_ew_ex`)
- **VW exposure** (e.g., `aiie_21_vw_z`) → **VW returns** (`ret_vw_ex`)

---

## Key Dates Reference

| Date | Event Time (τ) | Description |
|------|----------------|-------------|
| 2012-12-01 | -120 | Start of 120-month pre-window |
| 2017-12-01 | -60 | Start of 60-month pre-window |
| 2022-11-01 | -1 | Reference period (omitted in event study) |
| 2022-11-30 | - | ChatGPT release date |
| 2022-12-01 | 0 | Event month (post1 = 1) |
| 2023-01-01 | +1 | January 2023 (post2 = 1 if included) |
| 2023-02-01 | +2 | End of sample |
| 2023-03-01 | +3 | ChatGPT API release (alternative event) |
| 2023-04-01 | +4 | GPT-4 release (alternative event) |

---

## Data Sources

| Dataset | Source | URL | Original Download |
|---------|--------|-----|-------------------|
| FF49 Portfolios | Ken French Data Library | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | `01_download_data.py` |
| FF3/FF5/MOM Factors | Ken French Data Library | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | `01_download_data.py` |
| AIOE 2021 | Felten et al. (AIOE-Data) | https://github.com/AIOE-Data/AIOE | `01_download_data.py` |
| LM AIOE 2023 | Felten et al. (AIOE-Data) | https://github.com/AIOE-Data/AIOE | `01_download_data.py` |
| OEWS May 2021 | BLS | https://www.bls.gov/oes/ | `01_download_data.py` |
| BEA VA 2021 | BEA | https://www.bea.gov/data/gdp/gdp-industry | `01_download_data.py` |
| Crosswalks | Census, BLS | Various | `01_download_data.py` |

---

## Pipeline Summary

1. **Download raw data** → `scripts/01_download_data.py`
2. **Build exposure measures** → `scripts/02_build_exposures.py`
   - Creates NAICS-level measures
   - Aggregates to FF49 portfolios
   - Outputs `ff49_exposure_measures.csv`
3. **Build panel dataset** → `scripts/03_build_panel.py`
   - Merges returns, factors, exposures
   - Creates event variables
   - Outputs `final_dataset.csv`
4. **Run analysis** → `scripts/04_run_analysis.py`
   - 1 baseline + 10 robustness DiD regressions
   - Pre-trends event study
   - Outputs in `outputs/analysis/`

---

## Units and Conventions

### Returns
- All returns in **percentage points** (2.5 = 2.5%)
- Monthly frequency (except daily returns)

### Monetary Values
- Value added: **Billions USD** (multiply by 1e9 for dollars)
- Wage bills: **USD**
- Not standardized across datasets

### Exposure Scores
- AIIE, ExpWB: Dimensionless indices
- Z-score versions: Mean=0, Std=1

### Dates
- All dates in **YYYY-MM-DD** format
- Monthly data uses month-end (YYYY-MM-01)

---

## Variable Naming Conventions

- `_21`: 2021 version (e.g., `aiie_21_ew`)
- `_23`: 2023/LM version (e.g., `aiie_23_ew`)
- `_ew`: Equal-weighted aggregation
- `_vw`: Value-weighted aggregation
- `_z`: Z-score standardized
- `_ex`: Excess return (return minus risk-free rate)

---

**END OF DATA DICTIONARY**
