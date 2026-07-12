# Methodological Appendix: AI Exposure Measures

This document describes the construction of the two AI exposure measures used in the
ChatGPT shock event study: **AIIE** (AI Industry Exposure) and **ExpWB** (Exposure
Weighted by Wage Bill Share). Both are mapped to Fama-French 49 industry portfolios.

---

## 1. Data Sources

| Dataset | Source | Vintage | Description |
|---------|--------|---------|-------------|
| AIOE (AI Occupational Exposure) | Felten et al. (2021, 2023) | 2021, 2023 | AI exposure scores for ~800 occupations (SOC 6-digit) |
| AIIE (AI Industry Exposure) | Felten et al. (2021, 2023) | 2021, 2023 | AI exposure scores for ~270 industries (NAICS 4-digit) |
| OEWS (Occupational Employment and Wage Statistics) | BLS | May 2021 | Employment counts and mean wages by occupation-industry cell |
| GDP by Industry Value Added | BEA | 2021 | Value added by industry (for VW aggregation to FF49) |
| SIC-NAICS Crosswalk | U.S. Census Bureau | SIC 1987 to NAICS 2002 | Maps SIC codes to NAICS codes |
| FF49 Industry Definitions | Ken French Data Library | Current | Maps SIC code ranges to 49 industry portfolios |

### References

- Felten, E., Raj, M., & Seamans, R. (2021). "Occupational, Industry, and Geographic Exposure to Artificial Intelligence: A Novel Dataset and Its Potential Uses." *Strategic Management Journal*, 42(12), 2195-2217.
- Felten, E., Raj, M., & Seamans, R. (2023). "How Will Language Modelers Like ChatGPT Affect Occupations and Industries?" Working paper.

---

## 2. NAICS to FF49 Mapping

The mapping from NAICS industries to Fama-French 49 portfolios proceeds in two steps:

1. **SIC-to-NAICS crosswalk**: Use the Census Bureau's SIC 1987 to NAICS 2002 concordance to create a reverse mapping (NAICS-4 to SIC-4).

2. **SIC-to-FF49 assignment**: For each NAICS-4 code, find all SIC codes it maps to, look up which FF49 portfolio each SIC code belongs to (using Ken French's SIC range definitions), and assign the **modal** FF49 portfolio.

This produces a many-to-one mapping from ~300 NAICS-4 codes to 43 FF49 portfolios (6 portfolios have no NAICS match and are excluded from the analysis).

**Script**: `scripts/02_build_exposures.py`, functions `create_naics_to_ff49_mapping()`, `load_sic_to_ff49()`, `load_naics_to_sic_crosswalk()`.

---

## 3. AIIE (AI Industry Exposure) — Simple

AIIE is the direct industry-level AI exposure score from Felten et al. It represents how exposed an industry's tasks are to AI capabilities.

### Construction

Felten et al. provide AIIE scores at the NAICS level (mixed 3-6 digit). We:

1. Truncate all NAICS codes to 4 digits
2. Average AIIE within each NAICS-4 group (for codes with multiple sub-industries)
3. Map NAICS-4 to FF49 using the crosswalk above
4. Aggregate to FF49 portfolios (see Section 5)

### Versions

| Variable | Source | Year |
|----------|--------|------|
| `aiie_21` | Felten et al. (2021) original AIIE | 2021 |
| `aiie_23` | Felten et al. (2023) Language Model AIIE | 2023 |

The 2023 version specifically measures exposure to **language modeling** capabilities (relevant for ChatGPT), while the 2021 version captures broader AI exposure.

**Script**: `scripts/02_build_exposures.py`, function `load_aiie_data()`.

---

## 4. ExpWB (Exposure Weighted by Wage Bill Share)

ExpWB is a bottom-up exposure measure constructed from **occupation-level** AI exposure (AIOE), weighted by each occupation's wage bill share within an industry.

### Formula

For industry *i*:

```
ExpWB_i = sum_o [ AIOE_o * (wagebill_io / sum_o(wagebill_io)) ]
```

where:

- **AIOE_o** = AI Occupational Exposure score for occupation *o* (from Felten et al.)
- **wagebill_io** = employment_io * mean_wage_io (from BLS OEWS, May 2021)
- **sum_o(wagebill_io)** = total wage bill for industry *i* across all occupations
- The fraction *wagebill_io / sum_o(wagebill_io)* is the **wage bill share** of occupation *o* in industry *i*

### Interpretation

ExpWB is an **adimensional intensity index** (not a dollar amount). It represents the average AI exposure of an industry's workforce, weighted by each occupation's economic importance (wage bill share) within that industry.

### Key difference from AIIE

| Feature | AIIE | ExpWB |
|---------|------|-------|
| Level | Direct industry scores | Aggregated from occupation scores |
| Weighting | None (Felten's raw scores) | Occupation wage bill shares |
| Data needed | AIIE only | AIOE + OEWS (employment × wages) |
| Interpretation | Industry task exposure to AI | Workforce-weighted AI exposure |

### Versions

| Variable | AIOE Source | Year |
|----------|------------|------|
| `expwb_21` | Felten et al. (2021) AIOE | 2021 |
| `expwb_23` | Felten et al. (2023) LM-AIOE | 2023 |

**Script**: `scripts/02_build_exposures.py`, function `build_expwb()`.

---

## 5. FF49 Aggregation

Both AIIE and ExpWB are first computed at the NAICS-4 level, then aggregated to FF49 portfolios using two methods:

### Equal-Weighted (EW)

Simple arithmetic mean across all NAICS-4 industries in the portfolio:

```
Measure_EW_p = (1/N_p) * sum_i [ Measure_i ]
```

where *N_p* is the number of NAICS-4 industries mapped to FF49 portfolio *p*.

### Value-Added Weighted (VW)

Weighted average using BEA value added as weights:

```
Measure_VW_p = sum_i [ Measure_i * VA_i ] / sum_i [ VA_i ]
```

where *VA_i* is the BEA GDP-by-industry value added for NAICS industry *i* (2021 vintage).

If no value added data is available for a portfolio's industries, the VW measure falls back to EW.

**Script**: `scripts/02_build_exposures.py`, function `aggregate_to_ff49()`.

---

## 6. Z-Score Standardization

All measures are standardized to z-scores (mean = 0, std = 1) across the cross-section of FF49 portfolios:

```
z_i = (x_i - mean(x)) / std(x)
```

This allows coefficient interpretation as: "a one standard deviation increase in AI exposure is associated with a γ percentage point change in excess returns."

Z-scored versions are denoted with the `_z` suffix (e.g., `aiie_23_ew_z`).

---

## 7. Variable Naming Convention

All exposure variables follow the pattern: `{measure}_{year}_{aggregation}[_z]`

| Component | Values | Description |
|-----------|--------|-------------|
| `measure` | `aiie`, `expwb` | AIIE Simple or ExpWB (wage-bill weighted) |
| `year` | `21`, `23` | 2021 or 2023 (Language Model) vintage |
| `aggregation` | `ew`, `vw` | Equal-weighted or value-added weighted |
| `_z` | (optional) | Z-score standardized |

### Examples

| Variable | Description |
|----------|-------------|
| `aiie_23_ew` | AIIE 2023, equal-weighted, raw |
| `aiie_23_ew_z` | AIIE 2023, equal-weighted, z-scored |
| `expwb_21_vw_z` | ExpWB 2021, value-added weighted, z-scored |

---

## 8. IPP Intensity Moderator (Panel A)

### Definition

IPP (Intellectual Property Products) intensity measures the share of an industry's fixed assets that are intangible (software, R&D, entertainment originals):

```
Z^IPP_j = IPP_netstock_j / TotalPrivateFixedAssets_netstock_j
```

### Data Source

BEA Fixed Assets tables (FixedAssets dataset, not GDPbyIndustry):

| Table | Content | Year |
|-------|---------|-------|
| FAAt301I | IPP net stock by industry (numerator) | 2021 |
| FAAt301ESI | Total private fixed assets net stock (denominator) | 2021 |

**Pre-event year**: 2021 (last full year before ChatGPT release on Nov 30, 2022). Single-year data (no averaging).

### Mapping Chain

BEA Fixed Assets lines → GDPbyIndustry description bridge → BEA industry codes → BEA-NAICS concordance → NAICS4 → FF49 (via existing `naics_to_ff49_mapping.csv`).

Key steps:

1. **Description matching**: Match FA line descriptions to BEA industry codes using GDPbyIndustry as a lookup table, supplemented by manual mappings for FA-specific lines not in GDP.
2. **Leaf-line filtering**: Skip aggregate lines (Manufacturing, Durable goods, etc.) to avoid double-counting with their children. Exception: Wholesale trade is treated as a leaf because its FA sub-lines ("Durable goods", "Nondurable goods") share names with Manufacturing aggregates and would be misclassified.
3. **Concordance mapping**: Map BEA codes → NAICS4 via the concordance hierarchy, preferring DETAIL-level matches over coarser SUMMARY/SECTOR matches. The concordance provides NAICS codes at varying digit lengths (2-6 digit).
4. **Short-code expansion**: Expand 2- and 3-digit NAICS codes to all matching 4-digit codes using the FF49 mapping as the universe of valid codes. This recovers industries where the BEA concordance provides coarser-than-4-digit NAICS (e.g., "315" for Apparel, "23" for Construction, "531" for Real Estate).
5. **Deduplication**: When multiple BEA codes map to the same NAICS4, keep the most granular match.

### Aggregation to FF49

Same methods as AIIE/ExpWB (Section 5):

- **EW**: Simple mean of `Z^IPP_j` across NAICS4 in portfolio
- **VW**: BEA 2021 value-added weighted average (3-digit NAICS match, 2-digit fallback)
- **Stock-share**: `sum(IPP_stock_j) / sum(Total_stock_j)` across NAICS4 in portfolio

### HighIPP Binary

Within the baseline Q1/Q4 AIIE sample (~22 portfolios):

- `HighIPP = 1` if `Z_IPP_p >= median(Z_IPP_p)` computed over Q1 ∪ Q4 only
- NaN for portfolios outside the Q1/Q4 estimation sample

### Coverage and Granularity

45 FF49 portfolios receive IPP values (290 NAICS4 after short-code expansion).

**Important granularity note**: The BEA Fixed Assets table provides ~75 industry lines at SUMMARY level (e.g., "Wholesale trade", "Food manufacturing"), not at NAICS-4 detail. All NAICS4 codes within a BEA SUMMARY line inherit the same Z_IPP ratio. This yields only 61 distinct Z_IPP values across 290 NAICS4. The NAICS4 intermediary maintains consistency with the AIIE/ExpWB pipeline but does not add sub-industry variation. For the HighIPP binary moderator (median split within Q1/Q4), the 18 distinct values across 22 portfolios provide a clean split with no ties at the boundary.

**Script**: `scripts/02b_build_ipp.py`

---

## 9. Portfolio Coverage

Of the 49 Fama-French industry portfolios, **43** receive exposure measures. The 6 missing portfolios have no NAICS-4 industries mapped through the SIC crosswalk. These portfolios are excluded from the analysis when their exposure measure is missing.

---

## 10. Output Files

| File | Description |
|------|-------------|
| `data/processed/exposure/ff49_exposure_measures.csv` | All exposure measures at FF49 level |
| `data/processed/exposure/naics_measures.csv` | NAICS-level measures (intermediate) |
| `data/processed/exposure/naics_to_ff49_mapping.csv` | NAICS-4 to FF49 concordance |
| `data/processed/ipp/panelA_ipp_ff49.csv` | IPP intensity + HighIPP at FF49 level |
| `data/processed/ipp/ipp_naics4_pre_event.csv` | NAICS4-level IPP intensity (intermediate) |
| `data/processed/ipp/ipp_mapping_audit.csv` | BEA→NAICS4 mapping audit trail |
| `data/processed/ipp/ipp_coverage_report.json` | Coverage diagnostics |
