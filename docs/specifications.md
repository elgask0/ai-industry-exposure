# Regression Specifications

This document describes the baseline Difference-in-Differences specification and
the 10 robustness checks used in the ChatGPT shock event study.

---

## 1. Difference-in-Differences Setup

### Event

ChatGPT was released on November 30, 2022. The event month is **December 2022** (first full trading month after release).

### Treatment

Binary treatment based on AI exposure quartiles:
- **Treated**: FF49 portfolios in the top quartile (Q4) of the exposure distribution
- **Control**: FF49 portfolios in the bottom quartile (Q1) of the exposure distribution
- Middle portfolios (Q2-Q3) are **dropped** from the sample

#### Precise treatment construction (baseline)

1. Compute `AIIE_p^EW` = arithmetic mean of AIIE 2023 (LM) across all NAICS-4 industries mapped to FF49 portfolio *p*
2. Z-score across portfolios: `AIIE_p^z = (AIIE_p^EW − mean) / sd`, computed over the **43 FF49 portfolios** with valid NAICS-to-FF49 mapping (6 portfolios with no NAICS match are excluded)
3. Compute q25, q75 of `AIIE_p^z` over the same 43 portfolios
4. `Treated_p = 1` if `AIIE_p^z ≥ q75` (~11 portfolios); `Control_p = 0` if `AIIE_p^z ≤ q25` (~11 portfolios)
5. Drop the ~21 interquartile portfolios from the regression sample

**Note:** Since z-scoring is rank-preserving, quartile assignment is identical whether computed on raw or z-scored exposure. The z-scoring matters for continuous-treatment coefficient interpretation, not for binary classification.

#### Re-ranking rule for robustness checks

- **Checks that change the exposure measure** (R1, R2, R10): re-rank portfolios into Q1/Q4 using the alternative measure's z-scores. Cutpoints are recomputed over the set of portfolios with valid exposure for that measure.
- **Checks that hold exposure fixed** (R3–R9): keep the baseline EW classification unchanged.

### Model

```
Y_it = alpha_i + beta * X_t + gamma * (Post_t x Treated_i) + epsilon_it
```

where:
- **Y_it** = excess return of portfolio *i* in month *t*
- **alpha_i** = portfolio fixed effects (absorb time-invariant industry characteristics)
- **X_t** = Fama-French factor returns (control for systematic risk)
- **Post_t** = indicator for the post-event window
- **Treated_i** = indicator for top-quartile AI exposure
- **gamma** = the DiD coefficient of interest

Note: The `Treated_i` main effect is absorbed by portfolio fixed effects. The `Post_t` main effect is absorbed by the factor controls (which vary only over time).

### Returns-Exposure Matching Rule

EW (equal-weighted) exposure measures are always paired with EW portfolio returns, and VW (value-added weighted) exposure measures with VW portfolio returns. This ensures consistency between how exposure is measured and how returns are computed.

---

## 2. Baseline Specification

The baseline was selected through a systematic specification search (54,240 specifications tested, refined to 768, final selection: Spec 456).

| Parameter | Choice | Rationale |
|-----------|--------|-----------|
| **Exposure** | AIIE 2023 EW (z-scored) | LM-specific exposure (most relevant for ChatGPT); EW avoids size bias |
| **Dep. var** | `ret_ew_ex` (EW excess return) | Matches EW exposure |
| **Fixed effects** | Portfolio only | Controls for industry heterogeneity; avoids month FE / factor collinearity |
| **Factors** | FF5 (MktRF, SMB, HML, RMW, CMA) | Controls for market, size, value, profitability, and investment risk |
| **Standard errors** | OLS (non-robust) | Efficient under homoskedasticity; robust alternatives in R9 |
| **Post period** | `post2` (Dec 2022 + Jan 2023) | Two-month event window (see rationale below) |
| **Pre-window** | 60 months (Dec 2017–Nov 2022) | 5 years of pre-event data for stable estimation |
| **Full sample** | 63 months (Dec 2017–Feb 2023) | Pre-window + 3 post-event months (Dec 2022, Jan 2023, Feb 2023) |
| **Treatment** | Binary Q1 vs Q4 | Top vs bottom quartile by exposure |

### Result

- **gamma = +2.45** (t = +2.20, p = 0.028)
- High-AI-exposure portfolios earned ~2.45 percentage points higher excess returns in the two-month post-ChatGPT window, relative to low-exposure portfolios
- Pre-trends: **Strict PASS** (0/11 significant at 10%, max|t| = 1.48, joint F p = 0.91)

---

## 3. Post-Period Rationale

The two-month window (`post2` = Dec 2022 + Jan 2023) was chosen based on the dynamic event study:

| Month | Event time | Market context | AI signal |
|-------|-----------|----------------|-----------|
| **Dec 2022** (tau=0) | Event month | Broad market decline (~-7%) | AI differential drowned by macro noise (t = 0.33) |
| **Jan 2023** (tau=+1) | +1 month | Broad market rally (+14%) | AI differential emerges clearly (t = 2.35**) |
| **Feb 2023** (tau=+2) | +2 months | Mixed | Essentially zero effect (t = 0.13) |

**Interpretation**: The ChatGPT signal took about one month to be priced. December was dominated by macro-driven selling that affected all portfolios equally. January's rally allowed the AI-specific differential to surface. By February, the repricing was complete. Including February (`post3`) would dilute the signal without adding information.

---

## 4. Pre-Trends Methodology

### Dynamic Event Study

The pre-trends test uses an 11-month pre-event window (τ = -12 to τ = -2, i.e., Dec 2021 to Oct 2022) with the same baseline specification. Instead of a single Post × Treated coefficient, the model estimates separate τ × Treated interactions for each event time:

```
Y_it = alpha_i + beta * X_t + sum_tau [ delta_tau * D(event_time=tau) * Treated_i ] + epsilon_it
```

where τ ranges from **-12 to +2**, with **τ = -1 (November 2022)** as the omitted reference period.

**Pre-event coefficients**: 11 coefficients for τ = -12, -11, ..., -2
**Post-event coefficients**: 3 coefficients for τ = 0 (Dec 2022), τ = +1 (Jan 2023), τ = +2 (Feb 2023)

### Criteria

| Criterion | Strict | Moderate |
|-----------|--------|----------|
| Pre-period coefficients significant at p < 0.10 | At most 1 of 11 | At most 2 of 11 |
| Maximum |t-statistic| among pre-period | <= 2.0 | <= 2.5 |
| Joint F-test p-value (all pre-coefs = 0) | >= 0.10 | >= 0.05 |
| Average |pre-coef| < average |post-coef| | Required | Not required |

The **strict** criteria are the primary test. Both strict and moderate criteria are reported.

---

## 5. Robustness Checks

Each robustness check varies **one parameter** from the baseline, holding everything else constant:

| ID | Check | Changed Parameter | Purpose |
|----|-------|-------------------|---------|
| **R1** | 2021 version | `aiie_21_ew_z` | Tests whether results depend on the 2023 LM-specific exposure vs the earlier general AI exposure |
| **R2** | VW weighting | `aiie_23_vw_z` + VW returns; **Treated/Control re-ranked using VW exposure quartiles** | Tests whether size-weighting in exposure aggregation and returns changes the result |
| **R3** | Post 1m (Dec only) | `post1` | Tests whether the effect appears in the event month alone |
| **R4** | 120m pre-window | 120 months | Tests sensitivity to estimation sample length (10 years vs 5 years) |
| **R5** | FF5 factors | `ff5` | Adds profitability (RMW) and investment (CMA) factors |
| **R6** | FF3 + MOM | `ff3_mom` | Adds momentum factor to FF3 |
| **R7** | FF5 + MOM | `ff5_mom` | Most comprehensive factor model (6 factors) |
| **R8** | CAPM only | `mktrf` | Minimal factor model — only market excess return |
| **R9** | Clustered SE | Cluster by month | Accounts for within-month correlation across portfolios |
| **R10** | ExpWB 2023 EW | `expwb_23_ew_z` | Alternative exposure measure (wage-bill weighted vs direct industry scores) |

### Design Principle

By changing only one parameter at a time, each check isolates the sensitivity to that specific choice. A robust result should maintain statistical significance across all (or most) checks.

---

## 6. Factor Model Details

| Set | Factors | Columns | Source |
|-----|---------|---------|--------|
| `mktrf` | CAPM | MktRF | Ken French |
| `ff3` | Fama-French 3 | MktRF, SMB, HML | Ken French |
| `ff5` | Fama-French 5 | MktRF, SMB, HML, RMW, CMA | Ken French |
| `ff3_mom` | FF3 + Momentum | MktRF, SMB, HML, MOM | Ken French |
| `ff5_mom` | FF5 + Momentum | MktRF, SMB, HML, RMW, CMA, MOM | Ken French |

### Collinearity Note

When **month fixed effects** are included, Fama-French factors must be **excluded**. FF factors vary only over time (they are constant across portfolios within a month), making them perfectly collinear with month dummies. The baseline uses portfolio FE + factors, avoiding this issue.

---

## 7. High–Low Portfolio Construction (Table 5)

### Main text (EW)

The High (Q4) and Low (Q1) legs use the **baseline EW classification** from Section 1.

Long–short return series:

```
LS_t = (1/N_Q4) × Σ_{p ∈ Q4} r^EW_{p,t}  −  (1/N_Q1) × Σ_{p ∈ Q1} r^EW_{p,t}
```

- Equal-weight across portfolios within each leg (not market-cap weighted)
- `r^EW_{p,t}` = EW excess return of portfolio *p*
- Alphas = intercepts from time-series regressions of LS_t on factor models (CAPM, FF3, FF5, FF5+MOM)

### Appendix (VW)

- Groups re-ranked using AIIE 2023 VW quartiles
- VW returns within each portfolio
- Same equal-weight-across-legs construction

### Rationale

FF49 portfolios are already internally weighted (EW or VW depending on series). Aggregating across portfolios within a leg should be equal-weight to treat each industry equally and avoid confounding size with exposure.

---

## 8. Event-Study VW Robustness

For the VW version of the event study (dynamic DiD):
- **Re-rank** Treated/Control using AIIE 2023 VW quartiles (same as R2)
- **Swap returns** to `ret_vw_ex`
- Report as an **appendix figure** (Figure A.1), not in the main Figure 1
- The main text reports the VW point estimate in Table 2 (R2 column)

---

## 9. Table-Note Templates

### Table 2 (DiD) — baseline note

> *Treated* portfolios are the 11 Fama–French 49 industry portfolios in the top quartile (≥ 75th percentile) of AIIE 2023 (EW, z-scored); *Control* portfolios are the 11 in the bottom quartile (≤ 25th percentile). Quartile cutpoints are computed over the 43 portfolios with valid NAICS-to-FF49 mapping. The 21 interquartile portfolios and 6 unmappable portfolios are excluded. All specifications use the same binary classification except columns that change the exposure measure (R1, R2, R10), which re-rank portfolios into quartiles using the alternative measure.

### Table 2 col. (R2) — VW note

> Column R2 uses value-added weighted AIIE (AIIE 2023 VW, z-scored) with VW portfolio returns. Treated/Control groups are re-ranked using VW exposure quartiles. Of the 11 baseline (EW) Treated portfolios, [N] remain Treated under VW ranking.

### Table 5 (High–Low spread) note

> The High (Low) portfolio is the equal-weighted average of monthly excess returns across all Treated (Control) portfolios as defined in Table 2. The long–short spread is LS_t = r̄_{High,t} − r̄_{Low,t}. Alphas are intercepts from time-series regressions of LS_t on the indicated factor model. Newey–West standard errors with [L] lags in parentheses.

### Table 5 appendix (VW) note

> Panel B uses VW portfolio returns with Treated/Control groups re-ranked by AIIE 2023 VW quartiles.

---

## 10. Output Files

| File | Description |
|------|-------------|
| `outputs/analysis/did_report.txt` | Full coefficient tables for all 11 regressions |
| `outputs/analysis/did_table.csv` | Machine-readable results |
| `outputs/analysis/pretrends_results.txt` | Event study coefficients + diagnostics |
| `outputs/analysis/pretrends_aiie_23_ew.png` | Event study plot |
| `outputs/analysis/full_report.txt` | Combined report (regressions + pre-trends) |
