# ChatGPT Shock Event Study

An empirical project on whether US industry portfolios with greater exposure to artificial intelligence earned different returns around the public release of ChatGPT on 30 November 2022.

## Research question

Did the release of ChatGPT coincide with a differential repricing of industries that were more exposed to AI-related tasks?

## Method

The project combines:

- Fama-French 49 industry portfolio returns;
- FF3, FF5, and momentum factors;
- Felten, Raj, and Seamans AI-exposure measures;
- BLS occupational employment and wage data;
- BEA industry weights;
- difference-in-differences and event-study specifications.

The design compares top- and bottom-quartile exposure portfolios around the release. It is an event study, not a clean causal experiment.

## Current evidence

The baseline FF5 specification estimates a positive two-month differential of 2.36 percentage points (`p = 0.034`). The estimate is sensitive to several reasonable alternatives:

| Specification | Estimate | p-value |
|---|---:|---:|
| FF5 baseline | +2.36 | 0.034 |
| Value-weighted exposure | +0.24 | 0.825 |
| December only | -0.88 | 0.577 |
| FF5 + momentum | +1.58 | 0.161 |
| Standard errors clustered by month | +2.36 | 0.287 |

Pre-period coefficients are jointly insignificant in the reported event study, but two placebo dates are individually significant. The result should therefore be read as suggestive evidence under revision, not as a settled market effect.

Selected aggregate outputs are stored in `results/`. Raw and processed datasets are generated locally and excluded from Git.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/run_all.py
```

A free BEA API key is required for the BEA download step. Other inputs are downloaded from their cited public sources.

To reuse existing local downloads:

```bash
python scripts/run_all.py --skip-download --skip-ipp
```

## Repository map

- `scripts/`: pipeline orchestration, estimators, checks, and output generation.
- `src/data/`: data-source adapters and crosswalk construction.
- `docs/`: methodology and specification notes.
- `results/`: selected aggregate estimates and a pre-trends figure.
- `data/`: local raw and processed inputs; contents are ignored.
- `outputs/`: generated local outputs; contents are ignored.

## Status

Research project under methodological revision. The code reproduces the current estimates; the interpretation remains deliberately cautious.
