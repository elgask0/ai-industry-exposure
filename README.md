# AI Exposure Across Industries

A reproducible pipeline for measuring how exposed US industries are to artificial intelligence through their occupational composition.

## Research question

How can industry-level AI exposure be constructed from occupational exposure scores, employment, wages, and consistent industry crosswalks?

The project treats exposure as a measurement problem before using it in any outcome analysis. It produces two complementary measures:

- **AIIE:** a direct industry exposure measure derived from Felten, Raj, and Seamans.
- **ExpWB:** a bottom-up measure that weights occupational AI exposure by each occupation's wage bill within an industry.

For industry `i`, the wage-bill-weighted measure is:

```text
ExpWB_i = sum_o(AIOE_o * wage_bill_io) / sum_o(wage_bill_io)
```

## Construction

The pipeline combines:

1. occupational and industry AI-exposure scores from Felten, Raj, and Seamans;
2. BLS Occupational Employment and Wage Statistics;
3. BEA industry data and concordances;
4. NAICS, SIC, and Fama-French 49 industry mappings;
5. equal-weighted and value-weighted portfolio aggregation.

The public outputs currently cover 43 of the 49 Fama-French industry portfolios. The remaining portfolios do not pass the available industry-mapping and data-coverage checks.

## Public outputs

`results/` contains derived, aggregate research outputs:

- `ff49_exposure_measures.csv`: AI-exposure measures at the Fama-French 49 portfolio level;
- `naics_measures.csv`: industry measures before portfolio aggregation;
- `naics_to_ff49_mapping.csv`: the auditable industry-to-portfolio crosswalk.

No proprietary data or credentials are included. The repository downloads or rebuilds source inputs locally where licensing permits.

## Reproduce the index

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/01_download_data.py
python scripts/02_build_exposures.py
```

A free BEA API key is required for the BEA download step. Other inputs come from the public sources documented in `docs/methodology.md`.

The repository also retains downstream event-study code as an experimental application of the measures. Those return estimates are not the headline result and remain under methodological revision.

## Repository map

- `scripts/`: downloads, exposure construction, mapping checks, and downstream experiments.
- `src/data/`: source adapters and crosswalk logic.
- `docs/`: methodology, empirical-design, and specification notes.
- `results/`: selected aggregate exposure measures and mappings.
- `data/`: local source and intermediate data; contents are excluded from Git.
- `outputs/`: generated local analysis outputs; contents are excluded from Git.

## Status

The exposure-index construction is public and reproducible. Downstream research using the index is ongoing.
