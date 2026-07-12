"""
Smoke Test: IPP Intensity Moderator for Panel A

Standalone script that:
1. Downloads/caches BEA Fixed Assets tables (FAAt301I, FAAt301ESI, FAAt307I, FAAt307ESI)
2. Maps BEA industries to NAICS4 using GDPbyIndustry + concordance
3. Computes IPP intensity (Z^IPP = IPP_netstock / Total_netstock) at NAICS4 level
4. Aggregates to FF49 using existing NAICS→FF49 mapping + BEA VA weights
5. Computes HighIPP binary within the AIIE Q1/Q4 sample
6. Saves all outputs + diagnostics

Does NOT import from the main pipeline — reads existing CSV outputs directly.
Run: python scripts/smoke_test_ipp.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEA_API_URL = "https://apps.bea.gov/api/data"

FIXED_ASSETS_TABLES = {
    "FAAt301I": {"years": "2018,2019,2020,2021", "desc": "IPP net stock by industry"},
    "FAAt301ESI": {"years": "2018,2019,2020,2021", "desc": "Total private fixed assets net stock by industry"},
    "FAAt307I": {"years": "2017,2018,2019", "desc": "IPP investment flow by industry"},
    "FAAt307ESI": {"years": "2017,2018,2019", "desc": "Total investment flow by industry"},
}

MIN_NAICS4_WITH_IPP = 50
MIN_FF49_WITH_VALUES = 35
MIN_HIGH_IPP_GROUP_SIZE = 3

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bea" / "fixed_assets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ipp"
EXPOSURE_DIR = PROJECT_ROOT / "data" / "processed" / "exposure"
BEA_DIR = PROJECT_ROOT / "data" / "processed" / "bea"


def fail(msg):
    print(f"\n{'='*60}")
    print(f"=== SMOKE TEST: FAIL ({msg}) ===")
    print(f"{'='*60}")
    sys.exit(1)


# =============================================================================
# BEA API DOWNLOAD (with caching)
# =============================================================================

def load_api_key():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        fail(".env file not found. Create one with BEA_API_KEY=your_key")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEA_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key
    fail("BEA_API_KEY not found or empty in .env file")


def download_fixed_assets_table(api_key, table_name, years):
    cache_csv = RAW_DIR / f"bea_fixedassets_{table_name}_{years.replace(',','_')}.csv"
    if cache_csv.exists() and cache_csv.stat().st_size > 0:
        print(f"  [CACHE] {table_name} years={years}")
        return pd.read_csv(cache_csv)

    print(f"  [API]   {table_name} years={years}...")
    params = {
        "method": "GetData",
        "datasetname": "FixedAssets",
        "TableName": table_name,
        "Year": years,
        "UserID": api_key,
        "ResultFormat": "JSON",
    }
    resp = requests.get(BEA_API_URL, params=params, timeout=120)
    resp.raise_for_status()
    js = resp.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"bea_fixedassets_{table_name}_{years.replace(',','_')}.json", "w") as f:
        json.dump(js, f, indent=2)

    results = js.get("BEAAPI", {}).get("Results", None)
    if results is None:
        fail(f"No BEAAPI.Results for {table_name}")

    data_rows = None
    if isinstance(results, dict) and "Data" in results:
        data_rows = results["Data"]
    elif isinstance(results, list):
        for obj in results:
            if isinstance(obj, dict) and "Data" in obj:
                data_rows = obj["Data"]
                break
    if data_rows is None:
        fail(f"No Data block for {table_name}")

    df = pd.DataFrame(data_rows)
    df.to_csv(cache_csv, index=False)
    print(f"          -> {len(df)} rows cached")
    return df


def download_all_tables(api_key):
    tables = {}
    for name, info in FIXED_ASSETS_TABLES.items():
        df = download_fixed_assets_table(api_key, name, info["years"])
        if df.empty:
            fail(f"Empty data for {name}")
        tables[name] = df
    return tables


# =============================================================================
# BEA CODE → NAICS4 MAPPING (via concordance)
# =============================================================================

def parse_naics_string(naics_str):
    """Parse RELATED_NAICS_2017 into set of NAICS4 codes."""
    if pd.isna(naics_str) or not str(naics_str).strip():
        return set()
    naics4 = set()
    for part in str(naics_str).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\d+)", part)
        if m and len(m.group(1)) >= 4:
            naics4.add(m.group(1)[:4])
        elif m and len(m.group(1)) == 3:
            naics4.add(m.group(1))  # keep 3-digit for later matching
    return naics4


def build_bea_code_to_naics4(concordance_path):
    """
    Build BEA industry code → set(NAICS4) from the concordance.

    The concordance CSV has a shifted-column hierarchy:
    - Sector rows: SECTOR=code, SUMMARY=title
    - Summary rows: SUMMARY=code, U_SUMMARY=title
    - U_Summary rows: U_SUMMARY=code, DETAIL=title
    - Detail rows: DETAIL=code, INDUSTRY_TITLE=title, RELATED_NAICS_2017=naics
    """
    df = pd.read_csv(concordance_path)

    # Clean all columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": "", "": ""})

    # Step 1: Extract DETAIL-level mappings (they have RELATED_NAICS_2017)
    detail_to_naics4 = {}
    detail_to_parent = {}  # detail_code → (u_summary, summary, sector)

    current_sector = ""
    current_summary = ""
    current_u_summary = ""

    audit_rows = []

    for _, row in df.iterrows():
        s, su, us, de, title, notes, naics = (
            row["SECTOR"], row["SUMMARY"], row["U_SUMMARY"],
            row["DETAIL"], row["INDUSTRY_TITLE"], row["NOTES"],
            row["RELATED_NAICS_2017"],
        )

        # Track hierarchy
        if s:
            current_sector = s
        if su:
            # SUMMARY column: could be sector title (descriptive) or a summary code
            # A code is alphanumeric starting with digit: 111CA, 211, 311FT, etc.
            if re.match(r"^\d", su):
                current_summary = su
        if us:
            if re.match(r"^\d", us):
                current_u_summary = us

        # Identify detail rows: DETAIL has a code AND RELATED_NAICS_2017 is populated
        if de and naics and re.match(r"^\d", de):
            naics4_set = parse_naics_string(naics)
            if naics4_set:
                detail_to_naics4[de] = naics4_set
                detail_to_parent[de] = (current_u_summary, current_summary, current_sector)
                audit_rows.append({
                    "bea_code": de,
                    "bea_level": "DETAIL",
                    "industry_title": title,
                    "related_naics_2017_raw": naics,
                    "extracted_naics4": ";".join(sorted(naics4_set)),
                    "n_naics4": len(naics4_set),
                    "parent_u_summary": current_u_summary,
                    "parent_summary": current_summary,
                    "parent_sector": current_sector,
                })

    # Step 2: Propagate up — U_SUMMARY gets union of its children DETAIL codes' NAICS4
    u_summary_to_naics4 = {}
    for de_code, (u_summ, summ, sect) in detail_to_parent.items():
        if u_summ:
            u_summary_to_naics4.setdefault(u_summ, set()).update(detail_to_naics4[de_code])

    # Step 3: SUMMARY gets union of its U_SUMMARY children
    summary_to_naics4 = {}
    for de_code, (u_summ, summ, sect) in detail_to_parent.items():
        if summ:
            summary_to_naics4.setdefault(summ, set()).update(detail_to_naics4[de_code])

    # Step 4: SECTOR gets union of its SUMMARY children
    sector_to_naics4 = {}
    for de_code, (u_summ, summ, sect) in detail_to_parent.items():
        if sect:
            sector_to_naics4.setdefault(sect, set()).update(detail_to_naics4[de_code])

    # Combine into one dict: bea_code → naics4_set
    # Priority: DETAIL > U_SUMMARY > SUMMARY > SECTOR (for dedup later)
    bea_to_naics4 = {}
    for code, n4 in sector_to_naics4.items():
        bea_to_naics4[code] = ("SECTOR", n4)
    for code, n4 in summary_to_naics4.items():
        bea_to_naics4[code] = ("SUMMARY", n4)
    for code, n4 in u_summary_to_naics4.items():
        bea_to_naics4[code] = ("U_SUMMARY", n4)
    for code, n4 in detail_to_naics4.items():
        bea_to_naics4[code] = ("DETAIL", n4)

    all_naics4 = set()
    for _, (_, n4) in bea_to_naics4.items():
        all_naics4.update(n4)

    print(f"  BEA concordance: {len(detail_to_naics4)} detail, "
          f"{len(u_summary_to_naics4)} u_summary, {len(summary_to_naics4)} summary, "
          f"{len(sector_to_naics4)} sector codes -> {len(all_naics4)} unique NAICS4")

    return bea_to_naics4, audit_rows


# =============================================================================
# MAP FIXED ASSETS LINES → BEA CODES (via GDPbyIndustry description bridge)
# =============================================================================

def _normalize(s):
    """Normalize description for matching: lowercase, strip, collapse spaces."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def build_description_to_bea_code(va_path, bea_to_naics4):
    """
    Build a description → BEA code lookup using:
    1. GDPbyIndustry (description → BEA code, reliable)
    2. Additional manual mappings for FA-specific lines not in GDP
    """
    va = pd.read_csv(va_path)
    va.columns = va.columns.str.strip().str.upper()

    desc_to_bea = {}
    for _, row in va.iterrows():
        desc = _normalize(row["INDUSTRYDESCRIPTION"])
        code = str(row["INDUSTRY"]).strip()
        if desc and code:
            desc_to_bea[desc] = code

    # Additional mappings for FA lines not in GDP-by-Industry
    # (FA has finer detail in some areas)
    extra = {
        "food manufacturing": "311",
        "beverage and tobacco product manufacturing": "312",
        "electric power": "2211",
        "natural gas distribution": "2212",
        "water, sewage, and other systems": "2213",
        "other manufacturing": "339",
        "medical equipment and supplies": "3391",
        "durable goods": "33DG",
        "nondurable goods": "31ND",
        "manufacturing": "31G",
        "motor vehicle and parts dealers": "441",
        "food and beverage stores": "445",
        "general merchandise stores": "452",
        "other retail": "4A0",
        "federal reserve banks": "521",
        "credit intermediation and related activities": "522",
        "depository credit intermediation": "5221",
        "other intermediation": "5223",
        "nondepository credit intermediation": "52229",
        "insurance carriers": "5241",
        "insurance agencies and brokers and related services": "5242",
        "publishing industries (includes software)": "511",
        "information and data processing services": "514",
        "miscellaneous professional, scientific, and technical services": "5412OP",
        "hospitals": "622",
        "performing arts, spectator sports, museums, and related activities": "711AS",
        "other transportation and support activities": "487OS",
    }
    for desc, code in extra.items():
        desc_to_bea.setdefault(desc, code)

    return desc_to_bea


def identify_leaf_lines(fa_lines):
    """
    Identify leaf (non-aggregate) lines in the Fixed Assets table.

    Aggregate lines to skip: total, sector totals that have children,
    manufacturing/durable/nondurable sub-totals.
    """
    # These line descriptions are known aggregates (their values are sums of children)
    aggregate_keywords = {
        "private fixed assets",  # line 1: grand total
    }

    # Lines that are sector-level aggregates WITH children in the table
    sector_aggregates = {
        "agriculture, forestry, fishing, and hunting",
        "mining",
        "utilities",
        "manufacturing",
        "durable goods",
        "nondurable goods",
        "wholesale trade",
        "retail trade",
        "transportation and warehousing",
        "information",
        "finance and insurance",
        "real estate and rental and leasing",
        "professional, scientific, and technical services",
        "administrative and waste management services",
        "health and social assistance",  # GDP code "62" -- has children 621-624
        "arts, entertainment, and recreation",
        "accommodation and food services",
    }

    # Lines with children that are also aggregates within their sector
    sub_aggregates = {
        "credit intermediation and related activities",  # children: 5221, 5223, 52229
        "insurance carriers and related activities",  # children: 5241, 5242
    }

    all_agg = aggregate_keywords | sector_aggregates | sub_aggregates

    leaf_mask = []
    for desc in fa_lines:
        normalized = _normalize(desc)
        is_leaf = normalized not in all_agg
        leaf_mask.append(is_leaf)
    return leaf_mask


def map_fa_lines_to_bea(ipp_df, total_df, desc_to_bea, bea_to_naics4):
    """
    Match IPP and Total Fixed Assets tables, identify leaf lines,
    and map to NAICS4.
    """
    # Parse values
    for df in [ipp_df, total_df]:
        df["line_number"] = pd.to_numeric(df["LineNumber"], errors="coerce")
        df["year"] = pd.to_numeric(df["TimePeriod"], errors="coerce")
        df["value"] = pd.to_numeric(
            df["DataValue"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df["desc_norm"] = df["LineDescription"].apply(_normalize)

    # Merge IPP and Total on (line_number, year)
    merged = ipp_df[["line_number", "year", "value", "desc_norm", "LineDescription"]].merge(
        total_df[["line_number", "year", "value"]],
        on=["line_number", "year"],
        suffixes=("_ipp", "_total"),
    )
    merged = merged.rename(columns={
        "value_ipp": "ipp_value",
        "value_total": "total_value",
    })
    print(f"  Merged IPP+Total: {len(merged)} rows ({merged['line_number'].nunique()} lines × "
          f"{merged['year'].nunique()} years)")

    # Identify leaf lines
    unique_descs = merged.drop_duplicates("line_number").sort_values("line_number")
    leaf_mask_lookup = dict(zip(
        unique_descs["desc_norm"],
        identify_leaf_lines(unique_descs["desc_norm"]),
    ))
    merged["is_leaf"] = merged["desc_norm"].map(leaf_mask_lookup)

    n_leaf = merged[merged["is_leaf"]]["line_number"].nunique()
    n_total = merged["line_number"].nunique()
    print(f"  Leaf lines: {n_leaf}/{n_total}")

    # Keep only leaf lines
    leaf = merged[merged["is_leaf"]].copy()

    # Match descriptions to BEA codes
    leaf["bea_code"] = leaf["desc_norm"].map(desc_to_bea)

    matched = leaf["bea_code"].notna().sum()
    unmatched = leaf[leaf["bea_code"].isna()]["desc_norm"].unique()
    print(f"  Description→BEA code: {matched}/{len(leaf)} rows matched")
    if len(unmatched) > 0:
        print(f"  UNMATCHED descriptions ({len(unmatched)}):")
        for d in sorted(unmatched):
            print(f"    - {d}")

    leaf = leaf[leaf["bea_code"].notna()].copy()

    # Map BEA codes to NAICS4
    results = []
    unmatched_bea = set()

    for _, row in leaf.iterrows():
        bea_code = row["bea_code"]
        if bea_code in bea_to_naics4:
            level, naics4_set = bea_to_naics4[bea_code]
        else:
            # Try prefix matching: "5221" → check "522", then "52"
            found = False
            for plen in range(len(bea_code) - 1, 1, -1):
                prefix = bea_code[:plen]
                if prefix in bea_to_naics4:
                    level, naics4_set = bea_to_naics4[prefix]
                    level = f"{level}_prefix"
                    found = True
                    break
            if not found:
                unmatched_bea.add(bea_code)
                continue

        # Filter to 4-digit NAICS only
        naics4_filtered = {n for n in naics4_set if len(n) == 4}
        if not naics4_filtered:
            # If concordance gave 3-digit codes, expand: all 4-digit starting with those 3
            naics4_filtered = naics4_set  # keep as-is, will match at 3-digit level later

        for n4 in naics4_filtered:
            results.append({
                "naics4": n4,
                "year": int(row["year"]),
                "ipp_value": row["ipp_value"],
                "total_value": row["total_value"],
                "bea_code": bea_code,
                "match_level": level,
                "description": row["LineDescription"],
            })

    if unmatched_bea:
        print(f"  BEA codes not in concordance ({len(unmatched_bea)}): {sorted(unmatched_bea)}")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        fail("No data after mapping to NAICS4")

    # Deduplicate: when multiple BEA codes map to the same NAICS4,
    # keep the most granular (DETAIL > U_SUMMARY > SUMMARY > SECTOR)
    level_rank = {"DETAIL": 0, "U_SUMMARY": 1, "SUMMARY": 2, "SECTOR": 3}
    result_df["level_rank"] = result_df["match_level"].apply(
        lambda x: level_rank.get(x.split("_")[0] if "_" in x else x, 4)
    )
    result_df = result_df.sort_values(["naics4", "year", "level_rank"])
    result_df = result_df.drop_duplicates(subset=["naics4", "year"], keep="first")
    result_df = result_df.drop(columns=["level_rank"])

    print(f"  Final NAICS4 data: {len(result_df)} rows, "
          f"{result_df['naics4'].nunique()} unique NAICS4")

    return result_df


# =============================================================================
# BUILD IPP INTENSITY AT NAICS4 LEVEL
# =============================================================================

def build_ipp_naics4(naics4_data, target_years, va_data):
    """
    Build Z_IPP = IPP/Total at NAICS4 level, averaged across target years.
    """
    # Average across BEA codes within same NAICS4×year (shouldn't happen after dedup, but safety)
    agg = naics4_data.groupby(["naics4", "year"]).agg(
        ipp_value=("ipp_value", "mean"),
        total_value=("total_value", "mean"),
        bea_code=("bea_code", "first"),
        match_level=("match_level", "first"),
    ).reset_index()

    # Compute ratio
    agg["Z_IPP"] = np.where(agg["total_value"] > 0, agg["ipp_value"] / agg["total_value"], np.nan)

    # Filter to target years and average
    agg = agg[agg["year"].isin(target_years)]
    valid = agg[agg["Z_IPP"].notna()]

    naics4_avg = valid.groupby("naics4").agg(
        Z_IPP_pre=("Z_IPP", "mean"),
        IPP_stock_pre=("ipp_value", "mean"),
        Total_stock_pre=("total_value", "mean"),
        n_years=("year", "count"),
        bea_code_matched=("bea_code", "first"),
        match_level=("match_level", "first"),
    ).reset_index()
    naics4_avg["source_years"] = ",".join(str(y) for y in sorted(target_years))

    # Merge VA data (repo convention: 2021 VA, matched at 3-digit then 2-digit)
    naics4_avg = _merge_va(naics4_avg, va_data)

    print(f"  NAICS4 IPP: {len(naics4_avg)} industries")
    print(f"  Z_IPP range: [{naics4_avg['Z_IPP_pre'].min():.4f}, {naics4_avg['Z_IPP_pre'].max():.4f}]")
    print(f"  VA coverage: {(naics4_avg['VA_pre'] > 0).sum()}/{len(naics4_avg)}")

    return naics4_avg


def _merge_va(naics4_avg, va_data):
    """Merge VA at 3-digit level with 2-digit fallback (same as 02_build_exposures.py)."""
    va = va_data.copy()
    va.columns = va.columns.str.strip().str.upper()
    va = va.rename(columns={"INDUSTRY": "naics", "DATAVALUE": "value_added"})
    va["value_added"] = pd.to_numeric(
        va["value_added"].astype(str).str.replace(",", ""), errors="coerce"
    ) * 1e9  # billions → dollars

    va["naics_clean"] = va["naics"].astype(str).str.extract(r"^(\d+)")[0]
    va = va.dropna(subset=["naics_clean"])
    va = va[va["naics_clean"].str.len() <= 3]

    va_3 = va.groupby(va["naics_clean"].str[:3])["value_added"].sum().reset_index()
    va_3.columns = ["naics_3", "value_added"]

    va_2 = va.groupby(va["naics_clean"].str[:2])["value_added"].sum().reset_index()
    va_2.columns = ["naics_2", "value_added"]

    naics4_avg["naics_3"] = naics4_avg["naics4"].str[:3]
    naics4_avg["naics_2"] = naics4_avg["naics4"].str[:2]

    naics4_avg = naics4_avg.merge(va_3, on="naics_3", how="left")
    naics4_avg = naics4_avg.rename(columns={"value_added": "VA_pre"})

    missing = naics4_avg["VA_pre"].isna()
    if missing.any():
        va2_map = va_2.set_index("naics_2")["value_added"]
        naics4_avg.loc[missing, "VA_pre"] = naics4_avg.loc[missing, "naics_2"].map(va2_map)

    naics4_avg["VA_pre"] = naics4_avg["VA_pre"].fillna(0)
    naics4_avg = naics4_avg.drop(columns=["naics_3", "naics_2"])

    return naics4_avg


# =============================================================================
# AGGREGATE TO FF49
# =============================================================================

def aggregate_to_ff49(naics4_ipp, naics_to_ff49):
    """EW, VW, and stock-share aggregation of IPP to FF49."""
    # Ensure naics4 is string in both
    naics4_ipp["naics4"] = naics4_ipp["naics4"].astype(str)
    naics_to_ff49["naics_4"] = naics_to_ff49["naics_4"].astype(str)

    merged = naics4_ipp.merge(
        naics_to_ff49[["naics_4", "ff49", "ff49_abbrev", "ff49_name"]],
        left_on="naics4", right_on="naics_4", how="inner",
    )

    print(f"  NAICS4 matched to FF49: {merged['naics4'].nunique()}")
    print(f"  FF49 portfolios: {merged['ff49'].nunique()}")

    results = []
    for ff49, grp in merged.groupby("ff49"):
        valid = grp[grp["Z_IPP_pre"].notna()]
        if len(valid) == 0:
            continue

        row = {
            "ff49": int(ff49),
            "ff49_abbrev": grp["ff49_abbrev"].iloc[0],
            "ff49_name": grp["ff49_name"].iloc[0],
            "n_naics_ipp": len(valid),
            "n_naics_mapped": len(grp),
        }

        # EW
        row["Z_IPP_p_EW"] = valid["Z_IPP_pre"].mean()

        # VW
        va_valid = valid[valid["VA_pre"] > 0]
        if len(va_valid) > 0 and va_valid["VA_pre"].sum() > 0:
            row["Z_IPP_p_VW"] = np.average(va_valid["Z_IPP_pre"], weights=va_valid["VA_pre"])
            row["portfolio_va"] = va_valid["VA_pre"].sum()
        else:
            row["Z_IPP_p_VW"] = valid["Z_IPP_pre"].mean()
            row["portfolio_va"] = 0.0

        # Stock-share
        total_sum = valid["Total_stock_pre"].sum()
        row["Z_IPP_p_stock_share"] = (
            valid["IPP_stock_pre"].sum() / total_sum if total_sum > 0 else np.nan
        )

        results.append(row)

    ff49_ipp = pd.DataFrame(results).sort_values("ff49").reset_index(drop=True)
    print(f"  FF49 with IPP: {len(ff49_ipp)}")
    if len(ff49_ipp) > 0:
        print(f"  Z_IPP_EW range: [{ff49_ipp['Z_IPP_p_EW'].min():.4f}, "
              f"{ff49_ipp['Z_IPP_p_EW'].max():.4f}]")

    return ff49_ipp


# =============================================================================
# COMPUTE HighIPP WITHIN Q1/Q4
# =============================================================================

def compute_high_ipp(ff49_ipp, ff49_exposure):
    """Median split of IPP within AIIE Q1/Q4 sample."""
    exp = ff49_exposure[["ff49", "aiie_23_ew_z"]].dropna()

    q25 = exp["aiie_23_ew_z"].quantile(0.25)
    q75 = exp["aiie_23_ew_z"].quantile(0.75)
    q4 = set(exp[exp["aiie_23_ew_z"] >= q75]["ff49"])
    q1 = set(exp[exp["aiie_23_ew_z"] <= q25]["ff49"])
    q1q4 = q1 | q4
    print(f"  AIIE Q1/Q4: Q4={len(q4)}, Q1={len(q1)}, total={len(q1q4)}")

    ff49_ipp["in_aiie_sample_q1q4"] = ff49_ipp["ff49"].isin(q1q4).astype(int)

    q1q4_data = ff49_ipp[ff49_ipp["ff49"].isin(q1q4) & ff49_ipp["Z_IPP_p_EW"].notna()]

    if len(q1q4_data) == 0:
        print("  WARNING: No IPP data for Q1/Q4 portfolios")
        ff49_ipp["HighIPP_EW"] = np.nan
        ff49_ipp["HighIPP_VW"] = np.nan
        return ff49_ipp

    # EW median split
    med_ew = q1q4_data["Z_IPP_p_EW"].median()
    ff49_ipp["HighIPP_EW"] = np.where(
        ff49_ipp["ff49"].isin(q1q4) & ff49_ipp["Z_IPP_p_EW"].notna(),
        (ff49_ipp["Z_IPP_p_EW"] >= med_ew).astype(int),
        np.nan,
    )

    # VW median split
    vw_data = q1q4_data["Z_IPP_p_VW"].dropna()
    if len(vw_data) > 0:
        med_vw = vw_data.median()
        ff49_ipp["HighIPP_VW"] = np.where(
            ff49_ipp["ff49"].isin(q1q4) & ff49_ipp["Z_IPP_p_VW"].notna(),
            (ff49_ipp["Z_IPP_p_VW"] >= med_vw).astype(int),
            np.nan,
        )
    else:
        ff49_ipp["HighIPP_VW"] = np.nan

    q = ff49_ipp[ff49_ipp["in_aiie_sample_q1q4"] == 1]
    print(f"  HighIPP_EW: med={med_ew:.4f}, High={(q['HighIPP_EW']==1).sum()}, "
          f"Low={(q['HighIPP_EW']==0).sum()}")

    return ff49_ipp


# =============================================================================
# FLOW-SHARE ROBUSTNESS
# =============================================================================

def build_flow_share(tables, desc_to_bea, bea_to_naics4, naics_to_ff49, va_data, ff49_ipp):
    """Build flow-share variant from FAAt307I/FAAt307ESI (2017-2019 avg)."""
    if "FAAt307I" not in tables or "FAAt307ESI" not in tables:
        ff49_ipp["Z_IPP_flow_EW"] = np.nan
        ff49_ipp["Z_IPP_flow_VW"] = np.nan
        return ff49_ipp

    print("\n--- Flow-Share Variant (FAAt307I/FAAt307ESI, 2017-2019) ---")
    try:
        n4_flow = map_fa_lines_to_bea(
            tables["FAAt307I"].copy(), tables["FAAt307ESI"].copy(),
            desc_to_bea, bea_to_naics4,
        )
        n4_flow_avg = build_ipp_naics4(n4_flow, [2017, 2018, 2019], va_data)
        ff49_flow = aggregate_to_ff49(n4_flow_avg, naics_to_ff49)

        flow_cols = ff49_flow[["ff49", "Z_IPP_p_EW", "Z_IPP_p_VW"]].rename(columns={
            "Z_IPP_p_EW": "Z_IPP_flow_EW",
            "Z_IPP_p_VW": "Z_IPP_flow_VW",
        })
        ff49_ipp = ff49_ipp.merge(flow_cols, on="ff49", how="left")
    except Exception as e:
        print(f"  Flow-share failed: {e}")
        ff49_ipp["Z_IPP_flow_EW"] = np.nan
        ff49_ipp["Z_IPP_flow_VW"] = np.nan

    return ff49_ipp


# =============================================================================
# DIAGNOSTICS & PASS/FAIL
# =============================================================================

def build_report(tables, bea_to_naics4, naics4_ipp, ff49_ipp, naics_to_ff49):
    report = {
        "bea_tables": {n: {"rows": len(df)} for n, df in tables.items()},
        "concordance": {
            "total_bea_codes": len(bea_to_naics4),
            "total_naics4": len({n for _, (_, s) in bea_to_naics4.items() for n in s}),
        },
        "naics4_coverage": {
            "with_ipp": len(naics4_ipp) if not naics4_ipp.empty else 0,
            "in_ff49_mapping": naics_to_ff49["naics_4"].nunique(),
        },
        "ff49_coverage": {
            "with_ipp": len(ff49_ipp) if not ff49_ipp.empty else 0,
            "in_mapping": naics_to_ff49["ff49"].nunique(),
        },
        "design_decisions": {
            "va_year": "2021 (repo convention)",
            "z_ipp_method": "average-of-ratios across target years",
            "leaf_only": "aggregate BEA lines skipped to avoid double-counting",
            "matching": "GDPbyIndustry description bridge + manual extras",
        },
    }
    if "HighIPP_EW" in ff49_ipp.columns:
        q = ff49_ipp[ff49_ipp["in_aiie_sample_q1q4"] == 1]
        report["high_ipp"] = {
            "q1q4_count": len(q),
            "high_ew": int((q["HighIPP_EW"] == 1).sum()),
            "low_ew": int((q["HighIPP_EW"] == 0).sum()),
        }
    return report


def run_pass_fail(naics4_ipp, ff49_ipp):
    failures = []
    n4 = len(naics4_ipp) if not naics4_ipp.empty else 0
    if n4 < MIN_NAICS4_WITH_IPP:
        failures.append(f"NAICS4 with IPP: {n4} < {MIN_NAICS4_WITH_IPP}")
    if naics4_ipp.empty or naics4_ipp["Z_IPP_pre"].isna().all():
        failures.append("All Z_IPP ratios are NA")
    n_ff49 = len(ff49_ipp) if not ff49_ipp.empty else 0
    if n_ff49 < MIN_FF49_WITH_VALUES:
        failures.append(f"FF49 with values: {n_ff49} < {MIN_FF49_WITH_VALUES}")
    if "HighIPP_EW" in ff49_ipp.columns:
        q = ff49_ipp[ff49_ipp["in_aiie_sample_q1q4"] == 1]
        hi = (q["HighIPP_EW"] == 1).sum()
        lo = (q["HighIPP_EW"] == 0).sum()
        if hi < MIN_HIGH_IPP_GROUP_SIZE:
            failures.append(f"HighIPP high group: {hi} < {MIN_HIGH_IPP_GROUP_SIZE}")
        if lo < MIN_HIGH_IPP_GROUP_SIZE:
            failures.append(f"HighIPP low group: {lo} < {MIN_HIGH_IPP_GROUP_SIZE}")
    return len(failures) == 0, failures


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("SMOKE TEST: IPP Intensity Moderator (Panel A)")
    print("=" * 60)

    # --- Prerequisites ---
    print("\n--- Prerequisites ---")
    api_key = load_api_key()
    print("  BEA_API_KEY: loaded")

    for path, name in [
        (BEA_DIR / "bea_naics_concordance.csv", "concordance"),
        (EXPOSURE_DIR / "naics_to_ff49_mapping.csv", "NAICS→FF49 mapping"),
        (EXPOSURE_DIR / "ff49_exposure_measures.csv", "FF49 exposures"),
        (BEA_DIR / "gdp_by_industry_va_2021.csv", "VA data"),
    ]:
        if not path.exists():
            fail(f"{name} not found at {path}")
    print("  All prerequisite files present")

    # --- 1: Download ---
    print("\n--- Step 1: Download BEA Fixed Assets ---")
    tables = download_all_tables(api_key)

    # --- 2: Build BEA code → NAICS4 mapping ---
    print("\n--- Step 2: BEA concordance → NAICS4 ---")
    bea_to_naics4, audit_rows = build_bea_code_to_naics4(str(BEA_DIR / "bea_naics_concordance.csv"))

    # --- 3: Build description → BEA code bridge ---
    print("\n--- Step 3: Description → BEA code bridge ---")
    va_data = pd.read_csv(BEA_DIR / "gdp_by_industry_va_2021.csv")
    desc_to_bea = build_description_to_bea_code(str(BEA_DIR / "gdp_by_industry_va_2021.csv"), bea_to_naics4)
    print(f"  Description lookup: {len(desc_to_bea)} entries")

    # --- 4: Map Fixed Assets → NAICS4 ---
    print("\n--- Step 4: Map Fixed Assets → NAICS4 ---")
    naics4_data = map_fa_lines_to_bea(
        tables["FAAt301I"].copy(), tables["FAAt301ESI"].copy(),
        desc_to_bea, bea_to_naics4,
    )

    # --- 5: Build IPP intensity ---
    print("\n--- Step 5: Build IPP at NAICS4 ---")
    naics4_ipp = build_ipp_naics4(naics4_data, [2018, 2019, 2020, 2021], va_data)

    # --- 6: Aggregate to FF49 ---
    print("\n--- Step 6: Aggregate to FF49 ---")
    naics_to_ff49 = pd.read_csv(EXPOSURE_DIR / "naics_to_ff49_mapping.csv")
    ff49_ipp = aggregate_to_ff49(naics4_ipp, naics_to_ff49)

    # --- 7: HighIPP ---
    print("\n--- Step 7: HighIPP within Q1/Q4 ---")
    ff49_exposure = pd.read_csv(EXPOSURE_DIR / "ff49_exposure_measures.csv")
    ff49_ipp = compute_high_ipp(ff49_ipp, ff49_exposure)

    # --- 8: Flow-share robustness ---
    ff49_ipp = build_flow_share(tables, desc_to_bea, bea_to_naics4, naics_to_ff49, va_data, ff49_ipp)

    # --- 9: Save ---
    print("\n--- Step 9: Save outputs ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    naics4_path = OUTPUT_DIR / "ipp_naics4_pre_event.csv"
    naics4_ipp.to_csv(naics4_path, index=False)
    print(f"  {naics4_path.name}: {len(naics4_ipp)} rows")

    ff49_path = OUTPUT_DIR / "panelA_ipp_ff49.csv"
    ff49_ipp.to_csv(ff49_path, index=False)
    print(f"  {ff49_path.name}: {len(ff49_ipp)} rows")

    audit_path = OUTPUT_DIR / "ipp_mapping_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"  {audit_path.name}: {len(audit_rows)} rows")

    report = build_report(tables, bea_to_naics4, naics4_ipp, ff49_ipp, naics_to_ff49)
    report_path = OUTPUT_DIR / "ipp_coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  {report_path.name}")

    # --- 10: PASS/FAIL ---
    print("\n--- PASS/FAIL ---")
    passed, failures = run_pass_fail(naics4_ipp, ff49_ipp)

    if passed:
        print(f"\n{'='*60}")
        print("=== SMOKE TEST: PASS ===")
        print(f"{'='*60}")
        print(f"\n  NAICS4 with IPP: {len(naics4_ipp)}")
        print(f"  FF49 with IPP:   {len(ff49_ipp)}")
        if "HighIPP_EW" in ff49_ipp.columns:
            q = ff49_ipp[ff49_ipp["in_aiie_sample_q1q4"] == 1]
            print(f"  Q1/Q4 with IPP:  {len(q)}")
            print(f"  HighIPP split:   {int((q['HighIPP_EW']==1).sum())} high / "
                  f"{int((q['HighIPP_EW']==0).sum())} low")

        print(f"\n  Top 5 by Z_IPP_p_EW:")
        for _, r in ff49_ipp.nlargest(5, "Z_IPP_p_EW").iterrows():
            print(f"    {r['ff49_abbrev']:8s}: {r['Z_IPP_p_EW']:.4f}")

        print(f"\n  Bottom 5 by Z_IPP_p_EW:")
        for _, r in ff49_ipp.nsmallest(5, "Z_IPP_p_EW").iterrows():
            print(f"    {r['ff49_abbrev']:8s}: {r['Z_IPP_p_EW']:.4f}")
    else:
        print(f"\n{'='*60}")
        for reason in failures:
            print(f"=== SMOKE TEST: FAIL ({reason}) ===")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
