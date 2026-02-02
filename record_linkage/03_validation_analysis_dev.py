#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generated from: 03_validation_analysis_dev.ipynb

This script was automatically generated from a Jupyter notebook.
Markdown cells have been converted to comments.
"""

################################################################################
# Validation resolutions
# Final step in this record linkage exercise, we look at all possible donor-patient match pairs and identify the best matches based on IMV support, Vasopressor administration, and withdrawl of life support treatment.
# This code also computes final table one for various subsets of population- Donor-Patient matches by confidence, and for a subset of DCD donors. And looks at various metrics of concordance between the two populations.
################################################################################

################################################################################
# 1. Setup and Imports
################################################################################

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

import os
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path.cwd().parent))
from utils.io import read_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from utils.config import config
site_name = config['site_name']
tables_path = config['tables_path']
file_type = config['file_type']
project_root = config['project_root']
sys.path.insert(0, project_root)
print(f"Site Name: {site_name}")
print(f"Tables Path: {tables_path}")
print(f"File Type: {file_type}")
from pathlib import Path
PROJECT_ROOT = Path(config['project_root'])
UTILS_DIR = PROJECT_ROOT / "utils"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FINAL_DIR = OUTPUT_DIR / "final"
OUTPUT_INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

# Create the output directories if they do not exist
for dir_path in [OUTPUT_DIR, OUTPUT_FINAL_DIR, OUTPUT_INTERMEDIATE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')
print("Libraries imported successfully")
print(f"Polars version: {pl.__version__}")
print(f"Pandas version: {pd.__version__}")

################################################################################
# 2. Load Data
################################################################################

# ------------------------------------------------------------------
# Load matched datasets
# ------------------------------------------------------------------
all_matches_df = pl.read_parquet(
    OUTPUT_INTERMEDIATE_DIR / "matches_df.parquet"
)

final_clif_df = pl.read_parquet(
    OUTPUT_INTERMEDIATE_DIR / "final_clif_data.parquet"
)

srtr_df = pl.read_parquet(
    OUTPUT_INTERMEDIATE_DIR / "final_srtr_data.parquet"
)

encounter_mapping_df = pl.read_parquet(
    OUTPUT_INTERMEDIATE_DIR / "encounter_mapping_matched.parquet"
)

wide_df = pl.read_parquet(
    OUTPUT_INTERMEDIATE_DIR / "wide_df.parquet"
)

# ------------------------------------------------------------------
# Quick sanity checks
# ------------------------------------------------------------------
print(f"Matches: {all_matches_df.shape}")
print(f"CLIF cohort: {final_clif_df.shape}")
print(f"SRTR donors: {srtr_df.shape}")
print(f"Encounter mapping: {encounter_mapping_df.shape}")
print(f"Wide events: {wide_df.shape}")

wide_df.columns

################################################################################
# 4. Life support
################################################################################

COLUMNS_TO_KEEP = [
 'encounter_block',
 'DONOR_ID',
 'is_dead',
 'final_outcome_dttm',
 'event_dttm',
 'med_cont_dobutamine',
 'med_cont_dopamine',
 'med_cont_epinephrine',
 'med_cont_milrinone',
 'med_cont_norepinephrine',
 'med_cont_phenylephrine',
 'med_cont_vasopressin',
 "med_cont_morphine",
 "med_cont_fentanyl",
 "med_cont_hydromorphone",
 "med_cont_remifentanil",
 "med_int_fentanyl",
 "med_int_hydromorphone",
 "med_int_morphine",
 'resp_device_name',
 'resp_device_category',
 'resp_mode_category',
 'resp_tracheostomy',
 'resp_fio2_set',
 'resp_lpm_set',
 'resp_tidal_volume_set',
 'resp_resp_rate_set']

logger.info("Extracting life support features...")
VASOPRESSOR_COLS = [
    "med_cont_dobutamine",
    "med_cont_dopamine",
    "med_cont_epinephrine",
    "med_cont_milrinone",
    "med_cont_norepinephrine",
    "med_cont_vasopressin",
]

INOTROPE_COLS = [
    "med_cont_dobutamine",
    "med_cont_milrinone",
]

OPIOID_COLS = [
    "med_cont_morphine",
    "med_cont_fentanyl",
    "med_cont_hydromorphone",
    "med_cont_remifentanil",
]

mapping = (
    encounter_mapping_df
    .select([
        "hospitalization_id",
        "encounter_block",
        "patient_id",
        "DONOR_ID",
    ])
    .unique()
)

# attach death info from clif
death_info = final_clif_df.select([
    "encounter_block",
    "is_dead",
    "final_outcome_dttm",
])

mapping = mapping.join(
    death_info,
    on="encounter_block",
    how="left"
)

df = (
    mapping
    .join(
        wide_df,
        on="hospitalization_id",
        how="left"
    )
    .with_columns([
        pl.col("event_dttm").cast(pl.Datetime),
        pl.col("final_outcome_dttm").cast(pl.Datetime),
    ])
)

# Add missing columns as null columns before selecting
existing_cols = set(df.columns)
missing_cols = [col for col in COLUMNS_TO_KEEP if col not in existing_cols]

# Create null columns for any missing columns
if missing_cols:
    df = df.with_columns([
        pl.lit(None).alias(col) for col in missing_cols
    ])

# Now safely select all columns
df = df.select(COLUMNS_TO_KEEP)
print("df shape",df.shape)
print("Missing columns:", missing_cols)

# normalise resp devices
df = df.with_columns(
    pl.col("resp_device_category")
      .cast(pl.Utf8)
      .str.to_lowercase()
      .alias("resp_device_category_lc")
)
# Sort + forward-fill respiratory state
df = df.sort(["encounter_block", "event_dttm"])

df = df.with_columns(
    pl.col("resp_device_category_lc")
      .forward_fill()
      .over("encounter_block")
      .alias("resp_device_ffill")
)

# Define IMV and non-IMV states
df = df.with_columns([
    (pl.col("resp_device_ffill") == "imv")
        .fill_null(False)
        .alias("is_imv"),

    (
        pl.col("resp_device_ffill").is_not_null() &
        (pl.col("resp_device_ffill") != "imv")
    )
    .fill_null(False)
    .alias("is_non_imv"),
])

# ffill vasopressors 
df = df.sort(["encounter_block", "event_dttm"])
for c in VASOPRESSOR_COLS:
    if c in df.columns:
        df = df.with_columns(
            pl.col(c)
              .forward_fill()
              .over("encounter_block")
              .alias(c)
        )

# Create vasopressor “on” flags (time-varying)
for c in VASOPRESSOR_COLS:
    if c in df.columns:
        df = df.with_columns(
            (pl.col(c) > 0).alias(f"{c}_on")  # CORRECT - treats 0.0 as "off"
        )
    else:
        df = df.with_columns(
            pl.lit(False).alias(f"{c}_on")
        )

# Aggregate “any vasopressor on” per event
df = df.with_columns(
     pl.any_horizontal(
         [pl.col(f"{c}_on") for c in VASOPRESSOR_COLS]
     ).fill_null(False).alias("any_vasopressor_on")
 )

# Create lagged state (for transitions)
df = df.with_columns([
    pl.col("is_imv")
      .shift(1)
      .over("encounter_block")
      .fill_null(False)
      .alias("prev_is_imv"),

    pl.col("any_vasopressor_on")
      .shift(1)
      .over("encounter_block")
      .fill_null(False)
      .alias("prev_vasopressor_on"),
])
# define wlst candidate events
df = df.with_columns([
    (
        pl.col("prev_is_imv") &
        pl.col("is_non_imv") &
        (~pl.col("any_vasopressor_on"))
    ).alias("wlst_classic"),

    (
        pl.col("prev_is_imv") &
        pl.col("is_imv") &
        pl.col("prev_vasopressor_on") &
        (~pl.col("any_vasopressor_on"))
    ).alias("wlst_vasopressor_withdrawal"),

    (
        pl.col("is_imv") &
        (pl.col("is_dead") == 1)
    ).alias("is_terminal_imv"),
])
# label wlst type
df = df.with_columns(
    (
        pl.col("wlst_classic") |
        pl.col("wlst_vasopressor_withdrawal")
    ).alias("wlst_candidate"),

    pl.when(pl.col("wlst_classic"))
      .then(pl.lit("classic_transition"))
    .when(pl.col("wlst_vasopressor_withdrawal"))
      .then(pl.lit("vasopressor_withdrawal"))
    .when(pl.col("is_terminal_imv"))
      .then(pl.lit("terminal_imv"))
    .otherwise(None)
    .alias("wlst_type"),
)
features = (
    df.group_by("encounter_block")
      .agg([
          pl.col("is_imv").any().alias("imv_ever"),
          pl.col("any_vasopressor_on").any().alias("any_vasopressor_ever"),

          # ADD THESE TWO LINES:
          pl.col("event_dttm").filter(pl.col("is_imv")).min().alias("imv_start_dttm"),
          pl.col("event_dttm").filter(pl.col("is_imv")).max().alias("imv_end_dttm"),

          pl.when(pl.col("wlst_candidate").any())
            .then(pl.col("event_dttm").filter(pl.col("wlst_candidate")).max())
            .when((pl.col("is_dead") == 1).any() & pl.col("is_terminal_imv").any())
            .then(pl.col("final_outcome_dttm").max())
            .otherwise(None)
            .alias("wlst_dttm"),

          pl.col("wlst_type")
            .filter(pl.col("wlst_type").is_not_null())
            .first()
            .alias("wlst_type"),

          pl.col("is_dead").max().alias("is_dead"),
          pl.col("final_outcome_dttm").max().alias("final_outcome_dttm"),
      ])
)

features.columns

features_with_donor = (
    features
    .join(
        all_matches_df.unique(),
        on="encounter_block",
        how="inner"
    )
)

# Add PO2 and pH arterial values from final_clif_df
lab_values_for_matches = final_clif_df.select([
    "encounter_block",
    "po2_arterial_value",
    "ph_arterial_value"
])

# Add PO2 and pH arterial values
features_with_donor = features_with_donor.join(
    lab_values_for_matches,
    on="encounter_block",
    how="left"
)

"""
 Note:

WLST Type Definitions:
- classic_transition: Life support is withdrawn by extubating the patient and stopping vasopressors
- vasopressor_withdrawal: Vasopressors are discontinued, but mechanical ventilation is continued
- terminal_imv: The patient dies while still on mechanical ventilation, without withdrawal
"""

################################################################################
# 5. Best Matches
# For donors with multiple potential patient matches, we apply hierarchical filtering. First, if any match has documented WLST (withdrawal of life-sustaining treatment), we exclusively consider WLST matches. Among remaining candidates, we prioritize by:
# 1. WLST presence,
# 2. death status,
# 3. progressive matching score,
# 4. date difference, and
# 5. age difference.
# This ensures selection of the most clinically relevant match, particularly prioritizing end-of-life indicators that align with organ donation scenarios. Single-match donors retain their only match without additional scoring.
################################################################################

donor_ids_in_mapping = set(encounter_mapping_df["DONOR_ID"].unique().to_list())
donor_ids_in_features = set(features_with_donor["DONOR_ID"].unique().to_list())

# Find donor IDs in mapping that are NOT in features
donor_ids_missing_in_features = donor_ids_in_mapping - donor_ids_in_features
print(f"Number of donor IDs in mapping not present in features: {len(donor_ids_missing_in_features)}")
if donor_ids_missing_in_features:
    print("Example donor IDs not in features:", list(donor_ids_missing_in_features)[:10])

# Calculate whether each donor has multiple matches
features_with_donor = features_with_donor.with_columns([
    pl.col("DONOR_ID").count().over("DONOR_ID").alias("n_matches_per_donor")
]).with_columns([
    (pl.col("n_matches_per_donor") > 1).alias("multiple_matches")
])

# For donors with multiple matches, apply hierarchical selection
# Step 1: If ANY match has WLST, keep ONLY those with WLST
donors_with_wlst_options = (
    features_with_donor
    .filter(pl.col("multiple_matches") == True)
    .group_by("DONOR_ID")
    .agg([
        pl.col("wlst_type").is_not_null().any().alias("any_has_wlst")  # Changed to wlst_type
    ])
)

# Join back to mark which donors have WLST options
features_with_donor = features_with_donor.join(
    donors_with_wlst_options,
    on="DONOR_ID",
    how="left"
).with_columns([
    pl.col("any_has_wlst").fill_null(False)
])

# Apply WLST filter: For donors with WLST options, keep ONLY WLST matches
features_filtered = features_with_donor.filter(
    # Keep if: single match OR (multiple matches without WLST option) OR (has WLST)
    (pl.col("multiple_matches") == False) |  # Keep all single matches
    (pl.col("any_has_wlst") == False) |      # Keep all if no WLST options exist
    (pl.col("wlst_type").is_not_null())      # If WLST options exist, only keep those with WLST - Changed to wlst_type
)

# Now select best match per donor using simple criteria
best_matches_df = (
    features_filtered
    .sort(
        by=[
            "DONOR_ID",
            # First priority: WLST (not null first)
            pl.col("wlst_type").is_not_null(),  # Changed to wlst_type
            # Second priority: is_dead
            "is_dead",
            # Third priority: original score from progressive matching
            "score",
            # Fourth priority: date difference (lower is better)
            "date_diff",
            # Fifth priority: age difference (lower is better)
            "age_diff"
        ],
        descending=[False, True, True, True, False, False]
    )
    .group_by("DONOR_ID")
    .agg(pl.all().first())
)

# Validation
print(f"\nTotal unique donors in input: {features_with_donor['DONOR_ID'].n_unique()}")
print(f"Total unique donors in output: {best_matches_df['DONOR_ID'].n_unique()}")
print(f"\nMatches with WLST: {best_matches_df.filter(pl.col('wlst_type').is_not_null()).height}")  # Changed to wlst_type
print(f"Matches with is_dead=1: {best_matches_df.filter(pl.col('is_dead') == 1).height}")
print(f"Donors that had multiple matches: {best_matches_df.filter(pl.col('n_matches_per_donor') > 1).height}")

# Check specific cases where multiple matches existed
multi_match_summary = best_matches_df.filter(pl.col('n_matches_per_donor') > 1)
print(f"\nFor donors with multiple matches:")
print(f"  Selected with WLST: {multi_match_summary.filter(pl.col('wlst_type').is_not_null()).height}")  # Changed to wlst_type
print(f"  Selected without WLST: {multi_match_summary.filter(pl.col('wlst_type').is_null()).height}")  # Changed to wlst_type

best_matches_df.columns

################################################################################
# 6. Process wide_df
################################################################################

COLS_TO_KEEP = [
 'hospitalization_id',
 'event_dttm',
 'med_cont_dobutamine',
 'med_cont_dopamine',
 'med_cont_epinephrine',
 'med_cont_fentanyl',
 'med_cont_hydromorphone',
 'med_cont_morphine',
 'med_cont_norepinephrine',
 'med_cont_phenylephrine',
 'med_cont_isoproterenol',
 'med_cont_milrinone',
 'med_cont_propofol',
 'med_cont_remifentanil',
 'med_cont_vasopressin',
 'med_int_fentanyl',
 'med_int_morphine',
 'med_int_hydromorphone',
 'resp_device_category',
 'resp_fio2_set',
 'lab_po2_arterial',
 'crrt_mode_category'
]

existing_cols = set(wide_df.columns)
missing_cols = [col for col in COLS_TO_KEEP if col not in existing_cols]

# Create null columns for any missing columns
if missing_cols:
    wide_df = wide_df.with_columns([
        pl.lit(None).alias(col) for col in missing_cols
    ])
# Now safely select all columns
wide_df_filtered = wide_df.select(COLS_TO_KEEP)

mapping = (
    encounter_mapping_df
    .select([
        "hospitalization_id",
        "encounter_block"
    ])
    .unique()
)
# attach death info from clif
death_info = final_clif_df.select([
    "encounter_block",
    "is_dead",
    "final_outcome_dttm",
])

mapping = mapping.join(
    death_info,
    on="encounter_block",
    how="left"
)
wide_df_filtered = (
    mapping
    .join(
        wide_df_filtered,
        on="hospitalization_id",
        how="left"
    )
    .with_columns([
        pl.col("event_dttm").cast(pl.Datetime),
        pl.col("final_outcome_dttm").cast(pl.Datetime),
    ])
)

# ============================================================
# Medications to include
# ============================================================
MEDS = [
    "dobutamine",
    "dopamine",
    "epinephrine",
    "fentanyl",
    "hydromorphone",
    "morphine",
    "norepinephrine",
    "phenylephrine",
    "propofol",
    "remifentanil",
    "vasopressin",
    "milrinone",
    "isoproterenol"  
]

OPIOIDS = {
    "fentanyl",
    "hydromorphone",
    "morphine",
    "remifentanil",
}

VASOPRESSORS = {
    "dopamine",
    "epinephrine",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
}

INOTROPES = {
    "dobutamine",
    "milrinone",
    ## adding others because these are marked as inotropes in srtr
    "dopamine",
    "epinephrine",
    "norepinephrine",
    "phenylephrine",
    "isoproterenol",
}

# ============================================================
# 0. START FROM wide_df_filtered
# ============================================================
df = wide_df_filtered.with_columns([
    pl.col("event_dttm").cast(pl.Datetime),
    pl.col("final_outcome_dttm").cast(pl.Datetime),
])

# ============================================================
# 1. MEDICATION EVER FLAGS (continuous + intermittent)
# ============================================================
ever_exprs = []

for m in MEDS:
    cont_col = f"med_cont_{m}"
    int_col  = f"med_int_{m}"

    cont_exists = cont_col in df.columns
    int_exists  = int_col in df.columns

    if cont_exists and int_exists:
        expr = (
            pl.col(cont_col).is_not_null() |
            pl.col(int_col).is_not_null()
        )
    elif cont_exists:
        expr = pl.col(cont_col).is_not_null()
    elif int_exists:
        expr = pl.col(int_col).is_not_null()
    else:
        expr = pl.lit(False)

    ever_exprs.append(
        expr.any().alias(f"{m}_ever")
    )

df_ever = (
    df.group_by("encounter_block")
      .agg(ever_exprs)
)

# ------------------------------------------------------------
# Ensure ALL <med>_ever columns exist
# ------------------------------------------------------------
for m in MEDS:
    col = f"{m}_ever"
    if col not in df_ever.columns:
        df_ever = df_ever.with_columns(pl.lit(False).alias(col))

# ============================================================
# 2. MEDICATION CLASS FLAGS
# ============================================================
df_ever = df_ever.with_columns([
    pl.any_horizontal(
        [pl.col(f"{m}_ever") for m in OPIOIDS if f"{m}_ever" in df_ever.columns]
    ).alias("any_opioid_ever"),

    pl.any_horizontal(
        [pl.col(f"{m}_ever") for m in VASOPRESSORS if f"{m}_ever" in df_ever.columns]
    ).alias("any_vasopressor_ever"),

    pl.any_horizontal(
        [pl.col(f"{m}_ever") for m in INOTROPES if f"{m}_ever" in df_ever.columns]
    ).alias("any_inotrope_ever"),
])

# ============================================================
# 3. CRRT EVER
# ============================================================
df_crrt = (
    df.group_by("encounter_block")
      .agg(
          pl.col("crrt_mode_category")
            .is_not_null()
            .any()
            .alias("crrt_ever")
      )
)

# ============================================================
# 4. PF RATIO (last valid before outcome)
# ============================================================
# Check if mean FiO2 > 1 to determine format
fio2_mean = df.select(pl.col("resp_fio2_set").mean()).item()
if fio2_mean > 1:
    # Convert percentage to fraction
    df = df.with_columns(
        (pl.col("resp_fio2_set") / 100.0).alias("resp_fio2_set")
    )
else:
    # Already in fraction format
    pass

pf_df = (
    df
    .sort([ "encounter_block", "event_dttm"])
    .with_columns(
        pl.col("resp_fio2_set")
          .cast(pl.Float64)
          .forward_fill()
          .over([ "encounter_block"])
          .alias("fio2_ffill")
    )
    .with_columns(
        (
            pl.when(
                pl.col("lab_po2_arterial").is_not_null() &
                pl.col("fio2_ffill").is_not_null() &
                (pl.col("fio2_ffill") > 0)
            )
            .then(pl.col("lab_po2_arterial") / pl.col("fio2_ffill"))
            .otherwise(None)
        ).alias("pf_ratio")
    )
    .filter(
        (pl.col("event_dttm") <= pl.col("final_outcome_dttm")) |
        pl.col("final_outcome_dttm").is_null()
    )
)

df_pf = (
    pf_df
    .group_by([ "encounter_block"])
    .agg(
        pl.col("pf_ratio")
          .filter(pl.col("pf_ratio").is_not_null())
          .last()
          .alias("last_pf_ratio")
    )
)

# ============================================================
# 5. FINAL MERGE
# ============================================================
features_final = (
    df_ever
    .join(df_crrt, on=[ "encounter_block"], how="left")
    .join(df_pf, on=["encounter_block"], how="left")
)

# Optional: fill null booleans
features_final = features_final.with_columns([
    pl.col(c).fill_null(False)
    for c in features_final.columns
    if c.endswith("_ever") or c == "crrt_ever"
])

# ============================================================
# RESULT
# ============================================================
features_final.columns

final_clif_df = (
    final_clif_df
    .join(
        features_final,
        on=["encounter_block"],
        how="left"
    )
)

################################################################################
# 7. Generate table ones
################################################################################

# Step 5: Create Table One for each confidence level
print("\n" + "="*80)
print("GENERATING TABLE ONE BY CONFIDENCE")
print("="*80)

import importlib
import create_tableone_by_confidence
importlib.reload(create_tableone_by_confidence)
from create_tableone_by_confidence import create_table_one_by_confidence, create_table_one_dcd_only
for confidence in ["HIGH", "MEDIUM", "LOW"]:
    print(f"\n📊 Creating Table One for {confidence} confidence...")

    # Standard Table One (no p-values)
    table_one = create_table_one_by_confidence(
        patients_df=final_clif_df,
        donors_df=srtr_df,
        best_matches_df=best_matches_df,
        confidence_level=confidence
    )

    output_path = OUTPUT_FINAL_DIR / f"table_{confidence.lower()}_confidence.csv"
    table_one.to_csv(output_path, index=False,  encoding="utf-8-sig")
    json_path = OUTPUT_FINAL_DIR / f"table_{confidence.lower()}_confidence.json"
    table_one.to_json(json_path, orient="records", indent=2)
    print(f"   → Saved: {output_path.name}")

    # DCD-only Table One
    table_one_dcd = create_table_one_dcd_only(
        patients_df=final_clif_df,
        donors_df=srtr_df,
        best_matches_df=best_matches_df,
        confidence_level=confidence
    )

    output_path_dcd = OUTPUT_FINAL_DIR / f"table_dcd_{confidence.lower()}_confidence.csv"
    table_one_dcd.to_csv(output_path_dcd, index=False,  encoding="utf-8-sig")
    json_path_dcd = OUTPUT_FINAL_DIR / f"table_dcd_{confidence.lower()}_confidence.json"
    table_one_dcd.to_json(json_path_dcd, orient="records", indent=2)
    print(f"   → Saved: {output_path_dcd.name}")

################################################################################
# 8. Generate Table One with Agreement Metrics
################################################################################

import importlib
import create_tableone_by_confidence
importlib.reload(create_tableone_by_confidence)
from create_tableone_by_confidence import create_table_one_with_agreement, create_table_one_dcd_only_with_agreement

for confidence in ["HIGH", "MEDIUM", "LOW"]:
    print(f"\n📊 Creating Table One for {confidence} confidence...")

    # Standard Table One (no p-values)
    table_one = create_table_one_with_agreement(
        patients_df=final_clif_df,
        donors_df=srtr_df,
        best_matches_df=best_matches_df,
        confidence_level=confidence
    )

    output_path = OUTPUT_FINAL_DIR / f"table_{confidence.lower()}_confidence_agreement.csv"
    table_one.to_csv(output_path, index=False,  encoding="utf-8-sig")
    json_path = OUTPUT_FINAL_DIR / f"table_{confidence.lower()}_confidence_agreement.json"
    table_one.to_json(json_path, orient="records", indent=2)
    print(f"   → Saved: {output_path.name}")

    # DCD-only Table One
    table_one_dcd = create_table_one_dcd_only_with_agreement(
        patients_df=final_clif_df,
        donors_df=srtr_df,
        best_matches_df=best_matches_df,
        confidence_level=confidence
    )

    output_path_dcd = OUTPUT_FINAL_DIR / f"table_dcd_{confidence.lower()}_confidence_agreement.csv"
    table_one_dcd.to_csv(output_path_dcd, index=False,  encoding="utf-8-sig")
    json_path_dcd = OUTPUT_FINAL_DIR / f"table_dcd_{confidence.lower()}_confidence_agreement.json"
    table_one_dcd.to_json(json_path_dcd, orient="records", indent=2)
    print(f"   → Saved: {output_path_dcd.name}")

################################################################################
# 9. DCD Analysis
################################################################################

print("\n" + "="*80)
print(" DCD ANALYSIS")
print("="*80)
import importlib
import record_linkage.dcd_analysis
importlib.reload(record_linkage.dcd_analysis)
from record_linkage.dcd_analysis import (
    create_dcd_donor_summary,
    create_dcd_four_population_comparison
)

# 1. DCD Donor Summary - All Donors
print("\nCreating Donor Summary (All Donors)...")
all_donor_summary = create_dcd_donor_summary(
    srtr_df=srtr_df,
    patients_df=final_clif_df,
    best_matches_df=best_matches_df,
    dcd_only=False
)

output_path = OUTPUT_FINAL_DIR / "all_donor_summary.csv"
all_donor_summary.to_csv(output_path, index=False, encoding="utf-8-sig")
json_path = OUTPUT_FINAL_DIR / "all_donor_summary.json"
all_donor_summary.to_json(json_path, orient="records", indent=2)
print(f"   → Saved: {output_path.name}")

# Display first 20 rows
print("\nAll Donor Summary (first 20 rows):")
print(all_donor_summary.head(20).to_string(index=False))

# 2. DCD Donor Summary - DCD Only
print("\nCreating DCD Donor Summary (DCD Only)...")
dcd_summary = create_dcd_donor_summary(
    srtr_df=srtr_df,
    patients_df=final_clif_df,
    best_matches_df=best_matches_df,
    dcd_only=True
)

output_path = OUTPUT_FINAL_DIR / "dcd_donor_summary.csv"
dcd_summary.to_csv(output_path, index=False, encoding="utf-8-sig")
json_path = OUTPUT_FINAL_DIR / "dcd_donor_summary.json"
dcd_summary.to_json(json_path, orient="records", indent=2)
print(f"   → Saved: {output_path.name}")

# Display summary
print("\nDCD Donor Summary:")
print(dcd_summary.to_string(index=False))

# 2. Four Population Comparison
print("\nCreating DCD Four Population Comparison...")
dcd_comparison = create_dcd_four_population_comparison(
    srtr_df=srtr_df,
    patients_df=final_clif_df,
    best_matches_df=best_matches_df
)

output_path = OUTPUT_FINAL_DIR / "dcd_four_population_comparison.csv"
dcd_comparison.to_csv(output_path, index=False, encoding="utf-8-sig")
json_path = OUTPUT_FINAL_DIR / "dcd_four_population_comparison.json"
dcd_comparison.to_json(json_path, orient="records", indent=2)
print(f"   → Saved: {output_path.name}")

# Display first few rows
print("\nFour Population Comparison (first 10 rows):")
print(dcd_comparison.head(10).to_string(index=False))

################################################################################
# 10. Fentanyl & Morphine analysis
################################################################################

def create_analgesic_visualization(
    df,
    features_with_donor,
    all_matches_df,
    srtr_df,
    drug_name="fentanyl",
):
    """
    Create WLST-centered analgesic visualizations (DCD only):
      - ALL
      - Stratified by donor utilization
      - Stratified by donor utilization × death
    Saves one CSV + PNG per stratum.
    """

    # ------------------------------------------------------------
    # Helper: compute hourly summary
    # ------------------------------------------------------------
    def compute_hourly_viz(dcd_wlst_df):
        hourly_data = {h: [] for h in range(-12, 13)}
        hourly_patients = {h: set() for h in range(-12, 13)}

        drug_col = f"med_cont_{drug_name}"

        for row in dcd_wlst_df.iter_rows(named=True):
            enc_block = row["encounter_block"]
            wlst_time = row["wlst_dttm"]

            enc_events = df.filter(
                (pl.col("encounter_block") == enc_block)
                & (pl.col("event_dttm") >= wlst_time - timedelta(hours=12))
                & (pl.col("event_dttm") <= wlst_time + timedelta(hours=12))
            )

            if drug_col not in enc_events.columns:
                continue

            enc_events = enc_events.with_columns(
                ((pl.col("event_dttm") - pl.lit(wlst_time))
                 .dt.total_hours())
                .alias("hours_from_wlst"),
                pl.col(drug_col).alias("drug_dose"),
            )

            for ev in enc_events.filter(pl.col("drug_dose").is_not_null()).iter_rows(named=True):
                h = int(round(ev["hours_from_wlst"]))
                if -12 <= h <= 12:
                    hourly_data[h].append(ev["drug_dose"])
                    hourly_patients[h].add(enc_block)

        return pl.DataFrame([
            {
                "hour_from_wlst": h,
                "mean_dose": np.mean(hourly_data[h]) if hourly_data[h] else None,
                "sem": (
                    np.std(hourly_data[h]) / np.sqrt(len(hourly_data[h]))
                    if len(hourly_data[h]) > 1 else 0.0
                ),
                "n_patients": len(hourly_patients[h]),
                "n_doses": len(hourly_data[h]),
            }
            for h in range(-12, 13)
        ])

    # ------------------------------------------------------------
    # Base cohort: DCD + WLST
    # ------------------------------------------------------------
    base = (
        features_with_donor
        .filter(pl.col("wlst_dttm").is_not_null())
        .join(
            all_matches_df.select(["encounter_block", "DONOR_ID"]).unique(),
            on="encounter_block",
        )
        .join(
            srtr_df.select(["DONOR_ID", "DON_NON_HR_BEAT"]),
            on="DONOR_ID",
        )
        .filter(pl.col("DON_NON_HR_BEAT") == "Y")
        .select(
            ["encounter_block", "wlst_dttm", "don_utilized", "is_dead"]
        )
        .with_columns(
            pl.col("don_utilized")
              .cast(pl.Utf8)
              .str.to_uppercase()
              .alias("don_utilized")
        )
    )

    # ------------------------------------------------------------
    # Define strata
    # ------------------------------------------------------------
    strata = {
        "ALL": base,

        "UTILIZED_Y": base.filter(pl.col("don_utilized") == "Y"),
        "UTILIZED_N": base.filter(pl.col("don_utilized") == "N"),

        "UTILIZED_Y_DEAD_Y": base.filter(
            (pl.col("don_utilized") == "Y") & (pl.col("is_dead") == 1)
        ),
        "UTILIZED_Y_DEAD_N": base.filter(
            (pl.col("don_utilized") == "Y") & (pl.col("is_dead") == 0)
        ),
        "UTILIZED_N_DEAD_Y": base.filter(
            (pl.col("don_utilized") == "N") & (pl.col("is_dead") == 1)
        ),
        "UTILIZED_N_DEAD_N": base.filter(
            (pl.col("don_utilized") == "N") & (pl.col("is_dead") == 0)
        ),
    }

    outputs = {}

    # ------------------------------------------------------------
    # Loop over strata → CSV + figure
    # ------------------------------------------------------------
    for label, cohort in strata.items():
        if cohort.height == 0:
            continue

        viz_df = compute_hourly_viz(cohort)

        # ---- Save CSV
        csv_path = OUTPUT_FINAL_DIR / f"{drug_name}_wlst_dcd_{label.lower()}.csv"
        viz_df.write_csv(csv_path)

        # ---- Plot
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.set_xlabel("Hours from WLST")
        ax.set_ylabel(f"Average {drug_name.capitalize()} Dose")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="WLST")
        ax.axvspan(-12, 0, alpha=0.1, color="blue")
        ax.axvspan(0, 12, alpha=0.1, color="orange")

        df_plot = viz_df.filter(pl.col("mean_dose").is_not_null())
        if df_plot.height > 0:
            ax.errorbar(
                df_plot["hour_from_wlst"].to_list(),
                df_plot["mean_dose"].to_list(),
                yerr=df_plot["sem"].to_list(),
                marker="o",
                linewidth=2,
                capsize=5,
            )

        ax.set_title(
            f"{drug_name.capitalize()} Around WLST (DCD) — {label.replace('_', ' ')}"
        )
        ax.grid(alpha=0.3)

        # ---- Patient counts below x-axis (robust)
        ax.set_ylim(auto=True)
        y_min, y_max = ax.get_ylim()
        y_text = y_min - 0.08 * (y_max - y_min)

        for h in range(-12, 13):
            n_pat = viz_df.filter(pl.col("hour_from_wlst") == h)["n_patients"][0]
            if n_pat > 0:
                ax.text(
                    h,
                    y_text,
                    f"n={n_pat}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="gray",
                    clip_on=False,
                )

        ax.text(
            0,
            y_text - 0.04 * (y_max - y_min),
            "Patients per hour",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="gray",
            clip_on=False,
        )

        # ---- Overall N
        total_n = cohort.select("encounter_block").n_unique()
        ax.text(
            0.98,
            0.95,
            f"Total n = {total_n}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # ---- Save figure
        fig_path = OUTPUT_FINAL_DIR / f"{drug_name}_wlst_dcd_{label.lower()}.png"
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        outputs[label] = viz_df

    return outputs

# Create enhanced fentanyl visualization
print("\nCreating enhanced fentanyl visualization...")
fentanyl_viz_df = create_analgesic_visualization(df, features_with_donor, all_matches_df, srtr_df, "fentanyl")

# Create enhanced morphine visualization
print("Creating enhanced morphine visualization...")
morphine_viz_df = create_analgesic_visualization(df, features_with_donor, all_matches_df, srtr_df, "morphine")

################################################################################
# 11. WLST vs Agonal Times
################################################################################

dcd_donors = srtr_df.filter(pl.col("DON_NON_HR_BEAT") == "Y")
dcd_patients = best_matches_df.filter(pl.col("DON_NON_HR_BEAT") == "Y")
dcd_donors = dcd_donors.with_columns([
    pl.when(
        pl.col("DON_DCD_AGONAL_BEGIN_DT").is_not_null() &
        pl.col("DON_DCD_AGONAL_BEGIN_TM").is_not_null()
    )
    .then(
        # Parse date string → Date → Datetime
        pl.col("DON_DCD_AGONAL_BEGIN_DT")
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .cast(pl.Datetime)
        +
        # Add seconds since midnight
        pl.duration(
            seconds=pl.col("DON_DCD_AGONAL_BEGIN_TM")
        )

    )
    .otherwise(None)
    .alias("srtr_agonal_begin_dttm")
])

comparison_df = (
    dcd_patients
    .filter(pl.col("wlst_dttm").is_not_null())
    .join(
        dcd_donors.select(["DONOR_ID", "srtr_agonal_begin_dttm", "DON_NON_HR_BEAT"]),
        on="DONOR_ID",
        how="left"
    )
    .filter(pl.col("DON_NON_HR_BEAT") == "Y")  # DCD donors only
)

comparison_df = comparison_df.with_columns([
    # Time difference in hours
    (pl.col("wlst_dttm") - pl.col("srtr_agonal_begin_dttm"))
    .dt.total_hours()
    .alias("time_diff_hours"),

    # Absolute time difference
    (pl.col("wlst_dttm") - pl.col("srtr_agonal_begin_dttm"))
    .dt.total_hours()
    .abs()
    .alias("abs_time_diff_hours"),

    # Time difference in days
    (pl.col("wlst_dttm") - pl.col("srtr_agonal_begin_dttm"))
    .dt.total_days()
    .alias("time_diff_days")
])

print(f"\nComparison Dataset:")
print(f"  Total DCD matches with WLST: {comparison_df.height}")
print(f"  With both timestamps: {comparison_df.filter(pl.col('srtr_agonal_begin_dttm').is_not_null()).height}")

def calculate_statistics(df: pl.DataFrame, label: str = "Overall") -> dict:
    """
    Calculate summary statistics for time differences.

    Args:
        df: DataFrame with time_diff_hours column
        label: Label for this analysis subset

    Returns:
        Dictionary of statistics
    """
    valid_df = df.filter(
        pl.col("srtr_agonal_begin_dttm").is_not_null() &
        pl.col("wlst_dttm").is_not_null()
    )

    if valid_df.height == 0:
        return {
            "label": label,
            "n_comparisons": 0,
            "message": "No valid comparisons available"
        }

    # Convert to pandas for easier statistics
    valid_pd = valid_df.to_pandas()
    time_diffs = valid_pd["time_diff_hours"].dropna()
    abs_time_diffs = valid_pd["abs_time_diff_hours"].dropna()

    stats = {
        "label": label,
        "n_comparisons": len(time_diffs),
        "median_diff_hours": time_diffs.median(),
        "mean_diff_hours": time_diffs.mean(),
        "std_diff_hours": time_diffs.std(),
        "min_diff_hours": time_diffs.min(),
        "max_diff_hours": time_diffs.max(),
        "q25_diff_hours": time_diffs.quantile(0.25),
        "q75_diff_hours": time_diffs.quantile(0.75),
        "within_1_hour": (abs_time_diffs <= 1).sum(),
        "within_1_hour_pct": (abs_time_diffs <= 1).sum() / len(abs_time_diffs) * 100,
        "within_6_hours": (abs_time_diffs <= 6).sum(),
        "within_6_hours_pct": (abs_time_diffs <= 6).sum() / len(abs_time_diffs) * 100,
        "within_24_hours": (abs_time_diffs <= 24).sum(),
        "within_24_hours_pct": (abs_time_diffs <= 24).sum() / len(abs_time_diffs) * 100,
        "within_48_hours": (abs_time_diffs <= 48).sum(),
        "within_48_hours_pct": (abs_time_diffs <= 48).sum() / len(abs_time_diffs) * 100,
    }

    return stats

# Overall statistics
overall_stats = calculate_statistics(comparison_df, "Overall")
print(f"\nOverall Statistics:")
print(f"  N with both timestamps: {overall_stats.get('n_comparisons', 0)}")
if overall_stats.get('n_comparisons', 0) > 0:
    print(f"  Median difference: {overall_stats['median_diff_hours']:.1f} hours")
    print(f"  Mean difference: {overall_stats['mean_diff_hours']:.1f} ± {overall_stats['std_diff_hours']:.1f} hours")
    print(f"  Within ±1 hour: {overall_stats['within_1_hour_pct']:.1f}%")
    print(f"  Within ±6 hours: {overall_stats['within_6_hours_pct']:.1f}%")
    print(f"  Within ±24 hours: {overall_stats['within_24_hours_pct']:.1f}%")

# Statistics by confidence level
all_stats = [overall_stats]
for confidence in ['HIGH', 'MEDIUM', 'LOW']:
    subset = comparison_df.filter(pl.col("confidence") == confidence)
    stats = calculate_statistics(subset, f"{confidence} Confidence")
    all_stats.append(stats)

    print(f"\n{confidence} Confidence:")
    print(f"  N with both timestamps: {stats.get('n_comparisons', 0)}")
    if stats.get('n_comparisons', 0) > 0:
        print(f"  Median difference: {stats['median_diff_hours']:.1f} hours")
        print(f"  Within ±1 hour: {stats['within_1_hour_pct']:.1f}%")

# Define output path
output_file = Path(OUTPUT_FINAL_DIR) / "wlst_time_agreement_stats.json"

# Make sure all values are JSON-serializable
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    else:
        return obj

# Convert stats
json_data = make_json_safe(all_stats)

# Write to file
with open(output_file, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✓ Agreement statistics saved to: {output_file}")

import seaborn as sns
def create_boxplot_by_confidence(
    comparison_df: pl.DataFrame,
    output_dir: Path
):
    """
    Create and save box plot of time differences by confidence level.

    Args:
        comparison_df: Polars DataFrame with time comparisons
        output_dir: Directory to save plot
    """

    # Filter valid rows and convert to pandas
    plot_df = (
        comparison_df
        .filter(pl.col("time_diff_hours").is_not_null())
        .to_pandas()
    )

    if len(plot_df) == 0:
        print("No valid data for box plot")
        return

    # Prepare data
    confidence_levels = ["HIGH", "MEDIUM", "LOW"]
    colors = ["green", "orange", "red"]

    data = []
    labels = []

    for conf in confidence_levels:

        conf_data = (
            plot_df
            .loc[plot_df["confidence"] == conf, "time_diff_hours"]
            .dropna()
        )

        if len(conf_data) > 0:
            data.append(conf_data)
            labels.append(f"{conf}\n(n={len(conf_data)})")

    if not data:
        print("No confidence-level data available")
        return

    # Create figure
    plt.figure(figsize=(8, 6))

    bp = plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Reference line
    plt.axhline(
        0,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="Perfect Agreement"
    )

    # Labels
    plt.ylabel("Time Difference (hours)")
    plt.title("Time Differences by Match Confidence")

    # Save
    output_file = output_dir / "time_diff_boxplot_by_confidence.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved box plot to: {output_file}")

create_boxplot_by_confidence(comparison_df, OUTPUT_FINAL_DIR)
