#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generated from: 01_record_linkage_dev.ipynb

This script was automatically generated from a Jupyter notebook.
Markdown cells have been converted to comments.
"""

################################################################################
# Hierarchical Patient-Donor Matching Algorithm
# This notebook implements the hierarchical matching strategy to link CLIF patient records with SRTR donor records.
# The algorithm stops searching for each donor once exactly one validated match is found.
################################################################################

################################################################################
# 1. Setup and Imports
################################################################################

import polars as pl
import pandas as pd
import duckdb
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
from clifpy.utils.stitching_encounters import stitch_encounters
from utils.outlier_handler import apply_outlier_handling
import gc

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
# 2. Data
################################################################################

################################################################################
# SRTR Data
################################################################################

donor_ids_path = "../shared/donor_deceased_site_filtered_clif.csv"
donor_ids_df = pd.read_csv(donor_ids_path)
# Filter to only rows where clif_site matches site_name
donor_deceased_site_filtered = donor_ids_df[
    (donor_ids_df['clif_site'] == site_name) & (donor_ids_df['cliffed'] == "Y")
]
donor_deceased_site_filtered['DONOR_ID'] = donor_deceased_site_filtered['DONOR_ID'].astype('int32')
# Print count and ensure uniqueness
print(f"Total adult donors between 2018-20114 at '{site_name}': {len(donor_deceased_site_filtered):,}")

output_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, f"donor_deceased_site_filtered_{site_name}.csv")
donor_deceased_site_filtered.to_csv(output_path, index=False)
print(f"Saved donor_deceased_site_filtered to {output_path}")

# Convert DON_RECOV_DT to datetime if not already
donor_deceased_site_filtered['DON_RECOV_DT'] = pd.to_datetime(donor_deceased_site_filtered['DON_RECOV_DT'], errors='coerce')

# Filter donor_deceased_site_filtered to include only rows where DON_RECOV_DT is between 2018 and 2024 
# Sanity check - ideally all ids from SRTR are for adults between 2018-2024
donor_deceased_site_filtered = donor_deceased_site_filtered[
    (donor_deceased_site_filtered['DON_RECOV_DT'] >= '2018-01-01') &
    (donor_deceased_site_filtered['DON_RECOV_DT'] <= '2024-12-31') &
    (donor_deceased_site_filtered["DON_AGE"] >= 18) ].copy()
print(f"Filtered donor_deceased_site_filtered to dates between 2018 and 2024 and all adult donors. New shape: {donor_deceased_site_filtered.shape}")

# Print number of unique donors in the filtered dataframe
num_unique_donors = donor_deceased_site_filtered['DONOR_ID'].nunique()
print(f"Number of unique donors in filtered data: {num_unique_donors}")

################################################################################
# SRTR data donor summary
################################################################################

def fmt_median_iqr(s, digits=2):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "NA"
    return f"{s.median():.{digits}f} [{s.quantile(0.25):.{digits}f}, {s.quantile(0.75):.{digits}f}]"

def fmt_min_max_date(s):
    s = pd.to_datetime(s, errors="coerce").dropna()
    if s.empty:
        return "NA"
    return f"{s.min().date()} – {s.max().date()}"

def fmt_n_pct(s):
    """
    Format binary flags as N (%), robust to 0/1, Y/N, Yes/No, True/False.
    """
    if s is None:
        return "NA"

    s = s.copy()

    # Normalize strings
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.strip()
             .str.upper()
             .replace({
                 "Y": 1, "YES": 1, "TRUE": 1, "T": 1,
                 "N": 0, "NO": 0, "FALSE": 0, "F": 0,
                 "": np.nan, "NA": np.nan, "NAN": np.nan
             })
        )

    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    n = int(s.sum())
    denom = len(s)

    return f"{n} ({100*n/denom:.1f}%)"

labs = [
    "DON_CREAT","DON_FINAL_SERUM_CREAT","DON_PEAK_SERUM_CREAT",
    "DON_BUN","DON_TOT_BILI","DON_SGOT","DON_SGPT",
    "DON_SODIUM","DON_INR","DON_PH","DON_PO2","DON_PO2_FIO2","DON_PCO2",
    "DON_TROPONIN_I","DON_TROPONIN_T"
]

ino_flags = [
    "INO_MED_DOPAMINE","INO_MED_DOPUTAMINE","INO_MED_EPINEPHRINE",
    "INO_MED_LEVOPHED","INO_MED_NEOSYNEPHRINE","INO_MED_ISOPROTERENOL", 
    'DON_DOPAMINE', 'DON_DOBUTAMINE', 'DON_ARGININE',
]

df = donor_deceased_site_filtered.copy()

summary_rows = []

# --- N donors
summary_rows.append(("N donors", len(df)))

# --- Age
summary_rows.append(("Donor age (years)", fmt_median_iqr(df["DON_AGE"])))
summary_rows.append(("Donor age (months)", fmt_median_iqr(df["DON_AGE_IN_MONTHS"])))

# --- Race / Ethnicity (SRTR)
summary_rows.append(("Race (SRTR)", df["DON_RACE_SRTR"].value_counts(dropna=False).to_dict()))
summary_rows.append(("Ethnicity (SRTR)", df["DON_ETHNICITY_SRTR"].value_counts(dropna=False).to_dict()))

# --- Height / Weight
summary_rows.append(("Height (cm)", fmt_median_iqr(df["DON_HGT_CM"])))
summary_rows.append(("Weight (kg)", fmt_median_iqr(df["DON_WGT_KG"])))

# --- Recovery date
summary_rows.append(("Recovery date range", fmt_min_max_date(df["DON_RECOV_DT"])))

# --- Labs
for lab in labs:
    summary_rows.append((lab, fmt_median_iqr(df[lab])))

# --- Inotrope flags (binary)
summary_rows.append(("Any inotrope support (DON_INOTROP_SUPPORT)", fmt_n_pct(df["DON_INOTROP_SUPPORT"])))

for flag in ino_flags:
    summary_rows.append((flag, fmt_n_pct(df[flag])))

# --- DCD / non-heart-beating donor
summary_rows.append(("Non-heart-beating donor (DCD)", fmt_n_pct(df["DON_NON_HR_BEAT"])))

# --- Donor utilized
summary_rows.append(("Donor utilized", fmt_n_pct(df["don_utilized"])))

summary_df = pd.DataFrame(summary_rows, columns=["Variable", "Summary"])

summary_df

summary_df.to_csv(
    os.path.join(OUTPUT_FINAL_DIR, f"donor_summary_{site_name}.csv"),
    index=False
)

################################################################################
# CLIF data
################################################################################

# read required tables
adt_filepath = f"{tables_path}/clif_adt.{file_type}"
hospitalization_filepath = f"{tables_path}/clif_hospitalization.{file_type}"
patient_filepath = f"{tables_path}/clif_patient.{file_type}"
adt_df = read_data(adt_filepath, file_type)
hospitalization_df = read_data(hospitalization_filepath, file_type)
patient_df = read_data(patient_filepath, file_type)

################################################################################
# 2.1 Stitch ALL encounters
################################################################################

# ------------------------------------------------------------
# SITE-SPECIFIC ENCOUNTER LOGIC
# ------------------------------------------------------------
if site_name.lower() == "umn":

    print("Multi hospital site detected — using hospitalizations_joined_id as encounter_block")

    # -----------------------------
    # Hospitalization-level encounter definition
    # -----------------------------
    hospitalization_cols = [
        "patient_id",
        "hospitalization_id",
        "hospitalizations_joined_id",
        "admission_dttm",
        "discharge_dttm",
        "age_at_admission",
        "admission_type_category",
        "discharge_category",
    ]

    # Create integer mapping for hospitalizations_joined_id
    id_mapping = (
        hospitalization_df
        .select("hospitalizations_joined_id")
        .unique()
        .sort("hospitalizations_joined_id")
        .with_row_index(
            name="encounter_block",
            offset=1  # Start from 1 instead of 0
        )
        .with_columns(
            pl.col("encounter_block").cast(pl.Int32)
        )
    )

    # Apply mapping to create hosp_stitched with integer encounter_block
    hosp_stitched = (
        hospitalization_df
        .select(hospitalization_cols)
        .join(
            id_mapping,
            on="hospitalizations_joined_id",
            how="left"
        )
        .drop("hospitalizations_joined_id")  # Drop the string ID
        .unique()
    )

    # ADT: map to encounter_block via hospitalizations_joined_id
    adt_stitched = (
        adt_df
        .join(
            hospitalization_df
            .select(["hospitalization_id", "hospitalizations_joined_id"])
            .unique()
            .join(
                id_mapping,
                on="hospitalizations_joined_id",
                how="left"
            )
            .drop("hospitalizations_joined_id"),
            on="hospitalization_id",
            how="left"
        )
        .with_columns(
            pl.col("encounter_block").cast(pl.Int32)
        )
    )

    # Encounter mapping 
    encounter_mapping = (
        hosp_stitched
        .select([
            "hospitalization_id",
            "encounter_block",
            "patient_id",
        ])
        .unique()
    )
else:

    print("running encounter stitching")

    # -----------------------------
    # ORIGINAL STITCHING LOGIC
    # -----------------------------
    hospitalization_cols = [
        "patient_id",
        "hospitalization_id",
        "admission_dttm",
        "discharge_dttm",
        "age_at_admission",
        "admission_type_category",
        "discharge_category",
    ]

    hospitalization_df = (
        hospitalization_df
        .select(hospitalization_cols)
        .sort(["hospitalization_id", "admission_dttm"])
        .unique()
    )

    adt_df = adt_df.sort(["hospitalization_id", "in_dttm"])

    hosp_stitched_pd, adt_stitched_pd, encounter_mapping_pd = stitch_encounters(
        hospitalization=hospitalization_df.to_pandas(),
        adt=adt_df.to_pandas(),
        time_interval=12,
    )

    hosp_stitched = pl.from_pandas(hosp_stitched_pd)
    adt_stitched = pl.from_pandas(adt_stitched_pd)
    encounter_mapping = pl.from_pandas(encounter_mapping_pd)

    hosp_stitched = hosp_stitched.with_columns(
        pl.col("encounter_block").cast(pl.Int32)
    )
    adt_stitched = adt_stitched.with_columns(
        pl.col("encounter_block").cast(pl.Int32)
    )
    encounter_mapping = encounter_mapping.with_columns(
        pl.col("encounter_block").cast(pl.Int32)
    )

# ------------------------------------------------------------
# COMMON POST-PROCESSING (ALL SITES)
# ------------------------------------------------------------

patient_cols = [
    "patient_id",
    "race_name",
    "race_category",
    "ethnicity_name",
    "ethnicity_category",
    "sex_name",
    "sex_category",
    "birth_date",
    "death_dttm",
]

patient_df = (
    patient_df
    .select(patient_cols)
    .unique()
    .sort("patient_id")
)

hosp_stitched = (
    hosp_stitched
    .join(patient_df, on="patient_id", how="left")
    .unique()
)

final_df = hosp_stitched.filter(
    pl.col("hospitalization_id")
      .is_in(adt_stitched["hospitalization_id"].unique())
)

gc.collect()

################################################################################
# 2.2 Final outcome dttm
# To identify final_outcome_dttm for each patient, we use death_dttm from the patient table. If death_dttm is null, then use the last_vital_dttm for the stitched hospitalization as the final_outcome_dttm
################################################################################

vitals_filepath = f"{tables_path}/clif_vitals.{file_type}"
all_hosp_ids = final_df.select('hospitalization_id').to_series().to_list()
vitals_df = read_data(
      vitals_filepath,
      file_type,
      filter_ids=all_hosp_ids,
      id_column='hospitalization_id'
  )

vitals_df = apply_outlier_handling(vitals_df, 'vitals')

# First, sort by recorded_dttm within each hospitalization_id
vitals_df = (
    vitals_df
    .sort(['hospitalization_id', 'recorded_dttm'])
)

# Get first and last recorded_dttm, plus last weight and height for each hospitalization
vitals_first_last = (
    vitals_df
    .group_by('hospitalization_id')
    .agg([
        pl.col('recorded_dttm').min().alias('first_recorded_vital_dttm'),
        pl.col('recorded_dttm').max().alias('last_recorded_vital_dttm'),

        # Get last recorded weight
        pl.col('vital_value')
            .filter(pl.col('vital_category') == 'weight_kg')
            .last()
            .alias('last_weight_kg'),

        # Get last recorded height
        pl.col('vital_value')
            .filter(pl.col('vital_category') == 'height_cm')
            .last()
            .alias('last_height_cm')
    ])
)

# Calculate BMI
vitals_first_last = vitals_first_last.with_columns(
    (pl.col('last_weight_kg') / ((pl.col('last_height_cm') / 100) ** 2)).alias('bmi')
)

# Join with final_df
final_df = final_df.join(vitals_first_last, on='hospitalization_id', how='left')

# Define final_outcome_dttm as death_dttm, if missing then last_recorded_vital_dttm
final_df = final_df.with_columns(
    pl.when(pl.col("death_dttm").is_not_null())
      .then(pl.col("death_dttm"))
      .otherwise(pl.col("last_recorded_vital_dttm"))
      .alias("final_outcome_dttm")
)

# create a new is_dead variable where discharge category is expired or death dttm not null
# Create a new is_dead variable: True if discharge_category is 'expired' (case-insensitive) or death_dttm is not null, else False
eligible_discharge_categories = ['expired']
final_df = final_df.with_columns(
    (
        pl.col('discharge_category').str.to_lowercase().is_in(eligible_discharge_categories) |
        pl.col('death_dttm').is_not_null()
    ).alias('is_dead')
)

################################################################################
# 2.3 Timezone check
################################################################################

print("\n" + "=" * 80)
print("DATETIME COLUMNS - BEFORE TIMEZONE CONVERSION")
print("=" * 80)

# Find all columns ending with '_dttm'
dttm_cols_check = [col for col in final_df.columns if col.endswith('_dttm')]

print(f"\nFound {len(dttm_cols_check)} datetime columns: {dttm_cols_check}\n")

for col in dttm_cols_check:
    if col in final_df.columns:
        # Get non-null sample
        sample = final_df.select(pl.col(col)).drop_nulls().head(3)
        
        if len(sample) > 0:
            # Convert to pandas to check timezone info
            sample_pd = sample.to_pandas()
            
            print(f"Column: {col}")
            print(f"  Dtype: {final_df[col].dtype}")
            print(f"  Sample values:")
            for idx, val in enumerate(sample_pd[col]):
                tz_info = f" (tz: {val.tzinfo})" if hasattr(val, 'tzinfo') else ""
                print(f"    [{idx}] {val}{tz_info}")
            print()

print("=" * 80 + "\n")

# Handle timezone conversion for all *_dttm variables
import pytz
from datetime import datetime, date as datetime_date

# Get timezone from config (already defined above)
timezone = config['timezone']
target_tz = pytz.timezone(timezone)

# Find all columns ending with '_dttm'
dttm_cols = [col for col in final_df.columns if col.endswith('_dttm')]

print(f"Found {len(dttm_cols)} datetime columns: {dttm_cols}")
print(f"Target timezone: {timezone}")

# Process each datetime column
for col in dttm_cols:
    # Check if column exists and has data
    if col in final_df.columns:
        # Get sample to check timezone
        sample = final_df.select(pl.col(col)).drop_nulls().head(1)
        
        if len(sample) > 0:
            # Convert to pandas to check timezone info
            sample_pd = sample.to_pandas()
            sample_value = sample_pd[col].iloc[0]
            
            # Check if it's a datetime with timezone info
            if hasattr(sample_value, 'tzinfo'):
                if sample_value.tzinfo is None:
                    # No timezone info - assume it's already in local timezone, just localize
                    print(f"  {col}: No timezone info, assuming already in {timezone}, adding timezone metadata")
                    final_df = final_df.with_columns(
                        pl.col(col).map_elements(
                            lambda x: target_tz.localize(
                                datetime.combine(x, datetime.min.time()) if isinstance(x, datetime_date) and not isinstance(x, datetime) else x
                            ) if x is not None else None,
                            return_dtype=pl.Datetime(time_unit='us', time_zone=timezone)
                        )
                    )
                elif str(sample_value.tzinfo) == 'UTC' or 'UTC' in str(sample_value.tzinfo):
                    # Already UTC - convert to target timezone
                    print(f"  {col}: UTC detected, converting to {timezone}")
                    final_df = final_df.with_columns(
                        pl.col(col).map_elements(
                            lambda x: (
                                datetime.combine(x, datetime.min.time()) if isinstance(x, datetime_date) and not isinstance(x, datetime) else x
                            ).astimezone(target_tz) if x is not None else None,
                            return_dtype=pl.Datetime(time_unit='us', time_zone=timezone)
                        )
                    )
                else:
                    # Has timezone but not UTC - verify it's in target timezone
                    current_tz = str(sample_value.tzinfo)
                    if current_tz != timezone:
                        print(f"  {col}: Warning - timezone is {current_tz}, expected {timezone}")
                        # Convert to target timezone
                        final_df = final_df.with_columns(
                            pl.col(col).map_elements(
                                lambda x: (
                                    datetime.combine(x, datetime.min.time()) if isinstance(x, datetime_date) and not isinstance(x, datetime) else x
                                ).astimezone(target_tz) if x is not None else None,
                                return_dtype=pl.Datetime(time_unit='us', time_zone=timezone)
                            )
                        )
                    else:
                        print(f"  {col}: Already in {timezone} timezone")
            else:
                # Not a datetime object - try to parse and convert
                print(f"  {col}: Converting string/other type to datetime in {timezone}")
                final_df = final_df.with_columns(
                    pl.col(col).str.to_datetime().dt.replace_time_zone("UTC").dt.convert_time_zone(timezone)
                )

# Create death_date variable from final_outcome_dttm
final_df = final_df.with_columns(
    pl.col('final_outcome_dttm').dt.date().alias('death_date')
)

print(f"\nCreated death_date column from final_outcome_dttm")

print("\n" + "=" * 80)
print("DATETIME COLUMNS - AFTER TIMEZONE CONVERSION")
print("=" * 80)

# Find all columns ending with '_dttm'
dttm_cols = [col for col in final_df.columns if col.endswith('_dttm')]

print(f"\nChecking {len(dttm_cols)} datetime columns: {dttm_cols}\n")

for col in dttm_cols:
    if col in final_df.columns:
        # Get non-null sample
        sample = final_df.select(pl.col(col)).drop_nulls().head(3)
        
        if len(sample) > 0:
            # Convert to pandas to check timezone info
            sample_pd = sample.to_pandas()
            
            print(f"Column: {col}")
            print(f"  Dtype: {final_df[col].dtype}")
            print(f"  Sample values:")
            for idx, val in enumerate(sample_pd[col]):
                tz_info = f" (tz: {val.tzinfo})" if hasattr(val, 'tzinfo') else ""
                print(f"    [{idx}] {val}{tz_info}")
            
            # Check if timezone is consistent
            if hasattr(sample_pd[col].iloc[0], 'tzinfo'):
                if sample_pd[col].iloc[0].tzinfo is not None:
                    print(f"  ✓ Timezone: {sample_pd[col].iloc[0].tzinfo}")
                else:
                    print(f"  ⚠ Timezone: None (timezone-naive)")
            print()

print("=" * 80 + "\n")

# Also check death_date column if it exists
if 'death_date' in final_df.columns:
    print("Death Date Column:")
    print(f"  Dtype: {final_df['death_date'].dtype}")
    death_sample = final_df.select('death_date').drop_nulls().head(3).to_pandas()
    print(f"  Sample values:")
    for idx, val in enumerate(death_sample['death_date']):
        print(f"    [{idx}] {val}")
    print()

# Calculate age in months using final_outcome_dttm and birth_date in final_df
final_df = final_df.with_columns(
    (
        (pl.col('final_outcome_dttm').cast(pl.Datetime) - pl.col('birth_date').cast(pl.Datetime))
        .dt.total_days()
        .truediv(30.44)   # approximate average month length
        .round(2)
        .alias('age_at_death_months')
    )
)
print("Added age_at_death_months column to final_df")

################################################################################
# 2.4 Duplicate Check
################################################################################

# confirm final_df unique by encounter_block
# Check initial state
initial_count = len(final_df)
initial_encounters = final_df.select(pl.col('encounter_block').n_unique()).item()

# Identify duplicate encounter_blocks
encounter_counts = final_df.group_by('encounter_block').agg(pl.len().alias('count'))
duplicate_encounters = encounter_counts.filter(pl.col('count') > 1)
n_duplicate_encounters = duplicate_encounters.height

if n_duplicate_encounters > 0:
    print(f"Found {n_duplicate_encounters} encounter_blocks with multiple rows")
    
    # For each encounter_block, aggregate values
    final_df = final_df.group_by('encounter_block').agg([
        # Take first value for these
        pl.col('patient_id').first(),
        pl.col('hospitalization_id').first(),
        pl.col('admission_dttm').min().alias('admission_dttm'),  # first admission
        pl.col('age_at_admission').first(),
        pl.col('admission_type_category').first(),
        pl.col('discharge_category').first(),
        
        # Take last value for these
        pl.col('discharge_dttm').max().alias('discharge_dttm'),  # last discharge
        pl.col('final_outcome_dttm').max().alias('final_outcome_dttm'),  # last final_outcome_dttm
        pl.col('death_date').max().alias('death_date'),  # last death_date
        pl.col('age_at_death_months').max().alias('age_at_death_months'),  # last death date in months
        pl.col('last_recorded_vital_dttm').max().alias('last_recorded_vital_dttm'),  # last recorded vital
        
        # Take first for birth_date
        pl.col('birth_date').min().alias('birth_date'),  # first birth_date
        
        # Handle conflicts: take first non-null, or most common
        pl.col('race_name').filter(pl.col('race_name').is_not_null()).first().alias('race_name'),
        pl.col('race_category').filter(pl.col('race_category').is_not_null()).first().alias('race_category'),
        pl.col('ethnicity_name').filter(pl.col('ethnicity_name').is_not_null()).first().alias('ethnicity_name'),
        pl.col('ethnicity_category').filter(pl.col('ethnicity_category').is_not_null()).first().alias('ethnicity_category'),
        pl.col('sex_name').filter(pl.col('sex_name').is_not_null()).first().alias('sex_name'),
        pl.col('sex_category').filter(pl.col('sex_category').is_not_null()).first().alias('sex_category'),
        
        # For numeric values, take last non-null
        pl.col('last_weight_kg').filter(pl.col('last_weight_kg').is_not_null()).last().alias('last_weight_kg'),
        pl.col('last_height_cm').filter(pl.col('last_height_cm').is_not_null()).last().alias('last_height_cm'),
        pl.col('bmi').filter(pl.col('bmi').is_not_null()).last().alias('bmi'),
        pl.col('is_dead').filter(pl.col('is_dead').is_not_null()).last().alias('is_dead'),
        # Take first for these
        pl.col('death_dttm').first(),
        pl.col('first_recorded_vital_dttm').first(),
        
    ])
    
    # Count rows dropped
    final_count = len(final_df)
    rows_dropped = initial_count - final_count
    
    print(f"\nDeduplication Summary:")
    print(f"  Initial rows: {initial_count:,}")
    print(f"  Final rows: {final_count:,}")
    print(f"  Rows dropped: {rows_dropped:,}")
    print(f"  Encounter blocks with duplicates: {n_duplicate_encounters:,}")
else:
    print("No duplicate encounter_blocks found. Data is already unique by encounter_block.")

################################################################################
# 2.5 Subset decedents
################################################################################

eligible_discharge_categories = ['expired']
final_df_deceased = final_df.filter(
    pl.col('discharge_category').str.to_lowercase().is_in(eligible_discharge_categories) |
    pl.col('death_dttm').is_not_null()
)

# Arrange columns 
first_cols = [
    'patient_id', 'hospitalization_id', 'encounter_block', 'age_at_admission',
    'discharge_category', 'death_dttm', 'final_outcome_dttm', 'sex_category',
    'race_category', 'ethnicity_category'
]
# Any remaining columns not listed above
rest_cols = [col for col in final_df_deceased.columns if col not in first_cols]
ordered_cols = first_cols + rest_cols

final_df_deceased = final_df_deceased.select(ordered_cols)

################################################################################
# 3. Helper Functions
################################################################################

def standardize_demographics(df, source='clif'):
    """
    Standardize race and ethnicity categories
    Keep original column names for CLIF, add DON_ prefix for SRTR
    """

    if source == 'clif':
        # CLIF mappings - keep original column names
        race_map = {
            'Black or African American': 'BLACK',
            'White': 'WHITE',
            'Asian': 'ASIAN',
            'American Indian or Alaska Native': 'NATIVE',
            'Native Hawaiian or Other Pacific Islander': 'PACIFIC',
            'Other': 'OTHER',
            'Unknown': 'UNKNOWN',
            None: 'UNKNOWN'
        }

        ethnicity_map = {
            'Hispanic': 'HISPANIC',
            'Non-Hispanic': 'NON-HISPANIC',
            'Unknown': 'UNKNOWN',
            None: 'UNKNOWN'
        }

        df = df.with_columns([
            pl.col('race_category').replace(race_map).fill_null('UNKNOWN').alias('race_std'),
            pl.col('ethnicity_category').replace(ethnicity_map).fill_null('UNKNOWN').alias('ethnicity_std')
        ])

    else:  # srtr
        # SRTR mappings - add DON_ prefix to standardized columns
        race_map = {
            'BLACK': 'BLACK',
            'WHITE': 'WHITE',
            'ASIAN': 'ASIAN',
            'AMIND': 'NATIVE',
            'NATIVE': 'NATIVE',
            'PACIFIC': 'PACIFIC',
            'MULTI': 'OTHER',
            'OTHER': 'OTHER',
            None: 'UNKNOWN'
        }

        ethnicity_map = {
            'LATINO': 'HISPANIC',
            'NLATIN': 'NON-HISPANIC',
            None: 'UNKNOWN'
        }

        df = df.with_columns([
            pl.col('DON_RACE_SRTR').replace(race_map).fill_null('UNKNOWN').alias('DON_race_std'),
            pl.col('DON_ETHNICITY_SRTR').replace(ethnicity_map).fill_null('UNKNOWN').alias('DON_ethnicity_std')
        ])

    return df

print("Standardization functions defined")

################################################################################
# 4. Load and Standardize Data
################################################################################

donor_deceased_site_filtered = pl.from_pandas(donor_deceased_site_filtered)
# Standardize gender values to match SRTR format (M/F)
gender_map = {
    'Male': 'M',
    'Female': 'F',
    'Unknown': 'U',
    'male': 'M',
    'female': 'F',
    'unknown': 'U',
}
final_df = final_df.with_columns([
    pl.col('sex_category').replace(gender_map).alias('gender')
])

# Ensure dates are in correct format
final_df = final_df.with_columns([
    pl.col('death_date').cast(pl.Datetime),
    pl.col('age_at_admission').cast(pl.Float64),
    pl.col('age_at_death_months').cast(pl.Float64)
])

donor_deceased_site_filtered = donor_deceased_site_filtered.with_columns([
    pl.col('DON_RECOV_DT').str.to_datetime().alias('DON_RECOV_DT') if donor_deceased_site_filtered['DON_RECOV_DT'].dtype == pl.Utf8 else pl.col('DON_RECOV_DT'),
    pl.col('DON_AGE').cast(pl.Float64)
])

# Standardize demographics
print("Standardizing patient demographics...")
final_df_std = standardize_demographics(final_df, source='clif')
final_df_deceased_std = standardize_demographics(final_df_deceased, source='clif')
print("Standardizing donor demographics...")
donor_deceased_site_filtered_std = standardize_demographics(donor_deceased_site_filtered, source='srtr')

print(f"\nData ready for matching:")
print(f"  Patients: {len(final_df_std):,}")
print(f"  Donors: {len(donor_deceased_site_filtered_std):,}")

################################################################################
# 6. Progressive Heirarchical Matching
################################################################################

patient_cols = {
    'id': 'patient_id',
    'block': 'encounter_block',
    'death_date': 'death_date',
    'age': 'age_at_death_months',
    'gender': 'gender',
    'race': 'race_std',
    'ethnicity': 'ethnicity_std',
    'is_dead':'is_dead',
    'discharge_dttm':'discharge_dttm'
}

donor_cols = {
    'id': 'DONOR_ID',
    'recovery_date': 'DON_RECOV_DT',
    'age': 'DON_AGE_IN_MONTHS',
    'gender': 'DON_GENDER',
    'race': 'DON_race_std',
    'ethnicity': 'DON_ethnicity_std'
}

import importlib
import progressive_matching
importlib.reload(progressive_matching)

all_matches_df, summary_df = progressive_matching.progressive_match_donors(
    final_df_std,
    donor_deceased_site_filtered_std,
    patient_cols,
    donor_cols,
    return_all_matches=True,  # Returns all candidates
    debug=True
)

donor_meta = (
    donor_deceased_site_filtered_std
    .select([
        pl.col(donor_cols["id"]).alias("DONOR_ID"),
        pl.col("don_utilized"),
        pl.col("DON_NON_HR_BEAT")
    ])
    .unique("DONOR_ID")
)

all_matches_df = (
        all_matches_df
        .join(
            donor_meta,
            on="DONOR_ID",
            how="left"
        )
    )

best_matches_df = (
    all_matches_df
    .sort(
        by=["DONOR_ID", "score", "confidence", "date_tolerance"],
        descending=[False, True, True, False]
    )
    .group_by("DONOR_ID")
    .agg(pl.all().first())
)

# Keep only encounter_blocks in matches_df
encounter_blocks_in_matches = all_matches_df["encounter_block"].unique()
encounter_mapping_filtered = encounter_mapping.filter(
    pl.col("encounter_block").is_in(encounter_blocks_in_matches)
)

# Join encounter_mapping_filtered with matches_df on encounter_block
encounter_mapping_matched = encounter_mapping_filtered.join(
    all_matches_df, on="encounter_block", how="inner"
)

################################################################################
# 7. Labs
################################################################################

import duckdb
import polars as pl
import pandas as pd

# ============================================================
# 1. Prepare cohort for DuckDB (pandas)
# ============================================================
# Convert final_outcome_dttm to timezone-naive by removing timezone info
final_df_std = final_df_std.with_columns(
    pl.col("final_outcome_dttm").dt.replace_time_zone(None).alias("final_outcome_dttm")
)
final_cohort_for_sql = (
    final_df_std
    .filter(
        pl.col("encounter_block")
        .is_in(encounter_mapping_matched["encounter_block"])
    )
    .select([
        "patient_id",
        "hospitalization_id",
        "encounter_block",
        "final_outcome_dttm",
    ])
    .unique("encounter_block")   # safety
    .to_pandas()
)

# Register in DuckDB
duckdb.register("final_cohort_for_sql", final_cohort_for_sql)

print(f"✓ Cohort prepared for SQL: {len(final_cohort_for_sql):,} encounters")

# ============================================================
# 2. Build Labs Query (DuckDB SQL)
# ============================================================

labs_filepath = f"{tables_path}/clif_labs.{file_type}"

print("Processing Labs data with DuckDB...")

labs_query = f"""
WITH labs_with_cohort AS (
    SELECT
        l.hospitalization_id,
        l.lab_collect_dttm,
        l.lab_category,
        l.lab_value_numeric,
        f.encounter_block,
        f.final_outcome_dttm
    FROM read_parquet('{labs_filepath}') l
    INNER JOIN final_cohort_for_sql f
        ON l.hospitalization_id = f.hospitalization_id
),

labs_with_death AS (
    SELECT *
    FROM labs_with_cohort
    WHERE lab_collect_dttm <= final_outcome_dttm
),

-- Latest creatinine per encounter
latest_creatinine AS (
    SELECT
        encounter_block,
        lab_value_numeric AS creatinine_value,
        lab_collect_dttm AS creatinine_dttm
    FROM (
        SELECT
            encounter_block,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (
                PARTITION BY encounter_block
                ORDER BY lab_collect_dttm DESC
            ) AS rn
        FROM labs_with_death
        WHERE lab_category = 'creatinine'
    ) ranked
    WHERE rn = 1
),

-- Latest other organ labs
latest_other_labs AS (
    SELECT
        encounter_block,

        -- Liver
        MAX(CASE WHEN lab_category = 'bilirubin_total' THEN lab_value_numeric END) AS bilirubin_total_value,
        MAX(CASE WHEN lab_category = 'bilirubin_total' THEN lab_collect_dttm END) AS bilirubin_total_dttm,

        MAX(CASE WHEN lab_category = 'ast' THEN lab_value_numeric END) AS ast_value,
        MAX(CASE WHEN lab_category = 'ast' THEN lab_collect_dttm END) AS ast_dttm,

        MAX(CASE WHEN lab_category = 'alt' THEN lab_value_numeric END) AS alt_value,
        MAX(CASE WHEN lab_category = 'alt' THEN lab_collect_dttm END) AS alt_dttm,

        -- Renal / Electrolytes
        MAX(CASE WHEN lab_category = 'bun' THEN lab_value_numeric END) AS bun_value,
        MAX(CASE WHEN lab_category = 'bun' THEN lab_collect_dttm END) AS bun_dttm,

        MAX(CASE WHEN lab_category = 'sodium' THEN lab_value_numeric END) AS sodium_value,
        MAX(CASE WHEN lab_category = 'sodium' THEN lab_collect_dttm END) AS sodium_dttm,

        -- Coagulation
        MAX(CASE WHEN lab_category = 'inr' THEN lab_value_numeric END) AS inr_value,
        MAX(CASE WHEN lab_category = 'inr' THEN lab_collect_dttm END) AS inr_dttm,

        -- ABG / VBG
        MAX(CASE WHEN lab_category = 'ph_arterial' THEN lab_value_numeric END) AS ph_arterial_value,
        MAX(CASE WHEN lab_category = 'ph_arterial' THEN lab_collect_dttm END) AS ph_arterial_dttm,

        MAX(CASE WHEN lab_category = 'po2_arterial' THEN lab_value_numeric END) AS po2_arterial_value,
        MAX(CASE WHEN lab_category = 'po2_arterial' THEN lab_collect_dttm END) AS po2_arterial_dttm,

        MAX(CASE WHEN lab_category = 'pco2_venous' THEN lab_value_numeric END) AS pco2_venous_value,
        MAX(CASE WHEN lab_category = 'pco2_venous' THEN lab_collect_dttm END) AS pco2_venous_dttm,

        -- Cardiac
        MAX(CASE WHEN lab_category = 'troponin_i' THEN lab_value_numeric END) AS troponin_i_value,
        MAX(CASE WHEN lab_category = 'troponin_i' THEN lab_collect_dttm END) AS troponin_i_dttm,

        MAX(CASE WHEN lab_category = 'troponin_t' THEN lab_value_numeric END) AS troponin_t_value,
        MAX(CASE WHEN lab_category = 'troponin_t' THEN lab_collect_dttm END) AS troponin_t_dttm

    FROM (
        SELECT
            encounter_block,
            lab_category,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (
                PARTITION BY encounter_block, lab_category
                ORDER BY lab_collect_dttm DESC
            ) AS rn
        FROM labs_with_death
        WHERE lab_category IN (
            'bilirubin_total', 'ast', 'alt',
            'bun', 'sodium', 'inr',
            'ph_arterial', 'po2_arterial', 'pco2_venous',
            'troponin_i', 'troponin_t'
        )
    ) ranked
    WHERE rn = 1
    GROUP BY encounter_block
),

-- Combine all labs
organ_labs AS (
    SELECT
        f.encounter_block,

        c.creatinine_value,
        c.creatinine_dttm,

        l.*

    FROM final_cohort_for_sql f
    LEFT JOIN latest_creatinine c
        ON f.encounter_block = c.encounter_block
    LEFT JOIN latest_other_labs l
        ON f.encounter_block = l.encounter_block
)

SELECT *
FROM organ_labs
"""

# ============================================================
# 3. Run Query + Convert to Polars
# ============================================================

organ_labs_result = duckdb.sql(labs_query).df()
organ_labs = pl.from_pandas(organ_labs_result)

print(f"✓ Organ labs loaded: {len(organ_labs):,} encounters")

# ============================================================
# 4. QC: Missing Labs Warning
# ============================================================

n_original = final_cohort_for_sql["encounter_block"].nunique()
n_with_labs = len(organ_labs)

if n_with_labs < n_original:
    missing = n_original - n_with_labs
    pct = missing / n_original * 100

    print(
        f"⚠️ WARNING: {missing:,} encounters missing labs "
        f"({pct:.2f}%). {n_with_labs:,}/{n_original:,} retained."
    )
else:
    print("✓ No lab attrition detected")

# ============================================================
# 5. Join Back to Polars Cohort
# ============================================================

final_cohort_matched = (
    final_df_std
    .filter(
        pl.col("encounter_block")
        .is_in(encounter_mapping_matched["encounter_block"])
    )
    .join(
        organ_labs,
        on="encounter_block",
        how="left"
    )
)

print("✓ Labs processing complete")
print(f"✓ Final dataset: {len(final_cohort_matched):,} encounters")

################################################################################
# 7. Save intermediate data
################################################################################

all_matches_df.write_parquet(str(OUTPUT_INTERMEDIATE_DIR / "matches_df.parquet"))
final_cohort_matched.write_parquet(str(OUTPUT_INTERMEDIATE_DIR / "final_clif_data.parquet"))
donor_deceased_site_filtered_std.write_parquet(str(OUTPUT_INTERMEDIATE_DIR / "final_srtr_data.parquet"))
encounter_mapping_matched.write_parquet(str(OUTPUT_INTERMEDIATE_DIR / "encounter_mapping_matched.parquet"))

matched_fp =  str(OUTPUT_INTERMEDIATE_DIR / "matches_df.parquet")
all_matches_df_check = pl.read_parquet(
    matched_fp
)
all_matches_df_check.shape

"""
columns

all_matches_df.columns
['DONOR_ID',
 'patient_id',
 'encounter_block',
 'is_dead',
 'discharge_dttm',
 'date_diff',
 'age_diff',
 'gender_match',
 'race_match',
 'ethnicity_match',
 'race_unknown',
 'ethnicity_unknown',
 'confidence',
 'score',
 'recovery_date',
 'death_date',
 'tolerance_level',
 'date_tolerance',
 'age_tolerance_months',
 'n_alternatives',
 'match_confidence',
 'is_best',
 'match_rank',
 'don_utilized',
 'DON_NON_HR_BEAT']


 final_cohort_matched.columns:
 ['encounter_block',
 'patient_id',
 'hospitalization_id',
 'admission_dttm',
 'age_at_admission',
 'admission_type_category',
 'discharge_category',
 'discharge_dttm',
 'final_outcome_dttm',
 'death_date',
 'age_at_death_months',
 'last_recorded_vital_dttm',
 'birth_date',
 'race_name',
 'race_category',
 'ethnicity_name',
 'ethnicity_category',
 'sex_name',
 'sex_category',
 'last_weight_kg',
 'last_height_cm',
 'bmi',
 'is_dead',
 'death_dttm',
 'first_recorded_vital_dttm',
 'gender',
 'race_std',
 'ethnicity_std',
 'creatinine_value',
 'creatinine_dttm',
 'encounter_block_1',
 'bilirubin_total_value',
 'bilirubin_total_dttm',
 'ast_value',
 'ast_dttm',
 'alt_value',
 'alt_dttm',
 'bun_value',
 'bun_dttm',
 'sodium_value',
 'sodium_dttm',
 'inr_value',
 'inr_dttm',
 'ph_arterial_value',
 'ph_arterial_dttm',
 'po2_arterial_value',
 'po2_arterial_dttm',
 'pco2_venous_value',
 'pco2_venous_dttm',
 'troponin_i_value',
 'troponin_i_dttm',
 'troponin_t_value',
 'troponin_t_dttm']

 donor_deceased_site_filtered.columns:
['DONOR_ID',
 'DON_OPO_CTR_ID',
 'PERS_ID',
 'DON_AGE',
 'DON_AGE_IN_MONTHS',
 'DON_GENDER',
 'DON_RACE',
 'DON_RACE_SRTR',
 'DON_ETHNICITY_SRTR',
 'DON_HGT_CM',
 'DON_WGT_KG',
 'DON_RECOV_DT',
 'DON_CLAMP_DT',
 'DON_CLAMP_TM',
 'DON_CLAMP_TM_ZONE',
 'DON_DEATH_MECH',
 'DON_DEATH_CIRCUM',
 'DON_CAD_DON_COD',
 'DON_CREAT',
 'DON_FINAL_SERUM_CREAT',
 'DON_PEAK_SERUM_CREAT',
 'DON_BUN',
 'DON_TOT_BILI',
 'DON_SGOT',
 'DON_SGPT',
 'DON_PROTEIN_URINE',
 'DON_SODIUM',
 'DON_INR',
 'DON_PH',
 'DON_PO2',
 'DON_PO2_FIO2',
 'DON_PCO2',
 'DON_DOPAMINE',
 'DON_DOBUTAMINE',
 'DON_ARGININE',
 'DON_INOTROP_SUPPORT',
 'INO_MED_DOPAMINE',
 'INO_MED_DOPUTAMINE',
 'INO_MED_EPINEPHRINE',
 'INO_MED_LEVOPHED',
 'INO_MED_NEOSYNEPHRINE',
 'INO_MED_ISOPROTERENOL',
 'DON_TROPONIN_I',
 'DON_TROPONIN_T',
 'DON_NON_HR_BEAT',
 'DON_DCD_SUPPORT_WITHDRAW_DT',
 'DON_DCD_SUPPORT_WITHDRAW_TM',
 'DON_DCD_AGONAL_BEGIN_DT',
 'clif_site',
 'HOSPITAL_NAME',
 'HOSPITAL_ZIP',
 'don_utilized',
 'DON_race_std',
 'DON_ethnicity_std']

 encounter_mapping_matched.columns
 ['hospitalization_id',
 'encounter_block',
 'DONOR_ID',
 'patient_id',
 'is_dead',
 'discharge_dttm',
 'date_diff',
 'age_diff',
 'gender_match',
 'race_match',
 'ethnicity_match',
 'race_unknown',
 'ethnicity_unknown',
 'confidence',
 'score',
 'recovery_date',
 'death_date',
 'tolerance_level',
 'date_tolerance',
 'age_tolerance_months',
 'n_alternatives',
 'match_confidence',
 'is_best',
 'match_rank',
 'don_utilized',
 'DON_NON_HR_BEAT']
"""
