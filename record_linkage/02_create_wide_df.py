#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################################################
# CLIF SRTR Record Linkage
################################################################################

################################################################################
# Setup
# * Load libraries, create required directories, and load the config file
################################################################################

import sys 
import os
import polars as pl 
import matplotlib.pyplot as plt
import pandas as pd
import clifpy
from utils.config import config
from utils.io import read_data
from utils.outlier_handler import apply_outlier_handling
import gc
import yaml

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

# Load config - fix path to work from any directory
from pathlib import Path
script_dir = Path(__file__).parent
config_path = script_dir.parent / "config" / "srtr_clif_data_requirements.yaml"

if config_path.exists():
    with open(config_path, 'r') as f:
        srtr_config = yaml.safe_load(f)
else:
    # If config file doesn't exist, use default configuration
    print(f"Warning: {config_path} not found. Using default configuration.")
    srtr_config = {
        'data_requirements': {
            'labs': True,
            'vitals': True,
            'medications': True,
            'respiratory': True
        },
        'patient_required_columns': [
            'patient_id', 'sex_category', 'race_category',
            'ethnicity_category', 'date_of_birth'
        ],
        'hospitalization_required_columns': [
            'hospitalization_id', 'patient_id', 'admission_dttm',
            'discharge_dttm', 'death_dttm', 'age_at_admission'
        ],
        'labs_required_columns': [
            'hospitalization_id', 'lab_result_dttm',
            'lab_category', 'lab_value_numeric'
        ],
        'vitals_required_columns': [
            'hospitalization_id', 'recorded_dttm',
            'vital_category', 'vital_value'
        ],
        'meds_continuous_required_columns': [
            'hospitalization_id', 'admin_dttm',
            'med_category', 'med_dose'
        ],
        'respiratory_support_required_columns': [
            'hospitalization_id', 'recorded_dttm',
            'ventilator_mode', 'device_category'
        ],
        'adt_required_columns': [
            'hospitalization_id', 'in_dttm', 'out_dttm',
            'location_category', 'location_name'
        ],
        'labs_of_interest': [
            'hemoglobin', 'platelet_count', 'white_blood_cell_count',
            'creatinine', 'blood_urea_nitrogen', 'glucose',
            'sodium', 'potassium', 'chloride', 'bicarbonate',
            'bilirubin_total', 'albumin', 'lactate'
        ],
        'vitals_of_interest': [
            'heart_rate', 'respiratory_rate', 'temperature',
            'sbp', 'dbp', 'map', 'spo2'
        ],
        'meds_continuous_of_interest': [
            'norepinephrine', 'epinephrine', 'dopamine',
            'dobutamine', 'vasopressin', 'milrinone'
        ]
    }

################################################################################
# CLIF Possible Matches
################################################################################

final_cohort_matched_fp = str(OUTPUT_INTERMEDIATE_DIR / "final_clif_data.parquet")
final_cohort_matched = pd.read_parquet(final_cohort_matched_fp)
# Convert patient_id to object (string type), lowercase trial_name, and convert enrollment_dttm to datetime
final_cohort_matched['hospitalization_id'] = final_cohort_matched['hospitalization_id'].astype(str)
final_cohort_matched['encounter_block'] = final_cohort_matched['encounter_block'].astype('int32')

# Keep only rows where trial_name contains 'aps'
# trials_df = trials_df[trials_df['trial_name'].str.contains('aps')]
# Get all patient_ids in a list
final_patient_ids = final_cohort_matched['patient_id'].tolist()
final_hosp_ids = final_cohort_matched['hospitalization_id'].tolist()

final_cohort_matched

################################################################################
# CLIF Core
# * Patient, Hospitalization
################################################################################

print("\n" + "=" * 80)
print("Loading CLIF Tables")
print("=" * 80)

from clifpy.clif_orchestrator import ClifOrchestrator
# Initialize ClifOrchestrator
clif = ClifOrchestrator(
    data_directory=config['tables_path'],
    filetype=config['file_type'],
    timezone=config['timezone'],
    output_directory = OUTPUT_DIR
)

clif.load_table(
        'patient',
        filters={'patient_id': list(final_patient_ids)}
    )

clif.load_table(
        'hospitalization',
        filters={'hospitalization_id': list(final_hosp_ids)}
    )

patient_required_columns = srtr_config['patient_required_columns']
hosp_required_columns = srtr_config['hospitalization_required_columns']

patient_df = clif.patient.df.loc[:, patient_required_columns]
hosp_df = clif.hospitalization.df.loc[:, hosp_required_columns]

# Merge patient and hospitalization tables on 'patient_id'
static_df = hosp_df.merge(
    patient_df,
    on='patient_id',
    how='left'
)

# cleanup
print(clif.get_tables_obj_list())
# Delete the patient and hospitalization table objects
clif.patient = None
clif.hospitalization = None
del patient_df, hosp_df
# Verify they're gone
print(clif.get_tables_obj_list())

# Merge static_df with final_cohort_matched on 'patient_id'
static_df = static_df.merge(
    final_cohort_matched,
    on='patient_id',
    how='left'
)

################################################################################
# Wide Dataset
################################################################################

# Define columns for each table
columns_config = {
    'labs': srtr_config['labs_required_columns'],
    'vitals': srtr_config['vitals_required_columns'],
    'medication_admin_continuous': srtr_config['meds_continuous_required_columns'],
    'respiratory_support': srtr_config['respiratory_support_required_columns']
}

# Define filters for each table (hospitalization_id + category filters)
filters_config = {
    'labs': {
        'hospitalization_id': list(final_hosp_ids),
        'lab_category': srtr_config['labs_of_interest']
    },
    'vitals': {
        'hospitalization_id': list(final_hosp_ids),
        'vital_category': srtr_config['vitals_of_interest']
    },
    'medication_admin_continuous': {
        'hospitalization_id': list(final_hosp_ids),
        'med_category': srtr_config['meds_continuous_of_interest']
    }
}

# Load all tables in one call
clif.initialize(
    tables=['labs', 'vitals', 'medication_admin_continuous' ],
    columns=columns_config,
    filters=filters_config
)

# =============================================================================
# Extract DataFrames from loaded tables
# =============================================================================
labs_df = clif.labs.df.copy()
vitals_df = clif.vitals.df.copy()
meds_cont_df = clif.medication_admin_continuous.df.copy()

# =============================================================================
# Pivot Vitals: narrow → wide
# =============================================================================
vitals_wide = vitals_df.pivot_table(
    index=['hospitalization_id', 'recorded_dttm'],
    columns='vital_category',
    values='vital_value',
    aggfunc='first'
).reset_index()

# Rename datetime column to event_dttm and flatten column names
vitals_wide = vitals_wide.rename(columns={'recorded_dttm': 'event_dttm'})
vitals_wide.columns = ['hospitalization_id', 'event_dttm'] + \
    [f'vital_{col}' for col in vitals_wide.columns[2:]]

# =============================================================================
# Pivot Labs: narrow → wide
# =============================================================================
labs_wide = labs_df.pivot_table(
    index=['hospitalization_id', 'lab_result_dttm'],
    columns='lab_category',
    values='lab_value_numeric',
    aggfunc='first'
).reset_index()

# Rename datetime column to event_dttm
labs_wide = labs_wide.rename(columns={'lab_result_dttm': 'event_dttm'})
labs_wide.columns = ['hospitalization_id', 'event_dttm'] + \
    [f'lab_{col}' for col in labs_wide.columns[2:]]

# =============================================================================
# Pivot Meds Continuous: narrow → wide
# =============================================================================
meds_cont_wide = meds_cont_df.pivot_table(
    index=['hospitalization_id', 'admin_dttm'],
    columns='med_category',
    values='med_dose',
    aggfunc='first'
).reset_index()

# Rename datetime column to event_dttm
meds_cont_wide = meds_cont_wide.rename(columns={'admin_dttm': 'event_dttm'})
meds_cont_wide.columns = ['hospitalization_id', 'event_dttm'] + \
    [f'med_cont_{col}' for col in meds_cont_wide.columns[2:]]

# =============================================================================
# Merge all wide tables on hospitalization_id + event_dttm
# =============================================================================
wide_df = vitals_wide.merge(
    labs_wide,
    on=['hospitalization_id', 'event_dttm'],
    how='outer'
)

wide_df = wide_df.merge(
    meds_cont_wide,
    on=['hospitalization_id', 'event_dttm'],
    how='outer'
)

# Sort by hospitalization and time
wide_df = wide_df.sort_values(['hospitalization_id', 'event_dttm']).reset_index(drop=True)

print(f"Final wide dataframe: {wide_df.shape}")
print(f"Columns: {list(wide_df.columns)}")

################################################################################
# Respiratory Support
################################################################################

clif.load_table(
        'respiratory_support',
        columns= srtr_config['respiratory_support_required_columns'], 
        filters={'hospitalization_id': list(final_hosp_ids)}
    )

clif.respiratory_support = clif.respiratory_support.waterfall()

# =============================================================================
# Respiratory Support (already wide format)
# =============================================================================
resp_df = clif.respiratory_support.df.copy()

# Rename datetime column to event_dttm
resp_df = resp_df.rename(columns={'recorded_dttm': 'event_dttm'})

# Add prefix to all columns except hospitalization_id and event_dttm
resp_cols_to_rename = [col for col in resp_df.columns if col not in ['hospitalization_id', 'event_dttm']]
resp_df = resp_df.rename(columns={col: f'resp_{col}' for col in resp_cols_to_rename})

# Merge with wide_df
wide_df = wide_df.merge(
    resp_df,
    on=['hospitalization_id', 'event_dttm'],
    how='outer'
)

# Re-sort by hospitalization and time
wide_df = wide_df.sort_values(['hospitalization_id', 'event_dttm']).reset_index(drop=True)

wide_df.columns

################################################################################
# ADT
################################################################################

clif.load_table(
        'adt',
        columns= srtr_config['adt_required_columns'], 
        filters={'hospitalization_id': list(final_hosp_ids)}
    )
adt_df = clif.adt.df

# =============================================================================
# Add ADT rows to wide_df (location transitions as events)
# =============================================================================
adt_df['in_dttm'] = pd.to_datetime(adt_df['in_dttm'])
adt_df['out_dttm'] = pd.to_datetime(adt_df['out_dttm'])

# Sort ADT by hospitalization and in_dttm
adt_sorted = adt_df.sort_values(['hospitalization_id', 'in_dttm'])

# Create ADT wide format with in_dttm as event_dttm
adt_wide = adt_df[['hospitalization_id', 'in_dttm', 'location_category', 'location_name']].copy()
adt_wide = adt_wide.rename(columns={'in_dttm': 'event_dttm'})

# Add prefix to ADT columns
adt_wide = adt_wide.rename(columns={
    'location_category': 'adt_location_category',
    'location_name': 'adt_location_name'
})

# Merge with wide_df
wide_df = wide_df.merge(
    adt_wide,
    on=['hospitalization_id', 'event_dttm'],
    how='outer'
)

# Sort by hospitalization and time
wide_df = wide_df.sort_values(['hospitalization_id', 'event_dttm']).reset_index(drop=True)

# Forward fill location within each hospitalization so every row knows the current location
wide_df['adt_location_category'] = wide_df.groupby('hospitalization_id')['adt_location_category'].ffill()
wide_df['adt_location_name'] = wide_df.groupby('hospitalization_id')['adt_location_name'].ffill()

print(f"wide_df shape: {wide_df.shape}")

output_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, f"wide_df.parquet")
wide_df.to_parquet(output_path, index=False)

################################################################################
# Connect with match information
# Join the wide_df with matches df to get the match status for each patient.
################################################################################
