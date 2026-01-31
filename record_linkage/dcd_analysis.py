"""
DCD (Donation after Circulatory Death) Analysis Module

This module provides specialized analysis for DCD donors, including:
1. Donor characteristics summary with DCD utilization rates
2. Four-population comparison between SRTR and EHR DCD donors
"""

import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import json

# Import utilities from existing modules
from record_linkage.table_one_linkage import CONTINUOUS_TOLERANCES as TOLERANCE_DEFINITIONS


def load_table_mappings() -> Dict:
    """Load column mappings from JSON file"""
    mapping_file = Path(__file__).parent / "table_one_mappings.json"
    with open(mapping_file, 'r') as f:
        return json.load(f)


def _fmt_n_pct(n: int, total: int) -> str:
    """Format count and percentage"""
    if total == 0:
        return "0 (0.0%)"
    pct = (n / total) * 100
    return f"{n} ({pct:.1f}%)"


def _fmt_median_iqr(s: pd.Series, digits: int = 2) -> str:
    """Format median and IQR for continuous variables"""
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return "N/A"
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def create_dcd_donor_summary(
    srtr_df: pl.DataFrame,
    patients_df: pl.DataFrame,
    best_matches_df: pl.DataFrame,
    dcd_only: bool = False
) -> pd.DataFrame:
    """
    Create summary table comparing ALL SRTR donors with matched patients by confidence level.
    Shows all table one variables.

    Args:
        srtr_df: SRTR donor data (Polars DataFrame)
        patients_df: CLIF patient data (Polars DataFrame)  
        best_matches_df: Best matches with confidence levels (Polars DataFrame)
        dcd_only: If True, filter to show only DCD donors and their matches

    Returns:
        DataFrame with donor and patient comparisons by confidence level
    """
    # Convert to pandas for easier manipulation
    srtr_pd = srtr_df.to_pandas()
    patients_pd = patients_df.to_pandas()
    matches_pd = best_matches_df.to_pandas()

    # Load mappings
    mappings = load_table_mappings()

    # Filter for DCD only if requested
    if dcd_only:
        # Filter SRTR donors to DCD only
        srtr_pd = srtr_pd[srtr_pd['DON_NON_HR_BEAT'] == 'Y']
        
        # Filter matches to only DCD donors
        dcd_donor_ids = srtr_pd['DONOR_ID'].unique()
        matches_pd = matches_pd[matches_pd['DONOR_ID'].isin(dcd_donor_ids)]

    # ALL SRTR donors (filtered for DCD if requested)
    all_donors_pd = srtr_pd

    # Get patients and matches by confidence level
    confidence_levels = ['HIGH', 'MEDIUM', 'LOW']
    patient_groups = {}
    match_groups = {}
    
    for conf in confidence_levels:
        conf_matches = matches_pd[matches_pd['confidence'] == conf]
        conf_encounter_blocks = conf_matches['encounter_block'].unique()
        patient_groups[conf] = patients_pd[patients_pd['encounter_block'].isin(conf_encounter_blocks)]
        match_groups[conf] = conf_matches

    # Build summary table
    table = []

    # Title row
    title = "**DCD DONOR SUMMARY**" if dcd_only else "**ALL DONOR SUMMARY**"
    table.append({
        "Variable": title,
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # Sample size
    table.append({
        "Variable": "N",
        "SRTR Donors (All)": f"{len(all_donors_pd):,}",
        "HIGH Confidence Patients": f"{len(patient_groups['HIGH']):,}" if len(patient_groups['HIGH']) > 0 else "0",
        "MEDIUM Confidence Patients": f"{len(patient_groups['MEDIUM']):,}" if len(patient_groups['MEDIUM']) > 0 else "0",
        "LOW Confidence Patients": f"{len(patient_groups['LOW']):,}" if len(patient_groups['LOW']) > 0 else "0"
    })

    # Blank row
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # DEMOGRAPHICS
    table.append({
        "Variable": "**DEMOGRAPHICS**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # Age
    age_map = mappings["demographics"]["age_years"]
    if age_map["srtr"] in all_donors_pd.columns:
        row = {
            "Variable": f"{age_map['label']}, median [Q1-Q3]",
            "SRTR Donors (All)": _fmt_median_iqr(all_donors_pd[age_map["srtr"]])
        }
        for conf in confidence_levels:
            if age_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                row[f"{conf} Confidence Patients"] = _fmt_median_iqr(patient_groups[conf][age_map["ehr"]])
            else:
                row[f"{conf} Confidence Patients"] = "—"
        table.append(row)

    # Gender
    male_map = mappings["demographics"]["sex_male"]
    if male_map["srtr"] in all_donors_pd.columns:
        table.append({
            "Variable": "Gender, n (%)",
            "SRTR Donors (All)": "",
            "HIGH Confidence Patients": "",
            "MEDIUM Confidence Patients": "",
            "LOW Confidence Patients": ""
        })
        
        # Male
        n_male_donor = all_donors_pd[male_map["srtr"]].isin(male_map["srtr_positive_values"]).sum()
        row = {
            "Variable": f"  {male_map['label']}",
            "SRTR Donors (All)": _fmt_n_pct(n_male_donor, len(all_donors_pd))
        }
        for conf in confidence_levels:
            if male_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                n_male = patient_groups[conf][male_map["ehr"]].isin(male_map["ehr_positive_values"]).sum()
                row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_male, len(patient_groups[conf]))
            else:
                row[f"{conf} Confidence Patients"] = "—"
        table.append(row)

        # Female
        female_map = mappings["demographics"]["sex_female"]
        n_female_donor = all_donors_pd[female_map["srtr"]].isin(female_map["srtr_positive_values"]).sum()
        row = {
            "Variable": f"  {female_map['label']}",
            "SRTR Donors (All)": _fmt_n_pct(n_female_donor, len(all_donors_pd))
        }
        for conf in confidence_levels:
            if female_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                n_female = patient_groups[conf][female_map["ehr"]].isin(female_map["ehr_positive_values"]).sum()
                row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_female, len(patient_groups[conf]))
            else:
                row[f"{conf} Confidence Patients"] = "—"
        table.append(row)

    # Race
    race_map = mappings["demographics"]["race"]
    if race_map["srtr"] in all_donors_pd.columns:
        table.append({
            "Variable": f"{race_map['label']}, n (%)",
            "SRTR Donors (All)": "",
            "HIGH Confidence Patients": "",
            "MEDIUM Confidence Patients": "",
            "LOW Confidence Patients": ""
        })
        
        # Get all unique races
        all_races = set()
        all_races.update(all_donors_pd[race_map["srtr"]].dropna().unique())
        for conf in confidence_levels:
            if race_map["ehr"] in patient_groups[conf].columns:
                all_races.update(patient_groups[conf][race_map["ehr"]].dropna().unique())
        
        for race in sorted(all_races):
            n_donor = (all_donors_pd[race_map["srtr"]] == race).sum()
            row = {
                "Variable": f"  {race}",
                "SRTR Donors (All)": _fmt_n_pct(n_donor, len(all_donors_pd))
            }
            for conf in confidence_levels:
                if race_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                    n_race = (patient_groups[conf][race_map["ehr"]] == race).sum()
                    row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_race, len(patient_groups[conf]))
                else:
                    row[f"{conf} Confidence Patients"] = "—"
            table.append(row)

    # Ethnicity
    ethnicity_map = mappings["demographics"]["ethnicity"]
    if ethnicity_map["srtr"] in all_donors_pd.columns:
        table.append({
            "Variable": f"{ethnicity_map['label']}, n (%)",
            "SRTR Donors (All)": "",
            "HIGH Confidence Patients": "",
            "MEDIUM Confidence Patients": "",
            "LOW Confidence Patients": ""
        })
        
        # Get all unique ethnicities
        all_ethnicities = set()
        all_ethnicities.update(all_donors_pd[ethnicity_map["srtr"]].dropna().unique())
        for conf in confidence_levels:
            if ethnicity_map["ehr"] in patient_groups[conf].columns:
                all_ethnicities.update(patient_groups[conf][ethnicity_map["ehr"]].dropna().unique())
        
        for ethnicity in sorted(all_ethnicities):
            n_donor = (all_donors_pd[ethnicity_map["srtr"]] == ethnicity).sum()
            row = {
                "Variable": f"  {ethnicity}",
                "SRTR Donors (All)": _fmt_n_pct(n_donor, len(all_donors_pd))
            }
            for conf in confidence_levels:
                if ethnicity_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                    n_eth = (patient_groups[conf][ethnicity_map["ehr"]] == ethnicity).sum()
                    row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_eth, len(patient_groups[conf]))
                else:
                    row[f"{conf} Confidence Patients"] = "—"
            table.append(row)

    # Blank row
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # CLINICAL MEASUREMENTS
    table.append({
        "Variable": "**CLINICAL MEASUREMENTS**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # Height and Weight
    for measure_key in ["height", "weight"]:
        measure_map = mappings["clinical_measurements"][measure_key]
        if measure_map["srtr"] in all_donors_pd.columns:
            row = {
                "Variable": f"{measure_map['label']}, median [Q1-Q3]",
                "SRTR Donors (All)": _fmt_median_iqr(all_donors_pd[measure_map["srtr"]])
            }
            for conf in confidence_levels:
                if measure_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                    row[f"{conf} Confidence Patients"] = _fmt_median_iqr(patient_groups[conf][measure_map["ehr"]])
                else:
                    row[f"{conf} Confidence Patients"] = "—"
            table.append(row)

    # Blank row
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # LABORATORY VALUES
    table.append({
        "Variable": "**LABORATORY VALUES**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # Process all lab values
    for lab_key, lab_map in mappings["laboratory_values"].items():
        if lab_map["srtr"] in all_donors_pd.columns:
            digits = lab_map.get("digits", 1)
            row = {
                "Variable": f"{lab_map['label']}, median [Q1-Q3]",
                "SRTR Donors (All)": _fmt_median_iqr(all_donors_pd[lab_map["srtr"]], digits=digits)
            }
            for conf in confidence_levels:
                if lab_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                    row[f"{conf} Confidence Patients"] = _fmt_median_iqr(patient_groups[conf][lab_map["ehr"]], digits=digits)
                else:
                    row[f"{conf} Confidence Patients"] = "—"
            table.append(row)

    # Blank row
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # LIFE SUPPORT (EHR only)
    table.append({
        "Variable": "**LIFE SUPPORT (EHR only)**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    for ls_key, ls_map in mappings["life_support"].items():
        # Check data source - some variables come from matches_pd, others from patients_pd
        data_source = ls_map.get("data_source", "patients_pd")
        
        if ls_map["type"] == "binary":
            row = {
                "Variable": f"{ls_map['label']}, n (%)",
                "SRTR Donors (All)": "—"  # No SRTR data
            }
            for conf in confidence_levels:
                # Use appropriate data source based on JSON mapping
                if data_source == "matches_pd":
                    # Get data from matches for this confidence level
                    if ls_map["ehr"] in match_groups[conf].columns and len(match_groups[conf]) > 0:
                        n_positive = match_groups[conf][ls_map["ehr"]].isin(ls_map.get("ehr_positive_values", [True])).sum()
                        row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_positive, len(match_groups[conf]))
                    else:
                        row[f"{conf} Confidence Patients"] = "—"
                else:
                    # Get data from patients (existing logic)
                    if ls_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                        n_positive = patient_groups[conf][ls_map["ehr"]].isin(ls_map.get("ehr_positive_values", [True])).sum()
                        row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_positive, len(patient_groups[conf]))
                    else:
                        row[f"{conf} Confidence Patients"] = "—"
            table.append(row)
            
        elif ls_map["type"] == "presence":
            row = {
                "Variable": f"{ls_map['label']}, n (%)",
                "SRTR Donors (All)": "—"  # No SRTR data
            }
            for conf in confidence_levels:
                # Use appropriate data source based on JSON mapping
                if data_source == "matches_pd":
                    # Get data from matches for this confidence level
                    if ls_map["ehr"] in match_groups[conf].columns and len(match_groups[conf]) > 0:
                        n_with_value = match_groups[conf][ls_map["ehr"]].notna().sum()
                        row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_with_value, len(match_groups[conf]))
                    else:
                        row[f"{conf} Confidence Patients"] = "—"
                else:
                    # Get data from patients (existing logic)
                    if ls_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                        n_with_value = patient_groups[conf][ls_map["ehr"]].notna().sum()
                        row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_with_value, len(patient_groups[conf]))
                    else:
                        row[f"{conf} Confidence Patients"] = "—"
            table.append(row)

    # Blank row
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # MEDICATIONS
    table.append({
        "Variable": "**MEDICATIONS**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    for med_key, med_map in mappings["medications"].items():
        row = {"Variable": f"{med_map['label']}, n (%)"}
        
        # SRTR donor value
        if med_map["srtr"] and med_map["srtr"] in all_donors_pd.columns:
            n_positive = all_donors_pd[med_map["srtr"]].isin(med_map.get("srtr_positive_values", ["Y", 1])).sum()
            row["SRTR Donors (All)"] = _fmt_n_pct(n_positive, len(all_donors_pd))
        else:
            row["SRTR Donors (All)"] = "—"
        
        # Patient values by confidence
        for conf in confidence_levels:
            if med_map["ehr"] and med_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
                n_positive = patient_groups[conf][med_map["ehr"]].isin(med_map.get("ehr_positive_values", [True])).sum()
                row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_positive, len(patient_groups[conf]))
            else:
                row[f"{conf} Confidence Patients"] = "—"
        
        table.append(row)

    # Blank row
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # DONOR CHARACTERISTICS
    table.append({
        "Variable": "**DONOR CHARACTERISTICS**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # DCD status
    dcd_map = mappings["donor_characteristics"]["dcd"]
    if dcd_map["srtr"] in all_donors_pd.columns:
        n_dcd = all_donors_pd[dcd_map["srtr"]].isin(dcd_map.get("srtr_positive_values", [1, "Y", "YES"])).sum()
        table.append({
            "Variable": f"{dcd_map['label']}, n (%)",
            "SRTR Donors (All)": _fmt_n_pct(n_dcd, len(all_donors_pd)),
            "HIGH Confidence Patients": "—",
            "MEDIUM Confidence Patients": "—",
            "LOW Confidence Patients": "—"
        })

    # Donor utilized
    util_map = mappings["donor_characteristics"]["donor_utilized"]
    if util_map["srtr"] in all_donors_pd.columns:
        # Use == 'Y' for don_utilized
        n_utilized = (all_donors_pd[util_map["srtr"]] == 'Y').sum()
        table.append({
            "Variable": f"{util_map['label']}, n (%)",
            "SRTR Donors (All)": _fmt_n_pct(n_utilized, len(all_donors_pd)),
            "HIGH Confidence Patients": "—",
            "MEDIUM Confidence Patients": "—",
            "LOW Confidence Patients": "—"
        })

        # If DCD only, show DCD utilized
        if dcd_only:
            n_dcd_utilized = ((all_donors_pd['DON_NON_HR_BEAT'] == 'Y') & (all_donors_pd[util_map["srtr"]] == 'Y')).sum()
            n_dcd = (all_donors_pd['DON_NON_HR_BEAT'] == 'Y').sum()
            table.append({
                "Variable": "  DCD Utilized, n (% of DCD)",
                "SRTR Donors (All)": _fmt_n_pct(n_dcd_utilized, n_dcd) if n_dcd > 0 else "—",
                "HIGH Confidence Patients": "—",
                "MEDIUM Confidence Patients": "—",
                "LOW Confidence Patients": "—"
            })

    # Blank row
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # DEATH STATUS
    table.append({
        "Variable": "**DEATH STATUS**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # Death documented
    death_map = mappings["death_status"]["death_documented"]
    
    # All donors are deceased
    row = {
        "Variable": f"{death_map['label']}, n (%)",
        "SRTR Donors (All)": _fmt_n_pct(len(all_donors_pd), len(all_donors_pd))  # 100%
    }
    
    for conf in confidence_levels:
        if death_map["ehr"] in patient_groups[conf].columns and len(patient_groups[conf]) > 0:
            n_dead = patient_groups[conf][death_map["ehr"]].isin(death_map.get("ehr_positive_values", [1, "1", "TRUE"])).sum()
            row[f"{conf} Confidence Patients"] = _fmt_n_pct(n_dead, len(patient_groups[conf]))
        else:
            row[f"{conf} Confidence Patients"] = "—"
    
    table.append(row)

    # Add note about matched counts
    table.append({
        "Variable": "",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    # Summary of matching
    n_matched = len(matches_pd['DONOR_ID'].unique())
    table.append({
        "Variable": "**MATCHING SUMMARY**",
        "SRTR Donors (All)": "",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })
    
    pct_matched = (n_matched / len(all_donors_pd) * 100) if len(all_donors_pd) > 0 else 0
    table.append({
        "Variable": "Donors with matches, n (%)",
        "SRTR Donors (All)": f"{n_matched} ({pct_matched:.1f}%)",
        "HIGH Confidence Patients": "",
        "MEDIUM Confidence Patients": "",
        "LOW Confidence Patients": ""
    })

    return pd.DataFrame(table)


def create_dcd_four_population_comparison(
    srtr_df: pl.DataFrame,
    patients_df: pl.DataFrame,
    best_matches_df: pl.DataFrame
) -> pd.DataFrame:
    """
    Create comparison table for four DCD populations:
    a) SRTR DCD donor
    b) Matched EHR donor
    c) SRTR DCD utilized donor
    d) Matched EHR DCD utilized donor

    Restricted to HIGH confidence matches with patients who died.

    Args:
        srtr_df: SRTR donor data (Polars DataFrame)
        patients_df: CLIF patient data (Polars DataFrame)
        best_matches_df: Best matches with confidence levels (Polars DataFrame)

    Returns:
        DataFrame with four population comparisons
    """
    # Filter for HIGH confidence only
    high_conf_matches = best_matches_df.filter(pl.col("confidence") == "HIGH")

    # Filter for patients who died 
    # Get encounter blocks for patients who died
    dead_patients_df = patients_df.filter(pl.col("is_dead") == 1)
    dead_encounter_blocks = dead_patients_df["encounter_block"].unique().to_list()

    # Filter matches to only include dead patients
    matches_dead_patients = high_conf_matches.filter(
        pl.col("encounter_block").is_in(dead_encounter_blocks)
    )

    # Convert to pandas
    matches_pd = matches_dead_patients.to_pandas()
    srtr_pd = srtr_df.to_pandas()
    patients_pd = patients_df.to_pandas()

    # Load mappings
    mappings = load_table_mappings()

    # Create four populations
    # Get matched donor and patient IDs
    matched_donor_ids = matches_pd['DONOR_ID'].unique()
    matched_encounter_blocks = matches_pd['encounter_block'].unique()

    # Filter to matched data
    matched_srtr = srtr_pd[srtr_pd['DONOR_ID'].isin(matched_donor_ids)]
    matched_patients = patients_pd[patients_pd['encounter_block'].isin(matched_encounter_blocks)]

    # Create merged data for agreement calculations
    merged_data = pd.merge(
        matches_pd[['DONOR_ID', 'encounter_block']],
        matched_srtr,
        on='DONOR_ID',
        how='inner'
    )
    merged_data = pd.merge(
        merged_data,
        matched_patients,
        on='encounter_block',
        how='inner'
    )

    # Define four populations
    # For DCD populations, also keep track of the corresponding matches for life support data
    dcd_encounter_blocks = merged_data[merged_data['DON_NON_HR_BEAT'] == 'Y']['encounter_block'].unique()
    dcd_utilized_encounter_blocks = merged_data[
        (merged_data['DON_NON_HR_BEAT'] == 'Y') &
        (merged_data['don_utilized'] == 'Y')
    ]['encounter_block'].unique()
    
    # Store match data for life support variables that come from matches_pd
    dcd_matches = matches_pd[matches_pd['encounter_block'].isin(dcd_encounter_blocks)]
    dcd_utilized_matches = matches_pd[matches_pd['encounter_block'].isin(dcd_utilized_encounter_blocks)]
    
    populations = {
        'a) SRTR DCD': matched_srtr[matched_srtr['DON_NON_HR_BEAT'] == 'Y'],
        'b) Matched EHR': matched_patients[matched_patients['encounter_block'].isin(dcd_encounter_blocks)],
        'c) SRTR DCD Utilized': matched_srtr[
            (matched_srtr['DON_NON_HR_BEAT'] == 'Y') &
            (matched_srtr['don_utilized'] == 'Y')
        ],
        'd) Matched EHR Utilized': matched_patients[matched_patients['encounter_block'].isin(dcd_utilized_encounter_blocks)]
    }
    
    # Also store the matches data for life support variables
    match_populations = {
        'b) Matched EHR': dcd_matches,
        'd) Matched EHR Utilized': dcd_utilized_matches
    }

    # For agreement calculations
    dcd_merged = merged_data[merged_data['DON_NON_HR_BEAT'] == 'Y']
    dcd_utilized_merged = merged_data[
        (merged_data['DON_NON_HR_BEAT'] == 'Y') &
        (merged_data['don_utilized'] == 'Y')
    ]

    # Build comparison table
    table = []

    # Header with population counts
    header_row = {
        "Variable": "N",
        "a) SRTR DCD": f"{len(populations['a) SRTR DCD']):,}",
        "b) Matched EHR": f"{len(populations['b) Matched EHR']):,}",
        "Agreement (a-b)": f"{len(dcd_merged):,} paired",
        "c) SRTR DCD Utilized": f"{len(populations['c) SRTR DCD Utilized']):,}",
        "d) Matched EHR Utilized": f"{len(populations['d) Matched EHR Utilized']):,}",
        "Agreement (c-d)": f"{len(dcd_utilized_merged):,} paired"
    }
    table.append(header_row)

    # Blank row
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # Demographics section
    table.append({
        "Variable": "**DEMOGRAPHICS**",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # Age
    age_map = mappings["demographics"]["age_years"]
    if age_map["srtr"] in matched_srtr.columns and age_map["ehr"] in matched_patients.columns:
        # Calculate agreement for DCD
        if len(dcd_merged) > 0:
            age_diff_dcd = (dcd_merged[age_map["srtr"]] - dcd_merged[age_map["ehr"]]).dropna()
            if len(age_diff_dcd) > 0:
                mean_diff_dcd = age_diff_dcd.mean()
                std_diff_dcd = age_diff_dcd.std()
                se_diff_dcd = std_diff_dcd / np.sqrt(len(age_diff_dcd))
                ci_lower_dcd = mean_diff_dcd - 1.96 * se_diff_dcd
                ci_upper_dcd = mean_diff_dcd + 1.96 * se_diff_dcd
                agreement_dcd = f"{mean_diff_dcd:.1f} [{ci_lower_dcd:.1f}, {ci_upper_dcd:.1f}]"
            else:
                agreement_dcd = "—"
        else:
            agreement_dcd = "—"

        # Calculate agreement for DCD utilized
        if len(dcd_utilized_merged) > 0:
            age_diff_util = (dcd_utilized_merged[age_map["srtr"]] - dcd_utilized_merged[age_map["ehr"]]).dropna()
            if len(age_diff_util) > 0:
                mean_diff_util = age_diff_util.mean()
                std_diff_util = age_diff_util.std()
                se_diff_util = std_diff_util / np.sqrt(len(age_diff_util))
                ci_lower_util = mean_diff_util - 1.96 * se_diff_util
                ci_upper_util = mean_diff_util + 1.96 * se_diff_util
                agreement_util = f"{mean_diff_util:.1f} [{ci_lower_util:.1f}, {ci_upper_util:.1f}]"
            else:
                agreement_util = "—"
        else:
            agreement_util = "—"

        table.append({
            "Variable": "Age (years), median [Q1-Q3]",
            "a) SRTR DCD": _fmt_median_iqr(populations['a) SRTR DCD'][age_map["srtr"]]) if age_map["srtr"] in populations['a) SRTR DCD'].columns else "—",
            "b) Matched EHR": _fmt_median_iqr(populations['b) Matched EHR'][age_map["ehr"]]) if age_map["ehr"] in populations['b) Matched EHR'].columns else "—",
            "Agreement (a-b)": agreement_dcd,
            "c) SRTR DCD Utilized": _fmt_median_iqr(populations['c) SRTR DCD Utilized'][age_map["srtr"]]) if age_map["srtr"] in populations['c) SRTR DCD Utilized'].columns else "—",
            "d) Matched EHR Utilized": _fmt_median_iqr(populations['d) Matched EHR Utilized'][age_map["ehr"]]) if age_map["ehr"] in populations['d) Matched EHR Utilized'].columns else "—",
            "Agreement (c-d)": agreement_util
        })

    # Gender
    male_map = mappings["demographics"]["sex_male"]
    if male_map["srtr"] in matched_srtr.columns and male_map["ehr"] in matched_patients.columns:
        # Calculate percentages for each population
        def calc_male_pct(df, col, pos_vals):
            if col not in df.columns or len(df) == 0:
                return "—"
            n_male = df[col].isin(pos_vals).sum()
            return _fmt_n_pct(n_male, len(df))

        # Calculate agreement for binary variable
        def calc_binary_agreement(merged_df, srtr_col, ehr_col, srtr_vals, ehr_vals):
            if len(merged_df) == 0:
                return "—"
            paired = merged_df[[srtr_col, ehr_col]].copy()
            paired['srtr_binary'] = paired[srtr_col].isin(srtr_vals).astype(int)
            paired['ehr_binary'] = paired[ehr_col].isin(ehr_vals).astype(int)
            paired = paired[['srtr_binary', 'ehr_binary']].dropna()
            if len(paired) == 0:
                return "—"
            agree = (paired['srtr_binary'] == paired['ehr_binary']).sum()
            pct = (agree / len(paired)) * 100
            return f"{pct:.0f}%"

        table.append({
            "Variable": "Male, n (%)",
            "a) SRTR DCD": calc_male_pct(populations['a) SRTR DCD'], male_map["srtr"], male_map["srtr_positive_values"]),
            "b) Matched EHR": calc_male_pct(populations['b) Matched EHR'], male_map["ehr"], male_map["ehr_positive_values"]),
            "Agreement (a-b)": calc_binary_agreement(dcd_merged, male_map["srtr"], male_map["ehr"],
                                                    male_map["srtr_positive_values"], male_map["ehr_positive_values"]),
            "c) SRTR DCD Utilized": calc_male_pct(populations['c) SRTR DCD Utilized'], male_map["srtr"], male_map["srtr_positive_values"]),
            "d) Matched EHR Utilized": calc_male_pct(populations['d) Matched EHR Utilized'], male_map["ehr"], male_map["ehr_positive_values"]),
            "Agreement (c-d)": calc_binary_agreement(dcd_utilized_merged, male_map["srtr"], male_map["ehr"],
                                                    male_map["srtr_positive_values"], male_map["ehr_positive_values"])
        })

    # Add blank row
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # Clinical measurements
    table.append({
        "Variable": "**CLINICAL MEASUREMENTS**",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # Add Height and Weight
    for measure_key in ["height", "weight"]:
        measure_map = mappings["clinical_measurements"][measure_key]
        if measure_map["srtr"] in matched_srtr.columns and measure_map["ehr"] in matched_patients.columns:
            # Calculate agreement for continuous variables
            def calc_continuous_agreement(merged_df, srtr_col, ehr_col):
                if len(merged_df) == 0:
                    return "—"
                diff = (merged_df[srtr_col] - merged_df[ehr_col]).dropna()
                if len(diff) == 0:
                    return "—"
                mean_diff = diff.mean()
                std_diff = diff.std()
                se_diff = std_diff / np.sqrt(len(diff))
                ci_lower = mean_diff - 1.96 * se_diff
                ci_upper = mean_diff + 1.96 * se_diff
                return f"{mean_diff:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"

            table.append({
                "Variable": f"{measure_map['label']}, median [Q1-Q3]",
                "a) SRTR DCD": _fmt_median_iqr(populations['a) SRTR DCD'][measure_map["srtr"]]) if measure_map["srtr"] in populations['a) SRTR DCD'].columns else "—",
                "b) Matched EHR": _fmt_median_iqr(populations['b) Matched EHR'][measure_map["ehr"]]) if measure_map["ehr"] in populations['b) Matched EHR'].columns else "—",
                "Agreement (a-b)": calc_continuous_agreement(dcd_merged, measure_map["srtr"], measure_map["ehr"]),
                "c) SRTR DCD Utilized": _fmt_median_iqr(populations['c) SRTR DCD Utilized'][measure_map["srtr"]]) if measure_map["srtr"] in populations['c) SRTR DCD Utilized'].columns else "—",
                "d) Matched EHR Utilized": _fmt_median_iqr(populations['d) Matched EHR Utilized'][measure_map["ehr"]]) if measure_map["ehr"] in populations['d) Matched EHR Utilized'].columns else "—",
                "Agreement (c-d)": calc_continuous_agreement(dcd_utilized_merged, measure_map["srtr"], measure_map["ehr"])
            })

    # Add blank row
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # Laboratory values
    table.append({
        "Variable": "**LABORATORY VALUES**",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # Add selected lab values
    for lab_key in mappings["laboratory_values"].keys():
        lab_map = mappings["laboratory_values"][lab_key]
        if lab_map["srtr"] in matched_srtr.columns and lab_map["ehr"] in matched_patients.columns:
            # Calculate agreement
            def calc_lab_agreement(merged_df, srtr_col, ehr_col, tolerance_key=None):
                if len(merged_df) == 0:
                    return "—"
                diff = (merged_df[srtr_col] - merged_df[ehr_col]).dropna()
                if len(diff) == 0:
                    return "—"
                mean_diff = diff.mean()
                std_diff = diff.std()
                se_diff = std_diff / np.sqrt(len(diff))
                ci_lower = mean_diff - 1.96 * se_diff
                ci_upper = mean_diff + 1.96 * se_diff

                result = f"{mean_diff:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"

                # Add tolerance if available
                if tolerance_key and tolerance_key in TOLERANCE_DEFINITIONS:
                    tol = TOLERANCE_DEFINITIONS[tolerance_key]
                    if "absolute" in tol:
                        within_tol = (np.abs(diff) <= tol["absolute"]).mean() * 100
                        result += f" ({within_tol:.0f}%)"
                    elif "relative" in tol:
                        rel_diff = np.abs(diff / merged_df[srtr_col].dropna())
                        within_tol = (rel_diff <= tol["relative"]).mean() * 100
                        result += f" ({within_tol:.0f}%)"

                return result

            digits = lab_map.get("digits", 1)
            table.append({
                "Variable": f"{lab_map['label']}, median [Q1-Q3]",
                "a) SRTR DCD": _fmt_median_iqr(populations['a) SRTR DCD'][lab_map["srtr"]], digits=digits) if lab_map["srtr"] in populations['a) SRTR DCD'].columns else "—",
                "b) Matched EHR": _fmt_median_iqr(populations['b) Matched EHR'][lab_map["ehr"]], digits=digits) if lab_map["ehr"] in populations['b) Matched EHR'].columns else "—",
                "Agreement (a-b)": calc_lab_agreement(dcd_merged, lab_map["srtr"], lab_map["ehr"], lab_map.get("tolerance_key")),
                "c) SRTR DCD Utilized": _fmt_median_iqr(populations['c) SRTR DCD Utilized'][lab_map["srtr"]], digits=digits) if lab_map["srtr"] in populations['c) SRTR DCD Utilized'].columns else "—",
                "d) Matched EHR Utilized": _fmt_median_iqr(populations['d) Matched EHR Utilized'][lab_map["ehr"]], digits=digits) if lab_map["ehr"] in populations['d) Matched EHR Utilized'].columns else "—",
                "Agreement (c-d)": calc_lab_agreement(dcd_utilized_merged, lab_map["srtr"], lab_map["ehr"], lab_map.get("tolerance_key"))
            })

    # Blank row
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # LIFE SUPPORT (EHR only) - FIXED to check data_source
    table.append({
        "Variable": "**LIFE SUPPORT (EHR only)**",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    for ls_key, ls_map in mappings["life_support"].items():
        # Check data source for this variable
        data_source = ls_map.get("data_source", "patients_pd")
        
        if ls_map["type"] == "binary":
            # Choose the correct data based on data_source
            if data_source == "matches_pd":
                # Use match data
                dcd_data = match_populations['b) Matched EHR']
                util_data = match_populations['d) Matched EHR Utilized']
            else:
                # Use patient data (default)
                dcd_data = populations['b) Matched EHR']
                util_data = populations['d) Matched EHR Utilized']
            
            # Calculate values
            dcd_val = "—"
            util_val = "—"
            
            if ls_map["ehr"] in dcd_data.columns and len(dcd_data) > 0:
                n_positive = dcd_data[ls_map["ehr"]].isin(ls_map.get("ehr_positive_values", [True])).sum()
                dcd_val = _fmt_n_pct(n_positive, len(dcd_data))
            
            if ls_map["ehr"] in util_data.columns and len(util_data) > 0:
                n_positive = util_data[ls_map["ehr"]].isin(ls_map.get("ehr_positive_values", [True])).sum()
                util_val = _fmt_n_pct(n_positive, len(util_data))
            
            table.append({
                "Variable": f"{ls_map['label']}, n (%)",
                "a) SRTR DCD": "—",  # No SRTR data
                "b) Matched EHR": dcd_val,
                "Agreement (a-b)": "—",
                "c) SRTR DCD Utilized": "—",
                "d) Matched EHR Utilized": util_val,
                "Agreement (c-d)": "—"
            })
            
        elif ls_map["type"] == "presence":
            # Choose the correct data based on data_source
            if data_source == "matches_pd":
                # Use match data
                dcd_data = match_populations['b) Matched EHR']
                util_data = match_populations['d) Matched EHR Utilized']
            else:
                # Use patient data (default)
                dcd_data = populations['b) Matched EHR']
                util_data = populations['d) Matched EHR Utilized']
            
            # Calculate values
            dcd_val = "—"
            util_val = "—"
            
            if ls_map["ehr"] in dcd_data.columns and len(dcd_data) > 0:
                n_with_value = dcd_data[ls_map["ehr"]].notna().sum()
                dcd_val = _fmt_n_pct(n_with_value, len(dcd_data))
            
            if ls_map["ehr"] in util_data.columns and len(util_data) > 0:
                n_with_value = util_data[ls_map["ehr"]].notna().sum()
                util_val = _fmt_n_pct(n_with_value, len(util_data))
            
            table.append({
                "Variable": f"{ls_map['label']}, n (%)",
                "a) SRTR DCD": "—",
                "b) Matched EHR": dcd_val,
                "Agreement (a-b)": "—",
                "c) SRTR DCD Utilized": "—",
                "d) Matched EHR Utilized": util_val,
                "Agreement (c-d)": "—"
            })

    # Blank row
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # MEDICATIONS
    table.append({
        "Variable": "**MEDICATIONS**",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    for med_key, med_map in mappings["medications"].items():
        # SRTR columns
        srtr_dcd_val = "—"
        srtr_util_val = "—"
        if med_map["srtr"] and med_map["srtr"] in populations['a) SRTR DCD'].columns:
            n_positive_dcd = populations['a) SRTR DCD'][med_map["srtr"]].isin(med_map.get("srtr_positive_values", ["Y", 1])).sum()
            srtr_dcd_val = _fmt_n_pct(n_positive_dcd, len(populations['a) SRTR DCD']))

            if len(populations['c) SRTR DCD Utilized']) > 0:
                n_positive_util = populations['c) SRTR DCD Utilized'][med_map["srtr"]].isin(med_map.get("srtr_positive_values", ["Y", 1])).sum()
                srtr_util_val = _fmt_n_pct(n_positive_util, len(populations['c) SRTR DCD Utilized']))

        # EHR columns
        ehr_dcd_val = "—"
        ehr_util_val = "—"
        if med_map["ehr"] and med_map["ehr"] in populations['b) Matched EHR'].columns:
            n_positive_dcd = populations['b) Matched EHR'][med_map["ehr"]].isin(med_map.get("ehr_positive_values", [True])).sum()
            ehr_dcd_val = _fmt_n_pct(n_positive_dcd, len(populations['b) Matched EHR']))

            if len(populations['d) Matched EHR Utilized']) > 0:
                n_positive_util = populations['d) Matched EHR Utilized'][med_map["ehr"]].isin(med_map.get("ehr_positive_values", [True])).sum()
                ehr_util_val = _fmt_n_pct(n_positive_util, len(populations['d) Matched EHR Utilized']))

        # Agreement
        agreement_dcd = "—"
        agreement_util = "—"
        if med_map["srtr"] and med_map["ehr"] and med_map["srtr"] in dcd_merged.columns and med_map["ehr"] in dcd_merged.columns:
            agreement_dcd = calc_binary_agreement(dcd_merged, med_map["srtr"], med_map["ehr"],
                                                 med_map.get("srtr_positive_values", ["Y", 1]),
                                                 med_map.get("ehr_positive_values", [True]))
            if len(dcd_utilized_merged) > 0:
                agreement_util = calc_binary_agreement(dcd_utilized_merged, med_map["srtr"], med_map["ehr"],
                                                      med_map.get("srtr_positive_values", ["Y", 1]),
                                                      med_map.get("ehr_positive_values", [True]))

        table.append({
            "Variable": f"{med_map['label']}, n (%)",
            "a) SRTR DCD": srtr_dcd_val,
            "b) Matched EHR": ehr_dcd_val,
            "Agreement (a-b)": agreement_dcd,
            "c) SRTR DCD Utilized": srtr_util_val,
            "d) Matched EHR Utilized": ehr_util_val,
            "Agreement (c-d)": agreement_util
        })

    # Blank row
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # DONOR CHARACTERISTICS
    table.append({
        "Variable": "**DONOR CHARACTERISTICS**",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # DCD status (should be 100% for all since we filtered for DCD)
    dcd_map = mappings["donor_characteristics"]["dcd"]
    if dcd_map["srtr"] in populations['a) SRTR DCD'].columns:
        n_dcd = populations['a) SRTR DCD'][dcd_map["srtr"]].isin(dcd_map.get("srtr_positive_values", [1, "Y", "YES"])).sum()
        n_dcd_util = populations['c) SRTR DCD Utilized'][dcd_map["srtr"]].isin(dcd_map.get("srtr_positive_values", [1, "Y", "YES"])).sum() if len(populations['c) SRTR DCD Utilized']) > 0 else 0

        table.append({
            "Variable": f"{dcd_map['label']}, n (%)",
            "a) SRTR DCD": _fmt_n_pct(n_dcd, len(populations['a) SRTR DCD'])),
            "b) Matched EHR": "—",
            "Agreement (a-b)": "—",
            "c) SRTR DCD Utilized": _fmt_n_pct(n_dcd_util, len(populations['c) SRTR DCD Utilized'])) if len(populations['c) SRTR DCD Utilized']) > 0 else "—",
            "d) Matched EHR Utilized": "—",
            "Agreement (c-d)": "—"
        })

    # Donor utilized
    util_map = mappings["donor_characteristics"]["donor_utilized"]
    if util_map["srtr"] in populations['a) SRTR DCD'].columns:
        # Use == 'Y' for don_utilized
        n_utilized_dcd = (populations['a) SRTR DCD'][util_map["srtr"]] == 'Y').sum()
        n_utilized_util = (populations['c) SRTR DCD Utilized'][util_map["srtr"]] == 'Y').sum() if len(populations['c) SRTR DCD Utilized']) > 0 else 0

        table.append({
            "Variable": f"{util_map['label']}, n (%)",
            "a) SRTR DCD": _fmt_n_pct(n_utilized_dcd, len(populations['a) SRTR DCD'])),
            "b) Matched EHR": "—",
            "Agreement (a-b)": "—",
            "c) SRTR DCD Utilized": _fmt_n_pct(n_utilized_util, len(populations['c) SRTR DCD Utilized'])) if len(populations['c) SRTR DCD Utilized']) > 0 else "—",
            "d) Matched EHR Utilized": "—",
            "Agreement (c-d)": "—"
        })

    # Blank row
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # DEATH STATUS
    table.append({
        "Variable": "**DEATH STATUS**",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    # Death documented
    death_map = mappings["death_status"]["death_documented"]

    # All donors are deceased (100%)
    table.append({
        "Variable": f"{death_map['label']}, n (%)",
        "a) SRTR DCD": _fmt_n_pct(len(populations['a) SRTR DCD']), len(populations['a) SRTR DCD'])),  # 100%
        "b) Matched EHR": _fmt_n_pct(len(populations['b) Matched EHR']), len(populations['b) Matched EHR'])) if len(populations['b) Matched EHR']) > 0 else "—",  # All filtered for death
        "Agreement (a-b)": "100%" if len(dcd_merged) > 0 else "—",  # Perfect agreement since we filtered for dead patients
        "c) SRTR DCD Utilized": _fmt_n_pct(len(populations['c) SRTR DCD Utilized']), len(populations['c) SRTR DCD Utilized'])) if len(populations['c) SRTR DCD Utilized']) > 0 else "—",  # 100%
        "d) Matched EHR Utilized": _fmt_n_pct(len(populations['d) Matched EHR Utilized']), len(populations['d) Matched EHR Utilized'])) if len(populations['d) Matched EHR Utilized']) > 0 else "—",
        "Agreement (c-d)": "100%" if len(dcd_utilized_merged) > 0 else "—"
    })

    # Add note about agreement
    table.append({
        "Variable": "",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    table.append({
        "Variable": "**Note**: Agreement shows mean diff [95% CI] (% within tolerance where applicable) for continuous; % for categorical/binary",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    table.append({
        "Variable": "**Note**: All populations restricted to HIGH confidence matches with patients who died",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    return pd.DataFrame(table)