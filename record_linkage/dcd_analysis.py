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
    best_matches_df: pl.DataFrame
) -> pd.DataFrame:
    """
    Create summary table of donor characteristics focusing on DCD.

    Args:
        srtr_df: SRTR donor data (Polars DataFrame)
        best_matches_df: Best matches with confidence levels (Polars DataFrame)

    Returns:
        DataFrame with donor characteristics summary
    """
    # Convert to pandas for easier manipulation
    srtr_pd = srtr_df.to_pandas()
    matches_pd = best_matches_df.to_pandas()

    # Load mappings
    mappings = load_table_mappings()

    # Get unique matched donors
    matched_donor_ids = matches_pd['DONOR_ID'].unique()
    matched_donors_pd = srtr_pd[srtr_pd['DONOR_ID'].isin(matched_donor_ids)]

    # Calculate DCD statistics
    total_donors = len(matched_donors_pd)

    # DCD donors
    dcd_donors = matched_donors_pd[matched_donors_pd['DON_NON_HR_BEAT'] == 'Y']
    n_dcd = len(dcd_donors)

    # DCD utilized
    dcd_utilized = dcd_donors[dcd_donors['don_utilized'] == 1]
    n_dcd_utilized = len(dcd_utilized)

    # Get confidence level breakdown for DCD donors
    dcd_donor_ids = dcd_donors['DONOR_ID'].unique()
    dcd_matches = matches_pd[matches_pd['DONOR_ID'].isin(dcd_donor_ids)]

    # Count by confidence level
    confidence_counts = dcd_matches.groupby('confidence')['DONOR_ID'].nunique().to_dict()

    # Build summary table
    table = []

    # Header
    table.append({
        "Characteristic": "**DONOR CHARACTERISTICS SUMMARY**",
        "Value": "",
        "Notes": ""
    })

    # Total matched donors
    table.append({
        "Characteristic": "Total Matched Donors",
        "Value": f"{total_donors:,}",
        "Notes": ""
    })

    # Demographics
    table.append({
        "Characteristic": "",
        "Value": "",
        "Notes": ""
    })

    table.append({
        "Characteristic": "**Demographics**",
        "Value": "",
        "Notes": ""
    })

    # Age
    age_col = mappings["demographics"]["age_years"]["srtr"]
    if age_col in matched_donors_pd.columns:
        table.append({
            "Characteristic": "Age (years), median [Q1-Q3]",
            "Value": _fmt_median_iqr(matched_donors_pd[age_col]),
            "Notes": ""
        })

    # Gender
    gender_col = mappings["demographics"]["sex_male"]["srtr"]
    if gender_col in matched_donors_pd.columns:
        n_male = (matched_donors_pd[gender_col] == 'M').sum()
        table.append({
            "Characteristic": "Male, n (%)",
            "Value": _fmt_n_pct(n_male, total_donors),
            "Notes": ""
        })

    # Race
    race_col = mappings["demographics"]["race"]["srtr"]
    if race_col in matched_donors_pd.columns:
        table.append({
            "Characteristic": "Race, n (%)",
            "Value": "",
            "Notes": ""
        })
        race_counts = matched_donors_pd[race_col].value_counts()
        for race, count in race_counts.items():
            if pd.notna(race):
                table.append({
                    "Characteristic": f"  {race}",
                    "Value": _fmt_n_pct(count, total_donors),
                    "Notes": ""
                })

    # Blank row
    table.append({
        "Characteristic": "",
        "Value": "",
        "Notes": ""
    })

    # Donation Type
    table.append({
        "Characteristic": "**Donation Type**",
        "Value": "",
        "Notes": ""
    })

    # DCD status
    table.append({
        "Characteristic": "Donation after Circulatory Death (DCD), n (%)",
        "Value": _fmt_n_pct(n_dcd, total_donors),
        "Notes": ""
    })

    # DCD utilized
    table.append({
        "Characteristic": "DCD Donors Utilized, n (%)",
        "Value": _fmt_n_pct(n_dcd_utilized, n_dcd),
        "Notes": f"% of DCD donors"
    })

    # Blank row
    table.append({
        "Characteristic": "",
        "Value": "",
        "Notes": ""
    })

    # Confidence matching for DCD donors
    table.append({
        "Characteristic": "**DCD Donor Matching Confidence**",
        "Value": "",
        "Notes": ""
    })

    for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
        n_conf = confidence_counts.get(conf_level, 0)
        table.append({
            "Characteristic": f"{conf_level} confidence, n (%)",
            "Value": _fmt_n_pct(n_conf, n_dcd),
            "Notes": f"% of DCD donors"
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

    # Filter for patients who died (assuming is_dead column exists)
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
    populations = {
        'a) SRTR DCD': matched_srtr[matched_srtr['DON_NON_HR_BEAT'] == 'Y'],
        'b) Matched EHR': matched_patients[matched_patients['encounter_block'].isin(
            merged_data[merged_data['DON_NON_HR_BEAT'] == 'Y']['encounter_block']
        )],
        'c) SRTR DCD Utilized': matched_srtr[
            (matched_srtr['DON_NON_HR_BEAT'] == 'Y') &
            (matched_srtr['don_utilized'] == 1)
        ],
        'd) Matched EHR Utilized': matched_patients[matched_patients['encounter_block'].isin(
            merged_data[
                (merged_data['DON_NON_HR_BEAT'] == 'Y') &
                (merged_data['don_utilized'] == 1)
            ]['encounter_block']
        )]
    }

    # For agreement calculations
    dcd_merged = merged_data[merged_data['DON_NON_HR_BEAT'] == 'Y']
    dcd_utilized_merged = merged_data[
        (merged_data['DON_NON_HR_BEAT'] == 'Y') &
        (merged_data['don_utilized'] == 1)
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
    for lab_key in ["creatinine", "bilirubin", "ast", "alt", "sodium", "bun"]:
        if lab_key not in mappings["laboratory_values"]:
            continue
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
        "Variable": "**Note**: Agreement shows mean diff [95% CI] (% within tolerance where applicable)",
        "a) SRTR DCD": "",
        "b) Matched EHR": "",
        "Agreement (a-b)": "",
        "c) SRTR DCD Utilized": "",
        "d) Matched EHR Utilized": "",
        "Agreement (c-d)": ""
    })

    return pd.DataFrame(table)