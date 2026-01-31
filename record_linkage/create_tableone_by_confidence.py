"""
Create Table One by Confidence Level
This module contains functions for generating descriptive Table 1 comparing SRTR and EHR data.
P-values have been removed as requested - these tables focus on descriptive statistics only.
"""

import json
import pandas as pd
import polars as pl
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
from record_linkage.table_one_linkage import (
    CONTINUOUS_TOLERANCES as TOLERANCE_DEFINITIONS,
    calculate_continuous_agreement,
    calculate_binary_agreement,
    calculate_categorical_agreement
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_table_mappings() -> Dict:
    """Load column mappings from JSON file"""
    mapping_file = Path(__file__).parent / "table_one_mappings.json"
    with open(mapping_file, 'r') as f:
        return json.load(f)

def _to_num(s: pd.Series) -> pd.Series:
    """Convert series to numeric, coercing errors to NaN"""
    return pd.to_numeric(s, errors="coerce")


def _fmt_median_iqr(s: pd.Series, digits: int = 2) -> str:
    """Format median and IQR for continuous variables"""
    s = _to_num(s).dropna()
    if len(s) == 0:
        return "N/A"
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def _fmt_missing(s: pd.Series, denom: int) -> str:
    """Format missing data count and percentage"""
    s = _to_num(s)
    miss = int(s.isna().sum())
    pct = (miss / denom * 100) if denom else 0
    return f"{miss} ({pct:.1f}%)"


def _fmt_binary(series: pd.Series, true_vals, denom: int) -> str:
    """Helper function for formatting binary variables"""
    s = series.astype(str).str.strip().str.upper()
    true_vals = {str(v).upper() for v in true_vals}
    n = int(s.isin(true_vals).sum())
    pct = (n / denom * 100) if denom else 0
    return f"{n} ({pct:.1f}%)"

def _fmt_n_pct(count: int, total: int) -> str:
    """Format count and percentage"""
    if total == 0:
        return "0 (0.0%)"
    pct = (count / total) * 100
    return f"{count} ({pct:.1f}%)"


# ==============================================================================
# TABLE CREATION FUNCTIONS
# ==============================================================================

def create_table_one_by_confidence(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    best_matches_df: pl.DataFrame,
    confidence_level: str,
) -> pd.DataFrame:
    """
    Create Table 1 comparing:
    - SRTR Donors (all matched)
    - EHR encounters at a given confidence level (HIGH, MEDIUM, or LOW)

    Note: P-values have been removed from this version.
    This is a descriptive table only.
    """

    # -----------------------------
    # Filter matches
    # -----------------------------
    matches = best_matches_df.filter(pl.col("confidence") == confidence_level)

    donor_ids = matches["DONOR_ID"].unique().to_list()
    encounter_blocks = matches["encounter_block"].unique().to_list()

    donors_pd = (
        donors_df
        .filter(pl.col("DONOR_ID").is_in(donor_ids))
        .to_pandas()
    )

    patients_pd = (
        patients_df
        .filter(pl.col("encounter_block").is_in(encounter_blocks))
        .to_pandas()
    )

    # Get life support data from matches
    matches_pd = matches.to_pandas()

    # Load column mappings from JSON
    mappings = load_table_mappings()

    # Create merged dataset for agreement calculations
    if len(matches_pd) > 0:
        merged_data = matches_pd.merge(
            donors_pd,
            left_on="DONOR_ID",
            right_on="DONOR_ID",
            how="inner"
        ).merge(
            patients_pd,
            left_on="encounter_block",
            right_on="encounter_block",
            how="inner"
        )
    else:
        merged_data = pd.DataFrame()

    # Helper functions for agreement calculations
    def calc_agreement_continuous(srtr_col, ehr_col, tolerance_key=None):
        """Calculate continuous variable agreement"""
        if len(merged_data) == 0 or srtr_col not in merged_data.columns or ehr_col not in merged_data.columns:
            return {"n_paired": 0, "mean_diff_ci": "—", "agreement": "—"}

        srtr_vals = merged_data[srtr_col].values
        ehr_vals = merged_data[ehr_col].values

        # Remove NaN values
        mask = ~(np.isnan(pd.to_numeric(srtr_vals, errors='coerce')) |
                 np.isnan(pd.to_numeric(ehr_vals, errors='coerce')))

        if mask.sum() == 0:
            return {"n_paired": 0, "mean_diff_ci": "—", "agreement": "—"}

        srtr_clean = pd.to_numeric(srtr_vals[mask], errors='coerce')
        ehr_clean = pd.to_numeric(ehr_vals[mask], errors='coerce')

        # Get tolerance config
        tolerance_config = TOLERANCE_DEFINITIONS.get(tolerance_key) if tolerance_key else None

        # Calculate agreement
        result = calculate_continuous_agreement(srtr_clean, ehr_clean, tolerance_config)

        if result['n_paired'] == 0:
            return {"n_paired": 0, "mean_diff_ci": "—", "agreement": "—"}

        # Calculate mean diff with CI
        differences = ehr_clean - srtr_clean
        mean_diff = np.nanmean(differences)
        se = np.nanstd(differences) / np.sqrt(len(differences))
        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se
        mean_diff_str = f"{mean_diff:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"

        # Format agreement with tolerance
        if result['pct_within_tolerance'] is not None:
            agreement_str = f"{result['pct_within_tolerance']:.1f}% in tol"
        else:
            agreement_str = f"MAE: {result['mean_abs_error']:.1f}"

        return {
            "n_paired": result['n_paired'],
            "mean_diff_ci": mean_diff_str,
            "agreement": agreement_str
        }

    def calc_agreement_binary(srtr_col, ehr_col, srtr_positive_vals, ehr_positive_vals):
        """Calculate binary variable agreement"""
        if len(merged_data) == 0 or srtr_col not in merged_data.columns or ehr_col not in merged_data.columns:
            return {"n_paired": 0, "agreement": "—"}

        srtr_vals = merged_data[srtr_col].values
        ehr_vals = merged_data[ehr_col].values

        # Convert to binary based on positive values
        srtr_binary = np.array([1 if val in srtr_positive_vals else 0 for val in srtr_vals])
        ehr_binary = np.array([1 if val in ehr_positive_vals else 0 for val in ehr_vals])

        # Remove missing pairs
        mask = ~(pd.isna(srtr_vals) | pd.isna(ehr_vals))
        if mask.sum() == 0:
            return {"n_paired": 0, "agreement": "—"}

        srtr_clean = srtr_binary[mask]
        ehr_clean = ehr_binary[mask]

        n_paired = len(srtr_clean)
        n_agree = np.sum(srtr_clean == ehr_clean)
        percent_agreement = (n_agree / n_paired) * 100

        # Calculate breakdown
        both_yes = np.sum((srtr_clean == 1) & (ehr_clean == 1))
        both_no = np.sum((srtr_clean == 0) & (ehr_clean == 0))

        percent_both_yes = (both_yes / n_paired) * 100
        percent_both_no = (both_no / n_paired) * 100

        agreement_str = f"{percent_agreement:.1f}% ({percent_both_no:.0f}% No-No, {percent_both_yes:.0f}% Yes-Yes)"

        return {
            "n_paired": n_paired,
            "agreement": agreement_str
        }

    def calc_agreement_categorical(srtr_col, ehr_col, category):
        """Calculate agreement for a specific category"""
        if len(merged_data) == 0 or srtr_col not in merged_data.columns or ehr_col not in merged_data.columns:
            return {"n_paired": 0, "agreement": "—"}

        # Create binary flags for this category
        srtr_flag = (merged_data[srtr_col] == category).astype(int).values
        ehr_flag = (merged_data[ehr_col] == category).astype(int).values

        # Remove missing pairs
        mask = ~(pd.isna(merged_data[srtr_col].values) | pd.isna(merged_data[ehr_col].values))
        if mask.sum() == 0:
            return {"n_paired": 0, "agreement": "—"}

        srtr_clean = srtr_flag[mask]
        ehr_clean = ehr_flag[mask]

        n_paired = len(srtr_clean)
        n_agree = np.sum(srtr_clean == ehr_clean)
        percent_agreement = (n_agree / n_paired) * 100

        return {
            "n_paired": n_paired,
            "agreement": f"{percent_agreement:.1f}%"
        }

    # -----------------------------
    # Attach donor flags to EHR rows
    # -----------------------------
    ehr_with_donor_flags = (
        matches
        .select(["DONOR_ID", "encounter_block"])
        .join(
            donors_df.select(["DONOR_ID", "don_utilized", "DON_NON_HR_BEAT"]),
            on="DONOR_ID",
            how="left"
        )
        .to_pandas()
    )

    table = []

    # -----------------------------
    # Sample size
    # -----------------------------
    table.append({
        "Variable": "N",
        "SRTR Donors": f"{len(donors_pd):,}",
        f"EHR Encounters ({confidence_level})": f"{len(patients_pd):,}",
        "N Paired": f"{len(merged_data):,}",
        "Mean Diff [95% CI]": "—",
        "Agreement": "—"
    })

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # -----------------------------
    # DEMOGRAPHICS
    # -----------------------------
    table.append({
        "Variable": "**DEMOGRAPHICS**",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # Age (years) - using JSON mappings
    age_map = mappings["demographics"]["age_years"]
    age_agreement = calc_agreement_continuous(age_map["srtr"], age_map["ehr"], age_map.get("tolerance_key"))
    table.append({
        "Variable": age_map["label"],
        "SRTR Donors": _fmt_median_iqr(donors_pd[age_map["srtr"]]),
        f"EHR Encounters ({confidence_level})": _fmt_median_iqr(patients_pd[age_map["ehr"]]),
        "N Paired": str(age_agreement["n_paired"]) if age_agreement["n_paired"] > 0 else "—",
        "Mean Diff [95% CI]": age_agreement["mean_diff_ci"],
        "Agreement": age_agreement["agreement"]
    })

    # Age (months) - using JSON mappings
    age_months_map = mappings["demographics"]["age_months"]
    age_months_agreement = calc_agreement_continuous(age_months_map["srtr"], age_months_map["ehr"], age_months_map.get("tolerance_key"))
    table.append({
        "Variable": age_months_map["label"],
        "SRTR Donors": _fmt_median_iqr(donors_pd[age_months_map["srtr"]]),
        f"EHR Encounters ({confidence_level})": _fmt_median_iqr(patients_pd[age_months_map["ehr"]]),
        "N Paired": str(age_months_agreement["n_paired"]) if age_months_agreement["n_paired"] > 0 else "—",
        "Mean Diff [95% CI]": age_months_agreement["mean_diff_ci"],
        "Agreement": age_months_agreement["agreement"]
    })

    table.append({
        "Variable": "Sex/Gender, n (%)",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # Male - using JSON mappings
    male_map = mappings["demographics"]["sex_male"]
    if "gender" in patients_pd.columns:
        male_agreement = calc_agreement_binary(
            male_map["srtr"],
            male_map["ehr"],
            male_map["srtr_positive_values"],
            male_map["ehr_positive_values"]
        )
        ehr_male_str = _fmt_binary(patients_pd[male_map["ehr"]], set(male_map["ehr_positive_values"]), len(patients_pd))
    else:
        male_agreement = {"n_paired": 0, "agreement": "—"}
        ehr_male_str = _fmt_binary(patients_pd["sex_category"], {"MALE"}, len(patients_pd))

    table.append({
        "Variable": f"  {male_map['label']}",
        "SRTR Donors": _fmt_binary(donors_pd[male_map["srtr"]], set(male_map["srtr_positive_values"]), len(donors_pd)),
        f"EHR Encounters ({confidence_level})": ehr_male_str,
        "N Paired": str(male_agreement["n_paired"]) if male_agreement["n_paired"] > 0 else "—",
        "Mean Diff [95% CI]": "—",
        "Agreement": male_agreement["agreement"]
    })

    # Female - using JSON mappings
    female_map = mappings["demographics"]["sex_female"]
    if "gender" in patients_pd.columns:
        female_agreement = calc_agreement_binary(
            female_map["srtr"],
            female_map["ehr"],
            female_map["srtr_positive_values"],
            female_map["ehr_positive_values"]
        )
        ehr_female_str = _fmt_binary(patients_pd[female_map["ehr"]], set(female_map["ehr_positive_values"]), len(patients_pd))
    else:
        female_agreement = {"n_paired": 0, "agreement": "—"}
        ehr_female_str = _fmt_binary(patients_pd["sex_category"], {"FEMALE"}, len(patients_pd))

    table.append({
        "Variable": f"  {female_map['label']}",
        "SRTR Donors": _fmt_binary(donors_pd[female_map["srtr"]], set(female_map["srtr_positive_values"]), len(donors_pd)),
        f"EHR Encounters ({confidence_level})": ehr_female_str,
        "N Paired": str(female_agreement["n_paired"]) if female_agreement["n_paired"] > 0 else "—",
        "Mean Diff [95% CI]": "—",
        "Agreement": female_agreement["agreement"]
    })

    # -----------------------------
    # Race & Ethnicity
    # -----------------------------
    def add_cat(label, donor_col, ehr_col):
        table.append({
            "Variable": label,
            "SRTR Donors": "",
            f"EHR Encounters ({confidence_level})": "",
            "N Paired": "",
            "Mean Diff [95% CI]": "",
            "Agreement": ""
        })
        cats = set(donors_pd[donor_col].dropna()) | set(patients_pd[ehr_col].dropna())
        for c in sorted(cats):
            # Calculate category-specific agreement
            cat_agreement = calc_agreement_categorical(donor_col, ehr_col, c)
            table.append({
                "Variable": f"  {c}",
                "SRTR Donors": _fmt_binary(donors_pd[donor_col], {c}, len(donors_pd)),
                f"EHR Encounters ({confidence_level})": _fmt_binary(patients_pd[ehr_col], {c}, len(patients_pd)),
                "N Paired": str(cat_agreement["n_paired"]) if cat_agreement["n_paired"] > 0 else "—",
                "Mean Diff [95% CI]": "—",
                "Agreement": cat_agreement["agreement"]
            })

    # Use mappings for column names
    race_map = mappings["demographics"]["race"]
    ethnicity_map = mappings["demographics"]["ethnicity"]

    add_cat(f"{race_map['label']}, n (%)", race_map["srtr"], race_map["ehr"])
    add_cat(f"{ethnicity_map['label']}, n (%)", ethnicity_map["srtr"], ethnicity_map["ehr"])

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # -----------------------------
    # CLINICAL MEASUREMENTS
    # -----------------------------
    table.append({
        "Variable": "**CLINICAL MEASUREMENTS**",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # Height - using JSON mappings
    height_map = mappings["clinical_measurements"]["height"]
    height_agreement = calc_agreement_continuous(height_map["srtr"], height_map["ehr"], height_map.get("tolerance_key"))
    table.append({
        "Variable": height_map["label"],
        "SRTR Donors": _fmt_median_iqr(donors_pd[height_map["srtr"]]),
        f"EHR Encounters ({confidence_level})": _fmt_median_iqr(patients_pd[height_map["ehr"]]),
        "N Paired": str(height_agreement["n_paired"]) if height_agreement["n_paired"] > 0 else "—",
        "Mean Diff [95% CI]": height_agreement["mean_diff_ci"],
        "Agreement": height_agreement["agreement"]
    })

    # Weight - using JSON mappings
    weight_map = mappings["clinical_measurements"]["weight"]
    weight_agreement = calc_agreement_continuous(weight_map["srtr"], weight_map["ehr"], weight_map.get("tolerance_key"))
    table.append({
        "Variable": weight_map["label"],
        "SRTR Donors": _fmt_median_iqr(donors_pd[weight_map["srtr"]]),
        f"EHR Encounters ({confidence_level})": _fmt_median_iqr(patients_pd[weight_map["ehr"]]),
        "N Paired": str(weight_agreement["n_paired"]) if weight_agreement["n_paired"] > 0 else "—",
        "Mean Diff [95% CI]": weight_agreement["mean_diff_ci"],
        "Agreement": weight_agreement["agreement"]
    })

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # -----------------------------
    # LABS (with PO2 and pH)
    # -----------------------------
    table.append({
        "Variable": "**LABORATORY VALUES**",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # Process all labs from JSON mappings
    for lab_key, lab_config in mappings["laboratory_values"].items():
        srtr_col = lab_config["srtr"]
        ehr_col = lab_config["ehr"]

        # Skip if SRTR column doesn't exist
        if srtr_col not in donors_pd.columns:
            continue

        # Calculate agreement
        lab_agreement = calc_agreement_continuous(srtr_col, ehr_col, lab_config.get("tolerance_key"))

        # Main lab row
        table.append({
            "Variable": lab_config["label"],
            "SRTR Donors": _fmt_median_iqr(donors_pd[srtr_col], lab_config.get("digits", 2)),
            f"EHR Encounters ({confidence_level})": _fmt_median_iqr(
                patients_pd[ehr_col] if ehr_col in patients_pd.columns else pd.Series(),
                lab_config.get("digits", 2)
            ),
            "N Paired": str(lab_agreement["n_paired"]) if lab_agreement["n_paired"] > 0 else "—",
            "Mean Diff [95% CI]": lab_agreement["mean_diff_ci"],
            "Agreement": lab_agreement["agreement"]
        })

        # Missing row
        table.append({
            "Variable": "  Missing",
            "SRTR Donors": _fmt_missing(donors_pd[srtr_col], len(donors_pd)),
            f"EHR Encounters ({confidence_level})": _fmt_missing(
                patients_pd[ehr_col] if ehr_col in patients_pd.columns else pd.Series(),
                len(patients_pd)
            ),
            "N Paired": "",
            "Mean Diff [95% CI]": "",
            "Agreement": ""
        })

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # -----------------------------
    # LIFE SUPPORT & THERAPIES
    # -----------------------------
    table.append({
        "Variable": "**LIFE SUPPORT & THERAPIES**",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # Life support items from JSON mappings
    for life_support_key, life_support_config in mappings.get("life_support", {}).items():
        if life_support_config["type"] == "binary":
            # IMV and CRRT
            data_source = life_support_config.get("data_source", "patients_pd")
            source_df = matches_pd if data_source == "matches_pd" else patients_pd

            table.append({
                "Variable": f"{life_support_config['label']}, n (%)",
                "SRTR Donors": "—",
                f"EHR Encounters ({confidence_level})": _fmt_binary(
                    source_df[life_support_config["ehr"]],
                    set(life_support_config.get("ehr_positive_values", [True, 1])),
                    len(source_df)
                ) if life_support_config["ehr"] in source_df.columns else "N/A",
                "N Paired": "—",
                "Mean Diff [95% CI]": "—",
                "Agreement": "—"
            })
        elif life_support_config["type"] == "presence":
            # WLST
            if life_support_config["ehr"] in matches_pd.columns:
                wlst_detected = matches_pd[life_support_config["ehr"]].notna().sum()
                wlst_pct = (wlst_detected / len(matches_pd) * 100) if len(matches_pd) else 0
                table.append({
                    "Variable": f"{life_support_config['label']}, n (%)",
                    "SRTR Donors": "—",
                    f"EHR Encounters ({confidence_level})": f"{wlst_detected} ({wlst_pct:.1f}%)",
                    "N Paired": "—",
                    "Mean Diff [95% CI]": "—",
                    "Agreement": "—"
                })

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # -----------------------------
    # MEDICATIONS
    # -----------------------------
    table.append({
        "Variable": "**MEDICATIONS**",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # Process medications from JSON mappings
    for med_key, med_config in mappings["medications"].items():
        # Handle SRTR column
        if med_config["srtr"] and med_config["srtr"] in donors_pd.columns:
            srtr_value = _fmt_binary(
                donors_pd[med_config["srtr"]],
                set(med_config.get("srtr_positive_values", ["Y", 1])),
                len(donors_pd)
            )
        else:
            srtr_value = "—"

        # Handle EHR column
        if med_config["ehr"] in patients_pd.columns:
            ehr_value = _fmt_binary(
                patients_pd[med_config["ehr"]],
                set(med_config.get("ehr_positive_values", [True, 1])),
                len(patients_pd)
            )
        else:
            ehr_value = "N/A"

        # Calculate agreement if both columns exist
        if med_config["srtr"] and med_config["ehr"] and med_config["srtr"] in merged_data.columns and med_config["ehr"] in merged_data.columns:
            med_agreement = calc_agreement_binary(
                med_config["srtr"],
                med_config["ehr"],
                med_config.get("srtr_positive_values", ["Y", 1]),
                med_config.get("ehr_positive_values", [True, 1])
            )
        else:
            med_agreement = {"n_paired": 0, "agreement": "—"}

        table.append({
            "Variable": f"  {med_config['label']}",
            "SRTR Donors": srtr_value,
            f"EHR Encounters ({confidence_level})": ehr_value,
            "N Paired": str(med_agreement["n_paired"]) if med_agreement["n_paired"] > 0 else "—",
            "Mean Diff [95% CI]": "—",
            "Agreement": med_agreement["agreement"]
        })

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # -----------------------------
    # EHR-SPECIFIC
    # -----------------------------
    if "discharge_category" in patients_pd.columns:
        table.append({
            "Variable": "**EHR-SPECIFIC**",
            "SRTR Donors": "",
            f"EHR Encounters ({confidence_level})": "",
            "N Paired": "",
            "Mean Diff [95% CI]": "",
            "Agreement": ""
        })

        # Discharge category
        table.append({
            "Variable": "Discharge Category, n (%)",
            "SRTR Donors": "",
            f"EHR Encounters ({confidence_level})": "",
            "N Paired": "",
            "Mean Diff [95% CI]": "",
            "Agreement": ""
        })

        for c in patients_pd["discharge_category"].dropna().unique():
            table.append({
                "Variable": f"  {c}",
                "SRTR Donors": "—",
                f"EHR Encounters ({confidence_level})": _fmt_binary(
                    patients_pd["discharge_category"], {c}, len(patients_pd)
                ),
                "N Paired": "—",
                "Mean Diff [95% CI]": "—",
                "Agreement": "—"
            })

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Mean Diff [95% CI]": "",
        "Agreement": ""
    })

    # -----------------------------
    # DONOR CHARACTERISTICS (donor-only)
    # -----------------------------
    table.append({
        "Variable": "**DONOR CHARACTERISTICS**",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": ""
    })

    table.append({
        "Variable": "Donor utilized, n (%)",
        "SRTR Donors": _fmt_binary(
            donors_pd["don_utilized"],
            {1, "Y", "YES"},
            len(donors_pd)
        ),
        f"EHR Encounters ({confidence_level})": "—"
    })

    table.append({
        "Variable": "Donation after circulatory death (DCD), n (%)",
        "SRTR Donors": _fmt_binary(
            donors_pd["DON_NON_HR_BEAT"],
            {1, "Y", "YES"},
            len(donors_pd)
        ),
        f"EHR Encounters ({confidence_level})": "—"
    })

    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": ""})

    # -----------------------------
    # DEATH STATUS
    # -----------------------------
    table.append({
        "Variable": "**DEATH STATUS**",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": ""
    })

    table.append({
        "Variable": "Death documented, n (%)",
        "SRTR Donors": f"{len(donors_pd)} (100.0%)",  # All SRTR donors are deceased
        f"EHR Encounters ({confidence_level})": _fmt_binary(
            patients_pd["is_dead"], {1, "1", "TRUE"}, len(patients_pd)
        )
    })

    return pd.DataFrame(table)


def create_table_one_dcd_only(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    best_matches_df: pl.DataFrame,
    confidence_level: str,
) -> pd.DataFrame:
    """
    Create Table 1 for DCD patients only.
    This is a wrapper that filters for DCD donors before calling the main function.
    """

    # Filter for DCD only
    dcd_matches = best_matches_df.filter(
        (pl.col("confidence") == confidence_level) &
        (pl.col("DON_NON_HR_BEAT") == "Y")
    )

    # Use same logic as create_table_one_by_confidence but with DCD subset
    return create_table_one_by_confidence(
        patients_df=patients_df,
        donors_df=donors_df.filter(pl.col("DON_NON_HR_BEAT") == "Y"),
        best_matches_df=dcd_matches,
        confidence_level=confidence_level
    )


def create_table_one_dcd_only_with_agreement(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    best_matches_df: pl.DataFrame,
    confidence_level: str
) -> pd.DataFrame:
    """Create Table One with agreement metrics for DCD-only matched data.
    
    This is a convenience function that filters for DCD donors.
    
    Args:
        patients_df: EHR patients DataFrame
        donors_df: SRTR donors DataFrame  
        best_matches_df: Best matches/linkage DataFrame
        confidence_level: Confidence level to filter on
        
    Returns:
        Table One DataFrame with agreement metrics for DCD donors only
    """
    return create_table_one_with_agreement(
        patients_df=patients_df,
        donors_df=donors_df,
        best_matches_df=best_matches_df,
        confidence_level=confidence_level,
        dcd_only=True
    )

def create_table_one_with_agreement(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    best_matches_df: pl.DataFrame,
    confidence_level: str,
    dcd_only: bool = False
) -> pd.DataFrame:
    """
    Create Table 1 with agreement metrics using JSON mappings.
    
    Args:
        patients_df: Polars DataFrame with patient/encounter data
        donors_df: Polars DataFrame with SRTR donor data
        best_matches_df: Polars DataFrame with linkage results
        confidence_level: Confidence level for filtering ("HIGH", "MEDIUM", "LOW")
        dcd_only: If True, only include DCD donors
        
    Returns:
        Pandas DataFrame with table one statistics and agreement metrics
    """
    # Load JSON mappings
    mappings = load_table_mappings()
    
    # -----------------------------
    # Filter matches (same as original function)
    # -----------------------------
    matches = best_matches_df.filter(pl.col("confidence") == confidence_level)
    
    donor_ids = matches["DONOR_ID"].unique().to_list()
    encounter_blocks = matches["encounter_block"].unique().to_list()
    
    # Convert to pandas after filtering
    donors_pd = (
        donors_df
        .filter(pl.col("DONOR_ID").is_in(donor_ids))
        .to_pandas()
    )
    
    patients_pd = (
        patients_df
        .filter(pl.col("encounter_block").is_in(encounter_blocks))
        .to_pandas()
    )
    
    # Get matches as pandas for merging
    matches_pd = matches.to_pandas()
    
    # Create merged dataset for agreement calculations
    if len(matches_pd) > 0:
        # Merge donor and patient data through matches
        merged_data = matches_pd[["DONOR_ID", "encounter_block"]].copy()
        merged_data = pd.merge(
            merged_data,
            donors_pd,
            on="DONOR_ID",
            how="inner"
        )
        merged_data = pd.merge(
            merged_data,
            patients_pd,
            on="encounter_block",
            how="inner"
        )
    else:
        merged_data = pd.DataFrame()  # Empty if no linkage data
    
    # Filter for DCD only if requested
    if dcd_only:
        donors_pd = donors_pd[donors_pd["DON_NON_HR_BEAT"] == "Y"]
        if not merged_data.empty:
            merged_data = merged_data[merged_data["DON_NON_HR_BEAT"] == "Y"]
            # Also filter patients to only those matched to DCD donors
            dcd_encounter_blocks = merged_data["encounter_block"].unique()
            patients_pd = patients_pd[patients_pd["encounter_block"].isin(dcd_encounter_blocks)]
    
    table = []
    tolerance_footnotes = []
    footnote_counter = 1

    # -----------------------------
    # Sample size
    # -----------------------------
    table.append({
        "Variable": "N",
        "SRTR Donors": f"{len(donors_pd):,}",
        f"EHR Encounters ({confidence_level})": f"{len(patients_pd):,}",
        "N Paired": f"{len(merged_data):,}",
        "Agreement": "—"
    })

    table.append({
        "Variable": "",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Agreement": ""
    })

    # Helper functions for agreement calculations
    def calc_continuous_agreement_v2(donor_col, patient_col, tolerance_key=None, var_name=None):
        """Calculate continuous variable agreement."""
        if merged_data.empty or donor_col not in merged_data.columns or patient_col not in merged_data.columns:
            return "—", 0, ""
        
        # Get paired non-missing data
        paired_data = merged_data[[donor_col, patient_col]].dropna()
        n_paired = len(paired_data)
        
        if n_paired == 0:
            return "—", 0, ""
        
        # Calculate mean difference and CI
        differences = paired_data[donor_col] - paired_data[patient_col]
        mean_diff = differences.mean()
        std_diff = differences.std()
        se_diff = std_diff / np.sqrt(n_paired)
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        # Format mean difference with CI
        mean_diff_str = f"{mean_diff:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]"
        
        # Calculate agreement within tolerance if tolerance specified
        if tolerance_key and tolerance_key in TOLERANCE_DEFINITIONS:
            tolerance = TOLERANCE_DEFINITIONS[tolerance_key]
            if "absolute" in tolerance:
                within_tol = (np.abs(differences) <= tolerance["absolute"]).mean() * 100
            elif "relative" in tolerance:
                rel_diff = np.abs(differences / paired_data[donor_col])
                within_tol = (rel_diff <= tolerance["relative"]).mean() * 100
            else:
                within_tol = None
            
            if within_tol is not None:
                agreement_str = f"{within_tol:.0f}%"
                
                # Add footnote if not already added
                nonlocal footnote_counter
                if "absolute" in tolerance:
                    tol_desc = f"±{tolerance['absolute']} {tolerance.get('unit', 'units')}"
                else:
                    tol_desc = f"±{tolerance['relative']*100:.0f}%"
                footnote_text = f"†{footnote_counter} {var_name or tolerance_key}: {tol_desc}"
                if footnote_text not in tolerance_footnotes:
                    tolerance_footnotes.append(footnote_text)
                    footnote_ref = f"†{footnote_counter}"
                    footnote_counter += 1
                else:
                    # Find the existing footnote number
                    for i, ft in enumerate(tolerance_footnotes):
                        if var_name in ft or tolerance_key in ft:
                            footnote_ref = f"†{i+1}"
                            break
                
                # Combine mean diff and agreement
                return f"{mean_diff_str}; {agreement_str}", n_paired, footnote_ref
        
        return mean_diff_str, n_paired, ""
    
    def calc_binary_agreement_v2(donor_col, patient_col):
        """Calculate binary variable agreement."""
        if merged_data.empty or donor_col not in merged_data.columns or patient_col not in merged_data.columns:
            return "—", 0, ""
        
        # Get paired non-missing data
        paired_data = merged_data[[donor_col, patient_col]].dropna()
        n_paired = len(paired_data)
        
        if n_paired == 0:
            return "—", 0, ""
        
        # Calculate agreement
        both_no = ((paired_data[donor_col] == 0) & (paired_data[patient_col] == 0)).sum()
        both_yes = ((paired_data[donor_col] == 1) & (paired_data[patient_col] == 1)).sum()
        total_agree = both_no + both_yes
        
        pct_agree = (total_agree / n_paired) * 100
        pct_no_no = (both_no / n_paired) * 100
        pct_yes_yes = (both_yes / n_paired) * 100
        
        return f"{pct_agree:.0f}% ({pct_no_no:.0f}%, {pct_yes_yes:.0f}%)", n_paired, ""
    
    def calc_categorical_agreement_v2(donor_col, patient_col, category):
        """Calculate categorical variable agreement."""
        if merged_data.empty or donor_col not in merged_data.columns or patient_col not in merged_data.columns:
            return "—", 0
        
        # Get paired non-missing data
        paired_data = merged_data[[donor_col, patient_col]].dropna()
        n_paired = len(paired_data)
        
        if n_paired == 0:
            return "—", 0
        
        # Calculate agreement for this category
        both_match = ((paired_data[donor_col] == category) & (paired_data[patient_col] == category)).sum()
        pct_agree = (both_match / n_paired) * 100
        
        return f"{pct_agree:.0f}%", n_paired
    
    # -----------------------------
    # DEMOGRAPHICS
    # -----------------------------
    table.append({"Variable": "**DEMOGRAPHICS**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Age - Updated to use JSON mappings
    age_map = mappings["demographics"]["age_years"]
    if age_map["srtr"] in donors_pd.columns and age_map["ehr"] in patients_pd.columns:
        age_agreement, n_paired, footnote = calc_continuous_agreement_v2(
            age_map["srtr"], 
            age_map["ehr"], 
            age_map.get("tolerance_key"),
            age_map["label"]
        )
        table.append({
            "Variable": f"{age_map['label']}, median (Q1-Q3){footnote}",
            "SRTR Donors": _fmt_median_iqr(donors_pd[age_map["srtr"]]),
            f"EHR Encounters ({confidence_level})": _fmt_median_iqr(patients_pd[age_map["ehr"]]),
            "N Paired": str(n_paired) if n_paired > 0 else "—",
            "Agreement": age_agreement
        })
    
    # Age in months for pediatric
    age_months_map = mappings["demographics"]["age_months"]
    if age_months_map["srtr"] in donors_pd.columns and age_months_map["ehr"] in patients_pd.columns:
        # Only show for pediatric patients
        pediatric_donors = donors_pd[donors_pd[mappings["demographics"]["age_years"]["srtr"]] < 18]
        pediatric_patients = patients_pd[patients_pd[mappings["demographics"]["age_years"]["ehr"]] < 18]
        
        if len(pediatric_donors) > 0 or len(pediatric_patients) > 0:
            age_months_agreement, n_paired, footnote = calc_continuous_agreement_v2(
                age_months_map["srtr"],
                age_months_map["ehr"],
                age_months_map.get("tolerance_key"),
                age_months_map["label"]
            )
            table.append({
                "Variable": f"{age_months_map['label']}, median (Q1-Q3){footnote}",
                "SRTR Donors": _fmt_median_iqr(pediatric_donors[age_months_map["srtr"]] if age_months_map["srtr"] in pediatric_donors.columns else pd.Series()),
                f"EHR Encounters ({confidence_level})": _fmt_median_iqr(pediatric_patients[age_months_map["ehr"]] if age_months_map["ehr"] in pediatric_patients.columns else pd.Series()),
                "N Paired": str(n_paired) if n_paired > 0 else "—",
                "Agreement": age_months_agreement
            })
    
    # Gender - Updated to use JSON mappings
    table.append({"Variable": "Gender, n (%)", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Male
    male_map = mappings["demographics"]["sex_male"]
    if male_map["srtr"] in donors_pd.columns and male_map["ehr"] in patients_pd.columns:
        # Convert to binary for agreement calculation
        if not merged_data.empty and male_map["srtr"] in merged_data.columns and male_map["ehr"] in merged_data.columns:
            merged_data["donor_male"] = merged_data[male_map["srtr"]].isin(male_map["srtr_positive_values"]).astype(int)
            merged_data["patient_male"] = merged_data[male_map["ehr"]].isin(male_map["ehr_positive_values"]).astype(int)
            agreement, n_paired, _ = calc_binary_agreement_v2("donor_male", "patient_male")
        else:
            agreement = "—"
            n_paired = 0
        
        table.append({
            "Variable": f"  {male_map['label']}",
            "SRTR Donors": _fmt_binary(donors_pd[male_map["srtr"]], male_map["srtr_positive_values"], len(donors_pd)),
            f"EHR Encounters ({confidence_level})": _fmt_binary(patients_pd[male_map["ehr"]], male_map["ehr_positive_values"], len(patients_pd)),
            "N Paired": str(n_paired) if n_paired > 0 else "—",
            "Agreement": agreement
        })
    
    # Female  
    female_map = mappings["demographics"]["sex_female"]
    if female_map["srtr"] in donors_pd.columns and female_map["ehr"] in patients_pd.columns:
        # Convert to binary for agreement calculation
        if not merged_data.empty and female_map["srtr"] in merged_data.columns and female_map["ehr"] in merged_data.columns:
            merged_data["donor_female"] = merged_data[female_map["srtr"]].isin(female_map["srtr_positive_values"]).astype(int)
            merged_data["patient_female"] = merged_data[female_map["ehr"]].isin(female_map["ehr_positive_values"]).astype(int)
            agreement, n_paired, _ = calc_binary_agreement_v2("donor_female", "patient_female")
        else:
            agreement = "—"
            n_paired = 0
        
        table.append({
            "Variable": f"  {female_map['label']}",
            "SRTR Donors": _fmt_binary(donors_pd[female_map["srtr"]], female_map["srtr_positive_values"], len(donors_pd)),
            f"EHR Encounters ({confidence_level})": _fmt_binary(patients_pd[female_map["ehr"]], female_map["ehr_positive_values"], len(patients_pd)),
            "N Paired": str(n_paired) if n_paired > 0 else "—",
            "Agreement": agreement
        })
    
    # Race - Updated to use JSON mappings
    race_map = mappings["demographics"]["race"]
    if race_map["srtr"] in donors_pd.columns and race_map["ehr"] in patients_pd.columns:
        table.append({"Variable": f"{race_map['label']}, n (%)", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
        
        # Get all unique race categories from both datasets
        all_races = set()
        if race_map["srtr"] in donors_pd.columns:
            all_races.update(donors_pd[race_map["srtr"]].dropna().unique())
        if race_map["ehr"] in patients_pd.columns:
            all_races.update(patients_pd[race_map["ehr"]].dropna().unique())
        
        for race in sorted(all_races):
            agreement, n_paired = calc_categorical_agreement_v2(race_map["srtr"], race_map["ehr"], race)
            table.append({
                "Variable": f"  {race}",
                "SRTR Donors": _fmt_binary(donors_pd[race_map["srtr"]], {race}, len(donors_pd)),
                f"EHR Encounters ({confidence_level})": _fmt_binary(patients_pd[race_map["ehr"]], {race}, len(patients_pd)),
                "N Paired": str(n_paired) if n_paired > 0 else "—",
                "Agreement": agreement
            })
    
    # Ethnicity - Updated to use JSON mappings
    ethnicity_map = mappings["demographics"]["ethnicity"]
    if ethnicity_map["srtr"] in donors_pd.columns and ethnicity_map["ehr"] in patients_pd.columns:
        table.append({"Variable": f"{ethnicity_map['label']}, n (%)", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
        
        # Get all unique ethnicity categories from both datasets
        all_ethnicities = set()
        if ethnicity_map["srtr"] in donors_pd.columns:
            all_ethnicities.update(donors_pd[ethnicity_map["srtr"]].dropna().unique())
        if ethnicity_map["ehr"] in patients_pd.columns:
            all_ethnicities.update(patients_pd[ethnicity_map["ehr"]].dropna().unique())
        
        for ethnicity in sorted(all_ethnicities):
            agreement, n_paired = calc_categorical_agreement_v2(ethnicity_map["srtr"], ethnicity_map["ehr"], ethnicity)
            table.append({
                "Variable": f"  {ethnicity}",
                "SRTR Donors": _fmt_binary(donors_pd[ethnicity_map["srtr"]], {ethnicity}, len(donors_pd)),
                f"EHR Encounters ({confidence_level})": _fmt_binary(patients_pd[ethnicity_map["ehr"]], {ethnicity}, len(patients_pd)),
                "N Paired": str(n_paired) if n_paired > 0 else "—",
                "Agreement": agreement
            })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # -----------------------------
    # CLINICAL MEASUREMENTS
    # -----------------------------
    table.append({"Variable": "**CLINICAL MEASUREMENTS**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Height
    height_map = mappings["clinical_measurements"]["height"]
    if height_map["srtr"] in donors_pd.columns and height_map["ehr"] in patients_pd.columns:
        agreement, n_paired, footnote = calc_continuous_agreement_v2(
            height_map["srtr"],
            height_map["ehr"],
            height_map.get("tolerance_key"),
            height_map["label"]
        )
        table.append({
            "Variable": f"{height_map['label']}, median (Q1-Q3){footnote}",
            "SRTR Donors": _fmt_median_iqr(donors_pd[height_map["srtr"]]),
            f"EHR Encounters ({confidence_level})": _fmt_median_iqr(patients_pd[height_map["ehr"]]),
            "N Paired": str(n_paired) if n_paired > 0 else "—",
            "Agreement": agreement
        })
    
    # Weight
    weight_map = mappings["clinical_measurements"]["weight"]
    if weight_map["srtr"] in donors_pd.columns and weight_map["ehr"] in patients_pd.columns:
        agreement, n_paired, footnote = calc_continuous_agreement_v2(
            weight_map["srtr"],
            weight_map["ehr"],
            weight_map.get("tolerance_key"),
            weight_map["label"]
        )
        table.append({
            "Variable": f"{weight_map['label']}, median (Q1-Q3){footnote}",
            "SRTR Donors": _fmt_median_iqr(donors_pd[weight_map["srtr"]]),
            f"EHR Encounters ({confidence_level})": _fmt_median_iqr(patients_pd[weight_map["ehr"]]),
            "N Paired": str(n_paired) if n_paired > 0 else "—",
            "Agreement": agreement
        })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # -----------------------------
    # LABORATORY VALUES
    # -----------------------------
    table.append({"Variable": "**LABORATORY VALUES**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Process each lab value from mappings
    for lab_key, lab_map in mappings["laboratory_values"].items():
        if lab_map["srtr"] in donors_pd.columns and lab_map["ehr"] in patients_pd.columns:
            agreement, n_paired, footnote = calc_continuous_agreement_v2(
                lab_map["srtr"],
                lab_map["ehr"],
                lab_map.get("tolerance_key"),
                lab_map["label"]
            )
            
            # Format with appropriate decimal places
            digits = lab_map.get("digits", 1)
            if digits == 2:
                donor_val = _fmt_median_iqr(donors_pd[lab_map["srtr"]], digits=2)
                patient_val = _fmt_median_iqr(patients_pd[lab_map["ehr"]], digits=2)
            else:
                donor_val = _fmt_median_iqr(donors_pd[lab_map["srtr"]], digits=1)
                patient_val = _fmt_median_iqr(patients_pd[lab_map["ehr"]], digits=1)
            
            table.append({
                "Variable": f"{lab_map['label']}, median (Q1-Q3){footnote}",
                "SRTR Donors": donor_val,
                f"EHR Encounters ({confidence_level})": patient_val,
                "N Paired": str(n_paired) if n_paired > 0 else "—",
                "Agreement": agreement
            })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # -----------------------------
    # LIFE SUPPORT
    # -----------------------------
    table.append({"Variable": "**LIFE SUPPORT**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Process life support measures from mappings
    for ls_key, ls_map in mappings["life_support"].items():
        # Get data source (matches_pd or patients_pd)
        data_source = ls_map.get("data_source", "patients_pd")
        if data_source == "matches_pd":
            source_df = matches_pd
        else:
            source_df = patients_pd
            
        if ls_map["type"] == "binary":
            # For binary life support measures
            if ls_map["ehr"] in source_df.columns:
                patient_val = _fmt_binary(
                    source_df[ls_map["ehr"]], 
                    ls_map.get("ehr_positive_values", [True]), 
                    len(source_df)
                )
            else:
                patient_val = "—"
            
            table.append({
                "Variable": f"{ls_map['label']}, n (%)",
                "SRTR Donors": "—",  # No SRTR data for these
                f"EHR Encounters ({confidence_level})": patient_val,
                "N Paired": "—",
                "Agreement": "—"
            })
        elif ls_map["type"] == "presence":
            # For presence/absence indicators (like WLST)
            if ls_map["ehr"] in source_df.columns:
                n_with_value = source_df[ls_map["ehr"]].notna().sum()
                patient_val = _fmt_n_pct(n_with_value, len(source_df))
            else:
                patient_val = "—"
            
            table.append({
                "Variable": f"{ls_map['label']}, n (%)",
                "SRTR Donors": "—",
                f"EHR Encounters ({confidence_level})": patient_val,
                "N Paired": "—",
                "Agreement": "—"
            })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # -----------------------------
    # MEDICATIONS
    # -----------------------------
    table.append({"Variable": "**MEDICATIONS**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Process medications from JSON mappings
    for med_key, med_map in mappings["medications"].items():
        donor_has_col = med_map["srtr"] and med_map["srtr"] in donors_pd.columns
        patient_has_col = med_map["ehr"] and med_map["ehr"] in patients_pd.columns
        
        # Get donor value
        if donor_has_col:
            donor_val = _fmt_binary(
                donors_pd[med_map["srtr"]], 
                med_map.get("srtr_positive_values", ["Y", 1]),
                len(donors_pd)
            )
        else:
            donor_val = "—"
        
        # Get patient value
        if patient_has_col:
            patient_val = _fmt_binary(
                patients_pd[med_map["ehr"]],
                med_map.get("ehr_positive_values", [True]),
                len(patients_pd)
            )
        else:
            patient_val = "—"
        
        # Calculate agreement if both columns exist
        if donor_has_col and patient_has_col and not merged_data.empty:
            if med_map["srtr"] in merged_data.columns and med_map["ehr"] in merged_data.columns:
                # Convert to binary for agreement calculation
                merged_data[f"{med_key}_donor_binary"] = merged_data[med_map["srtr"]].isin(
                    med_map.get("srtr_positive_values", ["Y", 1])
                ).astype(int)
                merged_data[f"{med_key}_patient_binary"] = merged_data[med_map["ehr"]].isin(
                    med_map.get("ehr_positive_values", [True])
                ).astype(int)
                agreement, n_paired, _ = calc_binary_agreement_v2(
                    f"{med_key}_donor_binary",
                    f"{med_key}_patient_binary"
                )
            else:
                agreement = "—"
                n_paired = 0
        else:
            agreement = "—"
            n_paired = 0
        
        table.append({
            "Variable": f"  {med_map['label']}",
            "SRTR Donors": donor_val,
            f"EHR Encounters ({confidence_level})": patient_val,
            "N Paired": str(n_paired) if n_paired > 0 else "—",
            "Agreement": agreement
        })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # -----------------------------
    # EHR-SPECIFIC FIELDS
    # -----------------------------
    if not dcd_only:
        table.append({"Variable": "**EHR-SPECIFIC**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
        
        # Discharge category from mappings
        discharge_map = mappings.get("ehr_specific", {}).get("discharge_category", {})
        if discharge_map and discharge_map["ehr"] in patients_pd.columns:
            table.append({
                "Variable": f"{discharge_map['label']}, n (%)",
                "SRTR Donors": "",
                f"EHR Encounters ({confidence_level})": "",
                "N Paired": "",
                "Agreement": ""
            })
            for discharge_cat in patients_pd[discharge_map["ehr"]].unique():
                if pd.notna(discharge_cat):
                    table.append({
                        "Variable": f"  {discharge_cat}",
                        "SRTR Donors": "—",
                        f"EHR Encounters ({confidence_level})": _fmt_binary(
                            patients_pd[discharge_map["ehr"]], 
                            {discharge_cat}, 
                            len(patients_pd)
                        ),
                        "N Paired": "—",
                        "Agreement": "—"
                    })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # -----------------------------
    # DONOR CHARACTERISTICS
    # -----------------------------
    table.append({"Variable": "**DONOR CHARACTERISTICS**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Donor utilized from mappings
    donor_util_map = mappings["donor_characteristics"]["donor_utilized"]
    if donor_util_map["srtr"] in donors_pd.columns:
        donor_val = _fmt_binary(
            donors_pd[donor_util_map["srtr"]],
            donor_util_map.get("srtr_positive_values", [1, "Y", "YES"]),
            len(donors_pd)
        )
        table.append({
            "Variable": f"{donor_util_map['label']}, n (%)",
            "SRTR Donors": donor_val,
            f"EHR Encounters ({confidence_level})": "—",
            "N Paired": "—",
            "Agreement": "—"
        })
    
    # DCD from mappings  
    dcd_map = mappings["donor_characteristics"]["dcd"]
    if dcd_map["srtr"] in donors_pd.columns:
        donor_val = _fmt_binary(
            donors_pd[dcd_map["srtr"]],
            dcd_map.get("srtr_positive_values", [1, "Y", "YES"]),
            len(donors_pd)
        )
        table.append({
            "Variable": f"{dcd_map['label']}, n (%)",
            "SRTR Donors": donor_val,
            f"EHR Encounters ({confidence_level})": "—",
            "N Paired": "—",
            "Agreement": "—"
        })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # -----------------------------
    # DEATH STATUS
    # -----------------------------
    table.append({"Variable": "**DEATH STATUS**", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Death documented from mappings
    death_map = mappings["death_status"]["death_documented"]
    if death_map["ehr"] in patients_pd.columns:
        # All donors are deceased by definition
        donor_val = _fmt_n_pct(len(donors_pd), len(donors_pd))  # 100%
        
        # Calculate patient deaths
        patient_val = _fmt_binary(
            patients_pd[death_map["ehr"]],
            death_map.get("ehr_positive_values", [1, "1", "TRUE"]),
            len(patients_pd)
        )
        
        # Calculate agreement if we have merged data
        if not merged_data.empty and death_map["ehr"] in merged_data.columns:
            # Create binary columns for agreement
            merged_data["donor_death"] = 1  # All donors died
            merged_data["patient_death"] = merged_data[death_map["ehr"]].isin(
                death_map.get("ehr_positive_values", [1, "1", "TRUE"])
            ).astype(int)
            agreement, n_paired, _ = calc_binary_agreement_v2("donor_death", "patient_death")
        else:
            agreement = "—"
            n_paired = 0
        
        table.append({
            "Variable": f"{death_map['label']}, n (%)",
            "SRTR Donors": donor_val,
            f"EHR Encounters ({confidence_level})": patient_val,
            "N Paired": str(n_paired) if n_paired > 0 else "—",
            "Agreement": agreement
        })
    
    table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
    
    # Add footnote about medication agreement
    table.append({
        "Variable": "**Note**: Binary agreement shows: Total% (No-No%, Yes-Yes%)",
        "SRTR Donors": "",
        f"EHR Encounters ({confidence_level})": "",
        "N Paired": "",
        "Agreement": ""
    })
    
    # Add tolerance footnotes if any
    if tolerance_footnotes:
        table.append({"Variable": "", "SRTR Donors": "", f"EHR Encounters ({confidence_level})": "", "N Paired": "", "Agreement": ""})
        table.append({
            "Variable": "**Tolerance Definitions:**",
            "SRTR Donors": "",
            f"EHR Encounters ({confidence_level})": "",
            "N Paired": "",
            "Agreement": ""
        })
        for footnote_text in tolerance_footnotes:
            table.append({
                "Variable": footnote_text,
                "SRTR Donors": "",
                f"EHR Encounters ({confidence_level})": "",
                "N Paired": "",
                "Agreement": ""
            })
    
    # Convert to DataFrame and return
    return pd.DataFrame(table)
