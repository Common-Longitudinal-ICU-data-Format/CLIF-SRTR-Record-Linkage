"""
Table One Linkage Agreement Module
Provides functions for calculating record linkage agreement metrics
and generating validation tables for CLIF-SRTR matching
"""

import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from typing import Dict, Tuple, Optional, List, Any
import warnings

# Define variable-specific tolerances for continuous variables
CONTINUOUS_TOLERANCES = {
    'age': {'absolute': 1, 'relative': None, 'units': 'years'},
    'age_months': {'absolute': 12, 'relative': None, 'units': 'months'},
    'height': {'absolute': 5, 'relative': None, 'units': 'cm'},
    'weight': {'absolute': 5, 'relative': 0.1, 'units': 'kg'},  # 5kg or 10%
    'creatinine': {'absolute': 0.5, 'relative': 0.2, 'units': 'mg/dL'},  # 0.5 or 20%
    'bilirubin': {'absolute': 1.0, 'relative': 0.2, 'units': 'mg/dL'},
    'ast': {'absolute': 20, 'relative': 0.2, 'units': 'U/L'},
    'alt': {'absolute': 20, 'relative': 0.2, 'units': 'U/L'},
    'po2': {'absolute': 10, 'relative': None, 'units': 'mmHg'},
    'ph': {'absolute': 0.1, 'relative': None, 'units': ''},
    # Additional lab tolerances (can be adjusted based on clinical guidelines)
    'pco2': {'absolute': 5, 'relative': None, 'units': 'mmHg'},  # ±5 mmHg
    'bun': {'absolute': 5, 'relative': 0.2, 'units': 'mg/dL'},  # ±5 or 20%
    'sodium': {'absolute': 3, 'relative': None, 'units': 'mEq/L'},  # ±3 mEq/L
    'inr': {'absolute': 0.2, 'relative': None, 'units': ''},  # ±0.2
    'troponin_i': {'absolute': 0.1, 'relative': 0.2, 'units': 'ng/mL'},  # ±0.1 or 20%
    'troponin_t': {'absolute': 0.05, 'relative': 0.2, 'units': 'ng/mL'},  # ±0.05 or 20%
    'pf_ratio': {'absolute': 50, 'relative': None, 'units': ''},  # ±50
}


def calculate_continuous_agreement(
    srtr_values: np.ndarray,
    ehr_values: np.ndarray,
    tolerance_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculate agreement metrics for continuous variables

    Parameters:
    -----------
    srtr_values : array-like
        SRTR values
    ehr_values : array-like
        EHR values
    tolerance_config : dict
        Configuration for tolerance (absolute and/or relative)

    Returns:
    --------
    Dictionary with agreement metrics
    """
    # Remove pairs where either value is missing
    mask = ~(np.isnan(srtr_values) | np.isnan(ehr_values))
    srtr_clean = srtr_values[mask]
    ehr_clean = ehr_values[mask]

    if len(srtr_clean) == 0:
        return {
            'n_paired': 0,
            'median_abs_diff': None,
            'mean_abs_error': None,
            'correlation': None,
            'pct_within_tolerance': None,
            'bland_altman_mean': None,
            'bland_altman_lower': None,
            'bland_altman_upper': None
        }

    # Calculate differences
    differences = ehr_clean - srtr_clean
    abs_differences = np.abs(differences)

    # Basic metrics
    median_abs_diff = np.median(abs_differences)
    mean_abs_error = np.mean(abs_differences)

    # Correlation
    if len(srtr_clean) > 1:
        correlation = np.corrcoef(srtr_clean, ehr_clean)[0, 1]
    else:
        correlation = None

    # Calculate percentage within tolerance
    if tolerance_config:
        within_tolerance = 0
        # Convert to arrays to avoid index issues
        srtr_array = srtr_clean.values if hasattr(srtr_clean, 'values') else np.array(srtr_clean)
        abs_diff_array = abs_differences.values if hasattr(abs_differences, 'values') else np.array(abs_differences)

        for i in range(len(srtr_array)):
            abs_tol = tolerance_config.get('absolute', float('inf'))
            rel_tol = tolerance_config.get('relative')

            # Use the smaller of absolute or relative tolerance
            tolerance = abs_tol
            if rel_tol and srtr_array[i] != 0:
                rel_tolerance = abs(srtr_array[i] * rel_tol)
                tolerance = min(abs_tol, rel_tolerance) if abs_tol else rel_tolerance

            if abs_diff_array[i] <= tolerance:
                within_tolerance += 1

        pct_within_tolerance = (within_tolerance / len(srtr_clean)) * 100
    else:
        pct_within_tolerance = None

    # Bland-Altman limits of agreement
    mean_diff = np.mean(differences)
    sd_diff = np.std(differences)
    bland_altman_lower = mean_diff - 1.96 * sd_diff
    bland_altman_upper = mean_diff + 1.96 * sd_diff

    return {
        'n_paired': len(srtr_clean),
        'median_abs_diff': median_abs_diff,
        'mean_abs_error': mean_abs_error,
        'correlation': correlation,
        'pct_within_tolerance': pct_within_tolerance,
        'bland_altman_mean': mean_diff,
        'bland_altman_lower': bland_altman_lower,
        'bland_altman_upper': bland_altman_upper
    }


def calculate_binary_agreement(
    srtr_values: np.ndarray,
    ehr_values: np.ndarray,
    srtr_positive: Any,
    ehr_positive: Any
) -> Dict[str, Any]:
    """
    Calculate agreement metrics for binary variables

    Parameters:
    -----------
    srtr_values : array-like
        SRTR values
    ehr_values : array-like
        EHR values
    srtr_positive : Any
        Value in SRTR that represents positive/yes
    ehr_positive : Any
        Value in EHR that represents positive/yes

    Returns:
    --------
    Dictionary with agreement metrics
    """
    # Convert to binary
    srtr_binary = np.array([1 if str(v).upper() == str(srtr_positive).upper() else 0
                            for v in srtr_values])
    ehr_binary = np.array([1 if str(v).upper() == str(ehr_positive).upper() else 0
                          for v in ehr_values])

    # Remove pairs where either is missing
    mask = ~(pd.isna(srtr_values) | pd.isna(ehr_values))
    srtr_clean = srtr_binary[mask]
    ehr_clean = ehr_binary[mask]

    if len(srtr_clean) == 0:
        return {
            'n_paired': 0,
            'pct_agreement': None,
            'sensitivity': None,
            'specificity': None,
            'mcnemar_pvalue': None
        }

    # Calculate metrics
    agreement = np.sum(srtr_clean == ehr_clean)
    pct_agreement = (agreement / len(srtr_clean)) * 100

    # Sensitivity and specificity (treating SRTR as reference)
    true_positive = np.sum((srtr_clean == 1) & (ehr_clean == 1))
    true_negative = np.sum((srtr_clean == 0) & (ehr_clean == 0))
    false_positive = np.sum((srtr_clean == 0) & (ehr_clean == 1))
    false_negative = np.sum((srtr_clean == 1) & (ehr_clean == 0))

    sensitivity = (true_positive / (true_positive + false_negative) * 100
                  if (true_positive + false_negative) > 0 else None)
    specificity = (true_negative / (true_negative + false_positive) * 100
                  if (true_negative + false_positive) > 0 else None)

    # McNemar's test
    if false_positive + false_negative > 0:
        # Use binomial test for small samples
        if false_positive + false_negative < 25:
            from scipy.stats import binomtest
            mcnemar_result = binomtest(false_positive, false_positive + false_negative, 0.5)
            mcnemar_pvalue = mcnemar_result.pvalue
        else:
            # Use McNemar's chi-squared for larger samples
            chi2 = ((abs(false_positive - false_negative) - 1) ** 2) / (false_positive + false_negative)
            mcnemar_pvalue = 1 - stats.chi2.cdf(chi2, 1)
    else:
        mcnemar_pvalue = 1.0

    return {
        'n_paired': len(srtr_clean),
        'pct_agreement': pct_agreement,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcnemar_pvalue': mcnemar_pvalue
    }


def calculate_categorical_agreement(
    srtr_values: np.ndarray,
    ehr_values: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate agreement metrics for categorical variables

    Parameters:
    -----------
    srtr_values : array-like
        SRTR values
    ehr_values : array-like
        EHR values

    Returns:
    --------
    Dictionary with agreement metrics
    """
    # Remove pairs where either is missing
    mask = ~(pd.isna(srtr_values) | pd.isna(ehr_values))
    srtr_clean = srtr_values[mask]
    ehr_clean = ehr_values[mask]

    if len(srtr_clean) == 0:
        return {
            'n_paired': 0,
            'pct_exact_match': None,
            'cohen_kappa': None,
            'weighted_kappa': None
        }

    # Calculate exact match percentage
    exact_matches = np.sum(srtr_clean == ehr_clean)
    pct_exact_match = (exact_matches / len(srtr_clean)) * 100

    # Cohen's kappa
    if len(np.unique(srtr_clean)) > 1 or len(np.unique(ehr_clean)) > 1:
        cohen_kappa = cohen_kappa_score(srtr_clean, ehr_clean)
    else:
        cohen_kappa = None

    return {
        'n_paired': len(srtr_clean),
        'pct_exact_match': pct_exact_match,
        'cohen_kappa': cohen_kappa,
        'weighted_kappa': None  # Could implement if ordinal categories
    }


def create_linkage_agreement_table(
    best_matches_df: pl.DataFrame,
    srtr_df: pl.DataFrame,
    patients_pd: pd.DataFrame,
    confidence_level: str = "HIGH"
) -> pd.DataFrame:
    """
    Create the main linkage agreement table without p-values

    Parameters:
    -----------
    best_matches_df : pl.DataFrame
        Best matches dataframe (one per donor)
    srtr_df : pl.DataFrame
        SRTR donor data
    patients_pd : pd.DataFrame
        EHR patient data
    confidence_level : str
        Confidence level to filter by (HIGH, MEDIUM, LOW, or ALL)

    Returns:
    --------
    pd.DataFrame with linkage agreement metrics
    """
    # Filter by confidence if specified
    if confidence_level != "ALL":
        working_df = best_matches_df.filter(pl.col("confidence") == confidence_level)
    else:
        working_df = best_matches_df

    # Join with SRTR data
    matched_data = working_df.join(srtr_df, on="DONOR_ID", how="inner")

    # Convert to pandas for easier manipulation
    matched_pd = matched_data.to_pandas()

    # Join with patient data using encounter_block
    matched_pd = matched_pd.merge(
        patients_pd,
        left_on="encounter_block",
        right_on="encounter_block",
        how="left",
        suffixes=("", "_patient")
    )

    # Initialize results list
    results = []

    # Define ALL variable mappings and types (matching original table one)
    variable_mappings = {
        # ==================== DEMOGRAPHICS ====================
        'Age (years)': {
            'srtr': 'DON_AGE',
            'ehr': 'age_at_admission',
            'type': 'continuous',
            'tolerance_key': 'age'
        },
        'Age (months)': {
            'srtr': 'DON_AGE_IN_MONTHS',
            'ehr': 'age_at_admission',  # Will multiply by 12
            'type': 'continuous',
            'tolerance_key': 'age_months',
            'ehr_transform': lambda x: x * 12
        },
        'Sex (Male)': {
            'srtr': 'DON_GENDER',
            'ehr': 'sex_category',
            'type': 'binary',
            'srtr_positive': 'M',
            'ehr_positive': 'MALE'
        },
        'Race': {
            'srtr': 'DON_race_std',
            'ehr': 'race_std',
            'type': 'categorical'
        },
        'Ethnicity': {
            'srtr': 'DON_ethnicity_std',
            'ehr': 'ethnicity_std',
            'type': 'categorical'
        },

        # ==================== CLINICAL MEASUREMENTS ====================
        'Height (cm)': {
            'srtr': 'DON_HGT_CM',
            'ehr': 'last_height_cm',
            'type': 'continuous',
            'tolerance_key': 'height'
        },
        'Weight (kg)': {
            'srtr': 'DON_WGT_KG',
            'ehr': 'last_weight_kg',
            'type': 'continuous',
            'tolerance_key': 'weight'
        },

        # ==================== LABORATORY VALUES ====================
        'Creatinine (mg/dL)': {
            'srtr': 'DON_CREAT',
            'ehr': 'creatinine_value',
            'type': 'continuous',
            'tolerance_key': 'creatinine'
        },
        'Total Bilirubin (mg/dL)': {
            'srtr': 'DON_TOT_BILI',
            'ehr': 'bilirubin_total_value',
            'type': 'continuous',
            'tolerance_key': 'bilirubin'
        },
        'AST/SGOT (U/L)': {
            'srtr': 'DON_SGOT',
            'ehr': 'ast_value',
            'type': 'continuous',
            'tolerance_key': 'ast'
        },
        'ALT/SGPT (U/L)': {
            'srtr': 'DON_SGPT',
            'ehr': 'alt_value',
            'type': 'continuous',
            'tolerance_key': 'alt'
        },
        'PO2 Arterial (mmHg)': {
            'srtr': 'DON_PO2',
            'ehr': 'po2_arterial_value',
            'type': 'continuous',
            'tolerance_key': 'po2',
            'from_matches': True  # These are already in matched_pd from best_matches_df
        },
        'pH Arterial': {
            'srtr': 'DON_PH',
            'ehr': 'ph_arterial_value',
            'type': 'continuous',
            'tolerance_key': 'ph',
            'from_matches': True  # These are already in matched_pd from best_matches_df
        },

        # ==================== LIFE SUPPORT ====================
        'Invasive Mechanical Ventilation': {
            'srtr': 'DON_INO_INTUB',
            'ehr': 'imv_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },
        'Any Vasopressor Use': {
            'srtr': None,  # Not available in SRTR
            'ehr': 'any_vasopressor_ever',
            'type': 'ehr_only',
            'from_matches': True
        },
        'WLST Detected': {
            'srtr': None,  # Not available in SRTR
            'ehr': 'wlst_dttm',
            'type': 'ehr_only_notna',
            'from_matches': True
        },

        # ==================== MEDICATIONS ====================
        'Any Inotrope Support': {
            'srtr': 'DON_INOTROP_SUPPORT',
            'ehr': 'any_inotrope_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },
        'Dobutamine': {
            'srtr': 'INO_MED_DOPUTAMINE',  # Note SRTR typo
            'ehr': 'dobutamine_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },
        'Milrinone': {
            'srtr': None,  # Not in SRTR
            'ehr': 'milrinone_ever',
            'type': 'ehr_only',
            'from_matches': True
        },
        'Dopamine': {
            'srtr': 'INO_MED_DOPAMINE',
            'ehr': 'dopamine_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },
        'Epinephrine': {
            'srtr': 'INO_MED_EPINEPHRINE',
            'ehr': 'epinephrine_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },
        'Vasopressin/Arginine': {
            'srtr': 'DON_ARGININE',
            'ehr': 'vasopressin_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },
        'Norepinephrine/Levophed': {
            'srtr': 'INO_MED_LEVOPHED',
            'ehr': 'norepinephrine_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },
        'Phenylephrine/Neo-Synephrine': {
            'srtr': 'INO_MED_NEOSYNEPHRINE',
            'ehr': 'phenylephrine_ever',
            'type': 'binary',
            'srtr_positive': 'Y',
            'ehr_positive': True,
            'from_matches': True
        },

        # ==================== EHR-SPECIFIC ====================
        'CRRT Therapy': {
            'srtr': None,  # Not in SRTR
            'ehr': 'crrt_ever',
            'type': 'ehr_only',
            'from_matches': True
        },

        # ==================== DONOR CHARACTERISTICS ====================
        'Donor Utilized': {
            'srtr': 'don_utilized',
            'ehr': None,  # SRTR only
            'type': 'srtr_only'
        },
        'Donation after Circulatory Death (DCD)': {
            'srtr': 'DON_NON_HR_BEAT',
            'ehr': None,  # SRTR only
            'type': 'srtr_only'
        },

        # ==================== DEATH STATUS ====================
        'Death Documented': {
            'srtr': None,  # All SRTR donors are deceased by definition
            'ehr': 'is_dead',
            'type': 'death_comparison',
            'from_matches': True
        }
    }

    # Process each variable
    for var_name, var_config in variable_mappings.items():
        var_type = var_config['type']

        # Handle special types first
        if var_type == 'srtr_only':
            # Variables only in SRTR
            srtr_col = var_config['srtr']
            if srtr_col in matched_pd.columns:
                srtr_vals = matched_pd[srtr_col].values
                n_srtr = np.sum(~pd.isna(srtr_vals))
                if srtr_col == 'don_utilized':
                    n_yes = np.sum(matched_pd[srtr_col].astype(str).str.upper().isin(['Y', '1', 'YES']))
                    pct_yes = (n_yes / n_srtr * 100) if n_srtr > 0 else None
                    results.append({
                        'Variable': var_name,
                        'N': n_srtr,
                        'SRTR': f"{n_yes}/{n_srtr} ({pct_yes:.1f}%)" if pct_yes else "N/A",
                        'EHR': "—",
                        'Agreement': "—"
                    })
                elif srtr_col == 'DON_NON_HR_BEAT':
                    n_yes = np.sum(matched_pd[srtr_col] == 'Y')
                    pct_yes = (n_yes / n_srtr * 100) if n_srtr > 0 else None
                    results.append({
                        'Variable': var_name,
                        'N': n_srtr,
                        'SRTR': f"{n_yes}/{n_srtr} ({pct_yes:.1f}%)" if pct_yes else "N/A",
                        'EHR': "—",
                        'Agreement': "—"
                    })
            continue

        elif var_type == 'ehr_only':
            # Variables only in EHR
            ehr_col = var_config['ehr']
            from_matches = var_config.get('from_matches', False)

            if ehr_col in matched_pd.columns:
                ehr_vals = matched_pd[ehr_col].values
                n_ehr = np.sum(~pd.isna(ehr_vals))
                if ehr_col in ['any_vasopressor_ever', 'crrt_ever', 'milrinone_ever']:
                    n_yes = np.sum(ehr_vals == True)
                    pct_yes = (n_yes / n_ehr * 100) if n_ehr > 0 else None
                    results.append({
                        'Variable': var_name,
                        'N': n_ehr,
                        'SRTR': "—",
                        'EHR': f"{n_yes}/{n_ehr} ({pct_yes:.1f}%)" if pct_yes else "N/A",
                        'Agreement': "—"
                    })
            continue

        elif var_type == 'ehr_only_notna':
            # Variables where we check if not null (like WLST)
            ehr_col = var_config['ehr']
            if ehr_col in matched_pd.columns:
                n_total = len(matched_pd)
                n_detected = matched_pd[ehr_col].notna().sum()
                pct_detected = (n_detected / n_total * 100) if n_total > 0 else None
                results.append({
                    'Variable': var_name,
                    'N': n_total,
                    'SRTR': "—",
                    'EHR': f"{n_detected}/{n_total} ({pct_detected:.1f}%)" if pct_detected else "N/A",
                    'Agreement': "—"
                })
            continue

        elif var_type == 'death_comparison':
            # Special handling for death status
            ehr_col = var_config['ehr']
            if ehr_col in matched_pd.columns:
                n_total = len(matched_pd)
                n_died_ehr = np.sum(matched_pd[ehr_col] == 1)
                pct_died_ehr = (n_died_ehr / n_total * 100) if n_total > 0 else None
                results.append({
                    'Variable': var_name,
                    'N': n_total,
                    'SRTR': f"{n_total}/{n_total} (100.0%)",  # All SRTR donors are deceased
                    'EHR': f"{n_died_ehr}/{n_total} ({pct_died_ehr:.1f}%)" if pct_died_ehr else "N/A",
                    'Agreement': f"{pct_died_ehr:.1f}%" if pct_died_ehr else "N/A"
                })
            continue

        # Handle regular variables with both SRTR and EHR
        srtr_col = var_config.get('srtr')
        ehr_col = var_config.get('ehr')

        # Check if columns exist
        if srtr_col and srtr_col not in matched_pd.columns:
            continue
        if ehr_col and ehr_col not in matched_pd.columns:
            continue

        # Get values
        if srtr_col:
            srtr_vals = matched_pd[srtr_col].values
        else:
            srtr_vals = np.array([None] * len(matched_pd))

        if ehr_col:
            ehr_vals = matched_pd[ehr_col].values
            # Apply any transformations
            if 'ehr_transform' in var_config:
                ehr_vals = var_config['ehr_transform'](ehr_vals)
        else:
            ehr_vals = np.array([None] * len(matched_pd))

        # Calculate agreement based on type
        if var_type == 'continuous':
            # Get tolerance configuration
            tolerance_key = var_config.get('tolerance_key')
            tolerance_config = CONTINUOUS_TOLERANCES.get(tolerance_key)

            # Convert to numeric
            srtr_numeric = pd.to_numeric(srtr_vals, errors='coerce')
            ehr_numeric = pd.to_numeric(ehr_vals, errors='coerce')

            # Calculate agreement
            agreement_metrics = calculate_continuous_agreement(
                srtr_numeric, ehr_numeric, tolerance_config
            )

            # Format results
            n_paired = agreement_metrics['n_paired']
            if n_paired > 0:
                # Calculate medians and IQRs for display
                srtr_clean = srtr_numeric[~np.isnan(srtr_numeric)]
                ehr_clean = ehr_numeric[~np.isnan(ehr_numeric)]

                srtr_display = f"{np.median(srtr_clean):.1f} [{np.percentile(srtr_clean, 25):.1f}, {np.percentile(srtr_clean, 75):.1f}]" if len(srtr_clean) > 0 else "N/A"
                ehr_display = f"{np.median(ehr_clean):.1f} [{np.percentile(ehr_clean, 25):.1f}, {np.percentile(ehr_clean, 75):.1f}]" if len(ehr_clean) > 0 else "N/A"

                agreement_display = f"Diff: {agreement_metrics['median_abs_diff']:.1f}"
                if agreement_metrics['pct_within_tolerance']:
                    agreement_display += f" | {agreement_metrics['pct_within_tolerance']:.1f}% within tol"
                if agreement_metrics['correlation']:
                    agreement_display += f" | r={agreement_metrics['correlation']:.2f}"
            else:
                srtr_display = "N/A"
                ehr_display = "N/A"
                agreement_display = "No paired data"

            results.append({
                'Variable': var_name,
                'N': n_paired,
                'SRTR': srtr_display,
                'EHR': ehr_display,
                'Agreement': agreement_display
            })

        elif var_type == 'binary':
            srtr_positive = var_config['srtr_positive']
            ehr_positive = var_config['ehr_positive']

            agreement_metrics = calculate_binary_agreement(
                srtr_vals, ehr_vals, srtr_positive, ehr_positive
            )

            n_paired = agreement_metrics['n_paired']
            if n_paired > 0:
                # Calculate percentages for display
                srtr_positive_count = np.sum([str(v).upper() == str(srtr_positive).upper() for v in srtr_vals if not pd.isna(v)])
                ehr_positive_count = np.sum([str(v).upper() == str(ehr_positive).upper() for v in ehr_vals if not pd.isna(v)])

                srtr_display = f"{srtr_positive_count}/{n_paired} ({srtr_positive_count/n_paired*100:.1f}%)"
                ehr_display = f"{ehr_positive_count}/{n_paired} ({ehr_positive_count/n_paired*100:.1f}%)"

                agreement_display = f"{agreement_metrics['pct_agreement']:.1f}% agree"
                if agreement_metrics['sensitivity']:
                    agreement_display += f" | Sens: {agreement_metrics['sensitivity']:.1f}%"
                if agreement_metrics['specificity']:
                    agreement_display += f" | Spec: {agreement_metrics['specificity']:.1f}%"
            else:
                srtr_display = "N/A"
                ehr_display = "N/A"
                agreement_display = "No paired data"

            results.append({
                'Variable': var_name,
                'N': n_paired,
                'SRTR': srtr_display,
                'EHR': ehr_display,
                'Agreement': agreement_display
            })

        elif var_type == 'categorical':
            agreement_metrics = calculate_categorical_agreement(srtr_vals, ehr_vals)

            n_paired = agreement_metrics['n_paired']
            if n_paired > 0:
                # Show unique categories count
                srtr_unique = len(np.unique([v for v in srtr_vals if not pd.isna(v)]))
                ehr_unique = len(np.unique([v for v in ehr_vals if not pd.isna(v)]))

                srtr_display = f"{srtr_unique} categories"
                ehr_display = f"{ehr_unique} categories"

                agreement_display = f"{agreement_metrics['pct_exact_match']:.1f}% match"
                if agreement_metrics['cohen_kappa'] is not None:
                    agreement_display += f" | κ={agreement_metrics['cohen_kappa']:.2f}"
            else:
                srtr_display = "N/A"
                ehr_display = "N/A"
                agreement_display = "No paired data"

            results.append({
                'Variable': var_name,
                'N': n_paired,
                'SRTR': srtr_display,
                'EHR': ehr_display,
                'Agreement': agreement_display
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def create_supplementary_metrics(
    best_matches_df: pl.DataFrame,
    srtr_df: pl.DataFrame,
    patients_pd: pd.DataFrame,
    confidence_level: str = "HIGH"
) -> Dict[str, pd.DataFrame]:
    """
    Create supplementary tables with detailed metrics for continuous, binary, and categorical variables

    Returns:
    --------
    Dictionary with three DataFrames: 'continuous', 'binary', 'categorical'
    """
    # Filter by confidence if specified
    if confidence_level != "ALL":
        working_df = best_matches_df.filter(pl.col("confidence") == confidence_level)
    else:
        working_df = best_matches_df

    # Join with SRTR data
    matched_data = working_df.join(srtr_df, on="DONOR_ID", how="inner")

    # Convert to pandas
    matched_pd = matched_data.to_pandas()

    # Join with patient data using encounter_block
    matched_pd = matched_pd.merge(
        patients_pd,
        left_on="encounter_block",
        right_on="encounter_block",
        how="left",
        suffixes=("", "_patient")
    )

    # Initialize result dictionaries
    continuous_results = []
    binary_results = []
    categorical_results = []

    # Process continuous variables
    continuous_vars = {
        'Age (years)': ('DON_AGE', 'age_at_admission', 'age'),
        'Age (months)': ('DON_AGE_IN_MONTHS', 'age_at_admission', 'age_months', lambda x: x * 12),
        'Height (cm)': ('DON_HGT_CM', 'last_height_cm', 'height'),
        'Weight (kg)': ('DON_WGT_KG', 'last_weight_kg', 'weight'),
        'Creatinine (mg/dL)': ('DON_CREAT', 'creatinine_value', 'creatinine'),
        'Total Bilirubin (mg/dL)': ('DON_TOT_BILI', 'bilirubin_total_value', 'bilirubin'),
        'AST (U/L)': ('DON_SGOT', 'ast_value', 'ast'),
        'ALT (U/L)': ('DON_SGPT', 'alt_value', 'alt'),
        'PO2 Arterial (mmHg)': ('DON_PO2', 'po2_arterial_value', 'po2'),
        'pH Arterial': ('DON_PH', 'ph_arterial_value', 'ph'),
    }

    for var_name, var_config in continuous_vars.items():
        srtr_col = var_config[0]
        ehr_col = var_config[1]
        tolerance_key = var_config[2]
        transform = var_config[3] if len(var_config) > 3 else None

        if srtr_col in matched_pd.columns and ehr_col in matched_pd.columns:
            srtr_vals = pd.to_numeric(matched_pd[srtr_col], errors='coerce')
            ehr_vals = pd.to_numeric(matched_pd[ehr_col], errors='coerce')

            if transform:
                ehr_vals = transform(ehr_vals)

            tolerance_config = CONTINUOUS_TOLERANCES.get(tolerance_key)
            metrics = calculate_continuous_agreement(srtr_vals, ehr_vals, tolerance_config)

            continuous_results.append({
                'Variable': var_name,
                'N': metrics['n_paired'],
                'Median Abs Diff': f"{metrics['median_abs_diff']:.2f}" if metrics['median_abs_diff'] is not None else "—",
                'Mean Abs Error': f"{metrics['mean_abs_error']:.2f}" if metrics['mean_abs_error'] is not None else "—",
                '% Within Tolerance': f"{metrics['pct_within_tolerance']:.1f}" if metrics['pct_within_tolerance'] is not None else "—",
                'Correlation': f"{metrics['correlation']:.3f}" if metrics['correlation'] is not None else "—",
                'BA Mean': f"{metrics['bland_altman_mean']:.2f}" if metrics['bland_altman_mean'] is not None else "—",
                'BA Lower': f"{metrics['bland_altman_lower']:.2f}" if metrics['bland_altman_lower'] is not None else "—",
                'BA Upper': f"{metrics['bland_altman_upper']:.2f}" if metrics['bland_altman_upper'] is not None else "—"
            })

    # Process binary variables
    binary_vars = {
        'Sex (Male)': ('DON_GENDER', 'sex_category', 'M', 'MALE'),
        'Any Inotrope Support': ('DON_INOTROP_SUPPORT', 'any_inotrope_ever', 'Y', True),
        'Invasive Mechanical Ventilation': ('DON_INO_INTUB', 'imv_ever', 'Y', True),
        'Dobutamine': ('INO_MED_DOPUTAMINE', 'dobutamine_ever', 'Y', True),
        'Dopamine': ('INO_MED_DOPAMINE', 'dopamine_ever', 'Y', True),
        'Epinephrine': ('INO_MED_EPINEPHRINE', 'epinephrine_ever', 'Y', True),
        'Vasopressin': ('DON_ARGININE', 'vasopressin_ever', 'Y', True),
        'Norepinephrine': ('INO_MED_LEVOPHED', 'norepinephrine_ever', 'Y', True),
        'Phenylephrine': ('INO_MED_NEOSYNEPHRINE', 'phenylephrine_ever', 'Y', True),
    }

    for var_name, var_config in binary_vars.items():
        srtr_col = var_config[0]
        ehr_col = var_config[1]
        srtr_positive = var_config[2]
        ehr_positive = var_config[3]

        if srtr_col in matched_pd.columns and ehr_col in matched_pd.columns:
            srtr_vals = matched_pd[srtr_col]
            ehr_vals = matched_pd[ehr_col]

            metrics = calculate_binary_agreement(srtr_vals, ehr_vals, srtr_positive, ehr_positive)

            binary_results.append({
                'Variable': var_name,
                'N': metrics['n_paired'],
                '% Agreement': f"{metrics['pct_agreement']:.1f}" if metrics['pct_agreement'] is not None else "—",
                'Sensitivity': f"{metrics['sensitivity']:.1f}" if metrics['sensitivity'] is not None else "—",
                'Specificity': f"{metrics['specificity']:.1f}" if metrics['specificity'] is not None else "—",
                'McNemar p-value': f"{metrics['mcnemar_pvalue']:.3f}" if metrics['mcnemar_pvalue'] is not None else "—"
            })

    # Process categorical variables
    categorical_vars = {
        'Race': ('DON_race_std', 'race_std'),
        'Ethnicity': ('DON_ethnicity_std', 'ethnicity_std'),
    }

    for var_name, var_config in categorical_vars.items():
        srtr_col = var_config[0]
        ehr_col = var_config[1]

        if srtr_col in matched_pd.columns and ehr_col in matched_pd.columns:
            srtr_vals = matched_pd[srtr_col]
            ehr_vals = matched_pd[ehr_col]

            metrics = calculate_categorical_agreement(srtr_vals, ehr_vals)

            categorical_results.append({
                'Variable': var_name,
                'N': metrics['n_paired'],
                '% Exact Match': f"{metrics['pct_exact_match']:.1f}" if metrics['pct_exact_match'] is not None else "—",
                'Cohen\'s Kappa': f"{metrics['cohen_kappa']:.3f}" if metrics['cohen_kappa'] is not None else "—"
            })

    return {
        'continuous': pd.DataFrame(continuous_results),
        'binary': pd.DataFrame(binary_results),
        'categorical': pd.DataFrame(categorical_results)
    }