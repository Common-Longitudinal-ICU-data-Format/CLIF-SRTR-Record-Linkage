"""
Progressive Record Linkage for SRTR-CLIF Donor Matching

This module implements a fine-grained progressive matching algorithm that incrementally
relaxes date and age tolerances to find the best possible match for each donor.

Key Features:
- Fine-grained tolerance progression: (0,0) → (1,0), (0,1) → (2,0), (1,1), (0,2) → ...
- Flexible column mapping via dictionaries
- Binary search for efficient date window finding
- Probabilistic scoring with exponential decay
- Single best match per donor at lowest tolerance
- Match confidence indicators for alternative matches
- Optional debug mode for all matches
"""

import polars as pl
import pandas as pd
import numpy as np
import math
from typing import Dict, Tuple, List
from datetime import datetime, timedelta


def generate_tolerance_levels(max_date_tol: int = 30, max_age_tol: int = 2) -> List[Tuple[int, int, str]]:
    """
    Generate fine-grained tolerance levels following the pattern:
    (0,0) → (1,0), (0,1) → (2,0), (1,1), (0,2) → (3,0), (2,1), (1,2) → ...

    This explores tolerances where date_tol + age_tol increases progressively,
    exploring all combinations for each sum level with date prioritized.

    Note: Age tolerance is in MONTHS, not years!

    Args:
        max_date_tol: Maximum date tolerance in days (default 30)
        max_age_tol: Maximum age tolerance in MONTHS (default 2)

    Returns:
        List of (date_tolerance, age_tolerance, level_name) tuples
    """
    levels = []
    max_sum = max_date_tol + max_age_tol

    for total in range(0, max_sum + 1):
        # For each sum level, generate all valid combinations
        # Prioritize date tolerance (explore date first at each level)
        for date_tol in range(min(total + 1, max_date_tol + 1)):
            age_tol = total - date_tol
            if age_tol <= max_age_tol:
                # Create level name based on tolerances
                if total == 0:
                    level_name = 'EXACT'
                elif total <= 2:
                    level_name = 'STRICT'
                elif total <= 5:
                    level_name = 'STANDARD'
                elif total <= 10:
                    level_name = 'RELAXED'
                elif total <= 20:
                    level_name = 'EXPANDED'
                else:
                    level_name = 'MAXIMUM'

                levels.append((date_tol, age_tol, level_name))

    return levels


def validate_race_ethnicity(
    donor_race: str,
    donor_ethnicity: str,
    patient_race: str,
    patient_ethnicity: str
) -> Tuple[bool, bool, bool, bool]:
    """
    Validate race and ethnicity matching with special handling for UNKNOWN values.

    Returns:
    --------
    Tuple of (race_match, ethnicity_match, race_unknown, ethnicity_unknown)
    """
    # Check for UNKNOWN values
    race_unknown = (donor_race == 'UNKNOWN' or patient_race == 'UNKNOWN')
    ethnicity_unknown = (donor_ethnicity == 'UNKNOWN' or patient_ethnicity == 'UNKNOWN')

    # Race matching
    if race_unknown:
        race_match = True  # Consider UNKNOWN as potential match
    else:
        race_match = (donor_race == patient_race)

    # Ethnicity matching
    if ethnicity_unknown:
        ethnicity_match = True  # Consider UNKNOWN as potential match
    else:
        ethnicity_match = (donor_ethnicity == patient_ethnicity)

    return race_match, ethnicity_match, race_unknown, ethnicity_unknown


def calculate_continuous_score(
    date_diff: float,
    age_diff_months: float,
    gender_match: bool,
    race_match: bool,
    ethnicity_match: bool,
    race_unknown: bool,
    ethnicity_unknown: bool
) -> float:
    """
    Calculate continuous probabilistic score (0-100) for a match.
    """
    # Exponential decay with scale parameter = 3 days
    date_score = math.exp(-date_diff / 3.0) * 100

    # Age similarity score (linear decay over 60 months/5 years)
    age_score = max(0, (1 - age_diff_months / 60.0)) * 100

    # Gender match score (binary but important)
    gender_score = 100 if gender_match else 0

    # Demographics score (race and ethnicity combined)
    demo_score = 0
    if race_match and ethnicity_match:
        if race_unknown or ethnicity_unknown:
            demo_score = 70
        else:
            demo_score = 100
    elif race_match or ethnicity_match:
        demo_score = 50
    else:
        demo_score = 0

    total_score = (
        date_score * 0.40 +
        age_score * 0.30 +
        gender_score * 0.20 +
        demo_score * 0.10
    )

    return min(100, max(0, total_score))


def calculate_confidence(
    date_diff: float,
    age_diff_months: float,
    gender_match: bool,
    race_match: bool,
    ethnicity_match: bool,
    race_unknown: bool,
    ethnicity_unknown: bool,
    tolerance_level: str = 'STANDARD'
) -> Tuple[str, float]:
    """
    Calculate both categorical confidence and continuous score.
    """
    score = calculate_continuous_score(
        date_diff, age_diff_months, gender_match,
        race_match, ethnicity_match,
        race_unknown, ethnicity_unknown
    )

    # Convert age difference to years for confidence rules
    age_diff_years = age_diff_months / 12.0

    if date_diff <= 1 and age_diff_years < 0.5:
        if gender_match and (race_match or race_unknown) and (ethnicity_match or ethnicity_unknown):
            confidence = 'HIGH'
        elif gender_match:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
    elif date_diff <= 3 and age_diff_years <= 1:
        if gender_match and ((race_match or race_unknown) or (ethnicity_match or ethnicity_unknown)):
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
    elif date_diff > 5 and age_diff_years > 1:
        confidence = 'REVIEW'
    elif not gender_match:
        confidence = 'LOW'
    else:
        if score >= 85:
            confidence = 'HIGH'
        elif score >= 60:
            confidence = 'MEDIUM'
        elif score >= 40:
            confidence = 'LOW'
        else:
            confidence = 'REVIEW'

    # Apply caps based on tolerance level (for progressive matching)
    if tolerance_level not in ['EXACT', 'STRICT', 'STANDARD']:
        if tolerance_level == 'RELAXED' and confidence == 'HIGH':
            confidence = 'MEDIUM'
        elif tolerance_level in ['EXPANDED', 'MAXIMUM']:
            if confidence in ['HIGH', 'MEDIUM']:
                confidence = 'LOW' if tolerance_level == 'EXPANDED' else 'REVIEW'

    return confidence, score


def find_date_window_binary(
    death_dates: np.ndarray,
    recovery_date: np.datetime64,
    window_days: int
) -> Tuple[int, int]:
    """
    Find patients within date window using binary search.
    """
    window_td = np.timedelta64(window_days, 'D')
    lower_bound = recovery_date - window_td
    upper_bound = recovery_date + window_td
    start_idx = np.searchsorted(death_dates, lower_bound, side='left')
    end_idx = np.searchsorted(death_dates, upper_bound, side='right')
    return start_idx, end_idx


def perform_matching_pass(
    patients_df: pl.DataFrame,
    donors_to_match: List[Dict],
    death_dates_np: np.ndarray,
    date_tol: int,
    age_tol_months: int,
    date_window_days: int,
    tolerance_level: str,
    patient_cols: Dict[str, str],
    donor_cols: Dict[str, str],
    debug: bool
) -> List[Dict]:
    """
    Perform a single matching pass with given tolerances.
    """
    matches = []

    for donor_info in donors_to_match:
        donor_id = donor_info['donor_id']
        recovery_date = donor_info['recovery_date']
        donor_age_months = donor_info['age_months']
        donor_gender = donor_info['gender']
        donor_race = donor_info['race']
        donor_ethnicity = donor_info['ethnicity']

        if recovery_date is None:
            continue

        if not isinstance(recovery_date, np.datetime64):
            recovery_date = np.datetime64(recovery_date)

        start_idx, end_idx = find_date_window_binary(death_dates_np, recovery_date, date_window_days)

        patients_in_window = patients_df[start_idx:end_idx]

        matches_for_donor = []

        for patient_row in patients_in_window.iter_rows(named=True):
            patient_death_date = patient_row['death_date']
            if not isinstance(patient_death_date, np.datetime64):
                patient_death_date = np.datetime64(patient_death_date)

            # ---- HARD validity rule (directional) ----
            if patient_death_date > recovery_date + np.timedelta64(1, 'D'):
                continue

            # ---- similarity calculations (symmetric) ----
            date_diff_td = recovery_date - patient_death_date
            date_diff = abs(date_diff_td / np.timedelta64(1, 'D'))
            age_diff_months = abs(donor_age_months - patient_row['age_months'])

            if date_diff > date_tol or age_diff_months > age_tol_months:
                continue


            gender_match = (donor_gender == patient_row['gender'])
            race_match, ethnicity_match, race_unknown, ethnicity_unknown = validate_race_ethnicity(
                donor_race, donor_ethnicity,
                patient_row['race'], patient_row['ethnicity']
            )

            confidence, score = calculate_confidence(
                date_diff, age_diff_months, gender_match,
                race_match, ethnicity_match,
                race_unknown, ethnicity_unknown,
                tolerance_level
            )

            match_dict = {
                'donor_id': donor_id,
                'encounter_block': patient_row['encounter_block'],
                'patient_id': patient_row['patient_id'],
                'is_dead': patient_row['is_dead'],
                'discharge_dttm': patient_row['discharge_dttm'],
                'date_diff': date_diff,
                'age_diff_months': age_diff_months,
                'gender_match': gender_match,
                'race_match': race_match,
                'ethnicity_match': ethnicity_match,
                'race_unknown': race_unknown,
                'ethnicity_unknown': ethnicity_unknown,
                'confidence': confidence,
                'match_score': score,
                'recovery_date': pd.Timestamp(recovery_date).to_pydatetime()
                if isinstance(recovery_date, np.datetime64) else recovery_date,
                'death_date': pd.Timestamp(patient_death_date).to_pydatetime()
                if isinstance(patient_death_date, np.datetime64) else patient_row['death_date'],
                'tolerance_level': tolerance_level,
                'date_tolerance': date_tol,
                'age_tolerance_months': age_tol_months
            }

            matches_for_donor.append(match_dict)

        matches.extend(matches_for_donor)

    return matches


def progressive_match_donors(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    patient_cols: Dict[str, str],
    donor_cols: Dict[str, str],
    max_date_tolerance: int = 30,
    max_age_tolerance: int = 2,  # In MONTHS
    date_window: int = 60,        # UPDATED: tighter blocking window (was 365)
    return_all_matches: bool = False,
    debug: bool = False
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Perform progressive matching between patients and donors with fine-grained tolerance relaxation.
    """

    required_patient = {'id', 'block', 'death_date', 'age', 'gender', 'race', 'ethnicity'}
    required_donor = {'id', 'recovery_date', 'age', 'gender', 'race', 'ethnicity'}

    missing_patient = required_patient - set(patient_cols.keys())
    missing_donor = required_donor - set(donor_cols.keys())

    if missing_patient:
        raise ValueError(f"Missing patient column mappings: {missing_patient}")
    if missing_donor:
        raise ValueError(f"Missing donor column mappings: {missing_donor}")

    for key, col in patient_cols.items():
        if col not in patients_df.columns:
            if key == 'gender' and 'sex_category' in patients_df.columns:
                print(f"Warning: Column '{col}' not found, but 'sex_category' exists. Please update your mapping.")
            raise ValueError(
                f"Patient column '{col}' (for '{key}') not found in dataframe. "
                f"Available columns: {patients_df.columns}"
            )

    for key, col in donor_cols.items():
        if col not in donors_df.columns:
            raise ValueError(f"Donor column '{col}' (for '{key}') not found in dataframe")

    patients_internal = patients_df.select([
        pl.col(patient_cols['id']).alias('patient_id'),
        pl.col(patient_cols['block']).alias('encounter_block'),
        pl.col(patient_cols['death_date']).alias('death_date'),
        pl.col(patient_cols['age']).alias('age_months'),
        pl.col(patient_cols['gender']).alias('gender'),
        pl.col(patient_cols['race']).alias('race'),
        pl.col(patient_cols['ethnicity']).alias('ethnicity'),
        pl.col('is_dead').alias('is_dead'),
        pl.col('discharge_dttm').alias('discharge_dttm'),
    ])

    patients_internal = patients_internal.sort('death_date')
    death_dates_np = patients_internal['death_date'].to_numpy()

    donors_to_match = []
    for donor_row in donors_df.iter_rows(named=True):
        donor_info = {
            'donor_id': donor_row[donor_cols['id']],
            'recovery_date': donor_row[donor_cols['recovery_date']],
            'age_months': donor_row[donor_cols['age']],
            'gender': donor_row[donor_cols['gender']],
            'race': donor_row[donor_cols['race']],
            'ethnicity': donor_row[donor_cols['ethnicity']]
        }
        donors_to_match.append(donor_info)

    if debug:
        print("=" * 80)
        print("PROGRESSIVE MATCHING WITH FINE-GRAINED TOLERANCE RELAXATION")
        print("=" * 80)
        print(f"Total patients: {len(patients_internal):,}")
        print(f"Total donors: {len(donors_to_match):,}")
        print(f"Maximum tolerances: date=±{max_date_tolerance} days, age=±{max_age_tolerance} months")
        print(f"Date blocking window: ±{date_window} days")
        print()

    tolerance_levels = generate_tolerance_levels(max_date_tolerance, max_age_tolerance)

    if debug:
        print(f"Generated {len(tolerance_levels)} tolerance levels")
        print(f"First 10 levels: {[(d, a) for d, a, _ in tolerance_levels[:10]]}")
        print()

    matched_donors = set()
    all_matches = []
    best_matches_per_donor = {}
    alternative_counts = {}

    for level_idx, (date_tol, age_tol_months, level_name) in enumerate(tolerance_levels):

        unmatched_donors = [d for d in donors_to_match if d['donor_id'] not in matched_donors]

        if not unmatched_donors:
            if debug:
                print(f"All donors matched by level {level_idx} (date_tol={date_tol}, age_tol={age_tol_months} months)")
            break

        level_matches = perform_matching_pass(
            patients_internal, unmatched_donors, death_dates_np,
            date_tol, age_tol_months, date_window,
            level_name, patient_cols, donor_cols,
            False
        )

        if debug and level_matches:
            print(f"Level {level_idx} (date={date_tol}d, age={age_tol_months}mo, {level_name}): Found {len(level_matches)} candidate matches")

        matches_by_donor = {}
        for match in level_matches:
            donor_id = match['donor_id']
            if donor_id not in matches_by_donor:
                matches_by_donor[donor_id] = []
            matches_by_donor[donor_id].append(match)

        for donor_id, donor_matches in matches_by_donor.items():
            if donor_id not in matched_donors:
                donor_matches.sort(key=lambda x: (x['match_score'], x['confidence']), reverse=True)

                best_match = donor_matches[0]
                best_match['n_alternatives'] = len(donor_matches) - 1
                best_match['match_confidence'] = 'UNIQUE' if len(donor_matches) == 1 else 'MULTIPLE'

                best_matches_per_donor[donor_id] = best_match
                matched_donors.add(donor_id)

                alternative_counts[donor_id] = len(donor_matches)

                if return_all_matches:
                    for i, match in enumerate(donor_matches):
                        match['is_best'] = (i == 0)
                        match['match_rank'] = i + 1
                        all_matches.append(match)
                else:
                    all_matches.append(best_match)

        if debug and matches_by_donor:
            print(f"  → Matched {len(matches_by_donor)} new donors")
            print(f"  → Total matched so far: {len(matched_donors)}/{len(donors_to_match)}")

    if all_matches:
        matches_df = pl.DataFrame(all_matches)
        matches_df = matches_df.rename({
            'donor_id': 'DONOR_ID',
            'match_score': 'score',
            'age_diff_months': 'age_diff'
        })
    else:
        matches_df = pl.DataFrame({
            'DONOR_ID': [],
            'patient_id': [],
            'encounter_block': [],
            'score': [],
            'confidence': [],
            'date_diff': [],
            'age_diff': [],
            'date_tolerance': [],
            'age_tolerance_months': [],
            'tolerance_level': [],
            'gender_match': [],
            'race_match': [],
            'ethnicity_match': [],
            'n_alternatives': [],
            'match_confidence': []
        })

    if matches_df.height > 0:
        matches_df = matches_df.with_columns([
            pl.col("discharge_dttm").cast(pl.Datetime),
            pl.col("is_dead").cast(pl.Int8),
        ])

        # For each (DONOR_ID, patient_id), keep only the last terminal encounter
        matches_df = (
            matches_df
            .sort("discharge_dttm")
            .group_by(["DONOR_ID", "patient_id"])
            .agg(pl.all().last())
        )

    if matches_df.height > 0:
        summary_df = (
            matches_df
            .group_by('tolerance_level')
            .agg([
                pl.count().alias('n_matches'),
                pl.col('score').mean().alias('avg_score'),
                pl.col('score').min().alias('min_score'),
                pl.col('score').max().alias('max_score'),
                pl.col('n_alternatives').mean().alias('avg_alternatives')
            ])
            .sort('tolerance_level')
        )
    else:
        summary_df = pl.DataFrame({
            'tolerance_level': [],
            'n_matches': [],
            'avg_score': [],
            'min_score': [],
            'max_score': [],
            'avg_alternatives': []
        })

    total_donors = len(donors_to_match)
    total_matched = len(matched_donors)
    match_rate = (total_matched / total_donors * 100) if total_donors > 0 else 0

    if debug:
        print()
        print("=" * 80)
        print("FINAL MATCHING RESULTS")
        print("=" * 80)
        print(f"Total donors: {total_donors}")
        print(f"Matched donors: {total_matched} ({match_rate:.1f}%)")
        print(f"Unmatched donors: {total_donors - total_matched}")

        if matches_df.height > 0 and 'confidence' in matches_df.columns:
            confidence_counts = matches_df['confidence'].value_counts()
            print("\nConfidence distribution:")
            for row in confidence_counts.iter_rows(named=True):
                print(f"  {row['confidence']}: {row['count']}")

        print("=" * 80)

    return matches_df, summary_df
