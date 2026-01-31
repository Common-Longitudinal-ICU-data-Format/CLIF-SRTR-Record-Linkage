"""
Progressive Matching Algorithm 

This module implements a vectorized version of the progressive matching algorithm
using Polars' native operations for significantly improved performance.
"""

import polars as pl
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


def generate_tolerance_levels(max_date: int = 30, max_age_months: int = 2) -> List[Tuple[int, int, str]]:
    """
    Generate fine-grained tolerance levels for progressive matching.
    Returns: List of tuples (date_tolerance_days, age_tolerance_months, level_name)
    """
    tolerance_levels = [
        (0, 0, 'EXACT'),
        (1, 0, 'STRICT'),
        (1, 1, 'STANDARD')
    ]

    for date_tol in range(2, max_date + 1):
        for age_tol in range(0, min(date_tol, max_age_months) + 1):
            if date_tol <= 7 and age_tol <= 1:
                level_name = 'STANDARD'
            elif date_tol <= 14 and age_tol <= 1:
                level_name = 'RELAXED'
            elif date_tol <= 21 and age_tol <= 2:
                level_name = 'EXPANDED'
            else:
                level_name = 'MAXIMUM'

            if (date_tol, age_tol, level_name) not in tolerance_levels:
                tolerance_levels.append((date_tol, age_tol, level_name))

    return tolerance_levels


def validate_race_ethnicity(
    donor_race: str, donor_ethnicity: str,
    patient_race: str, patient_ethnicity: str
) -> Tuple[bool, bool, bool, bool]:
    """
    Validate race and ethnicity matching with unknown handling.
    """
    def normalize(value):
        if pd.isna(value) or value in ['None', 'Unknown', 'U', 'UNKNOWN', '']:
            return 'UNKNOWN'
        return str(value).upper()

    donor_race = normalize(donor_race)
    donor_ethnicity = normalize(donor_ethnicity)
    patient_race = normalize(patient_race)
    patient_ethnicity = normalize(patient_ethnicity)

    race_unknown = donor_race == 'UNKNOWN' or patient_race == 'UNKNOWN'
    ethnicity_unknown = donor_ethnicity == 'UNKNOWN' or patient_ethnicity == 'UNKNOWN'

    race_match = donor_race == patient_race or race_unknown
    ethnicity_match = donor_ethnicity == patient_ethnicity or ethnicity_unknown

    return race_match, ethnicity_match, race_unknown, ethnicity_unknown


def calculate_vectorized_scores(matches_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate match scores using vectorized operations.

    Scoring weights:
    - Date similarity: 45%
    - Age similarity: 35%
    - Gender match: 20%
    - Race/ethnicity: 0% (tracked but not scored)
    """
    return matches_df.with_columns([
        # Date score: exponential decay with scale parameter = 3 days
        ((-pl.col('date_diff') / 3.0).exp() * 100).alias('date_score'),

        # Age score: linear decay over 60 months (5 years)
        ((1 - pl.col('age_diff_months') / 60.0).clip(lower_bound=0) * 100).alias('age_score'),

        # Gender score: binary
        pl.when(pl.col('gender_match')).then(100).otherwise(0).alias('gender_score'),
    ]).with_columns([
        # Combined score with weights
        (
            pl.col('date_score') * 0.45 +
            pl.col('age_score') * 0.35 +
            pl.col('gender_score') * 0.20
        ).clip(0, 100).alias('match_score')
    ])


def calculate_vectorized_confidence(matches_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate confidence levels using vectorized operations.
    """
    # Convert age difference to years for confidence rules
    matches_df = matches_df.with_columns(
        (pl.col('age_diff_months') / 12.0).alias('age_diff_years')
    )

    # Apply confidence rules using Polars when/then/otherwise chains
    return matches_df.with_columns(
        pl.when(
            # HIGH confidence conditions
            (pl.col('date_diff') <= 1) &
            (pl.col('age_diff_years') < 0.5) &
            pl.col('gender_match') &
            (pl.col('race_match') | pl.col('race_unknown')) &
            (pl.col('ethnicity_match') | pl.col('ethnicity_unknown'))
        ).then(pl.lit('HIGH'))
        .when(
            # MEDIUM confidence conditions (first set)
            (pl.col('date_diff') <= 1) &
            (pl.col('age_diff_years') < 0.5) &
            pl.col('gender_match')
        ).then(pl.lit('MEDIUM'))
        .when(
            # MEDIUM confidence conditions (second set)
            (pl.col('date_diff') <= 3) &
            (pl.col('age_diff_years') <= 1) &
            pl.col('gender_match') &
            ((pl.col('race_match') | pl.col('race_unknown')) |
             (pl.col('ethnicity_match') | pl.col('ethnicity_unknown')))
        ).then(pl.lit('MEDIUM'))
        .when(
            # REVIEW conditions
            (pl.col('date_diff') > 5) & (pl.col('age_diff_years') > 1)
        ).then(pl.lit('REVIEW'))
        .when(
            # LOW for gender mismatch
            ~pl.col('gender_match')
        ).then(pl.lit('LOW'))
        .when(
            # Score-based confidence
            pl.col('match_score') >= 85
        ).then(pl.lit('HIGH'))
        .when(
            pl.col('match_score') >= 60
        ).then(pl.lit('MEDIUM'))
        .when(
            pl.col('match_score') >= 40
        ).then(pl.lit('LOW'))
        .otherwise(pl.lit('REVIEW'))
        .alias('confidence')
    ).with_columns(
        # Apply tolerance level caps
        pl.when(
            (pl.col('tolerance_level') == 'RELAXED') &
            (pl.col('confidence') == 'HIGH')
        ).then(pl.lit('MEDIUM'))
        .when(
            (pl.col('tolerance_level').is_in(['EXPANDED', 'MAXIMUM'])) &
            (pl.col('confidence').is_in(['HIGH', 'MEDIUM']))
        ).then(
            pl.when(pl.col('tolerance_level') == 'EXPANDED')
            .then(pl.lit('LOW'))
            .otherwise(pl.lit('REVIEW'))
        )
        .otherwise(pl.col('confidence'))
        .alias('confidence')
    )


def perform_vectorized_matching_batched(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    date_tol: int,
    age_tol_months: int,
    date_window_days: int,
    tolerance_level: str,
    batch_size: int = 50,
    debug: bool = False
) -> pl.DataFrame:
    """
    Perform vectorized matching with batching to handle large datasets.
    
    This version processes donors in batches to avoid memory issues with large cross joins.
    
    Args:
        batch_size: Number of donors to process at once (default 50)
    """
    
    all_matches = []
    n_donors = donors_df.height
    n_batches = (n_donors + batch_size - 1) // batch_size
    
    if debug:
        print(f"    Processing {n_donors} donors in {n_batches} batches of {batch_size}")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_donors)
        
        # Get batch of donors
        donors_batch = donors_df[start_idx:end_idx]
        
        # Process this batch
        batch_matches = perform_vectorized_matching(
            patients_df,
            donors_batch,
            date_tol,
            age_tol_months,
            date_window_days,
            tolerance_level,
            debug=False  # Don't debug individual batches
        )
        
        all_matches.append(batch_matches)
        
        if debug and batch_idx % 10 == 0:
            print(f"      Processed batch {batch_idx + 1}/{n_batches}")
    
    # Combine all batches
    if all_matches:
        result = pl.concat(all_matches, how='vertical')
        if debug:
            print(f"    Total matches from all batches: {result.height}")
        return result
    else:
        # Return empty dataframe with correct schema
        return pl.DataFrame({
            'donor_id': [],
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
        })

def perform_vectorized_matching(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    date_tol: int,
    age_tol_months: int,
    date_window_days: int,
    tolerance_level: str,
    debug: bool = False
) -> pl.DataFrame:
    """
    Perform vectorized matching using Polars join operations.

    This replaces the nested loops with efficient join operations.
    """

    # Step 1: Ensure consistent datetime types (convert to microseconds which is Polars default)
    donors_df = donors_df.with_columns([
        pl.col('recovery_date').cast(pl.Datetime('us')).alias('recovery_date')
    ])

    patients_df = patients_df.with_columns([
        pl.col('death_date').cast(pl.Datetime('us')).alias('death_date')
    ])

    # Step 2: Perform cross join to get ALL candidates within the date window
    # This is more memory intensive but ensures we don't miss any matches
    # For large datasets, we might need to process in chunks
    
    # Add a dummy column for cross join
    donors_df = donors_df.with_columns(pl.lit(1).alias('_join_key'))
    patients_df = patients_df.with_columns(pl.lit(1).alias('_join_key'))
    
    # Perform cross join
    candidates = donors_df.join(
        patients_df,
        on='_join_key',
        suffix='_patient'
    ).drop('_join_key')
    
    # Step 3: Apply date window filtering immediately to reduce memory usage
    candidates = candidates.filter(
        # Keep only patients within the date window
        (pl.col('death_date') >= pl.col('recovery_date') - pl.duration(days=date_window_days)) &
        (pl.col('death_date') <= pl.col('recovery_date') + pl.duration(days=date_window_days))
    )
    
    if debug:
        print(f"    Found {candidates.height} candidate pairs within date windows")

    # Step 4: Apply hard constraints using vectorized operations
    matches = candidates.filter(
        # Hard validity rule: patient death must be before or on recovery date + 1 day
        (pl.col('death_date') <= pl.col('recovery_date') + pl.duration(days=1))
    )

    # Step 5: Calculate differences and validate vitals
    matches = matches.with_columns([
        # Date difference (absolute value for matching)
        ((pl.col('recovery_date') - pl.col('death_date')).dt.total_days().abs()).alias('date_diff'),

        # Age difference in months
        (pl.col('age_months') - pl.col('age_months_patient')).abs().alias('age_diff_months'),

        # Gender matching
        (pl.col('gender') == pl.col('gender_patient')).alias('gender_match'),

        # Vitals validation
        pl.when(pl.col('last_recorded_vital_dttm').is_not_null())
        .then(
            ((pl.col('recovery_date') - pl.col('last_recorded_vital_dttm')).dt.total_days())
        )
        .otherwise(None)
        .alias('vitals_days_diff')
    ])

    # Step 6: Apply tolerance filters and vitals validation
    matches = matches.filter(
        # Date and age tolerances
        (pl.col('date_diff') <= date_tol) &
        (pl.col('age_diff_months') <= age_tol_months) &

        # Vitals validation: must be within -2 to 30 day window
        (
            pl.col('vitals_days_diff').is_null() |  # Allow null vitals (no vital recorded)
            ((pl.col('vitals_days_diff') >= -2) & (pl.col('vitals_days_diff') <= 30))
        )
    )

    # Step 7: Validate race and ethnicity (still row-wise but optimized)
    # For full vectorization, we could pre-normalize these columns
    race_ethnicity_validations = []
    for row in matches.select(['race', 'ethnicity', 'race_patient', 'ethnicity_patient']).iter_rows(named=True):
        race_match, ethnicity_match, race_unknown, ethnicity_unknown = validate_race_ethnicity(
            row['race'], row['ethnicity'],
            row['race_patient'], row['ethnicity_patient']
        )
        race_ethnicity_validations.append({
            'race_match': race_match,
            'ethnicity_match': ethnicity_match,
            'race_unknown': race_unknown,
            'ethnicity_unknown': ethnicity_unknown
        })

    if race_ethnicity_validations:
        race_ethnicity_df = pl.DataFrame(race_ethnicity_validations)
        matches = pl.concat([matches, race_ethnicity_df], how='horizontal')
    else:
        # Add empty columns if no matches
        matches = matches.with_columns([
            pl.lit(False).alias('race_match'),
            pl.lit(False).alias('ethnicity_match'),
            pl.lit(False).alias('race_unknown'),
            pl.lit(False).alias('ethnicity_unknown')
        ])

    # Step 8: Add tolerance level
    matches = matches.with_columns(
        pl.lit(tolerance_level).alias('tolerance_level')
    )

    # Step 9: Calculate scores and confidence using vectorized operations
    if matches.height > 0:
        matches = calculate_vectorized_scores(matches)
        matches = calculate_vectorized_confidence(matches)
    else:
        # Add empty score and confidence columns if no matches
        matches = matches.with_columns([
            pl.lit(0.0).alias('match_score'),
            pl.lit('').alias('confidence')
        ])

    # Step 10: Prepare final output format
    result_columns = {
        'donor_id': 'donor_id',
        'encounter_block': 'encounter_block_patient',
        'patient_id': 'patient_id_patient',
        'is_dead': 'is_dead_patient',
        'discharge_dttm': 'discharge_dttm_patient',
        'date_diff': 'date_diff',
        'age_diff_months': 'age_diff_months',
        'gender_match': 'gender_match',
        'race_match': 'race_match',
        'ethnicity_match': 'ethnicity_match',
        'race_unknown': 'race_unknown',
        'ethnicity_unknown': 'ethnicity_unknown',
        'confidence': 'confidence',
        'match_score': 'match_score',
        'recovery_date': 'recovery_date',
        'death_date': 'death_date',
        'tolerance_level': 'tolerance_level',
        'date_tolerance': pl.lit(date_tol),
        'age_tolerance_months': pl.lit(age_tol_months)
    }

    # Select and rename columns (only select columns that exist)
    existing_columns = set(matches.columns)
    select_expr = []
    for k, v in result_columns.items():
        if isinstance(v, str):
            if v in existing_columns:
                select_expr.append(pl.col(v).alias(k))
            else:
                # Add default value for missing columns
                if k == 'match_score':
                    select_expr.append(pl.lit(0.0).alias(k))
                elif k == 'confidence':
                    select_expr.append(pl.lit('').alias(k))
                else:
                    select_expr.append(pl.lit(None).alias(k))
        else:
            # It's a literal expression
            select_expr.append(v.alias(k))

    matches = matches.select(select_expr)

    if debug and matches.height > 0:
        print(f"    Vectorized matching found {matches.height} candidates after all filters")

    return matches


def progressive_match_donors(
    patients_df: pl.DataFrame,
    donors_df: pl.DataFrame,
    patient_cols: Dict[str, str],
    donor_cols: Dict[str, str],
    max_date_tolerance: int = 30,
    max_age_tolerance: int = 2,  # In MONTHS
    date_window: int = 60,
    return_all_matches: bool = False,
    debug: bool = False
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Optimized progressive matching using vectorized operations.

    This version replaces nested loops with efficient Polars operations,
    resulting in significantly improved performance for large datasets.
    """

    # Validate required columns
    required_patient = {'id', 'block', 'death_date', 'age', 'gender', 'race', 'ethnicity'}
    required_donor = {'id', 'recovery_date', 'age', 'gender', 'race', 'ethnicity'}

    missing_patient = required_patient - set(patient_cols.keys())
    missing_donor = required_donor - set(donor_cols.keys())

    if missing_patient:
        raise ValueError(f"Missing patient column mappings: {missing_patient}")
    if missing_donor:
        raise ValueError(f"Missing donor column mappings: {missing_donor}")

    # Prepare patient dataframe with standardized column names
    patients_internal = patients_df.select([
        pl.col(patient_cols['id']).alias('patient_id_patient'),
        pl.col(patient_cols['block']).alias('encounter_block_patient'),
        pl.col(patient_cols['death_date']).cast(pl.Datetime('us')).alias('death_date'),
        pl.col(patient_cols['age']).alias('age_months_patient'),
        pl.col(patient_cols['gender']).alias('gender_patient'),
        pl.col(patient_cols['race']).alias('race_patient'),
        pl.col(patient_cols['ethnicity']).alias('ethnicity_patient'),
        pl.col('is_dead').alias('is_dead_patient'),
        pl.col('discharge_dttm').cast(pl.Datetime('us')).alias('discharge_dttm_patient'),
        pl.col('last_recorded_vital_dttm').cast(pl.Datetime('us')).alias('last_recorded_vital_dttm'),
    ]).sort('death_date')

    # Prepare donor dataframe with standardized column names
    donors_internal = donors_df.select([
        pl.col(donor_cols['id']).alias('donor_id'),
        pl.col(donor_cols['recovery_date']).cast(pl.Datetime('us')).alias('recovery_date'),
        pl.col(donor_cols['age']).alias('age_months'),
        pl.col(donor_cols['gender']).alias('gender'),
        pl.col(donor_cols['race']).alias('race'),
        pl.col(donor_cols['ethnicity']).alias('ethnicity')
    ])

    if debug:
        print("=" * 80)
        print("OPTIMIZED PROGRESSIVE MATCHING WITH VECTORIZED OPERATIONS")
        print("=" * 80)
        print(f"Total patients: {len(patients_internal):,}")
        print(f"Total donors: {len(donors_internal):,}")
        print(f"Maximum tolerances: date=±{max_date_tolerance} days, age=±{max_age_tolerance} months")
        print(f"Date blocking window: ±{date_window} days")
        print()

    # Generate tolerance levels
    tolerance_levels = generate_tolerance_levels(max_date_tolerance, max_age_tolerance)

    if debug:
        print(f"Generated {len(tolerance_levels)} tolerance levels")
        print()

    # Track matched donors and collect all matches
    matched_donors = set()
    all_matches = []
    best_matches_per_donor = {}

    # Process each tolerance level
    for level_idx, (date_tol, age_tol_months, level_name) in enumerate(tolerance_levels):

        # Filter to unmatched donors
        unmatched_donors = donors_internal.filter(
            ~pl.col('donor_id').is_in(list(matched_donors))
        )

        if unmatched_donors.height == 0:
            if debug:
                print(f"All donors matched by level {level_idx}")
            break

        # Perform vectorized matching for this tolerance level
        # Use batched processing if we have many donors and patients to avoid memory issues
        n_unmatched = unmatched_donors.height
        n_patients = patients_internal.height

        # Use batched processing if cross join would be too large
        if n_unmatched * n_patients > 100000:  # Threshold for batching
            level_matches = perform_vectorized_matching_batched(
                patients_internal,
                unmatched_donors,
                date_tol,
                age_tol_months,
                date_window,
                level_name,
                batch_size=50,  # Process 50 donors at a time
                debug=debug and level_idx < 5
            )
        else:
            level_matches = perform_vectorized_matching(
                patients_internal,
                unmatched_donors,
                date_tol,
                age_tol_months,
                date_window,
                level_name,
                debug and level_idx < 5  # Only debug first few levels
            )

        if level_matches.height == 0:
            continue

        if debug:
            print(f"Level {level_idx} (date={date_tol}d, age={age_tol_months}mo, {level_name}): Found {level_matches.height} candidate matches")

        # Group matches by donor
        donor_groups = level_matches.group_by('donor_id').agg([
            pl.all().sort_by('match_score', descending=True)
        ])

        # Process each donor's matches
        new_matches_count = 0
        for donor_row in donor_groups.iter_rows(named=True):
            donor_id = donor_row['donor_id']

            if donor_id not in matched_donors:
                # Extract matches for this donor (they're already sorted by score)
                donor_matches = []
                for col_name in level_matches.columns:
                    if col_name != 'donor_id':
                        col_data = donor_row.get(col_name, [])
                        if isinstance(col_data, list):
                            for i, val in enumerate(col_data):
                                if len(donor_matches) <= i:
                                    donor_matches.append({'donor_id': donor_id})
                                donor_matches[i][col_name] = val

                if not donor_matches:
                    continue

                if return_all_matches:
                    # Keep ALL matches (no deduplication by patient)
                    best_match = donor_matches[0]
                    best_match['n_alternatives'] = len(donor_matches) - 1
                    best_match['match_confidence'] = 'UNIQUE' if len(donor_matches) == 1 else 'MULTIPLE'

                    best_matches_per_donor[donor_id] = best_match
                    matched_donors.add(donor_id)
                    new_matches_count += 1

                    # Add all matches with rankings
                    for i, match in enumerate(donor_matches):
                        match['is_best'] = (i == 0)
                        match['match_rank'] = i + 1
                        all_matches.append(match)
                else:
                    # Deduplicate by patient_id first
                    seen_patients = {}
                    deduplicated_matches = []
                    for match in donor_matches:
                        patient_id = match['patient_id']
                        if patient_id not in seen_patients:
                            seen_patients[patient_id] = match
                            deduplicated_matches.append(match)

                    if deduplicated_matches:
                        best_match = deduplicated_matches[0]
                        best_match['n_alternatives'] = len(deduplicated_matches) - 1
                        best_match['match_confidence'] = 'UNIQUE' if len(deduplicated_matches) == 1 else 'MULTIPLE'
                        best_match['is_best'] = True
                        best_match['match_rank'] = 1

                        best_matches_per_donor[donor_id] = best_match
                        matched_donors.add(donor_id)
                        new_matches_count += 1

                        # Only add the best match
                        all_matches.append(best_match)

        if debug and new_matches_count > 0:
            print(f"  → Matched {new_matches_count} new donors")
            print(f"  → Total matched so far: {len(matched_donors)}/{len(donors_internal)}")

    # Create final dataframe
    if all_matches:
        matches_df = pl.DataFrame(all_matches)

        # Rename columns to match expected output
        matches_df = matches_df.rename({
            'donor_id': 'DONOR_ID',
            'match_score': 'score',
            'age_diff_months': 'age_diff'
        })

        # Ensure proper data types
        if 'discharge_dttm' in matches_df.columns:
            matches_df = matches_df.with_columns([
                pl.col("discharge_dttm").cast(pl.Datetime),
                pl.col("is_dead").cast(pl.Int8),
            ])
    else:
        # Create empty dataframe with expected schema
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

    # Create summary dataframe
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

    # Final statistics
    total_donors = len(donors_internal)
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