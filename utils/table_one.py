"""
Table 1 utility for creating baseline characteristics summary tables.

This module provides functions to create Table 1 summarizing baseline
characteristics across Overall, CALC Donors, and CLIF Donors cohorts.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def create_table_one(final_cohort_df: pl.DataFrame, output_dir: str = 'output') -> pl.DataFrame:
    """
    Create Table 1 summarizing baseline characteristics across cohorts.

    Parameters
    ----------
    final_cohort_df : pl.DataFrame
        Final cohort dataframe with all variables
    output_dir : str
        Directory to save CSV output

    Returns
    -------
    pl.DataFrame
        Table 1 with characteristics for Overall, CALC, and CLIF cohorts
    """

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ============================================
    # Define cohorts
    # ============================================
    cohort_overall = final_cohort_df
    cohort_calc = final_cohort_df.filter(pl.col('calc_flag'))
    cohort_clif = final_cohort_df.filter(pl.col('clif_eligible_donors'))

    cohorts = {
        'Overall': cohort_overall,
        'CALC Donors': cohort_calc,
        'CLIF Donors': cohort_clif
    }

    # Get cohort sizes for display
    cohort_sizes = {
        name: f"{name} (n={cohort_df['patient_id'].n_unique()})"
        for name, cohort_df in cohorts.items()
    }

    # ============================================
    # Variables to summarize
    # ============================================
    categorical_vars = [
        'race_category',
        'ethnicity_category',
        'sex_category',
        'first_admission_location',
        # Combined organ eligibility criteria
        'kidney_eligible',  # Cr <4 AND not on CRRT
        'liver_eligible',   # Bili <4 AND AST <700 AND ALT <700
        'on_crrt_48h_before_death',  # CRRT status
        # Terminal lab thresholds (all patients)
        'creatinine_lt_4',
        'bilirubin_lt_4',
        'ast_lt_700',
        'alt_lt_700',
        # Terminal lab thresholds (BMI ≤50 only)
        'creatinine_lt_4_bmi50',
        'bilirubin_lt_4_bmi50',
        'ast_lt_700_bmi50',
        'alt_lt_700_bmi50'
    ]

    numerical_vars = [
        'age_at_death',
        'hospital_length_of_stay_days',
        'first_icu_los_days',
        'bmi',
        'creatinine_value',
        'bilirubin_total_value',
        'ast_value',
        'alt_value',
        'rass_value',
        'gcs_total_value'
    ]

    # ============================================
    # Build Table 1 data
    # ============================================
    summary_data = []

    # Unique patients and encounters
    summary_data.append({
        'Variable': 'Unique Patients',
        'Category': '',
        'Overall': cohort_overall['patient_id'].n_unique(),
        'CALC_Donors': cohort_calc['patient_id'].n_unique(),
        'CLIF_Donors': cohort_clif['patient_id'].n_unique()
    })

    # After patient-level deduplication, encounters should equal patients
    # but we include this for verification
    summary_data.append({
        'Variable': 'Unique Encounters',
        'Category': '',
        'Overall': len(cohort_overall),  # Should equal unique patients after deduplication
        'CALC_Donors': len(cohort_calc),
        'CLIF_Donors': len(cohort_clif)
    })

    # Data collection period (min-max years from admission_dttm)
    year_ranges = {}
    for cohort_name, cohort_df in cohorts.items():
        if 'admission_dttm' in cohort_df.columns:
            # Get min and max years from admission datetime
            admission_years = cohort_df.select(
                pl.col('admission_dttm').dt.year()
            ).to_series().drop_nulls()

            if len(admission_years) > 0:
                min_year = int(admission_years.min())
                max_year = int(admission_years.max())
                if min_year == max_year:
                    year_ranges[cohort_name] = str(min_year)
                else:
                    year_ranges[cohort_name] = f"{min_year} - {max_year}"
            else:
                year_ranges[cohort_name] = "N/A"
        else:
            year_ranges[cohort_name] = "N/A"

    # Debug print to see what we calculated
    print(f"Data Collection Period - Year ranges calculated: {year_ranges}")

    summary_data.append({
        'Variable': 'Data Collection Period',
        'Category': '',
        'Overall': year_ranges.get('Overall', 'N/A'),
        'CALC_Donors': year_ranges.get('CALC Donors', 'N/A'),
        'CLIF_Donors': year_ranges.get('CLIF Donors', 'N/A')
    })

    # Debug print to verify it was added
    print(f"Data Collection Period row added to summary_data")

    # Custom labels for boolean/threshold variables
    custom_labels = {
        # Combined eligibility
        'kidney_eligible': 'Kidney Eligible (Cr <4 AND not on CRRT)',
        'liver_eligible': 'Liver Eligible (Bili <4 AND AST <700 AND ALT <700)',
        'on_crrt_48h_before_death': 'On CRRT within 48h of death',
        # Terminal thresholds (all patients)
        'creatinine_lt_4': 'Terminal Creatinine < 4',
        'bilirubin_lt_4': 'Terminal Bilirubin < 4',
        'ast_lt_700': 'Terminal AST < 700',
        'alt_lt_700': 'Terminal ALT < 700',
        # Terminal thresholds (BMI ≤50 only)
        'creatinine_lt_4_bmi50': 'Terminal Creatinine < 4 (BMI ≤50)',
        'bilirubin_lt_4_bmi50': 'Terminal Bilirubin < 4 (BMI ≤50)',
        'ast_lt_700_bmi50': 'Terminal AST < 700 (BMI ≤50)',
        'alt_lt_700_bmi50': 'Terminal ALT < 700 (BMI ≤50)'
    }

    # ============================================
    # Categorical variables
    # ============================================
    for var_name in categorical_vars:
        # Check if this variable exists in the dataframe
        if var_name not in final_cohort_df.columns:
            print(f"Warning: Variable {var_name} not found in dataframe, skipping...")
            continue

        # Get unique categories
        all_categories = final_cohort_df.select(var_name).unique().to_series().to_list()

        # Check if this is a boolean flag
        is_boolean = all(v in [True, False, None] for v in all_categories if v is not None)

        if is_boolean:
            # For boolean flags, only show True counts
            var_label = custom_labels.get(var_name, var_name.replace('_', ' ').title())
            row = {'Variable': var_label, 'Category': ''}

            for cohort_name, cohort_df in cohorts.items():
                n = cohort_df.filter(pl.col(var_name) == True).shape[0]
                total = cohort_df.shape[0]
                pct = (n / total * 100) if total > 0 else 0
                row[cohort_name.replace(' ', '_')] = f"{n} ({pct:.1f}%)"

            summary_data.append(row)
        else:
            # Original categorical processing
            categories = sorted([c for c in all_categories if c is not None])

            for category in categories:
                row = {'Variable': var_name.replace('_', ' ').title(), 'Category': str(category)}

                for cohort_name, cohort_df in cohorts.items():
                    n = cohort_df.filter(pl.col(var_name) == category).shape[0]
                    total = cohort_df.shape[0]
                    pct = (n / total * 100) if total > 0 else 0
                    row[cohort_name.replace(' ', '_')] = f"{n} ({pct:.1f}%)"

                summary_data.append(row)

    # ============================================
    # Numerical variables
    # ============================================
    for var_name in numerical_vars:
        var_label = var_name.replace('_', ' ').title()

        # Median (IQR) row
        row_median = {'Variable': var_label, 'Category': 'Median (IQR)'}

        for cohort_name, cohort_df in cohorts.items():
            values = cohort_df.select(var_name).to_series().drop_nulls()

            if len(values) > 0:
                median = float(values.median())
                q1 = float(values.quantile(0.25))
                q3 = float(values.quantile(0.75))
                row_median[cohort_name.replace(' ', '_')] = f"{median:.1f} ({q1:.1f}-{q3:.1f})"
            else:
                row_median[cohort_name.replace(' ', '_')] = "N/A"

        summary_data.append(row_median)

        # N valid (missing) row
        row_missing = {'Variable': var_label, 'Category': 'N valid (% available)'}

        for cohort_name, cohort_df in cohorts.items():
            values = cohort_df.select(var_name).to_series().drop_nulls()
            n_total = cohort_df.shape[0]
            n_valid = len(values)
            pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0
            n_missing = n_total - n_valid
            row_missing[cohort_name.replace(' ', '_')] = f"{n_valid} ({pct_valid:.1f}%), {n_missing} missing"

        summary_data.append(row_missing)

    # ============================================
    # Convert to DataFrame
    # ============================================
    table_one_df = pl.DataFrame(summary_data)

    # Convert to pandas for export and printing
    table_one_pd = table_one_df.to_pandas()

    # Debug: Check if Data Collection Period is in the dataframe
    data_collection_rows = table_one_df.filter(pl.col('Variable') == 'Data Collection Period')
    print(f"Data Collection Period rows in DataFrame: {data_collection_rows}")

    # Save to CSV
    csv_path = output_path / 'table_one.csv'
    table_one_df.write_csv(csv_path)
    print(f"✓ Saved Table 1 to {csv_path}")

    # Save to HTML
    html_path = output_path / 'table_one.html'
    html_content = table_one_pd.to_html(index=False, border=0, classes='table table-striped')

    # Add CSS styling
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Table 1: Baseline Characteristics</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background-color: #4472C4;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                border: 1px solid #ddd;
            }}
            td {{
                padding: 10px 12px;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f0f0f0;
            }}
            /* Highlight summary rows */
            tr:nth-child(2), tr:nth-child(3), tr:nth-child(4) {{
                background-color: #e8f2ff;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>TABLE 1: BASELINE CHARACTERISTICS</h1>
        {html_content}
    </body>
    </html>
    """

    with open(html_path, 'w', encoding="utf-8") as f:
        f.write(html_with_style)
    print(f"✓ Saved Table 1 to {html_path}")

    # ============================================
    # Print to console
    # ============================================
    print("\n" + "="*160)
    print("TABLE 1: BASELINE CHARACTERISTICS")
    print("="*160)
    print(table_one_pd.to_string(index=False))
    print("="*160 + "\n")

    return table_one_df
