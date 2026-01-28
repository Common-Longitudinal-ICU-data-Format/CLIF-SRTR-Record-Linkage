#!/usr/bin/env python3
"""
SRTR-CLIF Donor-Patient Match Viewer
Interactive Streamlit app for exploring matched donor-patient pairs with longitudinal clinical data.
"""

import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import os
import sys
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SRTR-CLIF Match Viewer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = Path("../output/intermediate")
CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW", "REVIEW"]
CONFIDENCE_COLORS = {
    "HIGH": "#28a745",
    "MEDIUM": "#ffc107",
    "LOW": "#fd7e14",
    "REVIEW": "#dc3545"
}

@st.cache_data
def load_matching_results():
    """Load the matching results from the algorithm."""
    try:
        # Load from parquet file (output from 01_record_linkage_dev.py)
        matches_path = DATA_DIR / "matches_df.parquet"

        if not matches_path.exists():
            st.error(f"No match data found at {matches_path}. Please run the matching algorithm.")
            return None, None

        # Load with pandas for better compatibility
        matches_df = pd.read_parquet(matches_path)

        # Create results_df from matches
        results_df = matches_df.groupby('DONOR_ID').agg({
            'confidence': lambda x: x.iloc[0] if len(x) > 0 else 'UNKNOWN',
            'patient_id': 'count',
            'don_utilized': lambda x: x.iloc[0] if len(x) > 0 else None,
            'DON_NON_HR_BEAT': lambda x: x.iloc[0] if len(x) > 0 else None
        }).reset_index()
        results_df.rename(columns={'patient_id': 'n_matches'}, inplace=True)
        results_df['match_status'] = 'matched'

        # Convert date columns if they exist
        date_cols = ['recovery_date', 'death_date', 'discharge_dttm']
        for col in date_cols:
            if col in matches_df.columns:
                matches_df[col] = pd.to_datetime(matches_df[col], errors='coerce')
            if col in results_df.columns:
                results_df[col] = pd.to_datetime(results_df[col], errors='coerce')

        return matches_df, results_df
    except Exception as e:
        st.error(f"Error loading matching results: {str(e)}")
        return None, None

@st.cache_data
def load_encounter_mapping():
    """Load the encounter mapping that links hospitalization_id to encounter_block."""
    try:
        # Try final_clif_data first (has better mappings)
        cohort_path = DATA_DIR / "final_clif_data.parquet"
        if cohort_path.exists():
            cohort_df = pd.read_parquet(cohort_path)
            # Return only the columns we need
            if 'encounter_block' in cohort_df.columns and 'hospitalization_id' in cohort_df.columns:
                return cohort_df[['encounter_block', 'hospitalization_id', 'patient_id']].drop_duplicates()

        # Fallback to encounter_mapping_matched
        mapping_path = DATA_DIR / "encounter_mapping_matched.parquet"
        if mapping_path.exists():
            mapping_df = pd.read_parquet(mapping_path)
            return mapping_df
        else:
            st.warning("Encounter mapping file not found")
            return None
    except Exception as e:
        st.error(f"Error loading encounter mapping: {str(e)}")
        return None

@st.cache_data
def load_wide_df_with_mapping():
    """Load the wide longitudinal data from CLIF and merge with encounter mapping."""
    try:
        # Load the wide_df (output from 02_create_wide_df.py)
        wide_df_path = DATA_DIR / "wide_df.parquet"

        if not wide_df_path.exists():
            st.warning(f"Wide dataset not found at {wide_df_path}")
            return None

        wide_df = pd.read_parquet(wide_df_path)

        # Ensure event_dttm is datetime
        if 'event_dttm' in wide_df.columns:
            wide_df['event_dttm'] = pd.to_datetime(wide_df['event_dttm'], errors='coerce')

        # Now load and merge with encounter mapping
        mapping_df = load_encounter_mapping()
        if mapping_df is not None:
            # Merge to get encounter_block in wide_df
            # Keep only the encounter_block column from mapping
            mapping_subset = mapping_df[['hospitalization_id', 'encounter_block']].drop_duplicates()
            wide_df = wide_df.merge(mapping_subset, on='hospitalization_id', how='left')

            # Report how many records got mapped
            mapped_count = wide_df['encounter_block'].notna().sum()
            total_count = len(wide_df)
            if mapped_count > 0:
                st.success(f"✓ Linked {mapped_count:,} of {total_count:,} wide_df records to encounters")
            else:
                st.warning("No records could be linked to encounters")

        return wide_df

    except Exception as e:
        st.error(f"Error loading wide dataset: {str(e)}")
        return None

@st.cache_data
def load_srtr_donors():
    """Load the SRTR donor data."""
    try:
        # Load from parquet file (output from 01_record_linkage_dev.py)
        donors_path = DATA_DIR / "final_srtr_data.parquet"

        if not donors_path.exists():
            st.warning(f"SRTR donor file not found at {donors_path}")
            return None

        donors_df = pd.read_parquet(donors_path)

        # Convert date columns
        if 'DON_RECOV_DT' in donors_df.columns:
            donors_df['DON_RECOV_DT'] = pd.to_datetime(donors_df['DON_RECOV_DT'], errors='coerce')

        return donors_df
    except Exception as e:
        st.error(f"Error loading SRTR donor data: {str(e)}")
        return None

@st.cache_data
def load_table_one():
    """Load the pre-generated Table 1 donor comparison."""
    try:
        # Check for table1 files from 01_record_linkage_dev.py
        possible_files = [
            "../final/table1_high_matches.csv",
            "../final/table1_medium_matches.csv",
            "table_one_donor_comparison.csv"
        ]

        for filename in possible_files:
            table_one_path = DATA_DIR.parent / filename if filename.startswith("../") else DATA_DIR / filename
            if table_one_path.exists():
                return pd.read_csv(table_one_path)
        return None
    except Exception as e:
        st.error(f"Error loading Table 1: {str(e)}")
        return None

def create_summary_stats(matches_df, results_df, donors_df):
    """Create summary statistics for the About page."""
    stats = {}

    if results_df is not None:
        total_donors = len(results_df)
        matched_donors = len(results_df[results_df['match_status'] == 'matched'])
        stats['match_rate'] = (matched_donors / total_donors * 100) if total_donors > 0 else 0
        stats['total_donors'] = total_donors
        stats['matched_donors'] = matched_donors
        stats['unmatched_donors'] = total_donors - matched_donors

    if matches_df is not None:
        stats['total_matches'] = len(matches_df)
        stats['confidence_dist'] = matches_df['confidence'].value_counts().to_dict()
        stats['avg_score'] = matches_df['score'].mean() if 'score' in matches_df.columns else 0
        stats['avg_date_diff'] = matches_df['date_diff'].abs().mean() if 'date_diff' in matches_df.columns else 0
        stats['avg_age_diff'] = matches_df['age_diff'].abs().mean() if 'age_diff' in matches_df.columns else 0

    return stats

def display_donor_info(donor_id, donors_df, matches_df):
    """Display donor information in a formatted way."""
    if donors_df is None:
        st.warning("Donor data not available")
        return

    donor_info = donors_df[donors_df['DONOR_ID'] == donor_id]
    if donor_info.empty:
        st.warning(f"No donor information found for ID: {donor_id}")
        return

    donor = donor_info.iloc[0]

    # Get utilization and DCD status from matches_df
    donor_match_info = matches_df[matches_df['DONOR_ID'] == donor_id].iloc[0] if not matches_df[matches_df['DONOR_ID'] == donor_id].empty else {}

    # Create columns for donor info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Demographics")
        st.write(f"**Age:** {donor.get('DON_AGE', 'N/A')} years")
        st.write(f"**Gender:** {donor.get('DON_GENDER', 'N/A')}")
        st.write(f"**Race:** {donor.get('DON_RACE_SRTR', 'N/A')}")
        st.write(f"**Ethnicity:** {donor.get('DON_ETHNICITY_SRTR', 'N/A')}")

    with col2:
        st.markdown("### Clinical Info")
        st.write(f"**Recovery Date:** {donor.get('DON_RECOV_DT', 'N/A')}")
        st.write(f"**Height:** {donor.get('DON_HGT_CM', 'N/A')} cm")
        st.write(f"**Weight:** {donor.get('DON_WGT_KG', 'N/A')} kg")
        if 'DON_HGT_CM' in donor and 'DON_WGT_KG' in donor:
            if pd.notna(donor['DON_HGT_CM']) and pd.notna(donor['DON_WGT_KG']) and donor['DON_HGT_CM'] > 0:
                bmi = donor['DON_WGT_KG'] / ((donor['DON_HGT_CM']/100) ** 2)
                st.write(f"**BMI:** {bmi:.1f}")

    with col3:
        st.markdown("### Donor Status")
        # Donor utilization status
        if 'don_utilized' in donor_match_info:
            utilized = donor_match_info.get('don_utilized')
            utilized_display = '✅ Yes' if utilized == 1 or utilized == 'Y' else '❌ No' if utilized == 0 or utilized == 'N' else 'Unknown'
            st.write(f"**Donor Utilized:** {utilized_display}")

        # DCD status
        if 'DON_NON_HR_BEAT' in donor_match_info:
            dcd = donor_match_info.get('DON_NON_HR_BEAT')
            dcd_display = '✅ Yes (DCD)' if dcd == 1 or dcd == 'Y' else '❌ No (DBD)' if dcd == 0 or dcd == 'N' else 'Unknown'
            st.write(f"**DCD Donor:** {dcd_display}")

        st.write(f"**Death Mechanism:** {donor.get('DON_DEATH_MECH', 'N/A')}")
        st.write(f"**Death Circumstance:** {donor.get('DON_DEATH_CIRCUM', 'N/A')}")

def create_event_timeline(selected_match, clif_data, donor_data):
    """Create a timeline visualization of key events."""

    events = []
    colors = []

    # Helper function to standardize datetime - remove timezone info for consistency
    def standardize_datetime(dt):
        """Convert datetime to timezone-naive for consistent comparison."""
        if dt is None:
            return None
        dt = pd.to_datetime(dt)
        if dt is pd.NaT:
            return None
        # Remove timezone info if present
        if hasattr(dt, 'tz_localize'):
            if dt.tz is not None:
                dt = dt.tz_localize(None)
        return dt

    # Get patient data from CLIF
    if clif_data is not None and 'encounter_block' in selected_match:
        patient_data = clif_data[clif_data['encounter_block'] == selected_match['encounter_block']]
        if not patient_data.empty:
            patient = patient_data.iloc[0]

            # Add patient events
            if 'admission_dttm' in patient and pd.notna(patient['admission_dttm']):
                dt = standardize_datetime(patient['admission_dttm'])
                if dt is not None:
                    events.append({
                        'event': 'Admission',
                        'datetime': dt,
                        'type': 'patient',
                        'color': '#1f77b4'
                    })

            if 'discharge_dttm' in patient and pd.notna(patient['discharge_dttm']):
                dt = standardize_datetime(patient['discharge_dttm'])
                if dt is not None:
                    events.append({
                        'event': 'Discharge',
                        'datetime': dt,
                        'type': 'patient',
                        'color': '#ff7f0e'
                    })

            if 'death_dttm' in patient and pd.notna(patient['death_dttm']):
                dt = standardize_datetime(patient['death_dttm'])
                if dt is not None:
                    events.append({
                        'event': 'Death (EHR)',
                        'datetime': dt,
                        'type': 'patient',
                        'color': '#d62728'
                    })

            if 'last_recorded_vital_dttm' in patient and pd.notna(patient['last_recorded_vital_dttm']):
                dt = standardize_datetime(patient['last_recorded_vital_dttm'])
                if dt is not None:
                    events.append({
                        'event': 'Last Vital',
                        'datetime': dt,
                        'type': 'patient',
                        'color': '#9467bd'
                    })

    # Add donor recovery date
    if 'recovery_date' in selected_match and pd.notna(selected_match['recovery_date']):
        dt = standardize_datetime(selected_match['recovery_date'])
        if dt is not None:
            events.append({
                'event': 'Donor Recovery',
                'datetime': dt,
                'type': 'donor',
                'color': '#2ca02c'
            })
    elif donor_data is not None and 'DONOR_ID' in selected_match:
        donor_info = donor_data[donor_data['DONOR_ID'] == selected_match['DONOR_ID']]
        if not donor_info.empty and 'DON_RECOV_DT' in donor_info.columns:
            recovery_dt = donor_info.iloc[0]['DON_RECOV_DT']
            if pd.notna(recovery_dt):
                dt = standardize_datetime(recovery_dt)
                if dt is not None:
                    events.append({
                        'event': 'Donor Recovery',
                        'datetime': dt,
                        'type': 'donor',
                        'color': '#2ca02c'
                    })

    if not events:
        return None

    # Sort events by datetime
    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values('datetime')

    # Assign staggered y-positions to prevent label overlap
    # Patient events at different heights, donor event separate
    y_positions = {
        'Admission': 1.0,
        'Discharge': 1.3,
        'Death (EHR)': 1.6,
        'Last Vital': 0.7,
        'Donor Recovery': 0.3
    }

    # Create the timeline figure
    fig = go.Figure()

    # Add vertical drop lines for each event
    for _, event in events_df.iterrows():
        y_pos = y_positions.get(event['event'], 1.0)

        # Add vertical line from bottom to event
        fig.add_shape(
            type="line",
            x0=event['datetime'], x1=event['datetime'],
            y0=0.1, y1=y_pos - 0.05,
            line=dict(
                color=event['color'],
                width=2,
                dash="dot"
            ),
            opacity=0.5  # More visible on dark background
        )

    # Add events as markers with staggered positions
    for _, event in events_df.iterrows():
        y_pos = y_positions.get(event['event'], 1.0)

        fig.add_trace(go.Scatter(
            x=[event['datetime']],
            y=[y_pos],
            mode='markers+text',
            name=event['event'],
            text=f"<b>{event['event']}</b>",
            textposition="top center",
            textfont=dict(size=18, color='white'),  # Larger white text for dark background
            marker=dict(
                size=20,  # Larger marker size
                color=event['color'],
                symbol='diamond' if event['type'] == 'donor' else 'circle',
                line=dict(width=3, color='white')  # Thicker white border for better visibility
            ),
            showlegend=True,
            hovertemplate=f"<b>{event['event']}</b><br>Date: %{{x|%Y-%m-%d}}<br>Time: %{{x|%H:%M}}<extra></extra>",
            hoverlabel=dict(font_size=16, bgcolor=event['color'], font_color='white')  # Larger white hover text
        ))

    # Add connecting lines for patient stay
    admission_events = events_df[events_df['event'] == 'Admission']
    discharge_events = events_df[events_df['event'].isin(['Discharge', 'Death (EHR)'])]

    if not admission_events.empty and not discharge_events.empty:
        start_date = admission_events.iloc[0]['datetime']
        end_date = discharge_events.iloc[0]['datetime']

        # Get y-positions for line
        start_y = y_positions.get('Admission', 1.0)
        end_event = 'Death (EHR)' if 'Death (EHR)' in discharge_events['event'].values else 'Discharge'
        end_y = y_positions.get(end_event, 1.3)

        # Add horizontal line for patient stay
        fig.add_shape(
            type="line",
            x0=start_date, x1=end_date,
            y0=start_y, y1=start_y,
            line=dict(color="rgba(255, 255, 255, 0.4)", width=4, dash="solid")  # White line for dark background
        )

        # Add vertical connectors if death after discharge
        if 'Discharge' in events_df['event'].values and 'Death (EHR)' in events_df['event'].values:
            discharge_date = events_df[events_df['event'] == 'Discharge'].iloc[0]['datetime']
            death_date = events_df[events_df['event'] == 'Death (EHR)'].iloc[0]['datetime']

            fig.add_shape(
                type="line",
                x0=discharge_date, x1=death_date,
                y0=y_positions['Discharge'], y1=y_positions['Death (EHR)'],
                line=dict(color="rgba(255, 255, 255, 0.3)", width=2, dash="dot")  # White dotted line
            )

    # Update layout with larger fonts and height
    fig.update_layout(
        title=dict(
            text="<b>Patient-Donor Event Timeline</b>",
            font=dict(size=24, color='white')  # Larger white title
        ),
        xaxis=dict(
            title=dict(text="Date/Time", font=dict(size=20, color='white')),  # Larger white axis title
            type='date',
            tickformat='%Y-%m-%d %H:%M',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.2)',  # White grid lines
            tickfont=dict(size=16, color='white'),  # Larger white tick labels
            linecolor='rgba(255, 255, 255, 0.5)',  # White axis line
            tickcolor='rgba(255, 255, 255, 0.5)'  # White tick marks
        ),
        yaxis=dict(
            title="",
            range=[0, 2.0],  # Increased range for staggered positions
            showticklabels=False,
            showgrid=False,
            linecolor='rgba(255, 255, 255, 0.5)'  # White axis line
        ),
        height=450,  # Increased height
        hovermode='closest',  # Changed from 'x unified' to prevent overlap
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,  # Moved legend to top
            xanchor="center",
            x=0.5,
            font=dict(size=16, color='white'),  # Larger white legend text
            bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
            bordercolor="rgba(255, 255, 255, 0.3)",  # White subtle border
            borderwidth=0.5
        ),
        margin=dict(t=100, b=60, l=60, r=60),  # Adjusted margins for top legend
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)'  # Transparent paper background
    )

    return fig

def display_patient_timeline(selected_match, wide_df):
    """Display patient longitudinal data."""
    if wide_df is None:
        st.warning("Longitudinal data not available")
        return

    patient_data = pd.DataFrame()

    # Try to find patient data using different identifiers
    # Priority: hospitalization_id > encounter_block
    if 'hospitalization_id' in selected_match and pd.notna(selected_match['hospitalization_id']):
        # First try hospitalization_id (most reliable)
        hosp_id = selected_match['hospitalization_id']
        patient_data = wide_df[wide_df['hospitalization_id'] == hosp_id]
        if not patient_data.empty:
            st.success(f"Found data using hospitalization_id: {hosp_id}")

    if patient_data.empty and 'encounter_block' in wide_df.columns:
        # Try encounter_block if hospitalization_id didn't work
        enc_block = selected_match.get('encounter_block')
        if enc_block and pd.notna(enc_block):
            patient_data = wide_df[wide_df['encounter_block'] == enc_block]
            if not patient_data.empty:
                st.success(f"Found data using encounter_block: {enc_block}")

    if patient_data.empty:
        st.warning("No longitudinal data found for this patient")
        # Show debug info to help troubleshoot
        with st.expander("Debug Info"):
            st.write("Match information:")
            st.write(f"- Hospitalization ID: {selected_match.get('hospitalization_id', 'N/A')}")
            st.write(f"- Encounter Block: {selected_match.get('encounter_block', 'N/A')}")
            st.write(f"- Patient ID: {selected_match.get('patient_id', 'N/A')}")
            st.write(f"\nWide_df shape: {wide_df.shape}")
            st.write(f"Wide_df columns: {wide_df.columns.tolist()[:10]}...")
        return

    st.markdown(f"### Clinical Timeline - {len(patient_data)} events recorded")

    # Create tabs for different data categories
    tabs = st.tabs(["Vitals", "Labs", "Medications", "Respiratory", "Overview"])

    with tabs[0]:  # Vitals
        st.markdown("#### Vital Signs")
        vital_cols = [col for col in patient_data.columns if col.startswith('vital_')]
        if vital_cols:
            # Add toggle to show all rows
            show_all_vitals = st.checkbox("Show all vital signs", key="show_all_vitals")

            vitals_data = patient_data[['event_dttm'] + vital_cols].dropna(how='all', subset=vital_cols)
            if not vitals_data.empty:
                # Display either all rows or last 10 based on checkbox
                if show_all_vitals:
                    st.dataframe(vitals_data, use_container_width=True)
                    st.caption(f"Showing all {len(vitals_data)} vital sign measurements")
                else:
                    recent_vitals = vitals_data.tail(10)
                    st.dataframe(recent_vitals, use_container_width=True)
                    if len(vitals_data) > 10:
                        st.caption(f"Showing last 10 of {len(vitals_data)} vital sign measurements. Check the box above to see all.")
            else:
                st.info("No vital signs recorded")
        else:
            st.info("No vital sign columns found")

    with tabs[1]:  # Labs
        st.markdown("#### Laboratory Results")
        lab_cols = [col for col in patient_data.columns if col.startswith('lab_')]
        if lab_cols:
            # Add toggle to show all rows
            show_all_labs = st.checkbox("Show all lab results", key="show_all_labs")

            labs_data = patient_data[['event_dttm'] + lab_cols].dropna(how='all', subset=lab_cols)
            if not labs_data.empty:
                # Display either all rows or last 10 based on checkbox
                if show_all_labs:
                    st.dataframe(labs_data, use_container_width=True)
                    st.caption(f"Showing all {len(labs_data)} lab results")
                else:
                    recent_labs = labs_data.tail(10)
                    st.dataframe(recent_labs, use_container_width=True)
                    if len(labs_data) > 10:
                        st.caption(f"Showing last 10 of {len(labs_data)} lab results. Check the box above to see all.")
            else:
                st.info("No lab results recorded")
        else:
            st.info("No lab columns found")

    with tabs[2]:  # Medications
        st.markdown("#### Medications")
        med_cols = [col for col in patient_data.columns if col.startswith('med_')]
        if med_cols:
            # Add toggle to show all rows
            show_all_meds = st.checkbox("Show all medications", key="show_all_meds")

            meds_data = patient_data[['event_dttm'] + med_cols].dropna(how='all', subset=med_cols)
            if not meds_data.empty:
                # Display either all rows or last 10 based on checkbox
                if show_all_meds:
                    st.dataframe(meds_data, use_container_width=True)
                    st.caption(f"Showing all {len(meds_data)} medication records")
                else:
                    recent_meds = meds_data.tail(10)
                    st.dataframe(recent_meds, use_container_width=True)
                    if len(meds_data) > 10:
                        st.caption(f"Showing last 10 of {len(meds_data)} medication records. Check the box above to see all.")
            else:
                st.info("No medications recorded")
        else:
            st.info("No medication columns found")

    with tabs[3]:  # Respiratory
        st.markdown("#### Respiratory Support")
        resp_cols = [col for col in patient_data.columns if col.startswith('resp_')]
        if resp_cols:
            # Add toggle to show all rows
            show_all_resp = st.checkbox("Show all respiratory support data", key="show_all_resp")

            resp_data = patient_data[['event_dttm'] + resp_cols].dropna(how='all', subset=resp_cols)
            if not resp_data.empty:
                # Display either all rows or last 10 based on checkbox
                if show_all_resp:
                    st.dataframe(resp_data, use_container_width=True)
                    st.caption(f"Showing all {len(resp_data)} respiratory support records")
                else:
                    recent_resp = resp_data.tail(10)
                    st.dataframe(recent_resp, use_container_width=True)
                    if len(resp_data) > 10:
                        st.caption(f"Showing last 10 of {len(resp_data)} respiratory support records. Check the box above to see all.")
            else:
                st.info("No respiratory data recorded")
        else:
            st.info("No respiratory columns found")

    with tabs[4]:  # Overview
        st.markdown("#### Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Events", len(patient_data))
            if 'event_dttm' in patient_data.columns:
                st.metric("Date Range",
                         f"{patient_data['event_dttm'].min().strftime('%Y-%m-%d') if pd.notna(patient_data['event_dttm'].min()) else 'N/A'} to "
                         f"{patient_data['event_dttm'].max().strftime('%Y-%m-%d') if pd.notna(patient_data['event_dttm'].max()) else 'N/A'}")
        with col2:
            # Count non-null values per category
            vital_count = patient_data[vital_cols].notna().sum().sum() if 'vital_cols' in locals() else 0
            lab_count = patient_data[lab_cols].notna().sum().sum() if 'lab_cols' in locals() else 0
            st.metric("Vital Measurements", vital_count)
            st.metric("Lab Results", lab_count)

def main_page():
    """Main page for viewing donor-patient matches."""
    st.title("🏥 SRTR-CLIF Donor-Patient Match Viewer")

    # Load data
    with st.spinner("Loading data..."):
        matches_df, results_df = load_matching_results()
        wide_df = load_wide_df_with_mapping()  # Use the new function that includes mapping
        donors_df = load_srtr_donors()

        # Also load encounter mapping to get hospitalization_ids
        encounter_mapping = load_encounter_mapping()

        # Load CLIF patient data for timeline events
        clif_data_path = DATA_DIR / "final_clif_data.parquet"
        clif_data = pd.read_parquet(clif_data_path) if clif_data_path.exists() else None

        # Enrich matches with hospitalization_id from encounter mapping
        if matches_df is not None and encounter_mapping is not None:
            # Get the first hospitalization_id for each encounter_block
            enc_to_hosp = encounter_mapping.groupby('encounter_block')['hospitalization_id'].first().to_dict()
            matches_df['hospitalization_id'] = matches_df['encounter_block'].map(enc_to_hosp)

    if matches_df is None or results_df is None:
        st.error("⚠️ Unable to load matching results. Please ensure the matching algorithm has been run.")
        st.info("📝 **To generate the required data:**")
        st.markdown("""
        1. Run `01_record_linkage_dev.py` to perform the patient-donor matching
        2. Run `02_create_wide_df.py` to create the wide longitudinal dataset

        The following data files are required:
        - `output/intermediate/matches_df.parquet` - Match results
        - `output/intermediate/final_clif_data.parquet` - CLIF patient data
        - `output/intermediate/final_srtr_data.parquet` - SRTR donor data
        - `output/intermediate/wide_df.parquet` - Wide longitudinal dataset
        - `output/intermediate/encounter_mapping_matched.parquet` - Encounter mappings
        """)
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Data status in sidebar
    with st.sidebar.expander("Data Status", expanded=False):
        st.write(f"**Matches loaded:** {len(matches_df):,} records")
        st.write(f"**Donors loaded:** {len(results_df):,} records")
        if wide_df is not None:
            st.write(f"**Timeline data:** {len(wide_df):,} events")
            if 'encounter_block' in wide_df.columns:
                mapped_encounters = wide_df['encounter_block'].notna().sum()
                st.write(f"**Mapped events:** {mapped_encounters:,}")
                unique_encounters = wide_df['encounter_block'].nunique()
                st.write(f"**Unique patients:** {unique_encounters:,}")
        else:
            st.write("**Timeline data:** Not loaded")

    # Confidence level filter
    selected_confidence = st.sidebar.selectbox(
        "Select Confidence Level",
        options=CONFIDENCE_LEVELS,
        help="Filter matches by confidence level"
    )

    # Filter matches by confidence
    filtered_matches = matches_df[matches_df['confidence'] == selected_confidence]

    if filtered_matches.empty:
        st.warning(f"No matches found with {selected_confidence} confidence")
        return

    # Get unique donors for this confidence level
    unique_donors = filtered_matches['DONOR_ID'].unique()

    # Donor selection
    selected_donor = st.sidebar.selectbox(
        f"Select Donor ({len(unique_donors)} available)",
        options=sorted(unique_donors),
        help="Select a donor to view matches"
    )

    # Get matches for selected donor
    donor_matches = filtered_matches[filtered_matches['DONOR_ID'] == selected_donor]

    # Display header with match info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confidence", selected_confidence,
                 help="Match confidence level")
    with col2:
        st.metric("Matches for Donor", len(donor_matches),
                 help="Number of potential patient matches")
    with col3:
        if 'score' in donor_matches.columns:
            st.metric("Best Score", f"{donor_matches['score'].max():.1f}",
                     help="Highest match score (0-100)")
    with col4:
        if 'date_diff' in donor_matches.columns:
            st.metric("Best Date Diff", f"{donor_matches['date_diff'].abs().min():.1f} days",
                     help="Smallest date difference")

    # Display key indicators
    if not donor_matches.empty:
        first_match = donor_matches.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'don_utilized' in first_match:
                utilized = first_match.get('don_utilized')
                icon = '✅' if utilized == 1 or utilized == 'Y' else '❌'
                st.metric("Donor Utilized", icon,
                         help="Whether the donor organs were utilized")
        with col2:
            if 'DON_NON_HR_BEAT' in first_match:
                dcd = first_match.get('DON_NON_HR_BEAT')
                status = 'DCD' if dcd == 1 or dcd == 'Y' else 'DBD'
                st.metric("Donor Type", status,
                         help="DCD: Donation after Circulatory Death, DBD: Donation after Brain Death")
        with col3:
            # Count how many matched patients died
            if 'is_dead' in donor_matches.columns:
                dead_count = donor_matches['is_dead'].apply(lambda x: x == 1 or x == True).sum()
                st.metric("Deceased Matches", f"{dead_count}/{len(donor_matches)}",
                         help="Number of matched patients who died")
        with col4:
            # Show if best match patient died
            if 'is_dead' in donor_matches.columns:
                best_match = donor_matches.sort_values('score', ascending=False).iloc[0] if 'score' in donor_matches.columns else donor_matches.iloc[0]
                is_dead = best_match.get('is_dead')
                status = 'Deceased' if is_dead == 1 or is_dead == True else 'Alive'
                st.metric("Best Match EHR Status", status,
                         help="Whether the best matched patient died in EHR")

    # Display donor information
    st.markdown("---")
    st.header(f"Donor Information - ID: {selected_donor}")
    display_donor_info(selected_donor, donors_df, matches_df)

    # Display patient matches
    st.markdown("---")
    st.header("Patient Matches")

    if len(donor_matches) > 1:
        # If multiple matches, let user select
        match_options = []
        for idx, match in donor_matches.iterrows():
            score_text = f"Score: {match['score']:.1f}" if 'score' in match else ""
            date_diff_text = f"Date diff: {abs(match['date_diff']):.1f}d" if 'date_diff' in match else ""
            option_text = f"Patient {match.get('patient_id', match.get('encounter_block', 'Unknown'))} - {score_text} {date_diff_text}"
            match_options.append((idx, option_text))

        selected_match_idx = st.selectbox(
            "Select Patient Match",
            options=[idx for idx, _ in match_options],
            format_func=lambda x: [text for idx, text in match_options if idx == x][0]
        )

        selected_match = donor_matches.loc[selected_match_idx]
    else:
        selected_match = donor_matches.iloc[0]

    # Display match details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Match Details")
        st.write(f"**Patient ID:** {selected_match.get('patient_id', 'N/A')}")
        st.write(f"**Encounter:** {selected_match.get('encounter_block', 'N/A')}")
        if 'hospitalization_id' in selected_match:
            st.write(f"**Hospitalization ID:** {selected_match.get('hospitalization_id', 'N/A')}")

        # Patient death status
        if 'is_dead' in selected_match:
            is_dead = selected_match.get('is_dead')
            death_display = 'Yes' if is_dead == 1 or is_dead == True else 'No' if is_dead == 0 or is_dead == False else 'Unknown'
            st.write(f"**Deceased in EHR:** {death_display}")

        st.write(f"**Match Score:** {selected_match.get('score', 'N/A'):.1f}" if 'score' in selected_match and pd.notna(selected_match.get('score')) else "")
        st.write(f"**Confidence:** {selected_match.get('confidence', 'N/A')}")

    with col2:
        st.markdown("### Match Metrics")
        st.write(f"**Date Difference:** {selected_match.get('date_diff', 'N/A'):.1f} days" if 'date_diff' in selected_match else "")
        st.write(f"**Age Difference:** {selected_match.get('age_diff', 'N/A'):.1f} months" if 'age_diff' in selected_match else "")
        st.write(f"**Gender Match:** {'✅' if selected_match.get('gender_match') else '❌'}" if 'gender_match' in selected_match else "")
        st.write(f"**Race Match:** {'✅' if selected_match.get('race_match') else '❌'}" if 'race_match' in selected_match else "")

    # Display patient longitudinal data
    st.markdown("---")
    st.header("Patient Clinical Timeline")

    # Display event timeline visualization
    if clif_data is not None or donors_df is not None:
        timeline_fig = create_event_timeline(selected_match, clif_data, donors_df)
        if timeline_fig is not None:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("No timeline events available for visualization")

    if wide_df is not None:
        # Show data availability info
        hosp_id = selected_match.get('hospitalization_id')
        if hosp_id and pd.notna(hosp_id):
            available_events = wide_df[wide_df['hospitalization_id'] == hosp_id]
            if not available_events.empty:
                st.info(f"Found {len(available_events):,} timeline events for this patient")

        display_patient_timeline(selected_match, wide_df)
    else:
        st.warning("Unable to load patient timeline data")
        st.error("Wide dataset not loaded. Please check that wide_df files exist in ../output/intermediate/")

def about_page():
    """About page with algorithm explanation and summary statistics."""
    st.title("📊 About the Matching Algorithm")

    # Load data for statistics
    matches_df, results_df = load_matching_results()
    donors_df = load_srtr_donors()
    stats = create_summary_stats(matches_df, results_df, donors_df)

    # Algorithm overview
    st.header("Algorithm Overview")
    st.markdown("""
    The **Hierarchical Record Linkage with Progressive Tolerance Relaxation** algorithm matches
    SRTR organ donor records with CLIF patient records using a sophisticated approach that:

    - **Progressively relaxes tolerances** to achieve 100% match rate
    - **Calculates probabilistic scores** using weighted components (dates, age, gender, demographics)
    - **Assigns confidence categories** with penalties for relaxed tolerance matches
    - **Uses intelligent blocking** with date windows for computational efficiency
    """)

    # Performance metrics
    st.header("Performance Summary")
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Match Rate", f"{stats.get('match_rate', 0):.1f}%")
            st.metric("Total Donors", stats.get('total_donors', 0))
        with col2:
            st.metric("Matched Donors", stats.get('matched_donors', 0))
            st.metric("Unmatched", stats.get('unmatched_donors', 0))
        with col3:
            st.metric("Total Matches", stats.get('total_matches', 0))
            st.metric("Avg Score", f"{stats.get('avg_score', 0):.1f}")
        with col4:
            st.metric("Avg Date Diff", f"{stats.get('avg_date_diff', 0):.1f} days")
            st.metric("Avg Age Diff", f"{stats.get('avg_age_diff', 0):.1f} months")

    # Confidence distribution
    st.header("Confidence Distribution")
    if 'confidence_dist' in stats and stats['confidence_dist']:
        conf_df = pd.DataFrame.from_dict(stats['confidence_dist'], orient='index', columns=['Count'])
        conf_df['Percentage'] = (conf_df['Count'] / conf_df['Count'].sum() * 100).round(1)
        conf_df = conf_df.reindex(CONFIDENCE_LEVELS, fill_value=0)

        col1, col2 = st.columns([2, 1])
        with col1:
            # Create bar chart
            fig = px.bar(conf_df.reset_index(), x='index', y='Count',
                        color='index', color_discrete_map=CONFIDENCE_COLORS,
                        title="Match Distribution by Confidence Level")
            fig.update_layout(showlegend=False, xaxis_title="Confidence Level", yaxis_title="Number of Matches")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(conf_df, use_container_width=True)

    # Table 1 - Donor Characteristics Comparison
    st.header("Table 1: Donor Characteristics Comparison")

    # Load pre-generated table one if available
    table_one = load_table_one()

    if table_one is not None:
        st.dataframe(table_one, use_container_width=True, hide_index=True)

        # Add summary info below the table
        col1, col2, col3 = st.columns(3)
        with col1:
            # Extract match rate from table if present
            if 'Match Rate (%)' in table_one['Group'].values:
                match_rate_row = table_one[table_one['Group'] == 'Match Rate (%)']
                if not match_rate_row.empty:
                    st.metric("Overall Match Rate", f"{match_rate_row['N'].iloc[0]}%")

    else:
        # Fallback: generate simple comparison if table_one not found
        st.info("Pre-generated Table 1 not found. Generating comparison from current data...")

        if matches_df is not None and donors_df is not None:
            # Get HIGH confidence matches
            high_conf_matches = matches_df[matches_df['confidence'] == 'HIGH']
            high_conf_donors = high_conf_matches['DONOR_ID'].unique()

            # Calculate aggregates
            all_donors = donors_df
            matched_donors = donors_df[donors_df['DONOR_ID'].isin(matches_df['DONOR_ID'].unique())]
            high_donors = donors_df[donors_df['DONOR_ID'].isin(high_conf_donors)]

            comparison_data = {
                'Group': ['All Donors', 'Matched Donors', 'HIGH Confidence'],
                'N': [len(all_donors), len(matched_donors), len(high_donors)],
                'Mean Age': [
                    f"{all_donors['DON_AGE'].mean():.1f}" if 'DON_AGE' in all_donors else 'N/A',
                    f"{matched_donors['DON_AGE'].mean():.1f}" if len(matched_donors) > 0 and 'DON_AGE' in matched_donors else 'N/A',
                    f"{high_donors['DON_AGE'].mean():.1f}" if len(high_donors) > 0 and 'DON_AGE' in high_donors else 'N/A'
                ],
                'Male (%)': [
                    f"{(all_donors['DON_GENDER'] == 'M').mean() * 100:.1f}" if 'DON_GENDER' in all_donors else 'N/A',
                    f"{(matched_donors['DON_GENDER'] == 'M').mean() * 100:.1f}" if len(matched_donors) > 0 and 'DON_GENDER' in matched_donors else 'N/A',
                    f"{(high_donors['DON_GENDER'] == 'M').mean() * 100:.1f}" if len(high_donors) > 0 and 'DON_GENDER' in high_donors else 'N/A'
                ]
            }

            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Algorithm details
    with st.expander("Algorithm Technical Details"):
        st.markdown("""
        ### Scoring Components
        The matching score (0-100) is calculated using weighted components:

        | Component | Weight | Method |
        |-----------|--------|---------|
        | Date Match | 40% | Exponential decay (3-day half-life) |
        | Age Match | 30% | Linear decay (5-year tolerance) |
        | Gender | 20% | Binary match |
        | Demographics | 10% | Race + Ethnicity compatibility |

        ### Confidence Categories
        - **HIGH**: Score ≥ 85 or (date diff ≤ 1 day, age diff = 0, gender match)
        - **MEDIUM**: Score 60-84 or (date diff ≤ 3 days, age diff ≤ 1 year, gender match)
        - **LOW**: Score 40-59 or gender mismatch
        - **REVIEW**: Score < 40 or (date diff > 5 days and age diff > 1 year)

        ### Progressive Tolerance Levels
        The algorithm progressively relaxes tolerances for unmatched donors:

        | Level | Date Tolerance | Age Tolerance | Typical Coverage |
        |-------|---------------|---------------|------------------|
        | STANDARD | ±7 days | ±2 years | ~95.7% |
        | RELAXED | ±14 days | ±3 years | +3.9% |
        | EXPANDED | ±30 days | ±5 years | +0.4% |
        | BROAD | ±90 days | ±10 years | As needed |
        | MAXIMUM | ±365 days | ±20 years | As needed |

        Matches found at relaxed tolerances receive confidence penalties to ensure clinical safety.
        """)

def main():
    """Main application entry point."""

    # Create navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Match Viewer", "About"],
        label_visibility="collapsed"
    )

    if page == "Match Viewer":
        main_page()
    else:
        about_page()

if __name__ == "__main__":
    main()