# SRTR-CLIF Record Linkage Module

## Overview

This module performs hierarchical record linkage between SRTR (Scientific Registry of Transplant Recipients) organ donor records and CLIF (Critical Care Health Informatics Collaborative) patient records. The algorithm uses progressive tolerance relaxation to achieve 100% match rate while maintaining clinical interpretability through confidence categories.

## Key Features

- **100% Match Rate**: Progressive tolerance relaxation ensures every donor is matched
- **Probabilistic Scoring**: Continuous scores (0-100) based on weighted components
- **Confidence Categories**: HIGH/MEDIUM/LOW/REVIEW for clinical interpretation
- **Efficient Processing**: <5 seconds for 250+ donors using date-window blocking
- **Interactive Visualization**: Streamlit app for exploring matches and patient timelines

## Quick Start

### 1. Run the Matching Algorithm

```bash
python 01_match_hierarchical.py
```

This will:
- Load SRTR donor and CLIF patient data
- Run the progressive matching algorithm
- Generate Table 1 comparing donor characteristics
- Save results to `../output/intermediate/`

### 2. Create Wide Dataset (Optional)

```bash
python 02_create_wide_df.py
```

This creates a longitudinal timeline dataset with all clinical events for matched patients.

### 3. Launch the Interactive App

```bash
streamlit run donor_patient_viewer.py
```

Or use the convenience script:
```bash
./launch_app.sh
```

The app opens at http://localhost:8501

## Files

| File | Description |
|------|-------------|
| `01_match_hierarchical.py` | Main matching pipeline |
| `02_create_wide_df.py` | Creates longitudinal patient timeline data |
| `hierarchical_matching_optimized.py` | Core matching algorithm implementation |
| `donor_patient_viewer.py` | Streamlit app for visualization |
| `ALGORITHM_DETAILS.md` | Technical documentation of the algorithm |
| `requirements_app.txt` | Python dependencies for the app |
| `launch_app.sh` | Shell script to launch the app |

## Algorithm Summary

The matching algorithm:
1. **Standardizes demographics** for both datasets
2. **Applies date-window blocking** (±365 days) for efficiency
3. **Progressively relaxes tolerances** in 5 levels:
   - STANDARD: ±7 days, ±2 years (~95.7% matched)
   - RELAXED: ±14 days, ±3 years (+3.9%)
   - EXPANDED: ±30 days, ±5 years (+0.4%)
   - BROAD: ±90 days, ±10 years (as needed)
   - MAXIMUM: ±365 days, ±20 years (as needed)
4. **Calculates match scores** using:
   - Date similarity (40% weight, exponential decay)
   - Age similarity (30% weight, linear decay)
   - Gender match (20% weight, binary)
   - Demographics (10% weight, race/ethnicity)
5. **Assigns confidence categories** with penalties for relaxed matches

## Output Files

After running the matching algorithm:

```
../output/intermediate/
├── encounter_mapping_matched.parquet  # Linkage between patients and donors
├── table_one_donor_comparison.csv    # Donor characteristics comparison
├── srtr_clif_matches_all.csv        # All match pairs with scores
└── srtr_clif_matches_donor_summary.csv # Summary per donor
```

## Dependencies

```bash
pip install pandas polars numpy streamlit plotly
```

See `requirements_app.txt` for complete list.


## Configuration

The matching algorithm uses these initial parameters (automatically relaxed for unmatched donors):
- Date tolerance: ±7 days
- Age tolerance: ±2 years
- Date blocking window: ±365 days


## Technical Documentation

For detailed algorithm mathematics and implementation details, see [ALGORITHM_DETAILS.md](ALGORITHM_DETAILS.md).
