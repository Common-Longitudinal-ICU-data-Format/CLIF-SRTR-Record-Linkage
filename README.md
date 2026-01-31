# CLIF-SRTR Record Linkage

## Objective

Fuzzy match CLIF  patient records with SRTR (Scientific Registry of Transplant Recipients) organ donor records using a deterministic hierarchical matching algorithm.

## Setup

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

### 3. Configure Your Site

```bash
cp config/config_template.json config/config.json
```

Edit with your paths:
```json
{
  "site_name": "YOUR_SITE",
  "tables_path": "/path/to/clif/tables",
  "file_type": "parquet",
  "SRTR_data_path": "/path/to/srtr/data",
  "timezone": "America/Chicago",
  "project_root": "/path/to/this/repo"
}
```

## Run

### 1. Generate Matches
```bash
cd record_linkage
uv run  01_record_linkage.py
```

### 2. Create Wide Dataset
```bash
uv run  02_create_wide_df.py
```

### 3. View Results
```bash
uv run streamlit run donor_patient_viewer.py
```

Access at: http://localhost:8503