# Timezone Error Fix for Timeline Visualization

## ✅ Error Fixed

Fixed the `TypeError: Cannot compare tz-naive and tz-aware timestamps` error that occurred when running the viewer with real data.

## 🐛 The Problem

The error occurred in the timeline visualization when trying to sort events by datetime:
```python
events_df = events_df.sort_values('datetime')  # Line 322
```

The issue: Some datetime values from the CLIF data had timezone information (tz-aware) while others didn't (tz-naive), and pandas cannot compare/sort mixed timezone datetimes.

## ✅ The Solution

Added a `standardize_datetime()` helper function that:
1. Converts all datetime values to pandas datetime objects
2. **Removes timezone information** from all datetimes (makes them tz-naive)
3. Handles None/NaT values gracefully
4. Returns None for invalid dates (which are filtered out)

```python
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
```

## 🔧 What Changed

Before:
```python
events.append({
    'event': 'Admission',
    'datetime': pd.to_datetime(patient['admission_dttm']),  # Could be tz-aware or tz-naive
    ...
})
```

After:
```python
dt = standardize_datetime(patient['admission_dttm'])  # Always tz-naive
if dt is not None:
    events.append({
        'event': 'Admission',
        'datetime': dt,
        ...
    })
```

## ✅ Result

- All datetime values are now consistently timezone-naive
- Timeline can sort events without timezone comparison errors
- Visualization works with both real data (potentially tz-aware) and test data (tz-naive)
- No data is lost - only timezone information is stripped for display purposes

## 🚀 Impact

The viewer now works correctly with real CLIF data that may have timezone-aware datetime values from the 01_record_linkage_dev.py processing.