## Progressive Record Linkage Algorithm (SRTR–CLIF)

This algorithm links deceased organ donors from the SRTR registry to patient encounters in CLIF using a **progressive, tolerance-based record linkage strategy**. The objective is to identify the most plausible patient encounter corresponding to each donor while maintaining interpretability, clinical safety, and auditability.

---

### Overview of the Matching Strategy

For each donor \( d \), the algorithm searches for candidate patient encounters \( p \) such that:

- The donor recovery date is close to the patient death date
- Donor and patient ages are similar
- Core demographics (sex, race, ethnicity) are compatible

Rather than relying on a single fixed tolerance, the algorithm **progressively relaxes matching criteria** until a valid match is identified.  
Each donor is matched **greedily at the earliest (strictest) tolerance level** where at least one acceptable candidate exists.

---

## Blocking (Candidate Selection)

To reduce the search space, patients are first **blocked by date**:

\[
| \text{recovery\_date}_d - \text{death\_date}_p | \leq W
\]

where \( W \) is a fixed blocking window (default ±60 days).  
Only patients within this window are considered for detailed matching.

---

## Progressive Tolerance Levels

Matching is performed over a sequence of tolerance levels defined by:

- **Date tolerance** \( \Delta_d \) (days)
- **Age tolerance** \( \Delta_a \) (months)

Tolerance levels are explored in increasing order of:

\[
\Delta_d + \Delta_a
\]

This yields a progression such as:

\[
(0,0) \rightarrow (1,0), (0,1) \rightarrow (2,0), (1,1), (0,2) \rightarrow \dots
\]

Each tolerance level is labeled for interpretability:

| Label | Meaning |
|------|--------|
| **EXACT** | Exact date and age match |
| **STRICT** | Very small tolerances |
| **STANDARD** | Clinically reasonable variation |
| **RELAXED** | Broader but plausible |
| **EXPANDED** | Large tolerances, lower certainty |
| **MAXIMUM** | Last-resort matching |

Once a donor is matched at a given tolerance level, **no further relaxation is applied** for that donor.

---

## Continuous Match Score

For each donor–patient candidate pair, a **continuous score** \( S \in [0, 100] \) is computed.

### Date similarity (exponential decay)

\[
S_{\text{date}} = 100 \cdot e^{- \frac{| \Delta \text{date} |}{3}}
\]

This formulation strongly penalizes mismatches beyond a few days.

---

### Age similarity (linear decay)

\[
S_{\text{age}} = 100 \cdot \max\left(0, 1 - \frac{| \Delta \text{age (months)} |}{60} \right)
\]

Age differences greater than 5 years receive no credit.

---

### Demographics

- **Sex**: binary match (100 if matched, 0 otherwise)
- **Race / Ethnicity**:
  - Exact match → full credit
  - UNKNOWN → partial credit
  - Mismatch → reduced or zero credit

---

### Weighted score combination

\[
S =
0.40 \cdot S_{\text{date}} +
0.30 \cdot S_{\text{age}} +
0.20 \cdot S_{\text{sex}} +
0.10 \cdot S_{\text{demographics}}
\]

Weights reflect clinical relevance: **timing > age > sex > race/ethnicity**.

---

## Confidence Categories

In addition to the numeric score, each match is assigned a **categorical confidence** based on clinically motivated rules.

| Confidence | Interpretation |
|----------|----------------|
| **HIGH** | Very close date and age, sex match, compatible demographics |
| **MEDIUM** | Minor deviations, clinically plausible |
| **LOW** | Larger discrepancies or demographic mismatches |
| **REVIEW** | Weak match; requires manual inspection |

Confidence assignment uses **hard clinical thresholds**, not only the continuous score.

### Confidence caps by tolerance level

To avoid overstating certainty:

- **RELAXED** matches are capped at MEDIUM
- **EXPANDED / MAXIMUM** matches are capped at LOW or REVIEW

---

## Greedy Selection and Alternative Matches

At each tolerance level:

1. All valid matches are scored
2. Matches are ranked by:
   \[
   (\text{score}, \text{confidence}, -\text{tolerance})
   \]
3. The **top-ranked match** is selected
4. Remaining candidates are retained as **alternative matches** (if requested)

Each match includes:
- `is_best`: whether it is the selected best match
- `n_alternatives`: number of other viable candidates

---

## Handling Multiple Encounters per Patient

If a donor maps to multiple encounters for the **same patient**, the algorithm retains the **last terminal encounter** (latest discharge time) to ensure alignment with end-of-life clinical data.


