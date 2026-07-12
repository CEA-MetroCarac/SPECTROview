# Purpose

This is a reusable Markdown template for generating SPECTROview data analysis reports.

---

# SPECTROview Analysis Report

**Date:** YYYY-MM-DD
**Author:** [Name]
**Dataset:** [DataFrame Name]
**SPECTROview Version:** [Version]

---

## 1. Summary

Provide a brief 2–3 sentence overview of the analysis: what was measured, what dataset was used, and the key finding.

> **Example:** Raman spectroscopic measurements were performed on 150 silicon samples across 6 processing slots. Peak fitting results show a systematic shift in the Si peak centre correlated with Zone, indicating process-induced mechanical stress.

---

## 2. Dataset Overview

| Property | Value |
|----------|-------|
| **DataFrame name** | `fit_results` |
| **Rows** | N |
| **Columns** | M |
| **Key columns** | `center_Si`, `FWHM_Si`, `R_squared`, `Slot`, `Zone` |
| **Fit quality (mean R²)** | 0.XXX |

---

## 3. Descriptive Statistics

Insert the output of the `statistics` action here.

| Column | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
| center_Si | | | | | | | |
| FWHM_Si | | | | | | | |
| R_squared | | | | | | | |

---

## 4. Key Findings

### 4.1 [Finding Title]

Describe the finding with supporting data.

- **Observation:** ...
- **Evidence:** ...
- **Possible cause:** ...

### 4.2 [Finding Title]

Describe another finding.

---

## 5. Visualisations

List the graphs created during this analysis.

| Graph | Type | X | Y | Filter |
|-------|------|---|---|--------|
| 1 | Wafer map | X | center_Si | — |
| 2 | Box plot | Zone | FWHM_Si | — |
| 3 | Histogram | FWHM_Si | — | R_squared > 0.95 |

---

## 6. Conclusions

Summarise the main conclusions in bullet points.

- ...
- ...
- ...

---

## 7. Recommendations

List recommended next steps or further analyses.

1. ...
2. ...
3. ...

---

## 8. Notes

Any additional notes, caveats, or data quality observations.
