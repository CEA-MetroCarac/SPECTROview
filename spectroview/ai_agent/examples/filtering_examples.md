# Purpose

This file contains representative data filtering and statistics examples for the SPECTROview AI Agent. These illustrate how to translate natural-language conditions into pandas `.query()` expressions.

---

# Filtering Examples

## Example 1: Simple Numeric Condition

**User:** Show rows where FWHM_Si is greater than 5.

**AI Response:**
```json
{
  "action": "filter",
  "explanation": "Filtering rows where FWHM_Si is greater than 5.",
  "target_dataframe": "fit_results",
  "query": "FWHM_Si > 5"
}
```

---

## Example 2: Compound Condition

**User:** Find samples where R_squared > 0.95 and FWHM_Si < 4.

**AI Response:**
```json
{
  "action": "filter",
  "explanation": "Filtering rows where fit quality R_squared > 0.95 and FWHM_Si < 4.",
  "target_dataframe": "fit_results",
  "query": "R_squared > 0.95 and FWHM_Si < 4"
}
```

---

## Example 3: Categorical Filter

**User:** Show data from Zone == 'center'.

**AI Response:**
```json
{
  "action": "filter",
  "explanation": "Filtering rows where Zone equals 'center'.",
  "target_dataframe": "fit_results",
  "query": "Zone == 'center'"
}
```

---

## Example 4: Multiple Category Filter

**User:** Show rows where Slot is 5, 6, or 8.

**AI Response:**
```json
{
  "action": "filter",
  "explanation": "Filtering rows where Slot is one of 5, 6, or 8.",
  "target_dataframe": "fit_results",
  "query": "Slot in [5, 6, 8]"
}
```

---

## Example 5: Outlier Detection

**User:** Show outliers where peak intensity is more than 2 standard deviations above the mean.

**AI Response:**
```json
{
  "action": "answer",
  "explanation": "The filter action uses pandas .query() expressions which do not support inline statistical calculations like mean() and std(). Here is how to identify outliers:",
  "answer_text": "The AI Agent's filter action uses pandas .query() expressions and cannot compute mean/std inline. To find outliers, I recommend:\n\n1. First get statistics: ask 'Give me statistics for peak_intensity'\n2. Then filter using the computed threshold: 'Show rows where peak_intensity > <mean + 2*std>'\n\nFor example, if mean = 1000 and std = 150, ask: 'Show rows where peak_intensity > 1300'"
}
```

---

## Example 6: Combined Spatial and Quality Filter

**User:** Show rows from Zone 'edge' where R_squared > 0.98.

**AI Response:**
```json
{
  "action": "filter",
  "explanation": "Filtering edge zone rows with high fit quality (R_squared > 0.98).",
  "target_dataframe": "fit_results",
  "query": "Zone == 'edge' and R_squared > 0.98"
}
```

---

# Statistics Examples

## Example 7: Single Column Statistics

**User:** Give me statistics for the FWHM_Si column.

**AI Response:**
```json
{
  "action": "statistics",
  "explanation": "Computing descriptive statistics for FWHM_Si.",
  "target_dataframe": "fit_results",
  "stat_columns": ["FWHM_Si"]
}
```

---

## Example 8: Multiple Column Statistics

**User:** What are the statistics for peak center and FWHM?

**AI Response:**
```json
{
  "action": "statistics",
  "explanation": "Computing descriptive statistics for center_Si and FWHM_Si.",
  "target_dataframe": "fit_results",
  "stat_columns": ["center_Si", "FWHM_Si"]
}
```

---

## Example 9: General Data Question

**User:** What columns are available in my data?

**AI Response:**
```json
{
  "action": "answer",
  "explanation": "Listing the available columns in the active DataFrame.",
  "answer_text": "The DataFrame 'fit_results' contains the following columns:\n\n| Column | Type | Sample Values |\n|--------|------|---------------|\n| Slot | int64 | [1, 2, 3] |\n| Zone | object | ['center', 'edge'] |\n| center_Si | float64 | [519.2, 519.8, 518.9] |\n| FWHM_Si | float64 | [3.1, 3.4, 2.9] |\n| R_squared | float64 | [0.998, 0.997, 0.999] |"
}
```
