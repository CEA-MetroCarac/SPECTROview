# Purpose

This file contains representative data filtering and statistics examples for the SPECTROview AI Agent. These illustrate how to translate natural-language conditions into pandas `.query()` expressions.

**IMPORTANT:** You must call the provided tools (e.g., `query_dataframe`, `get_statistics`) to perform actions. Do NOT output JSON text — use the tool-calling mechanism of your API. Each example below shows which tool to call and which arguments to pass.

---

# Filtering Examples

## Example 1: Simple Numeric Condition

**User:** Show rows where FWHM_Si is greater than 5.

**Call tool:** `query_dataframe` with:
- `query` = `"FWHM_Si > 5"`
- `df_name` = `"fit_results"`

---

## Example 2: Compound Condition

**User:** Find samples where R_squared > 0.95 and FWHM_Si < 4.

**Call tool:** `query_dataframe` with:
- `query` = `"R_squared > 0.95 and FWHM_Si < 4"`
- `df_name` = `"fit_results"`

---

## Example 3: Categorical Filter

**User:** Show data from Zone == 'center'.

**Call tool:** `query_dataframe` with:
- `query` = `"Zone == 'center'"`
- `df_name` = `"fit_results"`

---

## Example 4: Multiple Category Filter

**User:** Show rows where Slot is 5, 6, or 8.

**Call tool:** `query_dataframe` with:
- `query` = `"Slot in [5, 6, 8]"`
- `df_name` = `"fit_results"`

---

## Example 5: Outlier Detection

**User:** Show outliers where peak intensity is more than 2 standard deviations above the mean.

**AI should respond with text answer** (no tool call needed):
The `query_dataframe` tool uses pandas `.query()` expressions and cannot compute mean/std inline. To find outliers, recommend the user:
1. First get statistics via `get_statistics` for `peak_intensity`
2. Then filter using the computed threshold

---

## Example 6: Combined Spatial and Quality Filter

**User:** Show rows from Zone 'edge' where R_squared > 0.98.

**Call tool:** `query_dataframe` with:
- `query` = `"Zone == 'edge' and R_squared > 0.98"`
- `df_name` = `"fit_results"`

---

# Statistics Examples

## Example 7: Single Column Statistics

**User:** Give me statistics for the FWHM_Si column.

**Call tool:** `get_statistics` with:
- `columns` = `["FWHM_Si"]`
- `df_name` = `"fit_results"`

---

## Example 8: Multiple Column Statistics

**User:** What are the statistics for peak center and FWHM?

**Call tool:** `get_statistics` with:
- `columns` = `["center_Si", "FWHM_Si"]`
- `df_name` = `"fit_results"`

---

## Example 9: General Data Question

**User:** What columns are available in my data?

**AI should respond with text answer** (no tool call needed):
List the columns from the DataFrame schema shown in the system context. For example:
"The DataFrame 'fit_results' contains columns: Slot (int64), Zone (object), center_Si (float64), FWHM_Si (float64), R_squared (float64)."
