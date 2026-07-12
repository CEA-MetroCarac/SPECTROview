# Plotting Instructions — Simplified

## How to Create Plots

When `action` is `"plot"`, add entries to the `plot_config` list.

### Required fields ONLY:
- `x`: column name for X axis
- `y`: column name for Y axis (use `null` for histogram)
- `plot_style`: one of `point`, `scatter`, `box`, `bar`, `line`, `trendline`, `histogram`, `wafer`, `2Dmap`

### When to use `query` before `plot`:
If the user asks for "the slot with highest strain" or "the wafer with lowest FWHM", you MUST:
1. First respond with `action: "query"` and a pandas expression like `df.groupby('Slot')['Strain (GPa)'].mean().idxmax()`
2. Then use the returned numeric value in your `filters` like `"Slot == 2"`

### Wafer plots:
- `x`: die X coordinate column, `y`: die Y coordinate column, `z`: measured value column
- For multiple distinct slots/wafers, create separate entries each with their own filter.

### Column names:
- **COPY PASTE** column names exactly from the DATAFRAME section. Same spelling, same case, same spaces.
- If the column is `fwhm_Si`, use `fwhm_Si` — not `FWHM_Si` or `fwhm_si`.

### Multi-style (same axes):
For "box and scatter of X vs Y", use one entry: `"plot_style": "box, scatter"`

### NEVER invent column names.
If a column doesn't exist, use `action: "answer"` to explain.
