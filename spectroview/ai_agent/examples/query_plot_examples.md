# Query → Plot Workflow Examples

These examples show the TWO-STEP workflow required when the user asks for plots based on computed values like "highest", "lowest", "max", "min".

---

## Example 1: Plot slot with highest strain (wafer plot)

**User:** Plot a wafer plot of the slot that has the highest strain values.

**AI Response — Step 1 (query):**
```json
{
  "action": "query",
  "explanation": "Finding the slot with the highest mean Strain (GPa).",
  "target_dataframe": "data_inline_sheet1",
  "query": "df.groupby('Slot')['Strain (GPa)'].mean().idxmax()"
}
```

*(System returns: `2`)*

**AI Response — Step 2 (plot):**
```json
{
  "action": "plot",
  "explanation": "Creating wafer plot for Slot 2 (highest strain).",
  "target_dataframe": "data_inline_sheet1",
  "plot_config": [
    {
      "x": "X",
      "y": "Y",
      "z": "Strain (GPa)",
      "filters": ["Slot == 2"],
      "plot_style": "wafer",
      "color_palette": "jet"
    }
  ]
}
```

---

## Example 2: Plot slot with lowest FWHM (wafer plot)

**User:** Plot a wafer map of the slot with the lowest fwhm_Si.

**AI Response — Step 1 (query):**
```json
{
  "action": "query",
  "explanation": "Finding the slot with the lowest mean fwhm_Si.",
  "target_dataframe": "data_inline_sheet1",
  "query": "df.groupby('Slot')['fwhm_Si'].mean().idxmin()"
}
```

*(System returns: `11`)*

**AI Response — Step 2 (plot):**
```json
{
  "action": "plot",
  "explanation": "Creating wafer plot for Slot 11 (lowest fwhm_Si).",
  "target_dataframe": "data_inline_sheet1",
  "plot_config": [
    {
      "x": "X",
      "y": "Y",
      "z": "fwhm_Si",
      "filters": ["Slot == 11"],
      "plot_style": "wafer",
      "color_palette": "jet"
    }
  ]
}
```

---

## Example 3: Multiple plots in one request (query then plot)

**User:** Add a wafer plot of the slot with highest strain and another wafer plot of the slot with lowest fwhm_Si. Also add two point plots comparing fwhm_Si and strain across all slots.

**AI Response — Step 1 (query — get both values at once using a tuple):**
```json
{
  "action": "query",
  "explanation": "Finding the slot with highest mean Strain and the slot with lowest mean fwhm_Si.",
  "target_dataframe": "data_inline_sheet1",
  "query": "(df.groupby('Slot')['Strain (GPa)'].mean().idxmax(), df.groupby('Slot')['fwhm_Si'].mean().idxmin())"
}
```

*(System returns: `(2, 11)`)*

**AI Response — Step 2 (plot all 4 graphs):**
```json
{
  "action": "plot",
  "explanation": "Creating 2 wafer plots (Slot 2 for highest strain, Slot 11 for lowest fwhm_Si) and 2 point plots comparing across all slots.",
  "target_dataframe": "data_inline_sheet1",
  "plot_config": [
    {
      "x": "X",
      "y": "Y",
      "z": "Strain (GPa)",
      "filters": ["Slot == 2"],
      "plot_style": "wafer",
      "color_palette": "jet"
    },
    {
      "x": "X",
      "y": "Y",
      "z": "fwhm_Si",
      "filters": ["Slot == 11"],
      "plot_style": "wafer",
      "color_palette": "jet"
    },
    {
      "x": "Slot",
      "y": "fwhm_Si",
      "z": null,
      "plot_style": "point"
    },
    {
      "x": "Slot",
      "y": "Strain (GPa)",
      "z": null,
      "plot_style": "point"
    }
  ]
}
```

---

## Example 4: Simple plot (no query needed)

**User:** Create a scatter plot of Slot vs fwhm_Si.

**AI Response (direct plot — no query needed):**
```json
{
  "action": "plot",
  "explanation": "Creating scatter plot of Slot vs fwhm_Si.",
  "target_dataframe": "data_inline_sheet1",
  "plot_config": [
    {
      "x": "Slot",
      "y": "fwhm_Si",
      "z": null,
      "plot_style": "scatter"
    }
  ]
}
```
