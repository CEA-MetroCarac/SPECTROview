# Purpose

This file contains representative plotting examples that demonstrate how the AI Agent responds to common plot requests. These examples serve as reference for prompt engineering and LLM few-shot context.

---

# Examples

## Example 1: Simple Scatter Plot

**User:** Create a scatter plot of Slot vs center_Si colored by Zone.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a scatter plot of Slot vs center_Si with Zone as the color grouping.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "Slot",
      "y": "center_Si",
      "z": "Zone",
      "plot_style": "scatter",
      "plot_title": null,
      "xlabel": null,
      "ylabel": null,
      "color_palette": "jet"
    }
  ]
}
```

---

## Example 2: Multi-Style Plot

**User:** Plot a box plot and a scatter plot of FWHM_Si vs Slot.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a box plot and scatter plot of FWHM_Si vs Slot using a single combined configuration.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "Slot",
      "y": "FWHM_Si",
      "z": null,
      "plot_style": "box, scatter",
      "plot_title": null,
      "xlabel": null,
      "ylabel": null
    }
  ]
}
```

---

## Example 3: Wafer Map

**User:** Show a wafer map of center_Si.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a wafer map of center_Si using X/Y die coordinates.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "X",
      "y": "Y",
      "z": "center_Si",
      "plot_style": "wafer",
      "plot_title": null,
      "color_palette": "jet"
    }
  ]
}
```

---

## Example 4: Histogram

**User:** Show a histogram of FWHM values with 30 bins.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a histogram of FWHM_Si with 30 bins.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "FWHM_Si",
      "y": null,
      "plot_style": "histogram",
      "hist_bins": 30,
      "hist_kde": false
    }
  ]
}
```

---

## Example 5: Trendline with Polynomial Fit

**User:** Create a second-order trendline of center_Si vs temperature.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a second-order polynomial trendline of center_Si vs temperature.",
  "target_dataframe": "measurements",
  "plot_config": [
    {
      "x": "temperature",
      "y": "center_Si",
      "z": null,
      "plot_style": "trendline",
      "trendline_order": 2
    }
  ]
}
```

---

## Example 6: Filtered Plot

**User:** Plot a box plot of FWHM_Si by Zone, only for Slot > 5.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a box plot of FWHM_Si grouped by Zone, filtered to Slot > 5.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "Zone",
      "y": "FWHM_Si",
      "z": null,
      "filters": ["Slot > 5"],
      "plot_style": "box"
    }
  ]
}
```

---

## Example 7: Update Graph Axis Limits

**User:** Set the Y-axis range of graph 3 to [3.5, 4.2].

**AI Response:**
```json
{
  "action": "update",
  "explanation": "Updating graph 3 to set the Y-axis range between 3.5 and 4.2.",
  "graph_update": [
    {
      "graph_id": 3,
      "properties": {
        "ymin": 3.5,
        "ymax": 4.2
      }
    }
  ]
}
```

---

## Example 8: Multiple Wafer Maps for Distinct Slots

**User:** Create wafer maps for slots 5, 6, and 8.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating separate wafer maps for slots 5, 6, and 8.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {"x": "X", "y": "Y", "z": "center_Si", "filters": ["Slot == 5"], "plot_style": "wafer"},
    {"x": "X", "y": "Y", "z": "center_Si", "filters": ["Slot == 6"], "plot_style": "wafer"},
    {"x": "X", "y": "Y", "z": "center_Si", "filters": ["Slot == 8"], "plot_style": "wafer"}
  ]
}
```
