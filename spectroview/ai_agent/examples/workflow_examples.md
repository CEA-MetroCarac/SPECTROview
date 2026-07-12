# Purpose

This file contains multi-step workflow examples for the SPECTROview AI Agent. These demonstrate how the agent maintains context across multiple conversation turns.

---

# Multi-Turn Workflow Examples

## Workflow 1: Filter → Analyse → Visualise

This workflow demonstrates progressive data exploration across three conversation turns.

---

**Turn 1 — Initial Question:**

> What does my data look like?

**AI Response:**
```json
{
  "action": "statistics",
  "explanation": "Computing descriptive statistics for all numeric columns to give an overview.",
  "target_dataframe": "fit_results",
  "stat_columns": ["center_Si", "FWHM_Si", "R_squared", "Slot"]
}
```

---

**Turn 2 — Follow-up Filter:**

> Show me only the poor fits (R_squared < 0.95).

**AI Response:**
```json
{
  "action": "filter",
  "explanation": "Filtering rows with poor fit quality where R_squared < 0.95.",
  "target_dataframe": "fit_results",
  "query": "R_squared < 0.95"
}
```

---

**Turn 3 — Visualise Results:**

> Now make a wafer map of those filtered samples.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a wafer map of R_squared values for spectra with poor fit quality (R_squared < 0.95).",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "X",
      "y": "Y",
      "z": "R_squared",
      "filters": ["R_squared < 0.95"],
      "plot_style": "wafer",
      "color_palette": "jet"
    }
  ]
}
```

*Note: The agent remembered the filter from Turn 2 and applied it to the wafer map.*

---

## Workflow 2: Plot → Add Style → Modify

---

**Turn 1 — Create Initial Plot:**

> Plot a box plot of center_Si by Zone.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a box plot of center_Si grouped by Zone.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "Zone",
      "y": "center_Si",
      "z": null,
      "plot_style": "box"
    }
  ]
}
```
*(Application creates Graph ID 1)*

---

**Turn 2 — Add Style:**

> Also show a scatter plot of the same data.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Adding a scatter plot of center_Si by Zone using the same parameters.",
  "target_dataframe": "fit_results",
  "plot_config": [
    {
      "x": "Zone",
      "y": "center_Si",
      "z": null,
      "plot_style": "scatter"
    }
  ]
}
```
*Note: The agent reuses x="Zone", y="center_Si" from Turn 1 without re-creating the box plot.*

---

**Turn 3 — Modify the Box Plot:**

> Change the Y-axis range of graph 1 to [519, 521].

**AI Response:**
```json
{
  "action": "update",
  "explanation": "Updating graph 1 to set the Y-axis range to [519, 521].",
  "graph_update": [
    {
      "graph_id": 1,
      "properties": {
        "ymin": 519,
        "ymax": 521
      }
    }
  ]
}
```

---

## Workflow 3: Comparative Analysis

---

**Turn 1:**

> Plot box plots comparing FWHM_Si between all Zone categories.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a box plot of FWHM_Si grouped by Zone.",
  "target_dataframe": "fit_results",
  "plot_config": [{"x": "Zone", "y": "FWHM_Si", "plot_style": "box"}]
}
```

---

**Turn 2:**

> Now do the same for center_Si.

**AI Response:**
```json
{
  "action": "plot",
  "explanation": "Creating a box plot of center_Si grouped by Zone, mirroring the previous plot structure.",
  "target_dataframe": "fit_results",
  "plot_config": [{"x": "Zone", "y": "center_Si", "plot_style": "box"}]
}
```

---

**Turn 3:**

> Delete all graphs and start fresh.

**AI Response:**
```json
{
  "action": "delete",
  "explanation": "Closing all open graphs to start fresh.",
  "graph_delete": {
    "delete_all": true,
    "graph_ids": []
  }
}
```
