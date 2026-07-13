# Purpose

This file contains representative plotting examples that demonstrate how the AI Agent responds to common plot requests. These examples serve as reference for prompt engineering and LLM few-shot context.

**IMPORTANT:** You must call the provided tools (e.g., `plot_graph`, `update_graph`, `delete_graph`) to perform actions. Do NOT output JSON text — use the tool-calling mechanism of your API. Each example below shows which tool to call and which arguments to pass.

---

# Examples

## Example 1: Simple Scatter Plot

**User:** Create a scatter plot of Slot vs center_Si colored by Zone.

**Call tool:** `plot_graph` with:
- `x` = `"Slot"`
- `y` = `"center_Si"`
- `z` = `"Zone"`
- `plot_style` = `"scatter"`
- `df_name` = `"fit_results"`

---

## Example 2: Multi-Style Plot

**User:** Plot a box plot and a scatter plot of FWHM_Si vs Slot.

**Call tool:** `plot_graph` with:
- `x` = `"Slot"`
- `y` = `"FWHM_Si"`
- `plot_style` = `"box, scatter"`
- `df_name` = `"fit_results"`

---

## Example 3: Wafer Map

**User:** Show a wafer map of center_Si.

**Call tool:** `plot_graph` with:
- `x` = `"X"`
- `y` = `"Y"`
- `z` = `"center_Si"`
- `plot_style` = `"wafer"`
- `df_name` = `"fit_results"`

---

## Example 4: Histogram

**User:** Show a histogram of FWHM values with 30 bins.

**Call tool:** `plot_graph` with:
- `x` = `"FWHM_Si"`
- `plot_style` = `"histogram"`
- `other_properties` = `{"hist_bins": 30}`
- `df_name` = `"fit_results"`

---

## Example 5: Trendline with Polynomial Fit

**User:** Create a second-order trendline of center_Si vs temperature.

**Call tool:** `plot_graph` with:
- `x` = `"temperature"`
- `y` = `"center_Si"`
- `plot_style` = `"trendline"`
- `other_properties` = `{"trendline_order": 2}`
- `df_name` = `"measurements"`

---

## Example 6: Filtered Plot

**User:** Plot a box plot of FWHM_Si by Zone, only for Slot > 5.

**Call tool:** `plot_graph` with:
- `x` = `"Zone"`
- `y` = `"FWHM_Si"`
- `filters` = `["Slot > 5"]`
- `plot_style` = `"box"`
- `df_name` = `"fit_results"`

---

## Example 7: Update Graph Axis Limits

**User:** Set the Y-axis range of graph 3 to [3.5, 4.2].

**Call tool:** `update_graph` with:
- `graph_id` = `"3"`
- `other_properties` = `{"ymin": 3.5, "ymax": 4.2}`

---

## Example 8: Multiple Wafer Maps for Distinct Slots

**User:** Create wafer maps for slots 5, 6, and 8.

For this request, make **three separate calls** to `plot_graph` — one per slot:

**Call 1** — `plot_graph` with:
- `x` = `"X"`
- `y` = `"Y"`
- `z` = `"center_Si"`
- `filters` = `["Slot == 5"]`
- `plot_style` = `"wafer"`
- `df_name` = `"fit_results"`

**Call 2** — `plot_graph` with:
- `x` = `"X"`
- `y` = `"Y"`
- `z` = `"center_Si"`
- `filters` = `["Slot == 6"]`
- `plot_style` = `"wafer"`
- `df_name` = `"fit_results"`

**Call 3** — `plot_graph` with:
- `x` = `"X"`
- `y` = `"Y"`
- `z` = `"center_Si"`
- `filters` = `["Slot == 8"]`
- `plot_style` = `"wafer"`
- `df_name` = `"fit_results"`

---

## Example 9: Point Plot with Grid and Filters

**User:** Create a point plot of fwhm_Si vs Slot (exclude slots 5, 6, 7, 10). Hue = Zone, add grid, no title.

**Call tool:** `plot_graph` with:
- `x` = `"Slot"`
- `y` = `"fwhm_Si"`
- `z` = `"Zone"`
- `filters` = `["Slot not in [5, 6, 7, 10]"]`
- `plot_style` = `"point"`
- `other_properties` = `{"grid": true}`
- `df_name` = `"fit_results"`

---

## Example 10: Box Plot with Multiple Filters

**User:** Box plot of x0_Si vs Slot (only slots 2, 6, 8, 10), hue = Quadrant, exclude Q4.

**Call tool:** `plot_graph` with:
- `x` = `"Slot"`
- `y` = `"x0_Si"`
- `z` = `"Quadrant"`
- `filters` = `["Slot in [2, 6, 8, 10]", "Quadrant != 'Q4'"]`
- `plot_style` = `"box"`
- `df_name` = `"fit_results"`

---

## Example 11: Update Graph Color Palette

**User:** Change graph 2 to use the viridis color palette.

**Call tool:** `update_graph` with:
- `graph_id` = `"2"`
- `other_properties` = `{"color_palette": "viridis"}`

---

## Example 12: Delete Specific Graphs

**User:** Delete graphs 1 and 3.

**Call tool:** `delete_graph` with:
- `graph_ids` = `[1, 3]`

---

## Example 13: Delete All Graphs

**User:** Close all open graphs.

**Call tool:** `delete_graph` with:
- `delete_all` = `true`
