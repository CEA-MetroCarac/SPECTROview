# Purpose

This file contains representative plotting examples that demonstrate how the AI Agent responds to common plot requests. These examples serve as reference for prompt engineering and LLM few-shot context.

**IMPORTANT:** You must call the provided tools (e.g., `plot_graph`, `update_graph`, `delete_graph`) to perform actions. Do NOT output JSON text — use the tool-calling mechanism of your API. Each example below shows which tool to call and which arguments to pass. Options like `grid`, `hist_bins`, `color_palette`, etc. are top-level arguments — use `other_properties` only for options without a dedicated parameter (see Example 14).

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
- `hist_bins` = `30`
- `df_name` = `"fit_results"`

---

## Example 5: Trendline with Polynomial Fit

**User:** Create a second-order trendline of center_Si vs temperature.

**Call tool:** `plot_graph` with:
- `x` = `"temperature"`
- `y` = `"center_Si"`
- `plot_style` = `"trendline"`
- `trendline_order` = `2`
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
- `ymin` = `3.5`
- `ymax` = `4.2`

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
- `grid` = `true`
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
- `color_palette` = `"viridis"`

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

---

## Example 14: Property Without a Dedicated Argument

**User:** Plot a histogram of FWHM_Si and set the figure DPI to 300.

**Call tool:** `plot_graph` with:
- `x` = `"FWHM_Si"`
- `plot_style` = `"histogram"`
- `other_properties` = `{"dpi": 300}`
- `df_name` = `"fit_results"`

(`dpi` has no dedicated parameter, so it goes inside `other_properties`. Compare to Example 4, where `hist_bins` *does* have a dedicated parameter and is passed directly.)

---

## Example 15: Group an Existing Graph by a Categorical Column

**User:** Plot 1: group the data by Zone, exclude slots 10, 11 and 12, increase the axis label font size, move the legend outside the plot, and rotate the x tick labels by 30°.

Graph 1 is currently `x="Slot"`, `y="fwhm_Si"`, `plot_style="point"`.

**Call tool:** `update_graph` with:
- `graph_id` = `"1"`
- `z` = `"Zone"`
- `filters` = `["Slot != 10", "Slot != 11", "Slot != 12"]`
- `other_properties` = `{"axis_label_fontsize": 14, "legend_outside": true, "x_rot": 30}`

Note what is **absent**: no `x`, no `y`, no `plot_style`. "Group by Zone" is the hue, so it is `z`; the X axis stays `"Slot"` because the user never asked to change it. Passing `x="Zone"` here would be wrong — it would replace the slot-by-slot view the user is looking at.

---

## Example 16: Group a New Plot by a Categorical Column

**User:** Point plot of Strain vs Slot, one colour per Zone.

**Call tool:** `plot_graph` with:
- `x` = `"Slot"`
- `y` = `"Strain (GPa)"`
- `z` = `"Zone"`
- `plot_style` = `"point"`
- `df_name` = `"fit_results"`

"one colour per Zone" / "grouped by Zone" / "split by Zone" all mean the same thing: `z="Zone"`, with `x` unchanged.
