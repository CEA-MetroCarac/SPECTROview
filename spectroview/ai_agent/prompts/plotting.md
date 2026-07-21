# Purpose

This file provides plotting-specific instructions for the SPECTROview AI Agent. It governs how the agent calls `plot_graph`, handles multiple plot styles, and generates spatial visualisations.

---

# Instructions

## Plotting with `plot_graph`

When creating a plot, call the `plot_graph` tool. Each tool call corresponds to one graph window in the SPECTROview Graphs workspace.

### Required Fields

- `x` — column name for the X axis (string)
- `y` — column name for the primary Y axis (string)
- `plot_style` — one of: `point`, `scatter`, `box`, `bar`, `line`, `trendline`, `histogram`, `wafer`, `2Dmap` (schema-validated — an invalid value is rejected)

### Optional Fields — Leave Unset Unless Explicitly Requested

All of the following are **optional, top-level tool arguments** — do NOT nest them inside `other_properties`. Leave them unset unless the user explicitly asks to set them; the application generates optimal axis labels and titles automatically:

- `plot_title` — only set if user provides a title
- `xlabel`, `ylabel`, `zlabel` — only set if user provides custom axis labels
- `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax` — only set if user provides axis limits
- `grid` — default is `false`; only set to `true` if user asks for grid lines
- `color_palette` — default is `"jet"`; only change if user requests a specific palette
- `xlogscale`, `ylogscale` — default is `false`
- `scatter_size` — marker size, for `scatter`/`point` styles
- `hist_bins` — number of bins, for `histogram` style
- `trendline_order` — polynomial order, for `trendline` style

For any property without a dedicated argument above (e.g. `x_rot`, `plot_width`, `plot_height`, `dpi`, `hist_kde`, `colormap_norm`/`colormap_center` for wafer/2Dmap, `axis_breaks`, `inset_enabled` and the other `inset_*` fields), pass it inside the `other_properties` dict instead.

## Multi-Style Plots

If the user requests multiple plot styles with **identical** axis columns and parameters (e.g., "create a box and scatter plot of X vs Y"), you can pass a comma-separated string to the `plot_style` argument in a **single** `plot_graph` tool call:

`plot_style: "box, scatter"`

This is the preferred compact form. The application will expand it into separate graphs automatically.

## Spatial Plots

For `"wafer"` and `"2Dmap"` plots, you MUST correctly map spatial coordinates and values:
- `x` MUST be the X-coordinate column (e.g., `"X"`, `"x_coord"`).
- `y` MUST be the Y-coordinate column (e.g., `"Y"`, `"y_coord"`). Do NOT assign the metric value to `y`.
- `z` MUST be the metric value you want to visualize (e.g., `"Strain (GPa)"`, `"FWHM_Si"`).

If the user requests multiple **distinct items** for spatial plots (e.g., "plot wafer maps for slots 5, 6, and 8"), you MUST execute **multiple separate tool calls** to `plot_graph` — one per item with a specific filter — since spatial plots cannot overlay distinct groupings on the same axes:

- Tool Call 1: `filters: ["Slot == 5"]`, `plot_style: "wafer"`
- Tool Call 2: `filters: ["Slot == 6"]`, `plot_style: "wafer"`
- Tool Call 3: `filters: ["Slot == 8"]`, `plot_style: "wafer"`

Do NOT set `spines_visible` (or any spine/border styling) for a `wafer` plot. A wafer plot shows only the **left** spine by convention — the top/right/bottom borders are intentionally hidden and the application applies this automatically. Passing `spines_visible` yourself risks re-enabling all four borders and producing an incorrect wafer plot.

## Updating Existing Graphs

When the user wants to **modify** an existing graph (change axis limits, title, style, filters, color palette), call the `update_graph` tool. Do NOT use `plot_graph`.

- Set `graph_id` to the specific integer ID (as a string), or `"all"` to apply to all open graphs.
- Only include the properties the user explicitly wants to change.
- When **adding** a filter to an existing graph, preserve the existing filters by including them in the new filters list.

## Filters

When providing filters, you must supply a list of valid pandas query strings. 
**CRITICAL**: You MUST use quotes around string values inside your query strings. Failure to do so will cause the query to fail. For example, use `["Zone != 'Edge'"]` instead of `["Zone != Edge"]` and `["Type == 'Control'"]` instead of `["Type == Control"]`. Smaller models in particular must pay close attention to this.

## Deleting Graphs

When the user wants to close or delete graphs, call the `delete_graph` tool.

- Set `delete_all: true` to close all graphs.
- Use `graph_ids: [1, 2, 3]` to delete specific graphs.
- For "delete all except graph 5", include all open graph IDs except 5 in `graph_ids`.

---

# Constraints

- Use EXACTLY these plot style strings: `point`, `scatter`, `box`, `bar`, `line`, `trendline`, `histogram`, `wafer`, `2Dmap`
- NEVER add a `plot_title` unless the user explicitly requests one.
- Do NOT add grid lines unless explicitly requested.
- For spatial plots with multiple distinct items, always generate separate entries.
