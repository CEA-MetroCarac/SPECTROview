# Purpose

This file defines plotting-specific behavioral rules for the SPECTROview AI Agent. These rules apply whenever calling `plot_graph`, `update_graph`, or `delete_graph`.

---

# Instructions

These rules govern how the AI agent creates, modifies, and deletes graphs.

---

# Rules

## Plot Creation

- **Do not add grid lines** to plots unless the user explicitly requests them. The default is `grid: false`.
- **Do not set a plot title** unless the user provides one. The application generates titles automatically. Leave `plot_title: null`.
- **Do not set custom axis labels** (`xlabel`, `ylabel`, `zlabel`) unless the user explicitly provides them. Leave them as `null`.
- **Do not set axis limits** (`xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`) unless the user explicitly provides numeric values.
- **Use descriptive column names** as-is from the DataFrame for axis mapping. Do not rename or abbreviate.
- **Default color palette** is `"jet"`. Only change it if the user requests a specific palette.
- **Filters**: When providing pandas query strings in `filters`, **always enclose string values in quotes** (e.g., `["Zone != 'Edge'"]`). Smaller models often omit quotes which causes evaluation errors.

## Plot Styles

- Use EXACTLY these strings: `point`, `scatter`, `box`, `bar`, `line`, `trendline`, `histogram`, `wafer`, `2Dmap`.
- For comma-separated multi-style requests with identical parameters, use a single `plot_graph` call with `plot_style: "box, scatter"`.
- For spatial plots (`wafer`, `2Dmap`) with multiple distinct groupings, always generate **separate entries** — one per group with a specific filter.

## Conversation Continuity

- When the user says "add also" or "do the same but", **reuse all arguments** from the previous turn (x, y, z, filters, df_name, axis limits). Change only what was explicitly requested.
- **Do not repeat `plot_graph` calls** for plots already created in a previous turn.
- When a user says "add a filter" to an existing graph, **include the existing filters** from the open graph summary plus the new filter in the `update_graph` call's `filters` argument.

## Spatial Plots

- `wafer` style: requires `x` (die X coordinate), `y` (die Y coordinate), `z` (measured value column).
- `2Dmap` style: requires `x` (column), `y` (row), `z` (heatmap value column).
- For multiple spatial items (e.g., multiple slots or wafers), generate one entry per item with an appropriate filter.

---

# Constraints

- Never use `seaborn` in any generated code or reference.
- Never add decorative elements (extra annotations, watermarks, legends) that the user did not request.
- Never generate a plot for data that does not exist in the loaded DataFrames.
