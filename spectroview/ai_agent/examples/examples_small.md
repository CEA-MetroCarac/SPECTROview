# Tool-Call Examples

These show which tool to call and which arguments to pass. Always use the
real tool-calling mechanism — never write these out as JSON or Python text.

---

**User:** Plot FWHM_Si vs Slot, only slots 2, 6, 8, 10.

**Call tool:** `plot_graph` with: `x="Slot"`, `y="FWHM_Si"`, `plot_style="point"`, `filters=["Slot in [2, 6, 8, 10]"]`

---

**User:** Box plot of x0_Si vs Slot, hue by Zone, exclude Zone == Edge, add grid.

**Call tool:** `plot_graph` with: `x="Slot"`, `y="x0_Si"`, `z="Zone"`, `plot_style="box"`, `filters=["Zone != 'Edge'"]`, `grid=true`

(Note: string values in `filters` are quoted — `'Edge'`, not `Edge`. `grid` is a top-level argument, not nested inside anything.)

---

**User:** Plot 1: group the data by Zone, exclude slots 10 and 11, rotate the x tick labels by 30°.

(Graph 1 is currently `x="Slot"`, `y="fwhm_Si"`, point plot.)

**Call tool:** `update_graph` with: `graph_id="1"`, `z="Zone"`, `filters=["Slot != 10", "Slot != 11"]`, `other_properties={"x_rot": 30}`

(Note: "group by Zone" is `z`, NOT `x`. No `x`, `y` or `plot_style` is sent — the user did not ask to change them, and omitted arguments keep their current value.)

---

**User:** Create a box and scatter plot of Strain vs Slot.

**Call tool:** `plot_graph` with: `x="Slot"`, `y="Strain (GPa)"`, `plot_style="box, scatter"` (comma-separated styles in one call create both graphs)

---

**User:** Which slot has the highest average Strain? Plot just that slot as a wafer map of Strain.

**Step 1 — Call tool:** `query_dataframe` with: `query="df.groupby('Slot')['Strain (GPa)'].mean().idxmax()"`
**Step 2 — read the result** (e.g. it returns `6`), then **call tool:** `plot_graph` with: `x="X"`, `y="Y"`, `z="Strain (GPa)"`, `plot_style="wafer"`, `filters=["Slot == 6"]`

---

**User:** Update graph 2 to use the viridis palette and a title "Strain Map".

**Call tool:** `update_graph` with: `graph_id="2"`, `color_palette="viridis"`, `plot_title="Strain Map"`

---

**User:** Close graph 3.

**Call tool:** `delete_graph` with: `graph_ids=[3]`

---

**User:** Give me stats for FWHM_Si and x0_Si.

**Call tool:** `get_statistics` with: `columns=["FWHM_Si", "x0_Si"]`

---

**User:** Show rows where R_squared > 0.95.

**Call tool:** `query_dataframe` with: `query="R_squared > 0.95"`
