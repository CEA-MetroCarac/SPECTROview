# Keyboard Shortcuts & Tips

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Rescale spectra plot |
| `Ctrl+Click` on Copy | Copy numerical data to clipboard |
| `Mouse wheel` | Quick Y-axis rescale in Spectra Viewer |

## Tips & Tricks

### General

- **Tooltips everywhere** — Hover over any GUI element for 1 second to see its function
- **Drag peaks** — Click and drag peaks directly in the Spectra Viewer to adjust position and height
- **Dark/Light mode** — Toggle via the toolbar button

### Fitting

- **Quick re-fit** — After adjusting parameters, click Fit again. The engine warm-starts from previous results
- **Fast preview** — Set `xtol`/`ftol` to `1e-2` in Settings for rapid previews during model building
- **Precision results** — Set `xtol`/`ftol` to `1e-6` for publication-quality fits

### Data Management

- **Multiple maps** — Load and switch between multiple maps; each map's spectra are managed independently
- **Profile extraction** — In Maps workspace, select exactly 2 points to extract an intensity profile
- **Data filtering** — Use pandas-style expressions in Graphs workspace to filter data

### Column Names

- **Special characters** — Use backticks (`` ` ``) around column names containing `()`, spaces, or hyphens
- **Case sensitive** — Column names are case-sensitive (`x0_p1` ≠ `X0_P1`)
