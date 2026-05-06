## 12. Keyboard Shortcuts & Tips

To maximize your efficiency within SPECTROview, a variety of keyboard shortcuts and power-user tips have been integrated directly into the application.

### Global Shortcuts

| Keyboard Shortcut | Resulting Action |
|----------|--------|
| `Ctrl + R` | Instantly rescales the axes of the current spectra plot to perfectly fit the visible data. |
| `Ctrl + Click` (on the Copy button) | Copies the raw numerical dataset of the current plot directly to your system clipboard (instead of an image). On macOS, use `Cmd + Click`. |
| `Mouse Wheel` | Quickly zooms and rescales the Y-axis interactively while hovering over the Spectra Viewer. |

### Advanced User Tips

- **Tooltips Everywhere**: If you are ever unsure what a specific button or parameter does, simply hover your mouse cursor over the GUI element for 1 second to reveal a descriptive tooltip.
- **Interactive Peak Dragging**: You do not need to manually type in initial peak parameters. You can directly click and drag the center point or the width of any peak inside the Spectra Viewer to adjust its initial guess dynamically.
- **Quick Re-fit (Warm Starting)**: If a fit does not converge perfectly, you can manually adjust the peak bounds and simply click the "Fit" button again. The engine will "warm-start" using the results of the previous optimization, making subsequent fits drastically faster.
- **Handling Complex Column Names**: If you are using the Data Filter or Computed Columns features and your target column name contains spaces or special characters, you must enclose the column name in backticks (`` ` ``) (e.g., `` `Laser Power` <= 5 ``).
- **Spatial Profile Extraction**: While inside the Maps workspace, if you select exactly 2 distinct spatial points on the MapViewer heatmap, SPECTROview will automatically extract and plot an interpolated intensity profile between those two coordinates.
