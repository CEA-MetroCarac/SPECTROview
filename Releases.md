### **Version v0.2.9 (22/09/2024) – Stable**

- **Reduce 'Maps' saved file size:**
    - Implemented data compression using gzip.
    - Removed redundant x0 and y0 values from saved files. These can now be retrieved directly from the saved `self.map_df`using the `spectrum.fnam` (i.e, `map_name`& `coord`).
    - Updated the `spectrum_to_dict` and `dict_to_spectrum` methods with an `is_map=False` argument to support this change in the Spectra tab.
- **Features related to 2Dmap processing:**
    - Plot heatmap from `Area`, `Intensity` or fitted parameters directly in Maps TAB.
    - `Auto scale` feature to filter out the outliers values of the heatmap.
    - Adjust `x-rang` and the heatmap color range via QRangSliders
    - Plot a line profile (height profile) directly in the heatmap plot whenever 2 points are selected.
    - Extract data of the line profile, send and plot it directly in Visualization TAB.
    - Copy the heatmap canvas in Clipboard. 
<p align="center">
    <img width=300 src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/doc\figures_release_notes\heatmap_widget.png">
</p>

- **Calibration of x-range for Spectrums TAB:**
    - Added new attributes `spectrum.is_corrected` and `spectrum.correction_value` for to track the calibration state of the `spectrum` object.
- **Optimization:**
    - Disable of “translation” of the labels of fit results to improuve the visibility of `df_fit_resutlts` dataframe. → Only the defaut prefix (m01) by `peak_labels` defined by user.
    - Refactored the `save_df_to_excel` method for improved performance and maintainability. Fill color for Excel tables with different colors to increase the visibility.
- **Additional Features:**
    - Fill DataFrame tables by different colors to increase the visibility.
    - Simplified df before displaying in GUI (for map_df)
    - Disabled color filling when viewing the “map_df” to prevent performance issues (lag).
    - Implemented  keyboard shortcuts for switching between tabs (Hold Ctrl + 1, 2, or 3).
- **Bug Fixes:**
    - Fixed an error related to the new attributes when saving **old_saved_work** files.