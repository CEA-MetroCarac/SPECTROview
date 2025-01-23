
### **Version v0.3.2 (23/01/2024)
- **Bug fixes and Optimization:**
    - Complete the features peak position correct with Si-ref peak (520.7cm-1) .

### **Version v0.3.0 (27/09/2024)

- **New Features:**
    - x-range correction is now integrated in the Maps Tab.
    - **Features related to 2Dmap processing:**
        - Plot heatmap from `Area`, `Intensity` or fitted parameters directly in Maps TAB.
        - `Auto scale` feature to filter out the outliers values of the heatmap.
        - Adjust `x-rang` and the heatmap color range via QRangSliders
        - Plot a line profile (height profile) directly in the heatmap plot whenever 2 points are selected.
        - Extract data of the line profile, send them (data and plot) to Visualization TAB.
        - Copy the heatmap canvas in Clipboard. 

<p align="center">
    <img width=700 src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/doc\figures_release_notes\heatmap_widget.png">
</p>

- **Optimization:**
    - Visualization tab: User can now modify/update the openned Excel file while it is opened.
    - Visualization tab: Dataframes will be compressed (gzip) before saving : update `save/load()` method
    - Maps tab: Reorganize the button, checkbox, comboxes of 2Dmap plot widget into a QToolbutton.

- **Bug Fixes:**
    - Bug related to malfunction of the graph do not effectively deleted when a `sub_window` is closed.
    - Fix bug arised when user delete or rename the default folder contaning all `fit_models`. A pop up message showed up to tell user to redefine the default folder for locking for `fit_models`.
    - Fix bug related to the noise correction not automatically applied : set `spectrum.baseline.noise=5` by default.


### **Version v0.2.9 (22/09/2024)

- **Reduce 'Maps' saved file size:**
    - Implemented data compression using gzip.
    - Removed redundant x0 and y0 values from saved files. These can now be retrieved directly from the saved `self.map_df`using the `spectrum.fnam` (i.e, `map_name`& `coord`).
    - Updated the `spectrum_to_dict` and `dict_to_spectrum` methods with an `is_map=False` argument to support this change in the Spectra tab.
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

**Full Changelog**: https://github.com/CEA-MetroCarac/SPECTROview/commits/v0.2.9