# Changelog

*(Dynamically synchronized from [GitHub Releases](https://github.com/CEA-MetroCarac/SPECTROview/releases))*

---

## [v26.18.3](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v26.18.3) - 2026-04-26

##### 🐞 Bug Fixes
- Fixed a bug related to plotting multiple wafer maps when using automatically detected wafer slot numbers.


##### 🎨 UI & Usability Improvements
- Improved GUI responsiveness when selecting thousands of spectra.
- Introduce **full Dark Mode** : 

  - The `Spectra Viewer` and `Maps Viewer` widgets now also support Dark Mode, matching the overall application style when Dark Mode is enabled.
  
  - User can also manually set the `Spectra Viewer` and `Maps Viewer` widgets to **Light** or **Dark** mode, independent of the application theme.

![image](https://github.com/user-attachments/assets/69d49625-596a-47e9-a43e-f717aedb901c)


- Users can choose the display mode (**Dark** or **Light**) for figures copied to the clipboard, regardless of the current application theme.

![image](https://github.com/user-attachments/assets/1ee8dfef-9edd-41cc-822a-d2e03361c738)


- Moved the **Graph Axis Limits** section from from the right Panel into the `Customize Graph Dialog`.

![image](https://github.com/user-attachments/assets/ad72fe93-82cc-42af-849f-97799779bb56)




---

## [v26.17.1](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v26.17.1) - 2026-04-22


##### ✨ New Features

- Added a **vertical/horizontal slider** to shift spectra along the X or Y axis for improved visual clarity.

![image](https://github.com/user-attachments/assets/036a212c-692a-4d70-8bc7-d88e332e0c46)

- Added the ability to **duplicate an existing graph**, creating a new graph instance.

![image](https://github.com/user-attachments/assets/2a02bafe-f8db-458e-bd4a-b0511e7ad657)


- Added support for **time series (temporal data)** in Renishaw `.wdf` files.  
  Time series files now open in the `Spectra Workspace`, with timestamps appended to the spectra names.

![image](https://github.com/user-attachments/assets/ec85a041-ddb9-48d9-8d2f-01953263f4e6)



##### 🎨 Improvements
- Increased the **height** of the DataFrame list box for better usability.
- Removed some toast notifications for certain actions.
- Improved overall GUI ergonomics with minor interface adjustments.

##### 🐞 Bug Fixes
- Fixed an issue in the **DataFilter display** where multiple filter expressions could overlap in the ListWidget (Graphs workspace).
- Fixed a crash when applying an invalid filter expression to a DataFrame (Graphs workspace).
- Fixed an issue where **Stop Fitting** did not work on macOS.
- Fixed a `RuntimeWarning: divide by zero` when fitting spectra with the **Pseudo-Voigt** function.

---

## [v26.9.6](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v26.9.6) - 2026-02-21

##### SPECTROview Release Notes **Version `26.9.6`** 
(This release notes includes updates from `26.8.1` versions released previously via PyPi)

##### 🚀 New Features & Enhancements

* **Universal Drag & Drop:** Drag and drop **any** supported file types **anywhere** in the GUI to open them instantly (before user need to drag and drop files into the listbox).
* **Expanded File Format Support:** Added full support for spectroscopic file formats **.WDF** (from Wire-Renishaw software) and **.SPC** (from Labspec6-HORIBA software), including comprehensive metadata extraction, which is displayed within the `More` Tab).

![More tab](https://github.com/user-attachments/assets/360b2226-81a7-4f2d-ab88-acd26b903e0e)

* **Intensity Normalization:** Introduced a new tool to `normalize intensity` feature (via the Tab `Metadata`).
* **Graph Annotations:** Add vertical lines, horizontal lines, and text annotations to graphs. Annotations are fully draggable, and double-clicking text opens a Customization Panel for quick editing.
![Plot Annotation](https://github.com/user-attachments/assets/620e5dca-3674-45ee-804f-05ebb23f6c07)

* **Spectra Export:** Save spectral data directly to a `.txt` file(s) from within the Spectra Workspace.

##### 🎨 UI & Usability Improvements

* **Interactive Legends:** 
  * Double-click on any Legend Box (including inside the `Spectra Workspace`) to quickly open its settings or modify its label and color.
* **Auto-Rescale:** The `SpectraViewer` now automatically rescales to best fit the data after cropping.
* **UI Layout Updates:** 
  * Moved the `BestFit` option (to show or not the `peaks` and `bestfit curve`) outside of the toolbar for quicker access. It is now placed next to the "Show Legend" button:
![image](https://github.com/user-attachments/assets/ec87004b-3cec-42d3-8cab-97ca570880d1)

  * Minor visual tweaks, including a new icon for some buttons within `Spectra Workspace`.
* Add **Y-Axis Labeling:** options: "Intensity (a.u.)" or "Intensity (a.u./s)"

- Added **quick access to the GitHub repository** and a direct link to the latest release notes from the `menu bar:` :

![menubar quick acces](https://github.com/user-attachments/assets/a58ac709-4adc-4a85-ae86-34b236890832)

##### 🐛 Bug Fixes

* **Fano Peak Shape (Bug #2):** Fixed issues relating to parameter scaling and intensity representation. Corrections are now accurately reflected in the plot, the peak table, and the hover tooltip (`on_hover` method).
* **2D Map Alignment:** improve 2Dmap visualization and interaction.
* Fixed a bug to preserve `metadata` when copy and paste `fit-model ` from one spectrum to another.
* **Plot Update Logic:** 
  * Fixed a bug where clicking the "Update plot" button would snap a dragged legend box back to its original position.
  * Fixed an issue where changes to the Z-axis were not triggering property updates in the legend.
  
* **"Esc" Key Behavior:** Fixed an issue where pressing the "Esc" key caused the plot (within MDI subwindows) to hide unexpectedly.

##### ⚙️ Under the Hood

* **Fitspy Compatibility:** Updated backend to support version 2026.2 of `fitspy` package (update p`yproject.toml`).
* **Matplotlib** : support lastest `matplotlib` version 3.10.8.
* **Codebase Refactoring:** Cleaned up code architecture by refactoring `customized_widgets.py` and migrating `customize_legend_widget` into the `Customize_graph_dialog` module.
* **Testing:** Updated the `pytest` test suite to improve application stability and code coverage.


---

## [v26.6.1](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v26.6.1) - 2026-02-05

> **Note:** This Release notes includes all changes and new features from versions **v0.9.4** to **v0.9.8** and from **v26.1.1** to **v26.5.1** previously released on PyPI.

##### Versioning Update

Starting with this release, we are adopting a new versioning format: **`YY.W.Z`**
- `YY` - Year (last two digits)
- `W` - Week number of the year
- `Z` - Build number within that week

##### 🎉 What's New

##### Built-in User Manual

- Added a detailed [User Manual ](https://github.com/CEA-MetroCarac/SPECTROview/blob/main/spectroview/resources/SPECTROview_UserManual.pdf) (PDF format) built into the application, accessible via the Manual button in the top toolbar.

---

##### Graphs Workspace:

##### UI/UX Improvements
- Optimized GUI layout for the right sidebar:

![image](https://github.com/user-attachments/assets/f88adc62-4027-4847-88b3-79b135cd4863)

- Rearranged GUI elements in the bottom toolbar for better user experience:

![image](https://github.com/user-attachments/assets/43779c67-c46f-474f-be94-31a608cad2ff)

- Moved the **Copy** button from the bottom toolbar to individual plot MDI subwindows of each plot for better accesibility:

![image](https://github.com/user-attachments/assets/6d46967a-8b6e-4736-b82c-3c80b95fa06f)


##### New Features
- **Log-scale option** for both X and Y axes
- **Data loading support** from Excel (single or multiple sheets) and CSV files
- **Refresh button**  ![image](https://github.com/user-attachments/assets/1e10c10b-2234-4c9b-a611-4d60760fad23)  for `DataFrame widget`: synchronize/reflect changes in Excel or CSV files without reloading
![image](https://github.com/user-attachments/assets/47bf9288-b70e-4131-95f6-d0ecd16510e8)

- **Delete All button** ![image](https://github.com/user-attachments/assets/df59609d-2333-4de1-bb1d-3be3d253c2b7) (in the bottom toolbar) to remove all existing plots at once

![image](https://github.com/user-attachments/assets/cd722b43-40bf-4a7e-9f22-f6dae62a0297)

- **Multiple wafer plots**: automatically detect available slots in the loaded dataset and display them dynamically in the bottom of the right-side panel.

![image](https://github.com/user-attachments/assets/ee0e59ed-2ed6-44ca-857a-7924f16b3bcb)

- **Auto-selection for WaferPlot style**: when WaferPlot style is selected, X and Y comboboxes are automatically configured, users only need to select the Z parameter
- **Customized legend dialog**: click on the legend box to show the dialog to customize legend box properties (color, label, marker)
- **Autocomplete feature** for DataFrame filter widget: type quickly and avoid mistyping column headers:

![image](https://github.com/user-attachments/assets/8faba1a5-0ba6-4ade-af7f-a9c01a67aaee)


##### Bug Fixes
- Fixed several minor bugs related to mismatched colors in box and bar plots


---

##### Spectra and Maps Workspaces:

##### General Updates
- New GUI layout for Maps Workspace with improved user experience
![image](https://github.com/user-attachments/assets/e866ecd6-0465-4acd-905e-37455e05e2a2)


- **TRPL data support**
- New peak shapes: **Fano**, **Single Exponential**, and **Bi-Exponential Decay**
- **Stop button** added to progress bar to halt ongoing fitting processes when needed
![image](https://github.com/user-attachments/assets/f93711d7-67d2-447e-83c7-bfcb2c5871cd)

##### `SpectraViewer` Widget

**Customizable legend box of SpectraViewer**: click on the legend box to show the dialog to customize legend box properties (color, label, marker)
![image](https://github.com/user-attachments/assets/6b66ecd1-5963-4715-a7c3-d93fe9a40787)

**New Option for Menu dropdown Options:**
- **LineWidth**: Adjust the width of spectral lines
- **Ratio**: Adjust the ratio (`width` x `heigh`) of copied figure
- **Additional X-axis labels**: 
  - Binding Energy (eV)
  - Emission Energy (eV)
  - Frequency (Hz)
  - 2theta (°)
  - Time (ns)
- **Plot style option**: select between `line `or `dot `style for Raw spectrum plot:

<p align="center">
  ![image](https://github.com/user-attachments/assets/b13c1901-d6d4-4ffa-9b72-4d50d77bc53f)
</p>

##### `MapViewer` Widget

**Multiple MapViewers:**
- Additional MapViewers are now displayed as floating dialog windows. Users can open multiple MapViewers simultaneously to view and compare different parameters

![image](https://github.com/user-attachments/assets/d24f9b86-26ed-447b-bf44-c2dea1a7fd14)

**Mask Features:**
- Create masked 2D map plots by filtering based on desired conditions of plotted parameters or other parameters

<p align="center">
  ![image](https://github.com/user-attachments/assets/bd536e72-c822-4c02-bbac-42d9cfb102f4)
</p>

**Selection Modes:**
- Added **rectangle selection mode** for 2D map plots (in addition to single point and multiple points selection)

**MenuOptions Updates:**

- Moved **Extract Profile** and **Remove Outlier** features inside MapViewer MenuOptions
- Display **statistics values** directly on wafer plots via `Show stats` checkbox within MenuOptions (cf. figure below)

<p align="center">
  ![image](https://github.com/user-attachments/assets/4296b3e4-061f-41b6-a09a-f6f471dda7e9)
</p>

##### FitResults Tab

**New Capabilities:**
- Compute and add new columns to fit results table via mathematical operations


![image](https://github.com/user-attachments/assets/6b6a9a9c-4f08-43eb-8915-d4ec9e9edbbb)


- **CSV export support** for best fit parameters (previously Excel-only)



---

##### Performance Improvements

- Optimized GUI performance and responsiveness across all workspaces





---

## [v0.9.3](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.9.3) - 2025-10-27

Wafer Plot Enhancements:
- Added support to plot multiple wafer maps based on selected slot checkboxes.
- Automatically display the `slot number `on the wafer map when a `Slot` filter is active.
- Whenever the `wafer` plot style is selected, the `X` and `Y` comboxes are now automatically assigned.
- The `wafer` plot style (and corresponding `X` and `Y` columns) is automatically selected when plotting.

Dataframe Filter Improvements:
- Added ability to copy `filter expressions `directly from the` filter list.`
- The selected filter’s text is now automatically reflected in the QLineEdit for easier editing.

---

## [v0.9.0](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.9.0) - 2025-10-11

Optimized GUI performance and responsiveness:
- Spectrum now updates in real time when the user types new `peak_model` parameters in the `PeakTable`.
- Mouse interactions now simultaneously adjust the peak position (`x0`) and peak intensiy (`ampli`) when dragging over the plot.
- Introduced a `SettingsPanel` class (QDialog) to gather all general settings in one place.

Bug Fixes:
- Customized the `apply_model` method to retain `xcorration_value`, `color`, and `label` attributes of `Spectrum` object when applying a new model.
- Fixed a bug related to the legend colors for `box` and `bar` plots of Graphs Tab.

Code Cleanup and Refactoring:
- Removed the `CommonUtilities` class.
- Revised and refactored the `FitModelManager` class.

---

## [v0.8.7](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.8.7) - 2025-10-05

- Optimized GUI performance and responsiveness.
- Fixed minor bugs related to the "x range correction" features.
- Update new color palette.
- Code cleaning and refactoring.


---

## [v0.8.5](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.8.5) - 2025-09-08

- Optimized GUI performance and responsiveness.
- Fixed minor bugs.
- Code cleaning and refactoring.

---

## [v0.8.0](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.8.0) - 2025-08-27

- Improved "MapViewWidget" class.
- Optimized GUI performance and responsiveness.
- Fixed minor bugs.

---

## [v0.7.1](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.7.1) - 2025-07-04

- Added support for converting WIRE 2D map format to a format compatible with SpectroView and LabSpec6.
- Fixed minor bugs.
- Optimized GUI performance and responsiveness.






---

## [v0.7.0](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.7.0) - 2025-07-03

- Added 'peak_area' to the fitting results.
- Fixed minor bugs.
- Optimized the GUI.

---

## [v0.6.2](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.6.2) - 2025-06-27

- Fixed minor bugs.
- Improved GUI responsiveness and layout consistency.

---

## [v0.4.5](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.4.5) - 2025-02-10

##### **New Features:**

* **Customized Toolbar**: The toolbar has been enhanced with an additional button for the  `Normalization" feature and more view options, improving usability (Previously, these options appeared in a pop-up when right-clicking on the figure canvas.)

* **Normalization Feature**: Users can now `normalize` spectra either to the maximum intensity or to a user-specified X value via a QLineEdits in `toolbar`.

<p align="center"> ![image](https://github.com/user-attachments/assets/d2acd0e9-5489-4439-8f03-2e33b5530268) </p> 


* Complete the **X axis Correction Feature** by using Si-Reference peak (recording in the same experimental session with other mesures).
* `Copy` only `baseline` or `peak_models`  from the selected spectrum, then `Paste` to other spectra with oneclick.

<p align="center"> ![image](https://github.com/user-attachments/assets/bfaefcf4-3e9f-4ca0-872b-7f8a4082d1d6) </p> 


* Support `waferplot` directly in the `Maps` Tab (previously only in the `Vizualization` Tab) allowing user to directly view the fitted parameters after fitting.

<p align="center"> ![image](https://github.com/user-attachments/assets/925aaade-df30-4559-a5c6-ea0d60b6dfcf) </p> 



##### **Bug Fixes & Optimizations:**
* GUI adjustments for a more user-friendly experience.
* Adjust DPI of spectra canvas within Class `SpectraViewWidget`  (from 90 to 80).
* Fixed various bugs related to the spectra view plot with new customized toolbar.









---

## [v0.3.2](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.3.2) - 2025-01-23

**Version v0.3.2 (23/01/2024)

New features: peak position correction using Si reference peak (520.7 cm-1).



---

## [v0.3.0](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.3.0) - 2024-09-27

##### **Version v0.3.0 (27/09/2024)
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
    ![image](https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/doc/figures_release_notes/heatmap_widget.png)
</p>

- **Optimization:**
    - Visualization tab: User can now modify/update the openned Excel file while it is opened.
    - Visualization tab: Dataframes will be compressed (gzip) before saving : update `save/load()` method
    - Maps tab: Reorganize the button, checkbox, comboxes of 2Dmap plot widget into a QToolbutton.

- **Bug Fixes:**
    - Bug related to malfunction of the graph do not effectively deleted when a `sub_window` is closed.
    - Fix bug arised when user delete or rename the default folder contaning all `fit_models`. A pop up message showed up to tell user to redefine the default folder for locking for `fit_models`.
    - Fix bug related to the noise correction not automatically applied : set `spectrum.baseline.noise=5` by default.
    
**Full Changelog**: https://github.com/CEA-MetroCarac/SPECTROview/compare/v0.2.9...v0.3.0
___



---

## [v0.2.9](https://github.com/CEA-MetroCarac/SPECTROview/releases/tag/v0.2.9) - 2024-09-24

##### **Version v0.2.9 (22/09/2024)

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

---

