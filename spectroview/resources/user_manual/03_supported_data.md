## 3. Supported Data Formats

Examples of all supported data types can be found in the [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) folder within the GitHub repository. Users can download these files to understand the supported file formats, data structures, and practice using the SPECTROview application with real examples.

### 3.1 .wdf (Renishaw) and .spc (HORIBA) Formats

Since version 26.7.1, SPECTROview natively supports:
- **`.wdf`** — Recorded with WiRE software from Renishaw instruments
- **`.spc`** — Recorded with LabSpec 6 software from HORIBA instruments

Both discrete spectra and hyperspectral datasets (2D maps) are fully supported. These files can be opened directly without any prior conversion. In addition to spectroscopic data, SPECTROview also reads and imports the associated **metadata**, including: excitation wavelength (nm), gratings (gr/mm), objective used, laser power, slit/hole, acquisition timestamp, etc. This ensures full access to both measurement data and experimental conditions within a single workflow.

### 3.2 Spectroscopic Data (.txt, .csv)

SPECTROview supports spectroscopic data in TXT or CSV format. Files must consist of **two columns** separated by semicolons, spaces, or tabs. Files can contain column headers or not:
- **Column 1**: X-axis values (e.g., Raman shift in cm⁻¹, wavelength in nm)
- **Column 2**: Corresponding intensity values (a.u.)

### 3.3 Hyperspectral Data (.txt, .csv)

SPECTROview supports hyperspectral data (2D maps or wafer maps) in TXT or CSV format, with the data structure arranged as follows:

| | | R1 | R2 | R3 | ... | Rn |
|---|---|---|---|---|---|---|
| **X1** | **Y1** | I1 | I1 | I1 | ... | I1 |
| **X2** | **Y2** | I2 | I2 | I2 | ... | I2 |
| **...** | **...** | ... | ... | ... | ... | ... |
| **Xn** | **Yn** | In | In | In | ... | In |

- The **first row** (R1 → Rn) is the X-axis values (e.g., Raman shift in cm⁻¹) of all spectra.
- The **first two columns** list the X and Y coordinates of the spectrum within the map.
- The **remaining columns** contain the corresponding intensity values for each spectrum (I1 → In), from the second row to the last.

> **Note**: 2D map formats from Renishaw tools must be converted before they can be used in SPECTROview. An integrated conversion tool is provided in the SPECTROview application for this purpose (see Section 4, File Convert Tool).

### 3.4 Datasheet (Excel or CSV)

Excel files (`.xlsx`, `.xls`) containing one or multiple sheets, or CSV files can be directly loaded into the **Graphs Workspace** for visualization.

### 3.5 Formats saved by SPECTROview

Depending on the active workspace, SPECTROview saves files with specific extensions so work can be resumed later:
- **Maps workspace** → `.maps`
- **Spectra workspace** → `.spectra`
- **Graphs workspace** → `.graphs`
