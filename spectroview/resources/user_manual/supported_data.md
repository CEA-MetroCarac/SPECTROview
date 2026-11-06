## **Supported Data Formats**

Examples of all supported data types can be found in the [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) folder of the GitHub repository. Users can download these files to familiarize themselves with supported file formats, data structures, and to practice using **SPECTROview** with real-world data.

| Format | Extension | Loader | Data Type |
|---|---|---|---|
| Delimited text | `.txt`, `.csv` | Auto-detected delimiter & header | Single spectra, 2D maps, wafer & time-resolved maps |
| Renishaw WiRE | `.wdf` | `renishawWiRE` library | Single spectra, temporal series, line profiles, and 2D maps |
| SPC | `.spc` | Built-in binary reader | Single spectra and 2D maps |
| TRPL | `.dat` | Built-in parser | Time-resolved photoluminescence decays |
| Tabular data | `.xlsx`, `.xls`, `.csv` | `Pandas` & `Openpyxl` | DataFrames for plotting |
| Fit model template | `.json` | Fit Model Builder | Peak models & baseline configuration |

### **1. `.wdf` (Renishaw) and `.spc` (HORIBA) Formats**

Since version 26.7.1, **SPECTROview** natively supports:

- **`.wdf`** — Data recorded using WiRE software on Renishaw instruments (discrete spectra, temporal/repeatability series, 1D line-scan profiles, and 2D hyperspectral maps).
- **`.spc`** — Data recorded using LabSpec 6 / T64000 software on HORIBA instruments (discrete Raman/PL spectra and 2D hyperspectral maps).

Both discrete spectra and hyperspectral datasets are routed automatically when opened or dragged into the main window without prior conversion. **SPECTROview** also parses and imports instrument **metadata** (excitation wavelength, grating, objective lens, laser power, slit/hole size, spectral units, spatial step sizes, and acquisition timestamps).

### **2. Spectroscopic Data (`.txt`, `.csv`)**

**SPECTROview** supports 1D spectroscopic data in TXT or CSV formats. Files must consist of **two numeric data columns** separated by semicolons (`;`), commas (`,`), tabs, or spaces. Leading comment lines (`#`) and column headers are optional and automatically detected without losing the first data point on headerless files:

- **Column 1**: X-axis values (e.g., Raman shift in cm⁻¹, wavelength in nm, 2θ in degrees). Descending X-axes are automatically sorted in ascending order.
- **Column 2**: Corresponding intensity values (a.u.).

### **3. Hyperspectral Data (`.txt`, `.csv`)**

**SPECTROview** supports hyperspectral data (2D stage maps, wafer maps, and multi-scan time-resolved wafer maps) in TXT or CSV formats, structured as follows:

| X    | Y    | R1 | R2 | R3 | ... | Rn |
|------|------|----|----|----|-----|----| 
| **X1** | **Y1** | i1 | i1 | i1 | ... | i1 |
| **X2** | **Y2** | i2 | i2 | i2 | ... | i2 |
| **...** | **...** | ... | ... | ... | ... | ... |
| **Xn** | **Yn** | in | in | in | ... | in |

- The **first row** (R1 → Rn) defines the X-axis values (e.g., Raman shift in cm⁻¹) shared across all spectra.
- The **first two columns** define the X and Y spatial coordinates of each spectrum within the map.
- The **remaining columns** contain the corresponding intensity values for each spectrum (I1 → In), extending from the second row to the last.

> **Note**: Text-exported 2D map formats from Renishaw tools often use a different structure and must be converted before they can be loaded into **SPECTROview**. An integrated conversion tool is provided within the application for this exact purpose (see the `Hyperspectral Data Converter Tool` section).

### **4. Datasheet (Excel or CSV)**

Excel files (`.xlsx`, `.xls`) containing one or multiple sheets, as well as CSV files, can be directly imported into the **`Graphs`** workspace for immediate visualization and plotting.

### **5. Native **SPECTROview** Formats**

To ensure workflows can be saved and resumed later, **SPECTROview** uses specific file extensions corresponding to the active workspace:

- **`Maps` Workspace** → `.maps`
- **`Spectra` Workspace** → `.spectra`
- **`Graphs` Workspace** → `.graphs`
- **Fit Model Templates** → `.json` (drag & drop or Load into `Spectra` and `Maps` workspaces)
