## **Supported Data Formats**

Examples of all supported data types can be found in the [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) folder of the GitHub repository. Users can download these files to familiarize themselves with supported file formats, data structures, and to practice using `SPECTROview` with real-world data.

| Format | Extension | Loader | Data Type |
|---|---|---|---|
| Delimited text | `.txt`, `.csv` | Auto-detected delimiter | Single spectra and maps |
| Renishaw WiRE | `.wdf` | `renishawWiRE` library | Single spectra and maps |
| SPC | `.spc` | Built-in binary reader | Single spectra and maps |
| TRPL | `.dat` | Built-in parser | Time-resolved decays |
| Tabular data | `.xlsx`, `.csv` | `Pandas` & `Openpyxl` | DataFrames for plotting |

### **1. `.wdf` (Renishaw) and `.spc` (HORIBA) Formats**

Since version 26.7.1, `SPECTROview` natively supports:

- **`.wdf`** — Data recorded using WiRE software on Renishaw instruments.
- **`.spc`** — Data recorded using LabSpec 6 software on HORIBA instruments.

Both discrete spectra and hyperspectral datasets (2D maps) are fully supported and can be opened directly without prior conversion. In addition to the raw spectroscopic data, `SPECTROview` automatically parses and imports associated **metadata**, such as excitation wavelength (nm), gratings (gr/mm), objective lens, laser power, slit/hole size, and acquisition timestamps. This provides comprehensive access to both measurement data and experimental conditions in a unified workflow.

### **2. Spectroscopic Data (`.txt`, `.csv`)**

`SPECTROview` supports 1D spectroscopic data in TXT or CSV formats. Files must consist of **two columns** separated by semicolons, spaces, or tabs. Column headers are optional but supported:

- **Column 1**: X-axis values (e.g., Raman shift in cm⁻¹, wavelength in nm).
- **Column 2**: Corresponding intensity values (a.u.).

### **3. Hyperspectral Data (`.txt`, `.csv`)**

`SPECTROview` supports hyperspectral data (2D maps or wafer maps) in TXT or CSV formats, provided the data is structured as follows:

| X    | Y    | R1 | R2 | R3 | ... | Rn |
|------|------|----|----|----|-----|----| 
| **X1** | **Y1** | i1 | i1 | i1 | ... | i1 |
| **X2** | **Y2** | i2 | i2 | i2 | ... | i2 |
| **...** | **...** | ... | ... | ... | ... | ... |
| **Xn** | **Yn** | in | in | in | ... | in |

- The **first row** (R1 → Rn) defines the X-axis values (e.g., Raman shift in cm⁻¹) shared across all spectra.
- The **first two columns** define the X and Y spatial coordinates of each spectrum within the map.
- The **remaining columns** contain the corresponding intensity values for each spectrum (I1 → In), extending from the second row to the last.

> **Note**: Text-exported 2D map formats from Renishaw tools often use a different structure and must be converted before they can be loaded into `SPECTROview`. An integrated conversion tool is provided within the application for this exact purpose (see the `Hyperspectral Data Converter Tool` section).

### **4. Datasheet (Excel or CSV)**

Excel files (`.xlsx`, `.xls`) containing one or multiple sheets, as well as CSV files, can be directly imported into the **`Graphs`** workspace for immediate visualization and plotting.

### **5. Native `SPECTROview` Formats**

To ensure workflows can be saved and resumed later, `SPECTROview` uses specific file extensions corresponding to the active workspace:

- **`Maps` Workspace** → `.maps`
- **`Spectra` Workspace** → `.spectra`
- **`Graphs` Workspace** → `.graphs`
