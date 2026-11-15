## **Supported Data Formats**

Examples of all supported data types can be found in the [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) folder of the GitHub repository. Users can download these files to familiarize themselves with supported file formats, data structures, and to practice using **SPECTROview** with real-world data.

| Format | Extension | Loader | Data Type |
|---|---|---|---|
| Delimited text | `.txt`, `.csv` | Auto-detected delimiter & header | Single spectra, 2D maps, wafer & time-resolved maps |
| Renishaw WiRE | `.wdf` | Built-in binary reader | Single spectra, temporal series, line profiles, and 2D maps |
| HORIBA LabSpec 6 | `.l6s`, `.l6m` | Built-in binary reader | Single spectra (`.l6s`) and 2D hyperspectral maps (`.l6m`) |
| HORIBA LabSpec 5 | `.ngs`, `.ngc` | Built-in binary reader | Single spectra (`.ngs`) and compressed 2D hyperspectral maps (`.ngc`) |
| SPC | `.spc` | Built-in binary reader | Single spectra and 2D maps |
| TRPL | `.dat` | Built-in parser | Time-resolved photoluminescence decays |
| Tabular data | `.xlsx`, `.xls`, `.csv` | `Pandas` & `Openpyxl` | DataFrames for plotting |
| Fit model template | `.json` | Fit Model Builder | Peak models & baseline configuration |

### **1. `.wdf` (Renishaw), `.spc`, `.l6s` / `.l6m` (HORIBA LabSpec 6), and `.ngs` / `.ngc` (HORIBA LabSpec 5) Formats**

**SPECTROview** natively supports binary spectrometer formats without requiring external vendor software:

- **`.wdf`** — Data recorded using WiRE software on Renishaw instruments (discrete spectra, temporal/repeatability series, 1D line-scan profiles, and 2D hyperspectral maps).
- **`.l6s` and `.l6m`** — Native binary spectra (`.l6s`) and 2D spectral images/maps (`.l6m`), both uncompressed and compressed, recorded using HORIBA LabSpec 6 software.
- **`.ngs` and `.ngc`** — Native binary spectra (`.ngs`) and compressed 2D spectral images/maps (`.ngc`) recorded using HORIBA LabSpec 5 software.
- **`.spc`** — Data recorded using LabSpec 6 / T64000 / GRAMS software on HORIBA and Thermo instruments (discrete Raman/PL spectra and 2D hyperspectral maps).

Both discrete spectra and hyperspectral datasets are routed automatically when opened or dragged into the main window without prior conversion. **SPECTROview** also parses and imports instrument **metadata** (excitation wavelength, grating, objective lens, laser power, exposure time, accumulations, slit/hole size, spectral units, spatial step sizes, acquisition timestamps, and additional instrument parameters such as detector model, detector temperature, SWIFT mode, and S/N ratio).

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

> **Note**: Native Renishaw `.wdf` map files can be loaded directly into **SPECTROview** without conversion. Long-format text-exported 2D maps from Renishaw WiRE (`#X`, `#Y`, `#Wave`, `#Intensity` columns) can be converted into wide-matrix `.txt` format using `spectroview.api.io.convert_renishaw_map(input_path, output_path)`.

### **4. Datasheet (Excel or CSV)**

Excel files (`.xlsx`, `.xls`) containing one or multiple sheets, as well as CSV files, can be directly imported into the **`Graphs`** workspace for immediate visualization and plotting.

### **5. Native **SPECTROview** Formats**

To ensure workflows can be saved and resumed later, **SPECTROview** uses specific file extensions corresponding to the active workspace:

- **`Maps` Workspace** → `.maps`
- **`Spectra` Workspace** → `.spectra`
- **`Graphs` Workspace** → `.graphs`
- **Fit Model Templates** → `.json` (drag & drop or Load into `Spectra` and `Maps` workspaces)
