# Supported Data Formats

Example files for all supported formats can be found in the [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) folder of the GitHub repository.

## .wdf (Renishaw) and .spc (HORIBA)

SPECTROview natively supports:

- **`.wdf`** — Recorded with WiRE software from Renishaw instruments
- **`.spc`** — Recorded with LabSpec 6 software from HORIBA instruments

Both discrete spectra and hyperspectral datasets (2D maps) are fully supported. These files can be opened directly without any prior conversion.

!!! info "Metadata"
    SPECTROview also reads associated metadata from `.wdf` and `.spc` files:
    excitation wavelength (nm), gratings (gr/mm), objective used, laser power, slit/hole, acquisition timestamp.

## Spectroscopic Data (.txt, .csv)

Two-column files separated by semicolons, spaces, or tabs:

| Column | Content |
|--------|---------|
| Column 1 | X-axis values (e.g., Raman shift in cm⁻¹) |
| Column 2 | Intensity values |

Files can include column headers or not. Multiple files can be loaded simultaneously.

## Hyperspectral Data (.txt, .csv)

For 2D maps or wafer maps:

| Column | Content |
|--------|---------|
| Column 1 | X position |
| Column 2 | Y position |
| Columns 3...N | Intensity at each wavelength/wavenumber |

The header row should contain wavenumber/wavelength values as column names.

## Datasheet (Excel or CSV)

Excel files (`.xlsx`, `.xls`) with one or multiple sheets, or CSV files can be loaded into the **Graphs Workspace** for visualization.

## TRPL Data (.dat)

Time-resolved photoluminescence data in `.dat` format.

## SPECTROview Workspace Files

| Workspace | Extension | Contents |
|-----------|-----------|----------|
| Maps | `.maps` | Map data, spectra, fit results, metadata |
| Spectra | `.spectra` | Spectra, fit models, baseline settings |
| Graphs | `.graphs` | Datasets and plot configurations |
