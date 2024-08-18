# SPECTROview user manual
<p align="center">
    <img width=100 src="icon3.png">
</p>

<!-- TOC -->

- [Introduction](#introduction)
- [1. Supported data formats](#1-supported-data-and-file-formats)
  - [1.1 Spectroscopic data (spectrum)](#11-spectroscopic-data-spectrum)
  - [1.2 Hyperspectral data (2Dmaps or wafer data)](#12-hyperspectral-data-2D-maps-or-wafer-data)
  - [1.3 Datasheet (Excel files)](#13-Datasheet-Excel-files)
- [2. "Spectra" and "Maps" tabs](#2-spectra-and-maps-tabs)
- [3. "Data Visualization" TAB](#3-data-visualization-tab)
- [4. Data filtering feature](#4-data-filtering-feature)

<!-- /TOC -->

------------ 
# Introduction

The SPECTROview application is designed for the efficient processing of spectroscopic data and easy visualization of fitted data. The application features three main modules (tabs), each developed for a specific purpose:

<p align="center">
    <img width=300 src="manual_figures/fig0.png">
</p>

- **Spectra**: for processing one or multiples discrete spectra.
- **Maps**: For processing one or multiple hyperspectral datasets, including wafer data and 2D maps.
- **Data visualization** : This module is dedicated to plotting and visualizing data.

Each module will be described in detail in the following sections.

A tutorial video on how to use the SPECTROview application in under 15 minutes is available here: [link].
# 1. Supported data and file formats
All supported spectroscopic data (cf. section 1) can be open via the universal "Open" button that and be found in the toolbar. 
The selected data/files will be opened and the modules (tabs) will be display accordingly depending on the format/type files. 

## 1.1 Spectroscopic data (spectrum)

Example structure of data, figure
Supported format (CSV, txt)

## 1.2 Hyperspectral data (2D maps or wafer data)

Example structure of data
Supported format (CSV, txt)

## 1.3 Datasheet (Excel files)

Excel files contaning datasheets within one or multiples Excel sheets is supportted. 
User can directly load an Excel files for plotting via the "data visualization" Tab.

# 2. "Spectra" and "Maps" tabs
The 'Spectra' and 'Maps' tab sharing almost the same feature and GUI except some specific GUi element for 2Dmaps/hyperspectral data navigation.

<p align="center">
    <img width=900 src="manual_figures/fig1.png">
</p>

# 3. "Data Visualization" TAB



# 4. Data filtering feature

The **pandas.DataFrame.query()** method allows you to filter rows from a
DataFrame based on a boolean expression. It's a powerful and flexible way to
subset your DataFrame to include only the rows that meet specific conditions.

<p align="center">
    <img width=350 src="manual_figures/dfr_filter.png">
</p>

In **SPECTROview**, `query()` method is integrated and can be used via GUI by
typing as following: **(column_name) (operator) (value)**:

1. **(column_name)**: is the exact header of the column containing the
   data to be filtered. When the column header contain 'space', you
   need to enclose them in single or double quotes (see example below).

2. **(operator)**: it could be comparison operators (
   e.g., `==`, `<`, `>`, `<=`, `>=`, `!=`)
   and logical operators (e.g., `and`, `or`, `not`) to build complex and
   multiples conditions.

3. **(value)**: it could be numeric or string values. String value must be
   enclosed in double quotes (cf. example below)

### Examples of using filtering features:

- Confocal != "high"
- Thickness == "1ML" or Thickness == "3ML"
- a3_LOM >= 1000
- `Laser Power <= 5
