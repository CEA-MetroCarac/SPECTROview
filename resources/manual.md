<!-- TOC -->
# SPECTROview manual
<p align="center">
    <img width=100 src="icon3.png">
</p>

- [Introduction](#introduction)
- [1. Supported data formats](#1-supported-data-formats)
  - [1.1 Discret spectroscopic data (spectrum)](#11-discret-spectroscopic-data-spectrum)
  - [1.2 Hyperspectral data (2Dmaps or wafer data)](#12-hyperspectral-data-2dmaps-or-wafer-data)
  - [1.3 Excel files containing datasheet](#13-excel-files-containing-datasheet)
- [2. "Spectra" and "Maps" tabs](#2-spectra-and-maps-tabs)
- [3. "Data Visualization" TAB](#3-data-visualization-tab)
- [4. Data filtering feature](#4-data-filtering-feature)

<!-- /TOC -->

------------ 
# Introduction

SPECTROview application's GUI have 3 main tabs dedicated for different purposes: 
- Spectra: for processing multiples discret spectroscopic data
- Maps: for processing multiples 2Dmaps or hyperspectral data
- Data visualization : for graphing fitted results and visualize Excel datasheet. 

<p align="center">
    <img width=300 src="manual_figures/fig0.png">
</p>
<p align="center">
    <img width=900 src="manual_figures/fig1.png">
</p>

# 1. Supported file/data formats

All file/data formats supported by SPECTROview can be opened via a signal push button in the toolbar

## 1.1 Discret spectroscopic data (spectrum)
## 1.2 Hyperspectral data (2Dmaps or wafer data)
## 1.3 Excel files contaning datasheet
Excel files contaning datasheets within one or multiples Excel sheets is supportted. 
User can directly load an Excel files for plotting via the "data visualization" Tab.

# 1. "Spectra" and "Maps" tabs
The 'Spectra' and 'Maps' tab sharing almost the same feature and GUI except some specific GUi element for 2Dmaps/hyperspectral data navigation.

# 2. "Data Visualization" TAB



# 3. Data filtering feature

The **pandas.DataFrame.query()** method allows you to filter rows from a
DataFrame based on a boolean expression. It's a powerful and flexible way to
subset your DataFrame to include only the rows that meet specific conditions.

<p align="center">
    <img width=400 src="manual_figures/dfr_filter.png">
</p>

In **SPECTROview**, `query()` method is integrated and can be used via GUI by
typing as following: **(column_name) (operator) (value)**:

1. **(column_name)**: is the exact header of the column containing the
   data/values to be filtered. When the column name contain 'space', you
   need to enclose them in single or double quotes (see example below).


2. **(operator)**: it could be comparison operators (
   e.g., `==`, `<`, `>`, `<=`, `>=`, `!=`)
   and logical operators (e.g., `and`, `or`, `not`) to build complex and
   multiples
   conditions.

3. **(value)**: it could be numeric or string values. String value must be
   enclosed in double quotes (cf. example below)

### Here are some examples of using filters:

- Confocal != "high"
- Thickness == "1ML" or Thickness == "3ML"
- a3_LOM >= 1000
- `Laser Power <= 5
