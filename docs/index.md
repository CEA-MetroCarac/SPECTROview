# SPECTROview

[![PyPI version](https://badge.fury.io/py/spectroview.svg)](https://badge.fury.io/py/spectroview)
[![Github](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](https://github.com/CEA-MetroCarac/spectroview)
[![Downloads](https://img.shields.io/pypi/dm/spectroview.svg)](https://pypi.org/project/spectroview/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14148070.svg)](https://doi.org/10.5281/zenodo.14147172)

## A Tool for Spectroscopic Data Processing and Visualization

SPECTROview is free, open-source software designed for spectroscopic data analysis. It supports a wide range of data types, including discrete spectra and hyperspectral data (such as 2D maps and wafer maps).

With its built-in visualization tools, SPECTROview streamlines your workflow by combining data processing and visualization into a single, efficient application.

For more information, please visit the **GitHub Repository**: [https://github.com/CEA-MetroCarac/SPECTROview](https://github.com/CEA-MetroCarac/SPECTROview)

---

## Features: 

- Cross-platform compatibility (Windows, macOS, Linux).
- Supports processing of spectral data (1D) and hyperspectral data (2D maps or wafer maps)*. 
- **Tensor Fit Engine**: 10–15× faster fitting using batched matrix operations
- Ability to fit multiple spectra or 2Dmaps using predefined models or by creating custom fit models*.

- Collect all best-fit results with one click.
- Optimized user interface for easy and quick inspection and comparison of spectra.
- Dedicated module for effortless, fast, and easy data visualization. 

**Fitting features are powered by the *fitspy* and *lmfit* open-source packages.*

---

## Demo

### Three separate tabs for processing discrete spectra, hyperspectral data, and data visualization

![General overview of SPECTROview](assets/general_demo.gif)

### Build a fit model, copy/paste to others, fit multiple spectra or maps, collect all best-fit results

![Fitting workflow demo](assets/fitting_demo.gif)

### Process hyperspectral maps and wafer data with high-performance tensor fitting

![Map workspace demo](assets/map_demo.gif)

### Plot and visualize data rapidly and easily

![Plotting and visualization demo](assets/plotting_demo.gif)

---

## Quick Start

```bash
pip install spectroview
spectroview
```

## Citation

Le, V.-H., & Quéméré, P. (2025). SPECTROview: A Tool for Spectroscopic Data Processing and Visualization. Zenodo. [https://doi.org/10.5281/zenodo.14147172](https://doi.org/10.5281/zenodo.14147172)
