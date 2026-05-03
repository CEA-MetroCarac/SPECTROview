[![PyPI version](https://badge.fury.io/py/spectroview.svg)](https://badge.fury.io/py/spectroview)
[![Doc](https://img.shields.io/badge/%F0%9F%95%AE-docs-green.svg)](https://CEA-MetroCarac.github.io/SPECTROview/)
[![Github](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](https://github.com/CEA-MetroCarac/spectroview)
[![Downloads](https://img.shields.io/pypi/dm/spectroview.svg)](https://pypi.org/project/spectroview/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14148070.svg)](https://doi.org/10.5281/zenodo.14147172) 

<p align="center">
    <img width=100 src="docs/assets/icon.png">
</p>

## SPECTROview : A Tool for Spectroscopic Data Processing and Visualization.

SPECTROview is free, open-source software designed for spectroscopic data analysis. It supports a wide range of data types, including discrete spectra and hyperspectral data (such as 2D maps and wafer maps).

With its built-in visualization tools, SPECTROview streamlines your workflow by combining data processing and visualization into a single, efficient application. 

- Full **documentation** is available at [**CEA-MetroCarac.github.io/SPECTROview**](https://CEA-MetroCarac.github.io/SPECTROview/). A PDF manual is also available [here](https://github.com/CEA-MetroCarac/SPECTROview/blob/main/spectroview/resources/SPECTROview_UserManual.pdf) or from within the application.
- **Installation**: Instructions can be found at the bottom of this page.
- Check out [**Releases**](https://github.com/CEA-MetroCarac/SPECTROview/releases) page for the latest updates and new features.
- **Getting Started**: Check out [this folder](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) to see the supported data formats and find example datasets for practice.




___
## Features: 

- Cross-platform compatibility (Windows, macOS, Linux).
- Supports processing of spectral data (1D) and hyperspectral data (2D maps or wafer maps)*. 
- **Tensor Fit Engine**: 10–15× faster fitting using batched matrix operations
- Ability to fit multiple spectra or 2Dmaps using predefined models or by creating custom fit models*.

- Collect all best-fit results with one click.
- Optimized user inferface for easy and quick inspection and comparison of spectra.
- Dedicated module for effortless, fast, and easy data visualization. 

**Fitting features are powered by the *fitspy* and *lmfit* open-source packages.*

______
### Three separate tabs for processing discrete spectra, hyperspectral data, and data visualization:
<p align="center">
    <img src="docs/assets/general_demo.gif">
</p>

### Build a fit model for later use, copy/paste to others, fit multiple spectra or maps, collect all best-fit results with one click:

<p align="center">
    <img src="docs/assets/fitting_demo.gif">
</p>

### Plot and visualize data radpily and easily:

<p align="center">
    <img src="docs/assets/plotting_demo.gif">
</p>

____

## Installation from PyPI:

Make sure that Python (version between 3.8 and 3.12) is already installed.

```bash
pip install spectroview
```

## Installation from Github:

```bash
pip install git+https://github.com/CEA-MetroCarac/SPECTROview.git
```


## To launch SPECTROview:
```bash
spectroview
```

## Acknowledgements

This work, carried out on the CEA - Platform for Nanocharacterisation (PFNC), was supported by the “Recherche Technologique de Base” program of the French National Research Agency (ANR).

---
## Citation

Le, V.-H., & Quéméré, P. (2025). SPECTROview : A Tool for Spectroscopic Data Processing and Visualization. Zenodo. https://doi.org/10.5281/zenodo.14147172
