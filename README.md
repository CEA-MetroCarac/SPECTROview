[![PyPI version](https://badge.fury.io/py/spectroview.svg)](https://badge.fury.io/py/spectroview)
[![Github](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](https://github.com/CEA-MetroCarac/spectroview)
[![Downloads](https://img.shields.io/pypi/dm/spectroview.svg)](https://pypi.org/project/spectroview/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14148070.svg)](https://doi.org/10.5281/zenodo.14147172) 

<p align="center">
    <img width=100 src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/icon3.png">
</p>

## SPECTROview : A Tool for Spectroscopic Data Processing and Visualization.

Spectroscopy techniques such as Raman Spectroscopy and Photoluminescence (PL) are widely used in various fields, including materials science, chemistry, biology, and geology. In recent years, these techniques have increasingly found their place in cleanroom environments, particularly within the microelectronics industry, where they serve as critical metrology tools for wafer-scale measurements. The data collected from these in-line measurements (wafer data) require specific processing, but existing software solutions are often not optimized for this type of data and typically lack advanced plotting and visualization capabilities. Additionally, the licensing requirements of these software solutions can restrict access for a broader community of users.

SPECTROview addresses these gap by offering free, open-source software that is compataible with both in-line data (wafer-map) as well as standard spectroscopic data (discret spectra, 2D maps). It also features a built-in visualization tool, enabling users to streamline both data processing and visualization in a single application, making the workflow more efficient.


___

## Features: 

- Cross-platform compatibility (Windows, macOS, Linux).
- Supports processing of spectral data (1D) and hyperspectral data (2D maps or wafer maps)*. 
- Ability to fit multiple spectra or 2Dmaps using predefined models or by creating custom fit models*.
- Collect all best-fit results with one click.
- Optimized user inferface for easy and quick inspection and comparison of spectra.
- Dedicated module for effortless, fast, and easy data visualization. 

**Fitting features are powered by the *fitspy* and *lmfit* open-source packages.*


<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/GIF/tab_maps.gif">
</p>

<p align="center">(Fitting multiple spectra / wafers / 2D-maps with predefined models)</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/GIF/collect_fit_results.gif">
</p>
<p align="center">(Collect data with one click)</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/GIF/2Dmap.png">
</p>
<p align="center">(2Dmap processing and visualization, extract and plot line profiles with ease, etc.)</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/GIF/visualization_tab.png">
</p>
<p align="center">(Support various plotting styles with automatic statistical calculations)</p>

## Installation from PyPI:

Make sure that Python (> 3.8) is already installed.

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

Le, V.-H., & Quéméré, P. (2025). SPECTROview : A Tool for Spectroscopic Data Processing and Visualization (0.4.6). Zenodo. https://doi.org/10.5281/zenodo.14147172
