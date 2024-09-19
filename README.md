[![PyPI version](https://badge.fury.io/py/spectroview.svg)](https://badge.fury.io/py/spectroview)
[![Github](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](https://github.com/CEA-MetroCarac/spectroview)
[![Downloads](https://img.shields.io/pypi/dm/spectroview.svg)](https://pypi.org/project/spectroview/)


<p align="center">
    <img width=100 src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/icon3.png">
</p>

## SPECTROview : A Tool for Spectroscopic Data Processing and Visualization.

Spectroscopy techniques such as Raman Spectroscopy and Photoluminescence (PL) are widely used in various fields, including materials science, chemistry, biology, and geology. In recent years, these techniques have increasingly found their place in cleanroom environments, particularly within the microelectronics industry, where they serve as critical metrology tools for wafer-scale measurements. The data obtained from these in-line measurements (wafer data) require specific processing procedures, but existing software solutions are often not optimized for this type of data and typically lack advanced plotting and visualization capabilities. Additionally, the licensing requirements of these software solutions can restrict access for a broader community of users.

SPECTROview addresses these gaps by providing free, open-source software designed to enhance both data processing and visualization workflows. Optimized for processing both wafer-type data and standard spectroscopic data, SPECTROview empowers users to streamline not only data processing but also data visualization, making the process more efficient, and rapid.

___

## Features:

- Cross-platform compatibility (Windows, macOS, Linux).
- Optimized user inferface for easy and quick inspection and comparison of spectra.
- Supports processing of spectral data (1D) and hyperspectral data (2D maps or wafer maps)*. 
- Ability to fit multiple spectra or 2Dmaps using predefined models or by creating custom fit models*.

**Fitting features are powered by the [fitspy](https://github.com/CEA-MetroCarac/fitspy) and [LMfit](https://lmfit.github.io/lmfit-py/) open-source packages.*


<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/GIF/tab_maps.gif">
</p>

- Collect all best-fit results with one click.

<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/GIF/collect_fit_results.gif">
</p>

- Dedicated module for effortless, fast, and easy data visualization. 

<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/spectroview/main/app/resources/GIF/visualization_tab.png">
</p>

## Installation from PyPI:

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
For any feedback, contact: [van-hoan.le@cea.fr](mailto:van-hoan.le@cea.fr)
