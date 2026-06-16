[![PyPI version](https://badge.fury.io/py/spectroview.svg)](https://badge.fury.io/py/spectroview)
[![Doc](https://img.shields.io/badge/%F0%9F%95%AE-docs-green.svg)](https://CEA-MetroCarac.github.io/SPECTROview/)
[![Github](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](https://github.com/CEA-MetroCarac/spectroview)
[![Downloads](https://img.shields.io/pypi/dm/spectroview.svg)](https://pypi.org/project/spectroview/)

<p align="center">
    <img width=100 src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/icon.png">
</p>
    
## **SPECTROview: A Tool for Spectroscopic Data Processing and Visualization**

**SPECTROview** is a free, open-source software application designed for advanced spectroscopic data analysis. It supports a diverse array of data types, including discrete spectra and hyperspectral datasets such as 2D maps and wafer maps.

With its integrated visualization tools, **SPECTROview** streamlines your analytical workflow by consolidating data processing and visualization into a single, cohesive application.

## **Documentation**

Full **documentation** is available at [**CEA-MetroCarac.github.io/SPECTROview**](https://CEA-MetroCarac.github.io/SPECTROview/):

- **For Users:** A comprehensive user manual is available [online](https://cea-metrocarac.github.io/SPECTROview/user_manual/) and can also be accessed directly within the application.
- **For Developers:** A detailed developer guide is available [here](https://cea-metrocarac.github.io/SPECTROview/developer/).
- **Changelog:** Review the latest updates and release notes [here](https://github.com/CEA-MetroCarac/SPECTROview/releases).
- **Getting Started**: Explore the [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) folder to familiarize yourself with supported data formats and find example datasets for practice.

---

### **Key Features**

- **High-Performance Vectorized Batch Fit Engine (`VBF Engine`):** Achieves very fast fitting speeds through batched matrix operations, capable of simultaneously fitting multiple spectra or large 2D maps.
- **Custom Fit Models:** Construct customized fit models for specific spectroscopic profiles and reuse them to rapidly analyze new datasets.
- **Versatile Data Processing:** Seamlessly process both 1D spectral data and 2D hyperspectral data.
- **Unified Results:** Collect and compile all best-fit results with a single click.
- **Advanced Visualization:** Dedicated workspace for generating fast, publication-ready data visualizations.
- **Optimized User Interface:** Designed for quick inspection, filtering, and comparison of large spectral datasets.

---

### **Dedicated workspaces for discrete spectra, hyperspectral data, and advanced visualization**
<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/overview.gif">
</p>

### **Design custom fit models and simultaneously process multiple spectra**

<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/fitting.gif">
</p>

### **Rapidly generate publication-ready data visualizations**

<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/graphs.gif">
</p>

---

## **Installation from PyPI**

`SPECTROview` requires Python (versions 3.8 through 3.12).

```bash
pip install spectroview
```

## **Installation from GitHub**

To install the latest development version directly from the source repository:

```bash
pip install git+https://github.com/CEA-MetroCarac/SPECTROview.git
```


## **Launch SPECTROview**
Open your terminal or command prompt and execute:
```bash
spectroview
```

## **Acknowledgements**

This work was carried out at the CEA — Platform for Nanocharacterisation (PFNC) and supported by the "Recherche Technologique de Base" program of the French National Research Agency (ANR).

---

