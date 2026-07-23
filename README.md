<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/top_banner.png" alt="SPECTROview Banner" width="100%">
</p>

<p align="center">
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/v/spectroview.svg?color=blue&logo=pypi&logoColor=white" alt="PyPI version"></a>
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/pyversions/spectroview.svg?color=yellow&logo=python&logoColor=white" alt="Python Versions"></a>
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/dm/spectroview.svg?color=green&label=downloads" alt="Downloads"></a>
    <a href="https://github.com/CEA-MetroCarac/SPECTROview/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL_v3-lightgrey.svg" alt="License: GPL v3"></a>
    <a href="https://CEA-MetroCarac.github.io/SPECTROview/"><img src="https://img.shields.io/badge/docs-latest-green.svg" alt="Docs"></a>
    <a href="https://github.com/CEA-MetroCarac/SPECTROview/stargazers"><img src="https://img.shields.io/github/stars/CEA-MetroCarac/SPECTROview?style=flat&logo=github&color=blue" alt="GitHub Stars"></a>
    <a href="https://github.com/CEA-MetroCarac/SPECTROview/issues"><img src="https://img.shields.io/github/issues/CEA-MetroCarac/SPECTROview?logo=github&color=blue" alt="GitHub Issues"></a>
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform"></a>
    <a href="https://doi.org/10.1016/j.softx.2026.102862"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2026.102862-blue.svg" alt="DOI"></a>
    <a href="https://doi.org/10.1016/j.softx.2026.102862"><img src="https://img.shields.io/badge/Published_in-SoftwareX-orange" alt="SoftwareX"></a>
</p>

<h1 align="center">SPECTROview</h1>

<p align="center">
   A free, open-source software designed for advanced spectroscopic data analysis.<br>
    It supports discrete spectra and hyperspectral datasets such as 2D maps and wafer maps,<br>
    with integrated visualization tools that streamline your analytical workflow.
</p>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Chat Agent** | Talk to SPECTROview in plain language to filter data, run analyses, and create or customize multiple plots — powered by your own local (Ollama) or cloud LLM |
| ⚡ **Vectorized Batch Fit Engine** | Ultra-fast fitting via batched matrix operations — simultaneously fit multiple spectra or large 2D maps |
| 🧩 **Custom Fit Models** | Build and reuse custom fit models tailored to your specific spectroscopic profiles |
| 📊 **Versatile Data Processing** | Seamlessly handle both 1D spectral data and 2D hyperspectral data |
| 📋 **Unified Results** | Collect and compile all best-fit results with a single click |
| 🎨 **Advanced Visualization** | Dedicated workspace for fast, publication-ready visualizations |
| 🖥️ **Optimized UI** | Designed for quick inspection, filtering, and comparison of large spectral datasets |

---

## 🎬 Demonstrations

<details open>
<summary><strong>📁 Dedicated workspaces for discrete spectra, hyperspectral data, and visualization</strong></summary>
<br>
<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/overview.gif" alt="SPECTROview Overview" width="100%">
</p>
</details>

<details open>
<summary><strong>📐 Design custom fit models and simultaneously process multiple spectra</strong></summary>
<br>
<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/fitting.gif" alt="Fitting Demo" width="100%">
</p>
</details>

<details open>
<summary><strong>📈 Rapidly generate publication-ready data visualizations</strong></summary>
<br>
<p align="center">
    <img src="https://raw.githubusercontent.com/CEA-MetroCarac/SPECTROview/main/docs/assets/graphs.gif" alt="Graphs Demo" width="100%">
</p>
</details>

---

## 🚀 Quick Start

**SPECTROview** requires Python 3.8 – 3.12.

**Install from PyPI:**
```bash
pip install spectroview
```

**Or install the latest development version from GitHub:**
```bash
pip install git+https://github.com/CEA-MetroCarac/SPECTROview.git
```

**Launch:**
```bash
spectroview
```

---

## 📚 Documentation

Full documentation is available at [**CEA-MetroCarac.github.io/SPECTROview**](https://CEA-MetroCarac.github.io/SPECTROview/):

| Resource | Link |
|---|---|
| 📖 **User Manual** | [Online manual](https://cea-metrocarac.github.io/SPECTROview/user_manual/) (also accessible within the app) |
| 🛠️ **Developer Guide** | [Developer docs](https://cea-metrocarac.github.io/SPECTROview/developer/) |
| 📝 **Changelog** | [Release notes](https://github.com/CEA-MetroCarac/SPECTROview/releases) |
| 📂 **Examples** | [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) — sample datasets and supported formats |

---

## 🙏 Acknowledgements

This work was carried out at the CEA — Platform for Nanocharacterisation (PFNC) and supported by the "Recherche Technologique de Base" program of the French National Research Agency (ANR).

---

## 📄 Citation

If you use **SPECTROview** in your work, please cite:

> Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization." *SoftwareX* 35 (September 2026): 102862. [https://doi.org/10.1016/j.softx.2026.102862](https://doi.org/10.1016/j.softx.2026.102862).

<details>
<summary><strong>BibTeX</strong></summary>

```bibtex
@article{le2026spectroview,
    title     = {SPECTROview: An Open-Source Application for Interactive
                 Spectroscopic Data Processing, Fitting, and Visualization},
    author    = {Le, Van Hoan},
    journal   = {SoftwareX},
    volume    = {35},
    pages     = {102862},
    year      = {2026},
    month     = sep,
    publisher = {Elsevier},
    doi       = {10.1016/j.softx.2026.102862}
}
```
</details>

---