---
hide:
  - navigation
  - toc
  - title
---

<p align="center" class="hero-badges">
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/v/spectroview.svg?logo=pypi&logoColor=white" alt="PyPI version"></a>
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/pyversions/spectroview.svg?logo=python&logoColor=white" alt="Python Versions"></a>
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/dm/spectroview.svg?color=blue&label=downloads" alt="Downloads"></a>
    <a href="https://github.com/CEA-MetroCarac/SPECTROview/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL_v3-blue.svg" alt="License"></a>
    <a href="https://doi.org/10.1016/j.softx.2026.102862"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2026.102862-blue.svg" alt="DOI"></a>
    <a href="https://doi.org/10.1016/j.softx.2026.102862"><img src="https://img.shields.io/badge/Published_in-SoftwareX-orange" alt="SoftwareX"></a>
</p>

**SPECTROview** is a free, open-source software application designed for advanced spectroscopic data analysis. It supports a diverse array of data types, including discrete spectra and hyperspectral datasets such as 2D maps and wafer maps. With its integrated visualization tools, **SPECTROview** streamlines your analytical workflow by consolidating data processing and visualization into a single, cohesive application.

---

### :material-star-shooting: **Key Features**

<div class="grid cards cards-2col" markdown>

-   :material-speedometer: __High-Performance Vectorized Batch Fit Engine (`VBF Engine`)__

    ---

    Achieves very fast fitting speeds through batched matrix operations, capable of simultaneously fitting multiple spectra or large 2D maps.

-   :material-chart-multiline: __Custom Fit Models__

    ---

    Construct customized fit models for specific spectroscopic profiles and reuse them to rapidly analyze new datasets.

-   :material-chart-bell-curve-cumulative: __Versatile Data Processing__

    ---

    Seamlessly process both 1D spectral data and 2D hyperspectral data.

-   :material-table-large-plus: __Unified Results__

    ---

    Collect and compile all best-fit results with a single click.

-   :material-chart-scatter-plot: __Advanced Visualization__

    ---

    Dedicated workspace for generating fast, publication-ready data visualizations.

-   :material-gesture-tap-button: __Optimized User Interface__

    ---

    Designed for quick inspection, filtering, and comparison of large spectral datasets.

</div>

---

### :material-play-circle-outline: **Demonstrations**

=== ":material-view-dashboard: Workspaces Overview"

    <p align="center">Dedicated workspaces for discrete spectra, hyperspectral data, and advanced visualization.</p>

    <div align="center" markdown>
    ![General overview of SPECTROview](assets/overview.gif){ loading=lazy width="90%" }
    </div>

=== ":material-chart-bell-curve: Fitting Engine"

    <p align="center">Design custom fit models and simultaneously process multiple spectra with the VBF Engine.</p>

    <div align="center" markdown>
    ![Fitting workflow demo](assets/fitting.gif){ loading=lazy width="90%" }
    </div>

=== ":material-chart-bar: Publication-Ready Graphs"

    <p align="center">Rapidly generate publication-ready data visualizations with the built-in Graphs workspace.</p>

    <div align="center" markdown>
    ![Plotting and visualization demo](assets/graphs.gif){ loading=lazy width="90%" }
    </div>

---

### :material-rocket-launch-outline: **Quick Start**

<div class="quick-start-block" markdown>

**SPECTROview** requires Python 3.8 – 3.12. Install and launch in seconds:

```bash
pip install spectroview
```

```bash
spectroview
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/CEA-MetroCarac/SPECTROview.git
```

</div>

---

### :material-book-open-page-variant: **Documentation**

<div class="grid cards cards-2col" markdown>

-   :material-account-hard-hat: __User Manual__

    ---

    Step-by-step guides covering all features, from data loading to advanced fitting and visualization.

    [:octicons-arrow-right-24: Read the User Manual](user_manual/index.md)

-   :material-code-tags: __Developer Guide__

    ---

    Architecture overview, data pipeline internals, and contribution guidelines for developers.

    [:octicons-arrow-right-24: Read the Developer Guide](developer/index.md)

-   :material-api: __API Reference__

    ---

    Detailed reference for all modules: spectra processing, 2D maps, graphs, and calculators.

    [:octicons-arrow-right-24: Browse the API](api/index.md)

-   :material-format-list-bulleted: __Changelog__

    ---

    Latest updates, release notes, bug fixes, and new feature announcements.

    [:octicons-arrow-right-24: View the Changelog](changelog.md)

</div>

---

### :material-hand-heart-outline: **Acknowledgements**

This work was carried out at the CEA — Platform for Nanocharacterisation (PFNC) and supported by the "Recherche Technologique de Base" program of the French National Research Agency (ANR).

---

### :material-format-quote-open: **Citation**

If you use **SPECTROview** in your work, please cite:

> Le, Van Hoan. "SPECTROview: An Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization." *SoftwareX* 35 (September 2026): 102862. [https://doi.org/10.1016/j.softx.2026.102862](https://doi.org/10.1016/j.softx.2026.102862).

??? note "BibTeX"

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
