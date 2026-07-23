---
hide:
  - navigation
  - toc
  - title
---

<p align="center" class="hero-badges">
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/v/spectroview.svg?color=blue&logo=pypi&logoColor=white" alt="PyPI version"></a>
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/pyversions/spectroview.svg?color=yellow&logo=python&logoColor=white" alt="Python Versions"></a>
    <a href="https://pypi.org/project/spectroview/"><img src="https://img.shields.io/pypi/dm/spectroview.svg?color=green&label=downloads" alt="Downloads"></a>
    <a href="https://github.com/CEA-MetroCarac/SPECTROview/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL_v3-lightgrey.svg" alt="License"></a>
    <a href="https://github.com/CEA-MetroCarac/SPECTROview/stargazers"><img src="https://img.shields.io/github/stars/CEA-MetroCarac/SPECTROview?style=flat&logo=github&color=blue" alt="GitHub Stars"></a>
    <a href="https://doi.org/10.1016/j.softx.2026.102862"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2026.102862-blue.svg" alt="DOI"></a>
    <a href="https://doi.org/10.1016/j.softx.2026.102862"><img src="https://img.shields.io/badge/Published_in-SoftwareX-orange" alt="SoftwareX"></a>
</p>

<p align="center">
    <b>SPECTROview</b> is an Open-Source Application for Interactive Spectroscopic Data Processing, Fitting, and Visualization.<br>
    It supports discrete spectra and hyperspectral datasets such as 2D maps and wafer maps,<br>
    with integrated visualization tools that streamline your analytical workflow.
</p>

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

-   :material-robot-happy-outline: __AI Chat Agent__

    ---

    Talk to SPECTROview in plain language to filter data, run analyses, and create or customize multiple plots — powered by your own local (Ollama) or cloud LLM.

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

=== ":material-robot-happy-outline: AI Chat Agent"

    <p align="center">Talk to SPECTROview in plain language to filter data and create or customize plots.</p>

    <div align="center" markdown>
    ![AI Chat Agent demo](assets/ai_agent.gif){ loading=lazy width="90%" }
    </div>

---

### :material-book-open-page-variant: **Documentation**

<div class="grid cards" markdown>

-   [:material-download-circle: **Installation**](user_manual/installation.md)
    <br> Install and launch in seconds via PyPI. Easily update to the latest version with a single command.

-   [:material-account-hard-hat: **User Manual**](user_manual/index.md)
    <br> Step-by-step guides covering all features, from data loading to advanced fitting and visualization.

-   [:material-code-tags: **Developer Guide**](developer/index.md)
    <br> Architecture overview, data pipeline internals, and contribution guidelines for developers.

-   [:material-api: **API Reference**](api/index.md)
    <br> Detailed reference for all modules: spectra processing, 2D maps, graphs, and calculators.

-   [:material-format-list-bulleted: **Changelog**](changelog.md)
    <br> Latest updates, release notes, bug fixes, and new feature announcements.

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
