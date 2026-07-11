# API

The SPECTROview Python API allows you to automate workflows and interact programmatically with the core spectroscopic processing engines of SPECTROview. This API is completely decoupled from the Qt GUI, making it perfect for Jupyter notebooks or custom scripts.

All modules are exposed through the `spectroview.api` package.

## API Modules

The API is organized into several functional modules:

- **[`spectroview.api.io`](spectra.md#loading-data)**: File readers and writers (WDF, SPC, CSV, TXT, DAT).
- **[`spectroview.api.processing`](spectra.md#preprocessing)**: Functions for baseline subtraction, cropping, and intensity normalization.
- **[`spectroview.api.fitting`](spectra.md#batch-fitting)**: The interface to the Vectorized Batch Fit (VBF) engine.
- **[`spectroview.api.analysis`](2dmap.md#multivariate-analysis-mva)**: Multivariate analysis tools like Principal Component Analysis (PCA) and Non-negative Matrix Factorization (NMF).
- **[`spectroview.api.graphs`](graphs.md)**: Functions for generating publication-quality plots replicating SPECTROview's native styles.
- **[`spectroview.api.calculators`](calculators.md)**: Physical calculators for optics and unit conversions.

## Exploring the API

Choose a topic below to see comprehensive code examples:

- [Spectra Processing & Fitting](spectra.md): How to load discrete spectra, subtract baselines, and run the batch fitting engine.
- [2D Map & Hyperspectral Analysis](2dmap.md): Loading wafer maps, dealing with spatial coordinates, and running PCA/NMF.
- [Data Visualization (Graphs)](graphs.md): Replicating SPECTROview plots natively in scripts using Seaborn/Matplotlib.
- [Quick Calculators](calculators.md): Using the programmatic optical and unit converters.
