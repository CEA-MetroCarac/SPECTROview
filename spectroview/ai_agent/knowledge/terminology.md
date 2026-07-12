# Purpose

This file defines the terminology used within SPECTROview. It ensures consistent language when the AI Agent explains concepts or references application components.

---

# Core Application Terms

| Term | Definition |
|------|-----------|
| **SPECTROview** | The scientific spectroscopy data analysis application. |
| **Workspace** | A dedicated tab in the main application window, each optimised for a specific workflow (e.g., Spectra, Graphs, Maps). |
| **Project** | A saved application state (`.specview` file) that preserves loaded data, fits, and graph configurations. |
| **Settings** | Application preferences accessible via `Ctrl+Shift+S` or the Settings menu. |

---

# Data Terms

| Term | Definition |
|------|-----------|
| **DataFrame** | A tabular data structure (pandas `DataFrame`) with named columns and rows. In the Graphs workspace, DataFrames are the primary data source for all plots and AI queries. |
| **Spectrum** | A single 1D dataset of intensity values as a function of an X axis (wavenumber, wavelength, energy, time, etc.). |
| **Hyperspectral Map** | A 2D grid of spectra, where each pixel contains a full spectrum. Also called a "map" or "spectral image". |
| **ROI** | Region of Interest — a user-defined spectral or spatial range used to limit processing or analysis. |
| **Active DataFrame** | The DataFrame currently selected in the Graphs workspace sidebar. AI queries default to this DataFrame when no specific name is provided. |

---

# Fitting Terms

| Term | Definition |
|------|-----------|
| **Peak** | A localised spectral feature characterised by its position (x0), width (FWHM), and amplitude. |
| **Peak Model** | An analytic function used to describe the shape of a peak (e.g., Gaussian, Lorentzian, PseudoVoigt). |
| **FWHM** | Full Width at Half Maximum — the width of a peak measured at 50% of its maximum intensity. Common column names: `fwhm`, `fwhm_Si`, `fwhm_l`, `fwhm_r`. |
| **x0** | Peak centre position, expressed in the X axis units. Common column names: `x0`, `center`, `center_Si`. |
| **Amplitude** | Peak maximum intensity. Common column names: `ampli`, `amplitude`, `amplitude_Si`. |
| **Area** | Integrated area under the peak curve. Common column names: `area`, `area_Si`. |
| **Baseline** | The background signal underlying the peaks, modelled and subtracted during fitting. |
| **R²** | Coefficient of determination — a measure of fit quality. Values close to 1.0 indicate a good fit. Common column name: `R_squared`. |
| **Fit Result** | A DataFrame containing the fitted parameters for each spectrum in a dataset (one row per spectrum). |
| **VBF** | Vectorized Batch Fit — SPECTROview's high-performance batch fitting engine that processes entire hyperspectral maps simultaneously using NumPy vectorisation. |
| **Residual** | The difference between the raw spectrum and the fitted model at each data point. |

---

# Plot Terms

| Term | Definition |
|------|-----------|
| **Graph** | A single plot window in the Graphs workspace MDI area. Each graph has a unique integer Graph ID displayed in its title bar. |
| **Graph ID** | A unique integer assigned to each open graph window. Used to target specific graphs for update or delete operations. |
| **Plot Style** | The type of visualisation: point, scatter, box, bar, line, trendline, histogram, wafer, or 2Dmap. |
| **X / Y / Z** | The DataFrame columns mapped to the horizontal axis, vertical axis, and color/grouping axis respectively. |
| **Filter** | A `pandas.query()` expression applied to a DataFrame before plotting, e.g., `"Zone == 'center'"`. |
| **Color Palette** | The colormap used for the Z axis or plot elements (e.g., jet, viridis, plasma). |
| **Wafer Map** | A spatial plot showing measurement values at die/chip positions on a semiconductor wafer. |
| **2D Map** | A heatmap where rows and columns correspond to spatial or categorical coordinates and the color encodes a measured value. |

---

# AI Agent Terms

| Term | Definition |
|------|-----------|
| **AI Agent** | The SPECTROview built-in AI assistant, powered by a configurable LLM backend. |
| **Provider** | The LLM backend (Ollama, OpenAI, Gemini, DeepSeek, Anthropic, or Custom). |
| **Conversation** | A multi-turn chat session between the user and the AI Agent. Conversations are saved automatically. |
| **System Prompt** | The instructions and context injected by the application before the conversation begins. |
| **Action** | The AI Agent's response type: `filter`, `statistics`, `plot`, `update`, `delete`, or `answer`. |
