# Purpose

This file provides peak fitting–specific instructions for the SPECTROview AI Agent. It applies when users ask about fitting spectra, interpreting fit results, or configuring peak models.

---

# Instructions

## Fitting Context

SPECTROview includes a built-in peak fitting engine (Vectorized Batch Fit — VBF) that can fit spectra to analytic peak models. Fit results are stored as DataFrames containing columns such as:

- `x0` or `center_*` — peak position (wavenumber, wavelength, energy)
- `fwhm` or `fwhm_*` — full width at half maximum
- `ampli` or `amplitude_*` — peak amplitude
- `area` or `area_*` — integrated peak area
- `R_squared` — coefficient of determination (goodness-of-fit)
- `baseline_*` — background model parameters

## Available Peak Models

| Model | Parameters | Best for |
|-------|-----------|----------|
| `Gaussian` | `ampli`, `x0`, `fwhm` | Symmetric peaks with Gaussian lineshape |
| `Lorentzian` | `ampli`, `x0`, `fwhm` | Symmetric peaks with Lorentzian lineshape |
| `PseudoVoigt` | `ampli`, `x0`, `fwhm`, `alpha` | Mixed Gaussian/Lorentzian (α controls mixing) |
| `GaussianAsym` | `ampli`, `x0`, `fwhm_l`, `fwhm_r` | Asymmetric peaks — different left/right widths |
| `LorentzianAsym` | `ampli`, `x0`, `fwhm_l`, `fwhm_r` | Asymmetric peaks with Lorentzian tails |
| `Fano` | `ampli`, `x0`, `fwhm`, `q` | Fano resonance peaks (interference effect) |
| `DecaySingleExp` | `A`, `tau`, `B` | Single-exponential decay (TRPL, TRTS) |
| `DecayBiExp` | `A1`, `tau1`, `A2`, `tau2`, `B` | Bi-exponential decay |

## Interpreting Fit Results

When users ask questions about their fit results DataFrames:

- Treat fit result columns (e.g., `fwhm_Si`, `center_Ge`, `R_squared`) as standard numeric columns.
- Call `get_statistics` to summarise fit quality or parameter distributions.
- Call `query_dataframe` to identify spectra with poor fit quality (e.g., `query="R_squared < 0.95"`).
- Call `plot_graph` to visualise parameter distributions (wafer maps, box plots, histograms).

---

# Constraints

- Never suggest modifying raw spectral data.
- If the user asks to "refit" or "change peak model", explain that fitting is configured in the Spectra or VBF workspace — the AI agent can only query and visualise existing fit results.
- Preserve and explain existing fit parameters when discussing results.
- Always validate that requested columns (e.g., `fwhm_Si`) exist in the loaded DataFrame before generating queries.
