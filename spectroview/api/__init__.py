"""SPECTROview Programmatic API
----------------------------
A comprehensive API to script and automate spectroscopic analysis workflows
without relying on the Qt Graphical User Interface.

Highlights:
    - `workspace.SpectraWorkspace` / `workspace.MapsWorkspace`: stateful
      sessions (load, preprocess, fit, collect results, save/load) that read
      and write the exact same files the GUI does.
    - `io`: file loading, dataset import, result export.
    - `preprocessing`: array-level baseline/crop/normalize.
    - `fitting`: fit-model construction, batch fitting, template CRUD.
    - `graphs`: publication-quality plots matching the GUI's own rendering,
      for all 9 supported plot styles, plus plot-template CRUD.
    - `analysis`: PCA / NMF.
    - `calculators`: quick spectroscopy calculators.
    - `settings`: headless access to persistent configuration.
    - `exceptions`: the `SpectroviewError` hierarchy raised across this package.

See docs/api/ for full usage examples.
"""
from spectroview.api import exceptions
from spectroview.api import settings
from spectroview.api import io
from spectroview.api import preprocessing
from spectroview.api import fitting
from spectroview.api import workspace
from spectroview.api import analysis
from spectroview.api import calculators
from spectroview.api import graphs

__all__ = [
    "exceptions",
    "settings",
    "io",
    "preprocessing",
    "fitting",
    "workspace",
    "analysis",
    "calculators",
    "graphs",
]
