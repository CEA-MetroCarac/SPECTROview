# spectroview/core/__init__.py
"""
High-performance fitting engine for hyperspectral data.

This package provides a batch fitting pipeline that operates on the full
(N_spectra × N_wavelengths) data matrix, using scipy.optimize.least_squares
directly and spatial neighbor propagation for 2D maps.
"""

from spectroview.core.batch_engine import BatchFittingEngine
from spectroview.core.hyper_fit_thread import HyperFitThread

__all__ = ["BatchFittingEngine", "HyperFitThread"]
