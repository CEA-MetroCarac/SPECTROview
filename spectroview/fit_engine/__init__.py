# spectroview/core2/__init__.py
"""Tensor-based high-performance fitting engine (core2).

Fits all spectra in a hyperspectral map simultaneously using a custom
Batched Levenberg-Marquardt optimizer with analytical Jacobians.
"""
from spectroview.fit_engine.tensor_engine import TensorFittingEngine
from spectroview.fit_engine.tensor_fit_thread import TensorFitThread

__all__ = ["TensorFittingEngine", "TensorFitThread"]
