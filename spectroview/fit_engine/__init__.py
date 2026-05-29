# spectroview/core2/__init__.py
"""Tensor-based high-performance fitting engine (core2).

Fits all spectra in a hyperspectral map simultaneously using a custom
Batched Levenberg-Marquardt optimizer with analytical Jacobians.
"""
from spectroview.fit_engine.vbf_engine import VBFengine
from spectroview.fit_engine.vbf_thread import VBFthread

__all__ = ["VBFengine", "VBFthread"]
