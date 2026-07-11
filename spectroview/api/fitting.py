"""Vectorized batch fitting capabilities."""
import numpy as np
from typing import Dict, Any, List, Optional
from spectroview.fit_engine.vbf_engine import VBFengine

def fit_batch(x: np.ndarray, Y: np.ndarray, fit_model: Dict[str, Any], 
              weights: Optional[np.ndarray] = None, 
              fit_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fit multiple spectra simultaneously using the VBF engine.
    
    Args:
        x: Wavenumber axis, shape (M,) or (N, M).
        Y: Intensity matrix, shape (N, M).
        fit_model: Dictionary defining the peak model and baseline.
        weights: Optional fit weights.
        fit_params: Optimizer parameters (xtol, ftol, max_ite).
        
    Returns:
        Dict containing 'params', 'success', 'r_squared', 'best_fits', 'peaks', 'param_names'.
    """
    engine = VBFengine()
    p_full, success, rsquared, best_fits, Y_peaks, param_names = engine.fit_spectra(
        x, Y, fit_model, weights=weights, fit_params=fit_params
    )
    
    return {
        "params": p_full,
        "success": success,
        "r_squared": rsquared,
        "best_fits": best_fits,
        "peaks": Y_peaks,
        "param_names": param_names
    }
