"""
Integration tests for the TensorFit engine using real benchmarking data.
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from spectroview.model.m_io import load_map_file, load_wdf_map
from spectroview.fit_engine.tensor_engine import TensorFittingEngine

# Paths to the benchmarking data
DATA_DIR = Path(__file__).parent.parent / "examples" / "fit_benchmarking_data"

def get_xy_from_map(df: pd.DataFrame):
    """Extract X array and Y matrix from map DataFrame."""
    # Columns are X, Y, followed by wavenumbers
    wavenumbers = [float(col) for col in df.columns[2:]]
    x = np.array(wavenumbers, dtype=np.float64)
    Y = df.iloc[:, 2:].to_numpy(dtype=np.float64)
    return x, Y

def load_fit_model(json_path: Path) -> dict:
    """Load fit model configuration from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # The JSON files usually use "0" as the root key for the model configuration
    if "0" in data:
        return data["0"]
    return data

@pytest.mark.skipif(not (DATA_DIR / "2_MoS2_map.txt").exists(), reason="Benchmarking data not found")
def test_tensor_engine_mos2_map():
    """Test convergence on MoS2 TXT map data."""
    # 1. Load data
    map_df = load_map_file(DATA_DIR / "2_MoS2_map.txt")
    x, Y = get_xy_from_map(map_df)
    
    # 2. Load model
    fit_model = load_fit_model(DATA_DIR / "2_MoS2_map.json")
    fit_params = fit_model.get("fit_params", {})
    
    # Use a subset of spectra to keep the test reasonably fast
    Y_sub = Y[:50]
    weights = np.ones_like(Y_sub)
    
    # Apply baseline if the JSON specifies it is attached and subtracted
    baseline_cfg = fit_model.get("baseline")
    if baseline_cfg and baseline_cfg.get("is_subtracted"):
        y_eval = np.array(baseline_cfg.get("y_eval", []), dtype=np.float64)
        if len(y_eval) == Y_sub.shape[1]:
            Y_sub = Y_sub - y_eval
    
    # 3. Fit
    engine = TensorFittingEngine()
    p_full, success, rsquared, best_fits, Y_peaks, param_names = engine.fit_spectra(
        x=x,
        Y=Y_sub,
        fit_model=fit_model,
        weights=weights,
        fit_params=fit_params,
        progress_callback=lambda c, t: None,
        cancel_check=lambda: False
    )
    
    # 4. Assertions
    assert len(success) == 50
    assert np.sum(success) >= 10, f"Too few fits converged ({np.sum(success)}/50)"
    
    # Check R2 only for successful fits
    good_r2 = rsquared[success]
    assert len(good_r2) > 0, "No valid R2 values returned"

@pytest.mark.skipif(not (DATA_DIR / "3_3721map.wdf").exists(), reason="Benchmarking data not found")
def test_tensor_engine_wdf_map():
    """Test convergence on WDF map data."""
    # 1. Load data
    map_df, _ = load_wdf_map(DATA_DIR / "3_3721map.wdf")
    x, Y = get_xy_from_map(map_df)
    
    # 2. Load model
    fit_model = load_fit_model(DATA_DIR / "3_3721map.json")
    fit_params = fit_model.get("fit_params", {})
    
    # Test a subset to keep the test reasonably fast
    Y_sub = Y[:50]
    weights = np.ones_like(Y_sub)
    
    # Apply baseline if the JSON specifies it is attached and subtracted
    baseline_cfg = fit_model.get("baseline")
    if baseline_cfg and baseline_cfg.get("is_subtracted"):
        y_eval = np.array(baseline_cfg.get("y_eval", []), dtype=np.float64)
        if len(y_eval) == Y_sub.shape[1]:
            Y_sub = Y_sub - y_eval
            
    # 3. Fit
    engine = TensorFittingEngine()
    p_full, success, rsquared, best_fits, Y_peaks, param_names = engine.fit_spectra(
        x=x,
        Y=Y_sub,
        fit_model=fit_model,
        weights=weights,
        fit_params=fit_params,
        progress_callback=lambda c, t: None,
        cancel_check=lambda: False
    )
    
    # 4. Assertions
    assert len(success) == 50
    assert np.sum(success) >= 10, f"Too few fits converged ({np.sum(success)}/50)"
    
    # Check R2 only for successful fits
    good_r2 = rsquared[success]
    assert np.mean(good_r2) > 0.8, f"Mean R2 too low: {np.mean(good_r2)}"
