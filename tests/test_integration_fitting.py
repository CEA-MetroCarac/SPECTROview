"""
Integration tests for the TensorFit engine using real benchmarking data.
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from spectroview.model.m_io import load_map_file, load_wdf_map
from spectroview.model.m_settings import MSettings
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.fit_engine.vbf_fit_thread import VBFthread
from spectroview.fit_engine.vbf_fit_engine import VBFengine

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
def test_vbf_fit_engine_mos2_map():
    """Test convergence on MoS2 TXT map data."""
    # 1. Setup VM and Store
    settings = MSettings()
    vm = VMWorkspaceSpectra(settings)
    store = vm.store

    # 2. Load data into store
    map_df = load_map_file(DATA_DIR / "2_MoS2_map.txt")
    x, Y = get_xy_from_map(map_df)
    coords = map_df.iloc[:, :2].to_numpy()
    fnames = [f"2_MoS2_map_{i}" for i in range(len(Y))]
    store.add_map("2_MoS2_map", x, Y, coords, fnames)

    # 3. Load model and apply
    fit_model = load_fit_model(DATA_DIR / "2_fit_MoS2map_NEW.json")
    md = store.get_map_data("2_MoS2_map")
    vm._apply_fit_model_to_mapdata(md, fit_model)

    # 4. Run fit thread
    tasks = [{
        "map_name": "2_MoS2_map",
        "indices": np.arange(len(Y)),
        "fit_model": fit_model
    }]
    thread = VBFthread(store, tasks)
    thread.run()  # Run synchronously for test

    # 5. Extract results from store
    success = md.fit_success
    rsquared = md.fit_r2
    param_names = md.param_names
    p_full = md.peak_params
    
    # 4. Verify physical parameter values against the known good fit from GUI benchmarking
    # The expected values are for the spectrum at (1764.6, 72.9) which is at index 739
    target_idx = 739
    if target_idx < len(Y):
        p_opt = p_full[target_idx]
        p_dict = {name: val for name, val in zip(param_names, p_opt)}
        print(f"Index {target_idx} actual values: {p_dict}")
        
        # Peak 1 (Lorentzian)
        assert abs(p_dict['P1_x0'] - 384.24) < 0.1
        assert abs(p_dict['P1_fwhm'] - 2.17) < 0.1
        assert abs(p_dict['P1_ampli'] - 404.98) < 25.0
        
        # Peak 2 (Lorentzian)
        assert abs(p_dict['P2_x0'] - 403.84) < 0.1
        assert abs(p_dict['P2_fwhm'] - 5.06) < 0.1
        assert abs(p_dict['P2_ampli'] - 393.86) < 15.0

        # Peak 3 (Lorentzian)
        assert abs(p_dict['P3_x0'] - 446.33) < 0.1
        assert abs(p_dict['P3_fwhm'] - 18.95) < 15.0
        assert abs(p_dict['P3_ampli'] - 34.61) < 10.0

    # 5. Assertions
    assert len(success) == len(Y)
    assert np.sum(success) > 1400, f"Too few fits converged ({np.sum(success)}/{len(Y)})"
    
    # Check R2 only for successful fits
    good_r2 = rsquared[success]
    assert len(good_r2) > 0, "No valid R2 values returned"


@pytest.mark.skipif(not (DATA_DIR / "3_3721map.wdf").exists(), reason="Benchmarking data not found")
def test_vbf_fit_engine_wdf_map():
    """Test convergence on WDF map data."""
    # 1. Load data
    map_df, _ = load_wdf_map(DATA_DIR / "3_3721map.wdf")
    x, Y = get_xy_from_map(map_df)
    
    # 2. Load model
    fit_model = load_fit_model(DATA_DIR / "3_3721map.json")
    fit_params = fit_model.get("fit_params", {})
    
    # Test a subset to keep the test reasonably fast
    Y_sub = Y[:3721]
    weights = np.ones_like(Y_sub)
    
    # Apply baseline if the JSON specifies it is attached and subtracted
    baseline_cfg = fit_model.get("baseline")
    if baseline_cfg and baseline_cfg.get("is_subtracted"):
        y_eval = np.array(baseline_cfg.get("y_eval", []), dtype=np.float64)
        if len(y_eval) == Y_sub.shape[1]:
            Y_sub = Y_sub - y_eval
            
    # 3. Fit
    engine = VBFengine()
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
    assert len(success) == len(Y_sub)
    assert np.sum(success) >= 10, f"Too few fits converged ({np.sum(success)}/50)"
    
    # Check R2 only for successful fits
    good_r2 = rsquared[success]
    assert np.mean(good_r2) > 0.8, f"Mean R2 too low: {np.mean(good_r2)}"
