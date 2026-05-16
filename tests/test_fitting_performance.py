"""
Performance and regression tests for the Tensor Fit Engine.
These tests use real benchmarking data to ensure fitting precision, speed,
and numerical stability scale correctly across various multi-peak scenarios.

Run these tests via:
  pytest tests/test_fitting_performance.py -v
  pytest tests/test_fitting_performance.py -v -m slow
"""

import json
import time
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from spectroview.model.m_io import load_map_file, load_wdf_map
from spectroview.model.m_spectrum import MSpectrum
from spectroview.model.m_spectra import MSpectra
from spectroview.fit_engine.tensor_engine import TensorFittingEngine

# --- Data Loading Helpers ---

def get_benchmarking_dir():
    return Path(__file__).parent.parent / "examples" / "fit_benchmarking_data"

def extract_spectra_from_df(map_df, map_name):
    """Convert a loaded map DataFrame into a list of MSpectrum objects."""
    wavenumber_cols = [col for col in map_df.columns if col not in ['X', 'Y']]
    
    # Handle legacy behavior (skipping last column if it's empty/invalid)
    x_values = pd.to_numeric(wavenumber_cols, errors='coerce').tolist()
    if len(x_values) > 1:
        x_values = x_values[:-1]
        wavenumber_cols = wavenumber_cols[:-1]
        
    x_data = np.asarray(x_values, dtype=np.float64)
    x_positions = map_df['X'].values
    y_positions = map_df['Y'].values
    intensity_data = map_df[wavenumber_cols].values
    
    spectra = []
    for idx in range(len(map_df)):
        s = MSpectrum()
        s.fname = f"{map_name}_({x_positions[idx]}, {y_positions[idx]})"
        s.x = x_data.copy()
        s.x0 = x_data.copy()
        s.y = np.asarray(intensity_data[idx], dtype=np.float64)
        s.y0 = s.y.copy()
        s.baseline.mode = "Linear"
        s.baseline.sigma = 4
        spectra.append(s)
    return spectra

def load_spectra_from_maps_file(maps_file_path):
    """Headless loading of spectra from a .maps saved workspace file."""
    with open(maps_file_path, 'r') as f:
        data = json.load(f)
    
    spectra = []
    spectrums_data = data.get('spectrums_data', {})
    
    # Simple extraction (assuming flat data representation for test purposes)
    # The actual application uses KD-Tree matching, but here we just need the 
    # spectrum objects initialized correctly for fitting.
    
    from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps
    from spectroview.model.m_settings import MSettings
    
    # We leverage the ViewModel's load method but in a headless way
    vm = VMWorkspaceMaps(MSettings())
    vm.load_work(str(maps_file_path))
    
    return list(vm.spectra)

# --- Benchmarks ---

@pytest.fixture(scope="module")
def cl_map_data():
    base_dir = get_benchmarking_dir()
    df = load_map_file(base_dir / "1_CL_map.txt")
    model = json.load(open(base_dir / "1_CL_map.json"))['0']
    spectra = extract_spectra_from_df(df, "1_CL_map")
    return spectra, model

@pytest.fixture(scope="module")
def mos2_map_data():
    base_dir = get_benchmarking_dir()
    df = load_map_file(base_dir / "2_MoS2_map.txt")
    model = json.load(open(base_dir / "2_MoS2_map.json"))['0']
    spectra = extract_spectra_from_df(df, "2_MoS2_map")
    return spectra, model

@pytest.fixture(scope="module")
def wdf_map_data():
    base_dir = get_benchmarking_dir()
    try:
        # load_wdf_map returns (DataFrame, metadata)
        result = load_wdf_map(base_dir / "3_3721map.wdf")
        df = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        pytest.skip(f"WDF loading failed (missing renishaw WiRE or dependencies): {e}")
        
    model = json.load(open(base_dir / "3_3721map.json"))['0']
    spectra = extract_spectra_from_df(df, "3_3721map")
    return spectra, model

@pytest.fixture(scope="module")
def d4_map_data():
    base_dir = get_benchmarking_dir()
    spectra = load_spectra_from_maps_file(base_dir / "4_D4_map.maps")
    model = json.load(open(base_dir / "4_D4_map.json"))['0']
    return spectra, model

@pytest.fixture(scope="module")
def mos2_wafers_data():
    base_dir = get_benchmarking_dir()
    spectra = load_spectra_from_maps_file(base_dir / "5_MoS2_wafers.maps")
    model = json.load(open(base_dir / "5_MoS2_wafers.json"))['0']
    return spectra, model

# --- Performance & Regression Tests ---

@pytest.mark.slow
def test_1_cl_map_performance(cl_map_data):
    """
    Benchmark 1: Large map, single peak.
    16,384 spectra, 1 Lorentzian.
    """
    spectra, model = cl_map_data
    assert len(spectra) == 16384
    
    engine = TensorFittingEngine()
    
    t0 = time.perf_counter()
    results = engine.fit_spectra(
        spectra=spectra,
        fit_model=model,
        fit_params={'method': 'leastsq', 'xtol': 1e-4, 'max_ite': 200, 'coef_noise': 1},
        apply_model_to_spectra=True
    )
    total_time = time.perf_counter() - t0
    
    # Performance assertion: Should be well under 8 seconds
    # (Typical optimized run is ~2-3s)
    assert total_time < 8.0, f"CL_map fitting is too slow: {total_time:.2f}s"
    
    # Correctness assertions
    success_count = sum(1 for r in results if r.success)
    assert success_count == len(spectra), "All spectra should converge"
    
    # Check that best_fit is non-empty
    assert len(results[0].best_fit) > 0


@pytest.mark.slow
def test_2_mos2_map_performance(mos2_map_data):
    """
    Benchmark 2: Medium map, multi-peak with noise masking.
    1,520 spectra, 3 Lorentzians.
    Tests noise threshold parameter restoration.
    """
    spectra, model = mos2_map_data
    assert len(spectra) == 1520
    
    # Verify initial param x0 is stored
    initial_x0 = model['peak_models']['0']['Lorentzian']['x0']['value']
    
    engine = TensorFittingEngine()
    
    t0 = time.perf_counter()
    results = engine.fit_spectra(
        spectra=spectra,
        fit_model=model,
        fit_params={'method': 'leastsq', 'xtol': 1e-4, 'max_ite': 200, 'coef_noise': 1},
        apply_model_to_spectra=True
    )
    total_time = time.perf_counter() - t0
    
    # Performance: Should be under 3 seconds (Typical optimized run ~0.5s)
    assert total_time < 3.0, f"MoS2_map fitting is too slow: {total_time:.2f}s"
    
    # Verify noise masking correctness
    noise_pixels = 0
    x0_restored = 0
    for r in results:
        if r.params['m01_ampli'].value == 0:
            noise_pixels += 1
            if r.params['m01_x0'].value == initial_x0:
                x0_restored += 1
                
    assert noise_pixels > 500, "Should have detected a significant amount of noise pixels"
    assert x0_restored == noise_pixels, "x0 must be restored to initial guess for all noise pixels"


@pytest.mark.slow
def test_3_wdf_map_performance(wdf_map_data):
    """
    Benchmark 3: Medium-large WDF map.
    3,721 spectra, 1 Lorentzian.
    """
    spectra, model = wdf_map_data
    assert len(spectra) == 3721
    
    engine = TensorFittingEngine()
    
    t0 = time.perf_counter()
    results = engine.fit_spectra(
        spectra=spectra,
        fit_model=model,
        fit_params={'method': 'leastsq', 'xtol': 1e-4, 'max_ite': 200, 'coef_noise': 1},
        apply_model_to_spectra=True
    )
    total_time = time.perf_counter() - t0
    
    # Performance: Should be under 6 seconds
    assert total_time < 6.0, f"WDF map fitting is too slow: {total_time:.2f}s"


@pytest.mark.slow
def test_4_d4_map_performance(d4_map_data):
    """
    Benchmark 4: .maps workspace.
    1,681 spectra, 2 Lorentzians.
    """
    spectra, model = d4_map_data
    assert len(spectra) == 1681
    
    engine = TensorFittingEngine()
    
    t0 = time.perf_counter()
    results = engine.fit_spectra(
        spectra=spectra,
        fit_model=model,
        fit_params={'method': 'leastsq', 'xtol': 1e-4, 'max_ite': 200, 'coef_noise': 1},
        apply_model_to_spectra=True
    )
    total_time = time.perf_counter() - t0
    
    # Performance: Should be under 4 seconds
    assert total_time < 4.0, f"D4 map fitting is too slow: {total_time:.2f}s"


@pytest.mark.slow
def test_5_mos2_wafers_multipeak_scaling(mos2_wafers_data):
    """
    Benchmark 5: High parameter count (.maps workspace).
    196 spectra, 6 Lorentzians (K=18).
    Ensures that Cholesky solver handles K=18 efficiently.
    """
    spectra, model = mos2_wafers_data
    assert len(spectra) == 196
    
    engine = TensorFittingEngine()
    
    t0 = time.perf_counter()
    results = engine.fit_spectra(
        spectra=spectra,
        fit_model=model,
        fit_params={'method': 'leastsq', 'xtol': 1e-4, 'max_ite': 200, 'coef_noise': 1},
        apply_model_to_spectra=True
    )
    total_time = time.perf_counter() - t0
    
    # Performance: Old linear algebra took ~3.7s for this. 
    # Optimized Cholesky takes ~1.0s. Setting threshold at 3.0s to catch regressions.
    assert total_time < 3.0, f"MoS2_wafers (K=18) fitting is too slow: {total_time:.2f}s"
    
    # Verify convergence rate isn't severely impacted by max() vs mean() criteria
    success_count = sum(1 for r in results if r.success)
    assert success_count > (len(spectra) * 0.5), "At least 50% of the complex 6-peak model should converge"
