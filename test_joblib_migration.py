#!/usr/bin/env python3
"""
Quick test script to verify joblib migration works correctly.
Tests both sequential (ncpus=1) and parallel (ncpus>1) fitting.
"""
import sys
import time
from pathlib import Path

# Add spectroview to path
sys.path.insert(0, str(Path(__file__).parent))

from spectroview.model.m_spectra import MSpectra
from spectroview.model.m_spectrum import MSpectrum
import numpy as np


def create_test_spectrum(i):
    """Create a simple test spectrum with a Gaussian-like peak."""
    x = np.linspace(100, 1000, 200)
    # Create a peak at position depending on i
    peak_pos = 500 + (i % 10) * 10
    y = 10 * np.exp(-((x - peak_pos) ** 2) / (2 * 50**2)) + 0.1 * np.random.randn(len(x))
    
    spectrum = MSpectrum()
    spectrum.x0 = x
    spectrum.y0 = y
    spectrum.fname = f"test_spectrum_{i}"
    return spectrum


def test_joblib_fitting():
    """Test joblib-based fitting."""
    print("=" * 60)
    print("Testing Joblib Migration")
    print("=" * 60)
    
    # Create test spectra
    n_spectra = 20
    print(f"\n1. Creating {n_spectra} test spectra...")
    spectra = MSpectra()
    for i in range(n_spectra):
        spectrum = create_test_spectrum(i)
        spectra.append(spectrum)
    print(f"   ✓ Created {len(spectra)} spectra")
    
    # Create a simple fit model
    print("\n2. Creating fit model...")
    fit_model = {
        'range_min': 200,
        'range_max': 800,
        'normalize': False,
        'baseline': {'mode': 'Linear', 'attached': True},
        'peak_models': {},  # Empty dict (fitspy expects dict, not list)
        'peak_labels': []
    }
    
    # Test sequential fitting (ncpus=1)
    print("\n3. Testing SEQUENTIAL fitting (ncpus=1)...")
    start = time.time()
    try:
        spectra.apply_model(fit_model, ncpus=1, show_progressbar=False)
        seq_time = time.time() - start
        print(f"   ✓ Sequential fitting completed in {seq_time:.2f}s")
        print(f"   ✓ Throughput: {n_spectra/seq_time:.1f} spectra/sec")
    except Exception as e:
        print(f"   ✗ Sequential fitting FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Reset spectra for parallel test
    spectra = MSpectra()
    for i in range(n_spectra):
        spectrum = create_test_spectrum(i)
        spectra.append(spectrum)
    
    # Test parallel fitting with joblib (ncpus>1)
    print("\n4. Testing PARALLEL fitting with joblib (ncpus=4)...")
    start = time.time()
    try:
        spectra.apply_model(fit_model, ncpus=4, show_progressbar=False)
        par_time = time.time() - start
        print(f"   ✓ Parallel fitting completed in {par_time:.2f}s")
        print(f"   ✓ Throughput: {n_spectra/par_time:.1f} spectra/sec")
        
        if seq_time > 0:
            speedup = seq_time / par_time
            print(f"   ✓ Speedup: {speedup:.2f}x")
            
    except Exception as e:
        print(f"   ✗ Parallel fitting FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_joblib_fitting()
    sys.exit(0 if success else 1)
