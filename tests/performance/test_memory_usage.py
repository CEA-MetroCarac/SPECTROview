"""Practical memory-usage regression checks for the Tensor Fit Engine.

Uses stdlib tracemalloc (cross-platform, no extra dependency) to catch a
gross memory blow-up (e.g. an accidental O(N^2) allocation, or a cache that
never gets released) rather than to finely tune bytes-per-spectrum. Runs
VBFengine.fit_spectra() directly (bypassing VMWorkspaceSpectra/VBFthread) so
the measurement isolates the engine itself.

The dominant term is the per-iteration Jacobian tensor J: (N, M, K_free)
float64 -- that shape (and therefore the peak memory footprint) is reached
within the first few Levenberg-Marquardt iterations, so max_ite is capped
low here purely to keep tracemalloc's severe per-allocation instrumentation
overhead bounded (a full 200-iteration fit under tracemalloc takes minutes,
not seconds); it does not change what's being measured.
"""
import tracemalloc

import numpy as np
import pytest

from spectroview.model.m_io import load_map_file
from spectroview.fit_engine.vbf_engine import VBFengine

pytestmark = pytest.mark.slow


def _get_xy_from_map(df):
    wavenumbers = [float(c) for c in df.columns[2:]]
    x = np.array(wavenumbers, dtype=np.float64)
    Y = df.iloc[:, 2:].to_numpy(dtype=np.float64)
    return x, Y


def _load_fit_model(json_path):
    import json
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("0", data)


def _peak_traced_memory_mb(fn):
    tracemalloc.start()
    try:
        result = fn()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / (1024 * 1024)


class TestMoS2MapMemory:
    def test_peak_memory_within_budget(self, mos2_map_txt, mos2_map_json):
        if not (mos2_map_txt.exists() and mos2_map_json.exists()):
            pytest.skip("MoS2 benchmark data not present")

        map_df = load_map_file(mos2_map_txt)
        x, Y = _get_xy_from_map(map_df)
        fit_model = _load_fit_model(mos2_map_json)
        N, M = Y.shape
        K_free = 9  # 3 Lorentzian peaks x 3 free params each
        fit_params = {**fit_model.get("fit_params", {}), "max_ite": 10}

        def _fit():
            engine = VBFengine()
            return engine.fit_spectra(x=x, Y=Y, fit_model=fit_model, fit_params=fit_params)

        result, peak_mb = _peak_traced_memory_mb(_fit)
        p_full, success, r2, best_fits, y_peaks, names = result
        assert success.shape[0] == N  # sanity: the fit actually ran

        jacobian_estimate_mb = N * M * K_free * 8 / (1024 * 1024)
        budget_mb = max(200.0, jacobian_estimate_mb * 8)  # generous 8x margin
        assert peak_mb < budget_mb, (
            f"Peak traced memory {peak_mb:.1f}MB exceeds budget {budget_mb:.1f}MB "
            f"(Jacobian estimate: {jacobian_estimate_mb:.1f}MB)"
        )


class TestCLMapMemoryScaling:
    """Uses a 2048-row subset (not the full 16384) to keep this practical
    check fast while still exercising realistic tensor sizes."""

    def test_peak_memory_within_budget_for_subset(self, cl_map_txt, cl_map_json):
        if not (cl_map_txt.exists() and cl_map_json.exists()):
            pytest.skip("CL map benchmark data not present")

        map_df = load_map_file(cl_map_txt)
        x, Y_full = _get_xy_from_map(map_df)
        Y = Y_full[:2048].copy()
        fit_model = _load_fit_model(cl_map_json)
        N, M = Y.shape
        K_free = 3  # single Lorentzian, all params free
        fit_params = {**fit_model.get("fit_params", {}), "max_ite": 10}

        def _fit():
            engine = VBFengine()
            return engine.fit_spectra(x=x, Y=Y, fit_model=fit_model, fit_params=fit_params)

        result, peak_mb = _peak_traced_memory_mb(_fit)
        p_full, success, r2, best_fits, y_peaks, names = result
        assert success.shape[0] == N

        jacobian_estimate_mb = N * M * K_free * 8 / (1024 * 1024)
        budget_mb = max(200.0, jacobian_estimate_mb * 8)
        assert peak_mb < budget_mb, (
            f"Peak traced memory {peak_mb:.1f}MB exceeds budget {budget_mb:.1f}MB "
            f"(Jacobian estimate: {jacobian_estimate_mb:.1f}MB)"
        )
