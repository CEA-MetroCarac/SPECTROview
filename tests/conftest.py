"""Pytest configuration and shared fixtures for the SPECTROview test suite.

Provides:
- An isolated QSettings backend so tests never touch the real user registry.
- A session QApplication for anything that touches Qt widgets/signals.
- Paths to real example/benchmark data files.
- Synthetic spectrum/map/fit-model builders shared by unit, integration and
  performance tests (kept here, not in a helper module, so every test file
  gets them via normal fixture discovery without import-path juggling).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the repository root is importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Render Qt widgets offscreen (no real native windows) during tests. Must be
# set before QApplication is constructed. Without this, a full test run
# creates enough real, natively-backed widgets (many VGraph canvases,
# range sliders, etc. across hundreds of tests in one process) to exhaust
# the platform's native window-handle budget and segfault -- setdefault()
# so a developer/CI setting a different platform explicitly still wins.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from spectroview.model.m_settings import MSettings
from spectroview.fit_engine.models import BATCHED_MODELS


# ═══════════════════════════════════════════════════════════════════════════
# Qt / settings isolation
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path, monkeypatch):
    """Redirect every QSettings(org, app) pair to a throwaway, per-test INI file.

    MSettings() constructs QSettings("CEA-Leti", "SPECTROview") directly. On
    this platform, QSettings.setDefaultFormat(IniFormat) alone does NOT change
    the format used by the (organization, application) convenience
    constructor -- verified empirically, it keeps resolving to NativeFormat
    (the real Windows registry) regardless. Only the fully-explicit
    QSettings(format, scope, organization, application) constructor honors a
    non-native format. So instead of relying on the default-format switch, we
    monkeypatch the `QSettings` name inside m_settings.py itself to a factory
    that always goes through the explicit form. Function-scoped (not
    session-scoped) so settings written by one test (e.g. save_fit_settings)
    can never leak into another test's "defaults" expectations.
    """
    ini_dir = tmp_path / "qsettings"
    ini_dir.mkdir()
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(ini_dir))
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, str(ini_dir))

    import spectroview.model.m_settings as m_settings_module

    def _isolated_qsettings(organization, application, *args, **kwargs):
        return QSettings(QSettings.IniFormat, QSettings.UserScope, organization, application)

    monkeypatch.setattr(m_settings_module, "QSettings", _isolated_qsettings)
    yield


@pytest.fixture(scope="session")
def qapp():
    """Session-wide QApplication for tests that touch Qt widgets/signals."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture(autouse=True)
def _release_qt_widgets_between_tests():
    """Force a GC pass (+ pending Qt deleteLater()s) after every test.

    Real GUI tests each construct several native-backed widgets (VGraph's
    matplotlib canvas, CustomizeAxis's range sliders, ...) as plain local
    variables with no explicit teardown; left to Python's lazy GC, hundreds
    of them can stay alive simultaneously across a full run and exhaust the
    platform's native surface budget (a real crash seen in practice: a
    segfault deep inside FigureCanvasQTAgg.__init__ once enough dead-but-
    not-yet-collected canvases had piled up). Prompting collection after
    each test keeps the live set small instead.
    """
    yield
    import gc
    from PySide6.QtCore import QEvent
    gc.collect()
    app = QApplication.instance()
    if app is not None:
        app.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        gc.collect()
        app.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


@pytest.fixture
def settings(qapp):
    """A fresh MSettings backed by the isolated QSettings store."""
    return MSettings()


# ═══════════════════════════════════════════════════════════════════════════
# Path fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def examples_dir(project_root):
    return project_root / "examples"


@pytest.fixture(scope="session")
def bench_dir(examples_dir):
    return examples_dir / "fit_benchmarking_data"


@pytest.fixture
def temp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


# ── Single-spectrum example files ───────────────────────────────────────────

@pytest.fixture(scope="session")
def single_spectrum_file(bench_dir):
    return bench_dir / "spectrum1_1ML.txt"


@pytest.fixture(scope="session")
def multiple_spectra_files(bench_dir):
    return [
        bench_dir / "spectrum1_1ML.txt",
        bench_dir / "spectrum2_1ML.txt",
        bench_dir / "spectrum3_3ML.txt",
    ]


# ── Map example files ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def map_2d_file(bench_dir):
    return bench_dir / "2Dmap_Si.txt"


@pytest.fixture(scope="session")
def wafer_file(bench_dir):
    return bench_dir / "wafer4_newformat.csv"


@pytest.fixture(scope="session")
def wafer_process1_file(bench_dir):
    return bench_dir / "wafer1_process1.csv"


@pytest.fixture(scope="session")
def wdf_map_file(bench_dir):
    return bench_dir / "3_3721map.wdf"


# ── Fit model files ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def predefined_fit_models_dir(bench_dir):
    return bench_dir / "predefined_fit_models"


@pytest.fixture(scope="session")
def fit_model_si_file(predefined_fit_models_dir):
    return predefined_fit_models_dir / "fit_model_Si_.json"


# ── Saved workspace files (real, checked-in) ────────────────────────────────

@pytest.fixture(scope="session")
def legacy_spectra_workspace(bench_dir):
    """Legacy raw-JSON .spectra workspace (format v1)."""
    return bench_dir / "baseline_features_2.spectra"


@pytest.fixture(scope="session")
def legacy_fano_spectra_workspace(bench_dir):
    return bench_dir / "test_fano_shape.spectra"


@pytest.fixture(scope="session")
def zip_maps_workspace(bench_dir):
    """Modern ZIP-streamed .maps workspace (format v2)."""
    return bench_dir / "batch_several_wafermap.maps"


# ── Dataframe example files (Graphs workspace, used sparingly here) ────────

@pytest.fixture(scope="session")
def dataframe_excel_file(examples_dir):
    return examples_dir / "datasets_for_plotting" / "dataset_Excel.xlsx"


# ── Benchmark datasets (performance/regression suite) ───────────────────────

@pytest.fixture(scope="session")
def cl_map_txt(bench_dir):
    return bench_dir / "1_CL_map.txt"


@pytest.fixture(scope="session")
def cl_map_json(bench_dir):
    return bench_dir / "1_CL_map.json"


@pytest.fixture(scope="session")
def mos2_map_txt(bench_dir):
    return bench_dir / "2_MoS2_map.txt"


@pytest.fixture(scope="session")
def mos2_map_json(bench_dir):
    return bench_dir / "2_fit_MoS2map_NEW.json"


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data builders
# ═══════════════════════════════════════════════════════════════════════════

def _peak_param(value, vmin, vmax, vary=True, expr=None):
    return {"value": float(value), "min": vmin, "max": vmax, "vary": vary, "expr": expr}


@pytest.fixture
def make_peak_hints():
    """Factory: canonical peak-parameter hints dict for a given shape.

    Returns {param_name: {"value", "min", "max", "vary", "expr"}} using each
    shape's canonical parameter order/names from BATCHED_MODELS, seeded with
    physically reasonable defaults so the resulting fit_model is directly
    fittable (bounds bracket the seed value).
    """
    def _make(shape: str, x0=500.0, ampli=100.0, fwhm=8.0, **overrides):
        if shape not in BATCHED_MODELS:
            raise ValueError(f"Unknown shape: {shape}")
        _, _, canonical = BATCHED_MODELS[shape]
        hints = {}
        for pname in canonical:
            if pname == "ampli":
                hints[pname] = _peak_param(ampli, 0.0, ampli * 1000)
            elif pname in ("fwhm", "fwhm_l", "fwhm_r"):
                hints[pname] = _peak_param(fwhm, 1e-3, 200.0)
            elif pname == "x0":
                hints[pname] = _peak_param(x0, x0 - 50, x0 + 50)
            elif pname == "alpha":
                hints[pname] = _peak_param(0.5, 0.0, 1.0)
            elif pname == "q":
                hints[pname] = _peak_param(5.0, -200.0, 200.0)
            elif pname in ("A", "A1", "A2"):
                hints[pname] = _peak_param(ampli, 0.0, ampli * 1000)
            elif pname in ("tau", "tau1", "tau2"):
                hints[pname] = _peak_param(fwhm, 1e-3, 500.0)
            elif pname == "B":
                hints[pname] = _peak_param(0.0, -ampli, ampli)
            else:
                hints[pname] = _peak_param(1.0, -np.inf, np.inf)
        for key, val in overrides.items():
            if key in hints:
                hints[key]["value"] = val
        return {shape: hints}
    return _make


@pytest.fixture
def make_fit_model(make_peak_hints):
    """Factory: build a full fit_model dict from a list of (shape, kwargs) peaks."""
    def _make(peaks, fit_params=None, range_min=None, range_max=None, baseline=None,
              peak_labels=None):
        peak_models = {}
        for i, (shape, kwargs) in enumerate(peaks):
            peak_models[str(i)] = make_peak_hints(shape, **kwargs)
        return {
            "fit_params": fit_params or {
                "fit_negative": False, "max_ite": 200, "xtol": 1e-4,
                "ftol": 1e-4, "coef_noise": 0.0,
            },
            "range_min": range_min,
            "range_max": range_max,
            "baseline": baseline,
            "peak_labels": peak_labels or [f"Peak{i + 1}" for i in range(len(peaks))],
            "peak_models": peak_models,
        }
    return _make


@pytest.fixture
def synth_x():
    """Shared 400-point wavenumber axis, 300-700 cm^-1."""
    return np.linspace(300.0, 700.0, 400)


def _eval_shape(shape, x, params_dict):
    """Evaluate one canonical peak shape at its 'value's for a single spectrum."""
    eval_fn, _, canonical = BATCHED_MODELS[shape]
    p = np.array([[params_dict[name]["value"] for name in canonical]])
    return eval_fn(x, p)[0]


@pytest.fixture
def make_synthetic_spectrum(synth_x):
    """Factory: build a single noise-free (or noisy) synthetic spectrum from peaks.

    peaks: list of (shape, kwargs) as accepted by make_peak_hints (via the
    make_fit_model factory's convention), so the same peak list can be used
    both to generate data and to build the fit_model that should recover it.
    """
    def _make(peaks_dict_list, x=None, noise_std=0.0, seed=0):
        x = synth_x if x is None else x
        y = np.zeros_like(x)
        for shape, hints in peaks_dict_list:
            y = y + _eval_shape(shape, x, hints)
        if noise_std > 0:
            rng = np.random.default_rng(seed)
            y = y + rng.normal(0.0, noise_std, size=y.shape)
        return x, y
    return _make


@pytest.fixture
def make_synthetic_map(make_peak_hints):
    """Factory: build a (x, Y, coords, fnames) synthetic hyperspectral map.

    Every row uses the same ground-truth peaks (optionally jittered) so
    fitted parameters can be checked against a known answer.
    """
    def _make(shape="Lorentzian", n_spectra=25, n_points=300,
              x0=500.0, ampli=100.0, fwhm=8.0, noise_std=0.5, seed=0,
              x_range=(300.0, 700.0)):
        rng = np.random.default_rng(seed)
        x = np.linspace(x_range[0], x_range[1], n_points)
        eval_fn, _, canonical = BATCHED_MODELS[shape]

        base = {"ampli": ampli, "fwhm": fwhm, "x0": x0, "fwhm_l": fwhm, "fwhm_r": fwhm,
                "alpha": 0.5, "q": 5.0, "A": ampli, "tau": fwhm, "B": 0.0,
                "A1": ampli * 0.6, "tau1": fwhm, "A2": ampli * 0.4, "tau2": fwhm * 2}
        jitter_ampli = ampli * rng.uniform(0.7, 1.3, size=n_spectra)
        jitter_x0 = x0 + rng.uniform(-1.0, 1.0, size=n_spectra)

        params = np.zeros((n_spectra, len(canonical)))
        for row in range(n_spectra):
            row_vals = dict(base)
            row_vals["ampli"] = jitter_ampli[row]
            row_vals["x0"] = jitter_x0[row]
            row_vals["A"] = jitter_ampli[row]
            for j, pname in enumerate(canonical):
                params[row, j] = row_vals[pname]

        Y = eval_fn(x, params)
        if noise_std > 0:
            Y = Y + rng.normal(0.0, noise_std, size=Y.shape)

        coords = np.column_stack([np.arange(n_spectra, dtype=float),
                                   np.zeros(n_spectra)])
        fnames = [f"synthmap_{i}" for i in range(n_spectra)]
        return x, Y, coords, fnames, params, canonical
    return _make


# ═══════════════════════════════════════════════════════════════════════════
# Misc sample data (Graphs-adjacent, kept for completeness/back-compat)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_dataframe():
    data = {
        "X": np.arange(10),
        "Y": np.random.randn(10),
        "Category": ["A", "B"] * 5,
        "Value": np.random.uniform(0, 100, 10),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_map_dataframe():
    x_coords, y_coords = [], []
    for i in range(5):
        for j in range(5):
            x_coords.append(i)
            y_coords.append(j)
    wavelengths = np.linspace(100, 500, 20)
    data = {"X": x_coords, "Y": y_coords}
    for wl in wavelengths:
        data[str(wl)] = np.random.uniform(0, 100, 25)
    return pd.DataFrame(data)
