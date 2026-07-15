"""Fixtures for the performance/regression benchmark suite.

Runs the Tensor Fit Engine (VBFengine/VBFthread) against the two real
benchmark datasets named in the test-suite brief:

    1_CL_map.txt   + 1_CL_map.json          (16384 spectra x 1024 points, single Lorentzian)
    2_MoS2_map.txt + 2_fit_MoS2map_NEW.json (1520 spectra x 575 points, 3 Lorentzians)

Each dataset is fit exactly twice per test module (module-scoped, so the
expensive full-map fit runs only once per file across every test in that
module) so that both an accuracy/stability check and a same-process
reproducibility check can be made without re-paying the load+fit cost.

QSettings isolation here is deliberately independent from the per-test
autouse fixture in tests/conftest.py (that one is function-scoped and would
force this module-scoped fixture to re-run per test, defeating the point of
caching); this fixture isolates once per module instead.
"""
import json
import time

import numpy as np
import pandas as pd
import pytest

from spectroview.model.m_io import load_map_file
from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra
from spectroview.fit_engine.vbf_thread import VBFthread


@pytest.fixture(scope="module", autouse=True)
def _isolate_qsettings_for_module(tmp_path_factory):
    from PySide6.QtCore import QSettings
    import spectroview.model.m_settings as m_settings_module

    mp = pytest.MonkeyPatch()
    ini_dir = tmp_path_factory.mktemp("perf_qsettings")
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(ini_dir))
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, str(ini_dir))

    def _isolated_qsettings(organization, application, *args, **kwargs):
        return QSettings(QSettings.IniFormat, QSettings.UserScope, organization, application)

    mp.setattr(m_settings_module, "QSettings", _isolated_qsettings)
    yield
    mp.undo()


def _get_xy_from_map(df: pd.DataFrame):
    wavenumbers = [float(c) for c in df.columns[2:]]
    x = np.array(wavenumbers, dtype=np.float64)
    Y = df.iloc[:, 2:].to_numpy(dtype=np.float64)
    return x, Y


def _load_fit_model(json_path) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("0", data)


def _run_full_map_fit(run_label, txt_path, json_path):
    """Load a real benchmark map, apply its fit model, run VBFthread
    synchronously over every spectrum, and time the fit step alone."""
    from spectroview.model.m_settings import MSettings

    map_df = load_map_file(txt_path)
    x, Y = _get_xy_from_map(map_df)
    coords = map_df.iloc[:, :2].to_numpy()
    fnames = [f"{run_label}_{i}" for i in range(len(Y))]

    vm = VMWorkspaceSpectra(MSettings())
    vm.store.add_map(run_label, x, Y, coords, fnames)
    md = vm.store.get_map_data(run_label)
    fit_model = _load_fit_model(json_path)
    vm._apply_fit_model_to_mapdata(md, fit_model)

    tasks = [{"map_name": run_label, "indices": np.arange(len(Y)), "fit_model": md.fit_model}]
    thread = VBFthread(vm.store, tasks)
    t0 = time.perf_counter()
    thread.run()
    elapsed = time.perf_counter() - t0
    return {"md": md, "elapsed": elapsed, "n_spectra": len(Y), "n_points": Y.shape[1]}


def _two_runs(name, txt_path, json_path):
    if not (txt_path.exists() and json_path.exists()):
        pytest.skip(f"benchmark data for {name} not present: {txt_path}, {json_path}")
    run1 = _run_full_map_fit(f"{name}_run1", txt_path, json_path)
    run2 = _run_full_map_fit(f"{name}_run2", txt_path, json_path)
    return {"run1": run1, "run2": run2}


@pytest.fixture(scope="module")
def cl_map_benchmark(cl_map_txt, cl_map_json):
    return _two_runs("cl_map", cl_map_txt, cl_map_json)


@pytest.fixture(scope="module")
def mos2_map_benchmark(mos2_map_txt, mos2_map_json):
    return _two_runs("mos2_map", mos2_map_txt, mos2_map_json)
