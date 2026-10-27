"""Regression tests for workspace clearing in Spectra and Maps workspaces."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def _shared_clear_window(qapp):
    import spectroview.main as main_mod
    original_prewarm = getattr(main_mod.Main, "_prewarm_ai_chat", None)
    main_mod.Main._prewarm_ai_chat = lambda self: None
    window = main_mod.Main()
    yield window
    window.close()
    if original_prewarm:
        main_mod.Main._prewarm_ai_chat = original_prewarm


@pytest.fixture
def main_window(_shared_clear_window, qapp):
    # Ensure all workspaces are clean before test starts
    _shared_clear_window.v_spectra_workspace.clear_workspace()
    _shared_clear_window.v_maps_workspace.clear_workspace()
    _shared_clear_window.v_graphs_workspace.clear_workspace()
    qapp.processEvents()
    return _shared_clear_window


def test_clear_spectra_workspace_clears_fit_results(main_window, qapp):
    """Test that Clear current workspace in Spectra workspace clears Fit Results table and VM df_fit_results."""
    ws = main_window.v_spectra_workspace
    vm = ws.vm

    # Add a synthetic spectrum
    x = np.linspace(100, 1000, 100)
    y = np.sin(x).reshape(1, -1)
    vm.store.add_map(
        name="spec1",
        x0=x,
        Y0=y,
        coords=np.array([[0.0, 0.0]]),
        fnames=["spec1"],
        is_active=np.array([True]),
    )
    vm.selected_fnames = ["spec1"]
    vm._emit_selected_spectra()
    qapp.processEvents()

    # Simulate fit results
    df_results = pd.DataFrame({"Filename": ["spec1"], "P1_ampli": [10.5], "R2": [0.99]})
    vm.df_fit_results = df_results
    vm.fit_results_updated.emit(df_results)
    qapp.processEvents()

    assert ws.v_fit_results.df_table.model.rowCount() == 1
    assert vm.df_fit_results is not None

    # Clear workspace
    ws.clear_workspace()
    qapp.processEvents()

    # Assert Fit Results table is cleared
    assert vm.df_fit_results is None
    assert ws.v_fit_results.df_table.model.rowCount() == 0
    assert ws.v_fit_results.df_table.model._df.empty


def test_clear_maps_workspace_clears_viewer_peak_table_and_fit_results(main_window, qapp):
    """Test that Clear current workspace in Maps workspace clears spectra viewer, peak table, and fit results."""
    ws = main_window.v_maps_workspace
    vm = ws.vm

    # Add a synthetic map
    x = np.linspace(100, 1000, 50)
    y = np.ones((4, 50))
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    fnames = [f"map1_({c[0]:.1f}, {c[1]:.1f})" for c in coords]
    
    vm.store.add_map(
        name="map1",
        x0=x,
        Y0=y,
        coords=coords,
        fnames=fnames,
        is_active=np.ones(4, dtype=bool),
    )
    # Add dummy map df
    cols = ["X", "Y"] + [str(val) for val in x]
    map_df_data = np.hstack([coords, y])
    map_df = pd.DataFrame(map_df_data, columns=cols)
    vm.maps["map1"] = map_df
    vm.current_map_name = "map1"
    vm.current_map_df = map_df
    vm.selected_fnames = [fnames[0]]

    # Add fit model with a peak to map data
    md = vm.store.get_map_data("map1")
    md.fit_model = {
        "fit_params": {"max_ite": 200, "xtol": 1e-5, "ftol": 1e-5},
        "peak_labels": ["Peak1"],
        "peak_models": {"0": {"Lorentzian": {
            "ampli": {"value": 80.0, "min": 0.0, "max": 1e6, "vary": True, "expr": None},
            "fwhm": {"value": 4.8, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
            "x0": {"value": 500.0, "min": 470, "max": 530, "vary": True, "expr": None},
        }}},
    }

    # Emit selected spectra so viewer and peak table get populated
    vm._emit_selected_spectra()
    qapp.processEvents()

    assert ws.v_spectra_viewer._tensor_data is not None
    assert ws.v_fit_model_builder.peak_table._spectrum is not None

    # Simulate fit results
    df_results = pd.DataFrame({"Filename": [fnames[0]], "P1_ampli": [10.0], "R2": [0.98]})
    vm.df_fit_results = df_results
    vm.fit_results_updated.emit(df_results)
    qapp.processEvents()

    assert ws.v_fit_results.df_table.model.rowCount() == 1

    # Clear workspace
    ws.clear_workspace()
    qapp.processEvents()

    # Verify everything is cleared
    assert vm.df_fit_results is None
    assert ws.v_fit_results.df_table.model.rowCount() == 0
    assert ws.v_fit_results.df_table.model._df.empty
    assert ws.v_spectra_viewer._tensor_data is None
    assert ws.v_fit_model_builder.peak_table._spectrum is None
    assert ws.v_fit_model_builder.peak_table.main_layout.count() == 0
    assert ws.v_map_viewer.map_df is None
    assert ws.lbl_count.text() == "0 spectra"


def test_clear_workspace_via_menu_action(main_window, qapp):
    """Test clearing current workspace via Main.clear_workspace() (menu bar action)."""
    # 1. Test Maps workspace active
    main_window.tabWidget.setCurrentWidget(main_window.v_maps_workspace)
    ws_maps = main_window.v_maps_workspace
    vm_maps = ws_maps.vm

    x = np.linspace(100, 1000, 20)
    y = np.ones((2, 20))
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    fnames = ["m1_(0.0, 0.0)", "m1_(1.0, 1.0)"]
    vm_maps.store.add_map(name="m1", x0=x, Y0=y, coords=coords, fnames=fnames, is_active=np.ones(2, dtype=bool))
    vm_maps.maps["m1"] = pd.DataFrame(np.hstack([coords, y]))
    vm_maps.current_map_name = "m1"
    vm_maps.selected_fnames = [fnames[0]]
    vm_maps._emit_selected_spectra()

    df_results = pd.DataFrame({"Filename": [fnames[0]], "P1_ampli": [10.0], "R2": [0.98]})
    vm_maps.df_fit_results = df_results
    vm_maps.fit_results_updated.emit(df_results)
    qapp.processEvents()

    assert ws_maps.v_fit_results.df_table.model.rowCount() == 1
    assert ws_maps.v_spectra_viewer._tensor_data is not None

    # Trigger menu action clear
    main_window.clear_workspace()
    qapp.processEvents()

    assert vm_maps.df_fit_results is None
    assert ws_maps.v_fit_results.df_table.model.rowCount() == 0
    assert ws_maps.v_spectra_viewer._tensor_data is None
    assert ws_maps.v_fit_model_builder.peak_table._spectrum is None
    assert ws_maps.lbl_count.text() == "0 spectra"
