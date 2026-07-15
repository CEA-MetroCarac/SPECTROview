"""Integration tests for fit_engine/vbf_thread.py - VBFthread against a real
SpectraStore, exercising all three task-dict shapes it dispatches on:

  A. {"map_name", "indices", "fit_model"}                       - single map
  B. {"grouped_2d_maps", "map_boundaries", "fit_model"}          - mega-batch across maps
  C. {"map_names", "fit_model"}                                  - batch of 1-row "maps"

Run() executes synchronously when called directly (not via .start()), so no
QApplication/event loop is needed here -- this is the same pattern the
production VMWorkspaceSpectra/VMWorkspaceMaps use under the hood.
"""
import numpy as np
import pytest

from spectroview.model.spectra_store import SpectraStore
from spectroview.fit_engine.vbf_thread import VBFthread


def _lorentzian_fit_model(ampli=100.0, fwhm=6.0, x0=500.0, seed_offset=-2.0):
    return {
        "fit_params": {"max_ite": 200, "xtol": 1e-5, "ftol": 1e-5},
        "peak_models": {"0": {"Lorentzian": {
            "ampli": {"value": ampli * 0.8, "min": 0.0, "max": 1e6, "vary": True, "expr": None},
            "fwhm": {"value": fwhm * 0.8, "min": 1e-3, "max": 100.0, "vary": True, "expr": None},
            "x0": {"value": x0 + seed_offset, "min": x0 - 30, "max": x0 + 30, "vary": True, "expr": None},
        }}},
    }


def _lorentzian_map(store, name, n_spectra=6, m=200, x0=500.0, ampli=100.0, fwhm=6.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(300.0, 700.0, m)
    Y = np.empty((n_spectra, m), dtype=np.float32)
    for i in range(n_spectra):
        a = ampli * rng.uniform(0.8, 1.2)
        Y[i] = a / (1 + 4 * ((x - x0) / fwhm) ** 2)
    coords = np.column_stack([np.arange(n_spectra, dtype=float), np.zeros(n_spectra)])
    fnames = [f"{name}_{i}" for i in range(n_spectra)]
    store.add_map(name, x, Y, coords, fnames)
    return store.get_map_data(name)


class TestSingleMapTask:
    def test_fits_all_requested_indices(self):
        store = SpectraStore()
        md = _lorentzian_map(store, "map1", n_spectra=8, seed=1)
        md.Y = md.Y0.astype(np.float64).copy()
        md.x = md.x0.copy()

        tasks = [{"map_name": "map1", "indices": np.arange(8), "fit_model": _lorentzian_fit_model()}]
        VBFthread(store, tasks).run()

        assert md.fit_success.all()
        assert md.peak_params.shape == (8, 3)
        assert md.Y_bestfit is not None
        assert md.Y_peaks is not None and len(md.Y_peaks) == 1

    def test_fits_only_a_subset_of_indices(self):
        store = SpectraStore()
        md = _lorentzian_map(store, "map1", n_spectra=6, seed=2)
        md.Y = md.Y0.astype(np.float64).copy()
        md.x = md.x0.copy()

        tasks = [{"map_name": "map1", "indices": np.array([1, 3]), "fit_model": _lorentzian_fit_model()}]
        VBFthread(store, tasks).run()

        assert md.fit_success[1] and md.fit_success[3]
        assert not md.fit_success[0]  # untouched row stays at allocated default (False)

    def test_missing_map_is_skipped_without_crash(self):
        store = SpectraStore()
        tasks = [{"map_name": "does_not_exist", "indices": np.arange(3), "fit_model": _lorentzian_fit_model()}]
        VBFthread(store, tasks).run()  # must not raise


class TestGroupedMegaBatchAcrossMaps:
    def test_stacks_and_scatters_results_correctly(self):
        store = SpectraStore()
        md_a = _lorentzian_map(store, "mapA", n_spectra=4, x0=500.0, seed=3)
        md_b = _lorentzian_map(store, "mapB", n_spectra=5, x0=500.0, seed=4)
        for md in (md_a, md_b):
            md.Y = md.Y0.astype(np.float64).copy()
            md.x = md.x0.copy()

        tasks = [{
            "grouped_2d_maps": ["mapA", "mapB"],
            "map_boundaries": [("mapA", 0, 4), ("mapB", 4, 9)],
            "indices": np.arange(9),
            "fit_model": _lorentzian_fit_model(),
        }]
        VBFthread(store, tasks).run()

        assert md_a.fit_success.all()
        assert md_b.fit_success.all()
        assert md_a.peak_params.shape == (4, 3)
        assert md_b.peak_params.shape == (5, 3)

    def test_results_match_fitting_each_map_separately(self):
        store_grouped = SpectraStore()
        md_a = _lorentzian_map(store_grouped, "mapA", n_spectra=4, x0=500.0, seed=5)
        md_b = _lorentzian_map(store_grouped, "mapB", n_spectra=4, x0=500.0, seed=6)
        for md in (md_a, md_b):
            md.Y = md.Y0.astype(np.float64).copy()
            md.x = md.x0.copy()
        VBFthread(store_grouped, [{
            "grouped_2d_maps": ["mapA", "mapB"],
            "map_boundaries": [("mapA", 0, 4), ("mapB", 4, 8)],
            "indices": np.arange(8),
            "fit_model": _lorentzian_fit_model(),
        }]).run()

        store_separate = SpectraStore()
        md_a2 = _lorentzian_map(store_separate, "mapA", n_spectra=4, x0=500.0, seed=5)
        md_b2 = _lorentzian_map(store_separate, "mapB", n_spectra=4, x0=500.0, seed=6)
        for md in (md_a2, md_b2):
            md.Y = md.Y0.astype(np.float64).copy()
            md.x = md.x0.copy()
        VBFthread(store_separate, [
            {"map_name": "mapA", "indices": np.arange(4), "fit_model": _lorentzian_fit_model()},
            {"map_name": "mapB", "indices": np.arange(4), "fit_model": _lorentzian_fit_model()},
        ]).run()

        np.testing.assert_allclose(md_a.peak_params, md_a2.peak_params, atol=1e-6)
        np.testing.assert_allclose(md_b.peak_params, md_b2.peak_params, atol=1e-6)


class TestGroupedSingleSpectraBatch:
    def test_groups_by_matching_x_axis_length(self):
        store = SpectraStore()
        # Two single-spectrum "maps" (Spectra workspace convention) sharing
        # the same x-axis length -> batched together in one group.
        x = np.linspace(300, 700, 150)
        y1 = 100.0 / (1 + 4 * ((x - 500.0) / 6.0) ** 2)
        y2 = 80.0 / (1 + 4 * ((x - 480.0) / 5.0) ** 2)
        store.add_map("s1", x, y1[None, :].astype(np.float32), np.zeros((1, 2)), ["s1"])
        store.add_map("s2", x, y2[None, :].astype(np.float32), np.zeros((1, 2)), ["s2"])
        for name in ("s1", "s2"):
            md = store.get_map_data(name)
            md.Y = md.Y0.astype(np.float64).copy()
            md.x = md.x0.copy()

        tasks = [{"map_names": ["s1", "s2"], "indices": np.arange(2), "fit_model": _lorentzian_fit_model()}]
        VBFthread(store, tasks).run()

        md1 = store.get_map_data("s1")
        md2 = store.get_map_data("s2")
        assert md1.fit_success[0] and md2.fit_success[0]
        idx_x0 = md1.param_names.index("P1_x0")
        assert md1.peak_params[0, idx_x0] == pytest.approx(500.0, abs=0.1)
        assert md2.peak_params[0, idx_x0] == pytest.approx(480.0, abs=0.1)

    def test_mismatched_x_axis_lengths_still_fit_correctly_in_separate_groups(self):
        store = SpectraStore()
        x_short = np.linspace(300, 700, 100)
        x_long = np.linspace(300, 700, 250)
        y_short = 100.0 / (1 + 4 * ((x_short - 500.0) / 6.0) ** 2)
        y_long = 100.0 / (1 + 4 * ((x_long - 500.0) / 6.0) ** 2)
        store.add_map("short", x_short, y_short[None, :].astype(np.float32), np.zeros((1, 2)), ["short"])
        store.add_map("long", x_long, y_long[None, :].astype(np.float32), np.zeros((1, 2)), ["long"])
        for name in ("short", "long"):
            md = store.get_map_data(name)
            md.Y = md.Y0.astype(np.float64).copy()
            md.x = md.x0.copy()

        tasks = [{"map_names": ["short", "long"], "indices": np.arange(2), "fit_model": _lorentzian_fit_model()}]
        VBFthread(store, tasks).run()  # must not raise despite mismatched lengths

        assert store.get_map_data("short").fit_success[0]
        assert store.get_map_data("long").fit_success[0]


class TestSignalsAndErrorHandling:
    def test_progress_and_timings_signals_emitted(self):
        store = SpectraStore()
        md = _lorentzian_map(store, "map1", n_spectra=3, seed=7)
        md.Y = md.Y0.astype(np.float64).copy()
        md.x = md.x0.copy()

        progress_events = []
        timings_events = []
        thread = VBFthread(store, [{"map_name": "map1", "indices": np.arange(3),
                                     "fit_model": _lorentzian_fit_model()}])
        thread.progress_changed.connect(lambda *args: progress_events.append(args))
        thread.timings_ready.connect(lambda s: timings_events.append(s))
        thread.run()

        assert len(progress_events) >= 1
        assert progress_events[-1][0] == progress_events[-1][1] == 3  # (current, total) at completion
        assert len(timings_events) == 1
        assert "Total Fit time" in timings_events[0]

    def test_malformed_fit_model_does_not_crash_run(self):
        store = SpectraStore()
        md = _lorentzian_map(store, "map1", n_spectra=2, seed=8)
        md.Y = md.Y0.astype(np.float64).copy()
        md.x = md.x0.copy()

        bad_fit_model = {"peak_models": {"0": {"NotARealShape": {"ampli": {"value": 1.0}}}}}
        tasks = [{"map_name": "map1", "indices": np.arange(2), "fit_model": bad_fit_model}]
        VBFthread(store, tasks).run()  # exception is caught & logged, not raised

        assert md.peak_params is None  # write-back never happened for the failed task

    def test_stop_sets_cancelled_flag(self):
        store = SpectraStore()
        md = _lorentzian_map(store, "map1", n_spectra=2, seed=9)
        md.Y = md.Y0.astype(np.float64).copy()
        md.x = md.x0.copy()
        thread = VBFthread(store, [{"map_name": "map1", "indices": np.arange(2),
                                     "fit_model": _lorentzian_fit_model()}])
        assert thread._is_cancelled is False
        thread.stop()
        assert thread._is_cancelled is True
