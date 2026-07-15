"""Unit tests for model/spectra_store.py - SpectraStore, MapData, SpectrumProxy, BaselineProxy.

SpectraStore is pure Python/NumPy (no Qt), so these tests need no QApplication.
"""
import numpy as np
import pytest

from spectroview.model.spectra_store import SpectraStore, SpectrumProxy, BaselineProxy


def _add_simple_map(store, name="map1", n=5, m=20, seed=0):
    rng = np.random.default_rng(seed)
    x0 = np.linspace(300.0, 700.0, m)
    Y0 = rng.normal(50, 5, size=(n, m)).astype(np.float32)
    coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    fnames = [f"{name}_{i}" for i in range(n)]
    store.add_map(name, x0, Y0, coords, fnames)
    return store.get_map_data(name)


class TestAddMap:
    def test_registers_map_and_populates_defaults(self):
        store = SpectraStore()
        md = _add_simple_map(store)
        assert store.map_names == ["map1"]
        assert md.n_spectra == 5
        assert md.n_wavenumbers == 20
        assert md.is_active.all()
        assert md.colors == [None] * 5
        assert md.labels == [None] * 5

    def test_store_reports_total_spectra_across_maps(self):
        store = SpectraStore()
        _add_simple_map(store, "map1", n=5)
        _add_simple_map(store, "map2", n=3)
        assert store.n_spectra == 8
        assert set(store.map_names) == {"map1", "map2"}

    def test_x0_and_y0_are_copied_and_dtype_cast(self):
        store = SpectraStore()
        x0 = np.linspace(0, 1, 10).astype(np.float32)  # deliberately wrong dtype
        Y0 = np.ones((2, 10), dtype=np.float64)
        coords = np.zeros((2, 2))
        store.add_map("m", x0, Y0, coords, ["a", "b"])
        md = store.get_map_data("m")
        assert md.x0.dtype == np.float64
        assert md.Y0.dtype == np.float32

    def test_custom_is_active_colors_labels(self):
        store = SpectraStore()
        x0 = np.linspace(0, 1, 5)
        Y0 = np.ones((2, 5), dtype=np.float32)
        coords = np.zeros((2, 2))
        store.add_map("m", x0, Y0, coords, ["a", "b"],
                       is_active=np.array([True, False]),
                       colors=["#fff", "#000"], labels=["L1", "L2"])
        md = store.get_map_data("m")
        assert list(md.is_active) == [True, False]
        assert md.colors == ["#fff", "#000"]
        assert md.labels == ["L1", "L2"]


class TestRemoveReorderMaps:
    def test_remove_map(self):
        store = SpectraStore()
        _add_simple_map(store, "map1")
        store.remove_map("map1")
        assert store.map_names == []
        assert store.get_map_data("map1") is None

    def test_remove_nonexistent_map_is_noop(self):
        store = SpectraStore()
        store.remove_map("does_not_exist")  # must not raise

    def test_reorder_maps(self):
        store = SpectraStore()
        _add_simple_map(store, "a")
        _add_simple_map(store, "b")
        _add_simple_map(store, "c")
        store.reorder_maps([2, 0, 1])
        assert store.map_names == ["c", "a", "b"]

    def test_reorder_with_mismatched_length_is_noop(self):
        store = SpectraStore()
        _add_simple_map(store, "a")
        _add_simple_map(store, "b")
        store.reorder_maps([0])  # wrong length
        assert store.map_names == ["a", "b"]


class TestColorAccessors:
    def test_get_set_color(self):
        store = SpectraStore()
        _add_simple_map(store, "map1", n=3)
        store.set_color("map1", 1, "#123456")
        assert store.get_color("map1", 1) == "#123456"
        assert store.get_color("map1", 0) is None


class TestGetXyBatch:
    def test_uses_raw_when_no_processed_data(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=4)
        x, Y = store.get_xy_batch("map1", np.array([0, 2]))
        np.testing.assert_array_equal(x, md.x0)
        np.testing.assert_allclose(Y, md.Y0[[0, 2]].astype(np.float64))

    def test_uses_processed_when_available(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=4, m=10)
        md.x = md.x0[2:8].copy()
        md.Y = md.Y0[:, 2:8].copy()
        x, Y = store.get_xy_batch("map1", np.array([0, 1]))
        np.testing.assert_array_equal(x, md.x)
        assert Y.shape == (2, 6)


class TestBatchPreprocess:
    def test_range_cropping(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=3, m=100)
        store.batch_preprocess("map1", baseline_config={}, range_min=400.0, range_max=500.0)
        assert md.x.min() >= 400.0 - 5  # within one bin of the boundary
        assert md.x.max() <= 500.0 + 5

    def test_linear_attached_baseline_subtraction(self):
        store = SpectraStore()
        x0 = np.linspace(0, 100, 101)
        y_line = 2.0 * x0 + 3.0
        Y0 = np.tile(y_line, (3, 1)).astype(np.float32)
        store.add_map("m", x0, Y0, np.zeros((3, 2)), ["a", "b", "c"])
        md = store.get_map_data("m")

        baseline_config = {"mode": "Linear", "attached": True, "sigma": 0,
                            "points": [[0.0, 100.0], [0, 0]]}
        store.batch_preprocess("m", baseline_config)
        # A perfectly linear signal minus its own linear baseline -> ~0
        np.testing.assert_allclose(md.Y, np.zeros_like(md.Y), atol=1e-3)

    def test_empty_range_aborts_without_crashing(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=2, m=20)
        store.batch_preprocess("map1", baseline_config={}, range_min=99999, range_max=999999)
        # aborted -- md.x/md.Y left untouched (still None, since never set before)
        assert md.x is None

    def test_missing_map_is_noop(self):
        store = SpectraStore()
        store.batch_preprocess("nope", {})  # must not raise

    def test_clear_preprocess_reverts_to_raw(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=2, m=20)
        store.batch_preprocess("map1", baseline_config={}, range_min=400, range_max=500)
        assert md.x is not None
        store.clear_preprocess("map1")
        assert md.x is None and md.Y is None


class TestSetFitResults:
    def test_allocates_and_writes(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=4)
        peak_params = np.random.default_rng(0).normal(size=(4, 3))
        store.set_fit_results("map1", np.arange(4), peak_params,
                               success=np.array([True, True, False, True]),
                               r2=np.array([0.9, 0.8, 0.1, 0.95]),
                               param_names=["P1_ampli", "P1_fwhm", "P1_x0"],
                               fit_model={"peak_models": {}})
        assert md.has_fit_results()
        np.testing.assert_array_equal(md.peak_params, peak_params)
        assert md.param_names == ["P1_ampli", "P1_fwhm", "P1_x0"]

    def test_partial_indices_write_only_those_rows(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=4)
        store.set_fit_results("map1", np.arange(4), np.zeros((4, 1)),
                               np.zeros(4, bool), np.zeros(4), ["p"], {})
        store.set_fit_results("map1", np.array([1]), np.array([[7.0]]),
                               np.array([True]), np.array([0.5]), ["p"], {})
        assert md.peak_params[1, 0] == 7.0
        assert md.peak_params[0, 0] == 0.0

    def test_has_fit_results_false_before_fitting(self):
        store = SpectraStore()
        _add_simple_map(store, "map1")
        assert store.has_fit_results() is False
        assert store.has_fit_results("map1") is False

    def test_has_fit_results_scoped_to_map_name(self):
        store = SpectraStore()
        _add_simple_map(store, "map1", n=2)
        _add_simple_map(store, "map2", n=2)
        store.set_fit_results("map1", np.arange(2), np.zeros((2, 1)),
                               np.ones(2, bool), np.ones(2), ["p"], {})
        assert store.has_fit_results("map1") is True
        assert store.has_fit_results("map2") is False
        assert store.has_fit_results() is True


class TestBuildFitResultsDf:
    def _fitted_store(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=4)
        md.coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        fit_model = {"peak_models": {"0": {"Lorentzian": {
            "ampli": {"value": 1}, "fwhm": {"value": 1}, "x0": {"value": 1}}}}}
        store.set_fit_results(
            "map1", np.arange(4),
            peak_params=np.array([[100.0, 5.0, 500.0]] * 4),
            success=np.array([True, True, False, True]),
            r2=np.array([0.9, 0.8, 0.1, 0.95]),
            param_names=["P1_ampli", "P1_fwhm", "P1_x0"],
            fit_model=fit_model,
        )
        return store

    def test_only_converged_filters_failed_rows(self):
        store = self._fitted_store()
        df = store.build_fit_results_df("map1", only_converged=True)
        assert len(df) == 3  # row index 2 (success=False) excluded

    def test_includes_all_rows_when_only_converged_false(self):
        store = self._fitted_store()
        df = store.build_fit_results_df("map1", only_converged=False)
        assert len(df) == 4

    def test_peak_labels_rename_columns(self):
        store = self._fitted_store()
        df = store.build_fit_results_df("map1", peak_labels=["Si"], only_converged=False)
        assert "ampli_Si" in df.columns
        assert "fwhm_Si" in df.columns
        assert "x0_Si" in df.columns

    def test_area_column_added_for_lorentzian(self):
        store = self._fitted_store()
        df = store.build_fit_results_df("map1", peak_labels=["Si"], only_converged=False)
        assert "area_Si" in df.columns
        expected_area = np.pi * 100.0 * 5.0 / 2
        assert df["area_Si"].iloc[0] == pytest.approx(round(expected_area, 4))

    def test_no_fit_results_returns_none(self):
        store = SpectraStore()
        _add_simple_map(store, "map1")
        assert store.build_fit_results_df("map1") is None

    def test_missing_map_returns_none(self):
        store = SpectraStore()
        assert store.build_fit_results_df("nope") is None

    def test_wafer_map_type_adds_zone_and_quadrant(self):
        store = self._fitted_store()
        df = store.build_fit_results_df("map1", map_type="wafer_300mm", only_converged=False)
        assert "Zone" in df.columns
        assert "Quadrant" in df.columns


class TestNpzSerializationRoundTrip:
    def test_to_npz_dict_and_load_map_from_npz(self):
        store = self._make_fitted_store()
        arrays = store.to_npz_dict("map1")
        meta = store.to_metadata_dict("map1")

        restored = SpectraStore.load_map_from_npz(arrays, meta, "map1")
        md_orig = store.get_map_data("map1")
        md_restored = restored.get_map_data("map1")

        np.testing.assert_array_equal(md_orig.x0, md_restored.x0)
        np.testing.assert_array_equal(md_orig.Y0, md_restored.Y0)
        np.testing.assert_array_equal(md_orig.coords, md_restored.coords)
        np.testing.assert_array_equal(md_orig.peak_params, md_restored.peak_params)
        assert md_restored.param_names == md_orig.param_names
        assert md_restored.baseline_config == md_orig.baseline_config
        assert md_restored.range_min == md_orig.range_min

    def test_append_to_existing_store(self):
        store = self._make_fitted_store()
        arrays = store.to_npz_dict("map1")
        meta = store.to_metadata_dict("map1")

        other_store = SpectraStore()
        _add_simple_map(other_store, "other_map")
        SpectraStore.load_map_from_npz(arrays, meta, "map1", store=other_store)
        assert set(other_store.map_names) == {"other_map", "map1"}

    @staticmethod
    def _make_fitted_store():
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=3, m=15)
        md.baseline_config = {"mode": "Linear", "points": [[0], [0]]}
        md.range_min, md.range_max = 300.0, 690.0
        store.set_fit_results(
            "map1", np.arange(3), np.ones((3, 2)), np.ones(3, bool),
            np.full(3, 0.9), ["P1_ampli", "P1_x0"], {"peak_models": {}},
        )
        return store


class TestBaselineProxy:
    def test_defaults_when_config_none(self):
        proxy = BaselineProxy(None, is_subtracted=False)
        assert proxy.mode == ""
        assert proxy.coef == 5
        assert proxy.points == [[], []]
        assert proxy.is_subtracted is False

    def test_reads_from_config(self):
        proxy = BaselineProxy({"mode": "Linear", "coef": 7, "points": [[1], [2]]}, is_subtracted=True)
        assert proxy.mode == "Linear"
        assert proxy.coef == 7
        assert proxy.points == [[1], [2]]
        assert proxy.is_subtracted is True


class TestSpectrumProxy:
    def test_label_is_none_when_unset(self):
        # md.labels defaults to [None]*N; the fname fallback only triggers
        # when md.labels itself is empty/falsy, not when an entry is None.
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=2)
        proxy = SpectrumProxy(md, idx=0, fname="map1_0")
        assert proxy.label is None

    def test_label_falls_back_to_fname_when_labels_list_empty(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=2)
        md.labels = []
        proxy = SpectrumProxy(md, idx=0, fname="map1_0")
        assert proxy.label == "map1_0"

    def test_label_setter_extends_list(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=2)
        proxy = SpectrumProxy(md, idx=1, fname="map1_1")
        proxy.label = "My Label"
        assert md.labels[1] == "My Label"

    def test_color_setter_extends_list(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=2)
        proxy = SpectrumProxy(md, idx=1, fname="map1_1")
        proxy.color = "#abcdef"
        assert md.colors[1] == "#abcdef"
        assert proxy.color == "#abcdef"

    def test_baseline_proxy_reflects_scalar_is_subtracted(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=1)
        md.baseline_config = {"mode": "Linear"}
        md.is_baseline_subtracted = True
        proxy = SpectrumProxy(md, idx=0, fname="map1_0")
        assert proxy.baseline.is_subtracted is True

    def test_baseline_proxy_reflects_per_row_array_is_subtracted(self):
        store = SpectraStore()
        md = _add_simple_map(store, "map1", n=3)
        md.is_baseline_subtracted = np.array([True, False, True])
        proxy0 = SpectrumProxy(md, idx=0, fname="map1_0")
        proxy1 = SpectrumProxy(md, idx=1, fname="map1_1")
        assert proxy0.baseline.is_subtracted is True
        assert proxy1.baseline.is_subtracted is False
