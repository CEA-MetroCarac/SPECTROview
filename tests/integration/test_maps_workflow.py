"""End-to-end integration tests for the Maps workspace.

Drives VMWorkspaceMaps through a complete real-world workflow on a real
hyperspectral map file: load -> ROI crop -> baseline -> multi-peak model ->
batch fit the whole map -> collect results -> save -> reload.
"""
import numpy as np
import pytest

from spectroview.viewmodel.vm_workspace_maps import VMWorkspaceMaps


@pytest.fixture
def vm(settings):
    return VMWorkspaceMaps(settings)


class TestCompleteMapWorkflow:
    def test_load_crop_baseline_fit_collect_save_reload(self, vm, qapp, map_2d_file, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        vm.load_map_files([str(map_2d_file)])
        map_name = map_2d_file.stem
        assert map_name in vm.maps
        vm.select_map(map_name)
        md = vm.store.get_map_data(map_name)
        n_spectra = md.n_spectra
        assert n_spectra > 0

        # ROI crop to the middle 80%.
        xmin_full, xmax_full = float(md.x0[0]), float(md.x0[-1])
        span = xmax_full - xmin_full
        vm.apply_spectral_range(xmin_full + 0.1 * span, xmax_full - 0.1 * span, apply_all=False)
        assert md.x is not None and len(md.x) < len(md.x0)

        # Baseline: two-point linear, attached, applied to the whole map.
        vm.select_all_current_map_spectra()
        vm.add_baseline_point(float(md.x[0]), 0.0)
        vm.add_baseline_point(float(md.x[-1]), 0.0)
        vm.subtract_baseline(apply_all=True)
        is_sub = md.is_baseline_subtracted
        assert (is_sub.all() if isinstance(is_sub, np.ndarray) else is_sub)

        # Add a peak at the strongest point of the mean spectrum.
        mean_spectrum = md.Y.mean(axis=0)
        peak_x = float(md.x[np.argmax(mean_spectrum)])
        vm.add_peak_at(peak_x)
        assert md.fit_model is not None

        # Batch-fit the whole map.
        vm.fit(apply_all=True)
        vm._fit_thread.wait()
        qapp.processEvents()
        assert md.fit_success is not None
        assert md.fit_success.shape[0] == n_spectra

        # Collect results into a DataFrame.
        vm.collect_fit_results()
        assert vm.df_fit_results is not None
        assert len(vm.df_fit_results) <= n_spectra
        assert "X" in vm.df_fit_results.columns and "Y" in vm.df_fit_results.columns

        # Save and reload.
        save_path = tmp_path / "map_workflow.maps"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        vm.save_work()
        assert save_path.exists()

        vm2 = VMWorkspaceMaps(vm.settings)
        vm2.load_work(str(save_path))
        md2 = vm2.store.get_map_data(map_name)
        assert md2 is not None
        np.testing.assert_allclose(md2.Y0, md.Y0)
        assert md2.fit_model is not None
        np.testing.assert_allclose(md2.peak_params, md.peak_params, atol=1e-6)


class TestPredefinedFitModelOnRealMap:
    def test_apply_predefined_si_model_and_fit(self, vm, qapp, map_2d_file, fit_model_si_file):
        from unittest.mock import MagicMock

        vm.load_map_files([str(map_2d_file)])
        map_name = map_2d_file.stem
        vm.select_map(map_name)

        builder = MagicMock()
        builder.get_current_model_path.return_value = fit_model_si_file
        vm.set_fit_model_builder(builder)

        vm.apply_fit_model(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        md = vm.store.get_map_data(map_name)
        assert md.fit_success is not None
        assert md.fit_success.sum() > 0  # at least some pixels converge


class TestMultiPeakMultiMapWorkflow:
    def test_two_maps_fit_via_apply_all_mega_batch(self, vm, qapp):
        x = np.linspace(300, 700, 150)

        def _map_df(name, x0=500.0, seed=0):
            rng = np.random.default_rng(seed)
            rows, coords = [], []
            for i in range(3):
                for j in range(3):
                    a = 100.0 * rng.uniform(0.8, 1.2)
                    y = a / (1 + 4 * ((x - x0) / 6.0) ** 2)
                    rows.append(np.append(y, 0.0))
                    coords.append((float(i), float(j)))
            import pandas as pd
            data = {"X": [c[0] for c in coords], "Y": [c[1] for c in coords]}
            cols = list(x) + [x[-1] + (x[1] - x[0])]
            for k, xc in enumerate(cols):
                data[str(xc)] = [row[k] for row in rows]
            return pd.DataFrame(data)

        for name, seed in (("mapA", 1), ("mapB", 2)):
            df = _map_df(name, seed=seed)
            vm.maps[name] = df
            vm.maps_metadata[name] = {}
            vm._extract_spectra_from_map(name, df)

        vm.select_map("mapA")
        vm.select_all_current_map_spectra()
        vm.add_peak_at(500.0)
        vm.copy_fit_model()
        vm.select_map("mapB")
        vm.paste_fit_model(apply_all=True)
        vm._fit_thread.wait()
        qapp.processEvents()

        for name in ("mapA", "mapB"):
            md = vm.store.get_map_data(name)
            assert md.fit_success is not None
            assert md.fit_success.all()
