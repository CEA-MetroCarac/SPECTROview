"""End-to-end integration tests for the Spectra workspace.

Drives VMWorkspaceSpectra through a complete real-world workflow: load a
real file -> crop ROI -> define a baseline -> subtract it -> add peaks ->
fit -> collect results -> save -> reload into a fresh ViewModel -> verify
everything survived the round trip.
"""
import numpy as np
import pytest

from spectroview.viewmodel.vm_workspace_spectra import VMWorkspaceSpectra


@pytest.fixture
def vm(settings):
    return VMWorkspaceSpectra(settings)


class TestCompleteSingleSpectrumWorkflow:
    def test_load_crop_baseline_fit_collect_save_reload(self, vm, qapp, single_spectrum_file, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        # 1. Load a real spectrum file.
        vm.load_files([str(single_spectrum_file)])
        fname = single_spectrum_file.stem
        assert fname in vm.store.map_names
        vm.set_selected_fnames([fname])
        md = vm.store.get_map_data(fname)
        full_range = (float(md.x0[0]), float(md.x0[-1]))

        # 2. ROI: crop to the middle 80% of the axis.
        span = full_range[1] - full_range[0]
        xmin, xmax = full_range[0] + 0.1 * span, full_range[1] - 0.1 * span
        vm.apply_spectral_range(xmin, xmax, apply_all=False)
        assert md.x.min() >= xmin - span * 0.02
        assert md.x.max() <= xmax + span * 0.02

        # 3. Baseline: two-point linear, attached.
        vm.add_baseline_point(float(md.x[0]), 0.0)
        vm.add_baseline_point(float(md.x[-1]), 0.0)
        assert md.baseline_config is not None
        vm.subtract_baseline(apply_all=False)
        assert md.is_baseline_subtracted is True

        # 4. Add a peak roughly at the spectrum's strongest point.
        peak_x = float(md.x[np.argmax(md.Y[0])])
        vm.add_peak_at(peak_x)
        assert md.fit_model is not None and len(md.fit_model["peak_models"]) == 1

        # 5. Fit.
        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()
        assert md.fit_success is not None

        # 6. Collect results.
        vm.collect_fit_results()
        assert vm.df_fit_results is not None
        assert len(vm.df_fit_results) == 1

        # 7. Save and reload into a brand-new ViewModel.
        save_path = tmp_path / "workflow.spectra"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(save_path), ""))
        vm.save_work()
        assert save_path.exists()

        vm2 = VMWorkspaceSpectra(vm.settings)
        vm2.load_work(str(save_path))
        md2 = vm2.store.get_map_data(fname)
        assert md2 is not None
        np.testing.assert_allclose(md2.Y0, md.Y0)
        assert md2.fit_model is not None
        assert md2.is_baseline_subtracted == md.is_baseline_subtracted
        np.testing.assert_allclose(md2.peak_params, md.peak_params, atol=1e-6)


class TestMultiSpectrumBatchWorkflow:
    def test_apply_shared_fit_model_to_several_spectra(self, vm, qapp, multiple_spectra_files):
        vm.load_files([str(p) for p in multiple_spectra_files])
        assert len(vm.store.map_names) == len(multiple_spectra_files)

        first_name = vm.store.map_names[0]
        vm.set_selected_fnames([first_name])
        md0 = vm.store.get_map_data(first_name)
        peak_x = float(md0.x0[np.argmax(md0.Y0[0])])
        vm.add_peak_at(peak_x)
        vm.copy_fit_model()

        vm.paste_fit_model(apply_all=True)  # applies to every active spectrum + fits
        vm._fit_thread.wait()
        qapp.processEvents()

        for name in vm.store.map_names:
            md = vm.store.get_map_data(name)
            assert md.fit_model is not None
            assert md.fit_success is not None


class TestPeakModelEditingWorkflow:
    """Different peak models + parameter bounds/constraints, end to end."""

    @pytest.mark.parametrize("shape", ["Lorentzian", "Gaussian", "PseudoVoigt", "GaussianAsym", "Fano"])
    def test_switch_shape_then_fit(self, vm, qapp, shape):
        x = np.linspace(300, 700, 300)
        y = 100.0 / (1 + 4 * ((x - 500.0) / 6.0) ** 2)
        vm.store.add_map("s1", x, y[None, :].astype(np.float32), np.array([[0.0, 0.0]]), ["s1"])
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        vm.update_peak_model(0, shape)

        md = vm.store.get_map_data("s1")
        assert list(md.fit_model["peak_models"]["0"].keys()) == [shape]

        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()
        assert md.fit_success is not None
        assert np.isfinite(md.peak_params).all()

    def test_bounded_parameter_stays_within_bounds_after_fit(self, vm, qapp):
        x = np.linspace(300, 700, 300)
        y = 100.0 / (1 + 4 * ((x - 500.0) / 6.0) ** 2)
        vm.store.add_map("s1", x, y[None, :].astype(np.float32), np.array([[0.0, 0.0]]), ["s1"])
        vm.set_selected_fnames(["s1"])
        vm.add_peak_at(500.0)
        vm.update_peak_param(0, "x0", "min", 490.0)
        vm.update_peak_param(0, "x0", "max", 495.0)  # excludes the true x0=500

        vm.fit(apply_all=False)
        vm._fit_thread.wait()
        qapp.processEvents()

        md = vm.store.get_map_data("s1")
        idx_x0 = md.param_names.index("P1_x0")
        assert md.peak_params[0, idx_x0] <= 495.0 + 1e-6
