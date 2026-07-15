"""Tests for the spectroview.api exception hierarchy and that real API
entry points raise the documented exception types instead of leaking raw
KeyError/ValueError."""
import numpy as np
import pytest

from spectroview.api import exceptions, fitting, io


class TestHierarchy:
    @pytest.mark.parametrize("cls", [
        exceptions.LoadError,
        exceptions.FitModelError,
        exceptions.FitError,
        exceptions.WorkspaceError,
        exceptions.TemplateError,
    ])
    def test_all_subclass_base_error(self, cls):
        assert issubclass(cls, exceptions.SpectroviewError)
        assert issubclass(cls, Exception)


class TestRealCallSitesRaiseTypedErrors:
    def test_load_spectra_unsupported_extension_raises_load_error(self, tmp_path):
        bogus = tmp_path / "spectrum.xyz"
        bogus.write_text("not a real spectrum file")
        with pytest.raises(exceptions.LoadError):
            io.load_spectra(bogus)

    def test_export_results_unsupported_extension_raises_load_error(self, tmp_path):
        with pytest.raises(exceptions.LoadError):
            io.export_results([{"a": 1}], tmp_path / "out.bogus")

    def test_fit_batch_with_no_peak_models_raises_fit_model_error(self):
        x = np.linspace(0, 10, 50)
        Y = np.zeros((2, 50))
        with pytest.raises(exceptions.FitModelError):
            fitting.fit_batch(x, Y, fit_model={"peak_models": {}})

    def test_build_fit_model_missing_model_key_raises_fit_model_error(self):
        with pytest.raises(exceptions.FitModelError):
            fitting.build_fit_model([{"x0": {"value": 1.0}}])

    def test_load_fit_model_template_missing_file_raises_template_error(self, tmp_path):
        with pytest.raises(exceptions.TemplateError):
            fitting.load_fit_model_template(tmp_path / "does_not_exist.json")
