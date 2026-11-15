"""Tests for the spectroview.api exception hierarchy."""
import pytest

from spectroview.api import exceptions


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
