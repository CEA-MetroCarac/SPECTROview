"""Unit tests for view options synchronization and persistence between workspaces."""

import pytest
from spectroview.main import Main


@pytest.fixture
def main_window(qapp, monkeypatch):
    # Disable prewarming or background timers for fast testing
    monkeypatch.setattr(Main, "_prewarm_ai_chat", lambda self: None)
    window = Main()
    yield window
    window.close()


def test_maps_workspace_view_options_saved_to_settings(main_window):
    """Confirm changes in Maps workspace spectra viewer are saved to settings and synced."""
    maps_viewer = main_window.v_maps_workspace.v_spectra_viewer
    spectra_viewer = main_window.v_spectra_workspace.v_spectra_viewer
    settings = main_window.settings

    # Verify initial default state is True
    assert maps_viewer.btn_bestfit.isChecked() is True
    assert spectra_viewer.btn_bestfit.isChecked() is True
    assert settings.load_view_options()["bestfit"] is True

    # User toggles bestfit in Maps workspace
    maps_viewer.btn_bestfit.setChecked(False)

    # Both workspaces should sync and settings should persist False
    assert maps_viewer.btn_bestfit.isChecked() is False
    assert spectra_viewer.btn_bestfit.isChecked() is False
    assert settings.load_view_options()["bestfit"] is False

    # Toggle back to True from Maps workspace
    maps_viewer.btn_bestfit.setChecked(True)
    assert maps_viewer.btn_bestfit.isChecked() is True
    assert spectra_viewer.btn_bestfit.isChecked() is True
    assert settings.load_view_options()["bestfit"] is True


def test_spectra_workspace_view_options_saved_to_settings(main_window):
    """Confirm changes in Spectra workspace spectra viewer are saved to settings and synced."""
    maps_viewer = main_window.v_maps_workspace.v_spectra_viewer
    spectra_viewer = main_window.v_spectra_workspace.v_spectra_viewer
    settings = main_window.settings

    # User toggles bestfit in Spectra workspace
    spectra_viewer.btn_bestfit.setChecked(False)

    # Both workspaces should sync and settings should persist False
    assert maps_viewer.btn_bestfit.isChecked() is False
    assert spectra_viewer.btn_bestfit.isChecked() is False
    assert settings.load_view_options()["bestfit"] is False
