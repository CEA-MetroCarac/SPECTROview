"""Unit tests for view options synchronization and persistence between workspaces."""

import pytest


@pytest.fixture(scope="module")
def _shared_main_window(qapp):
    import spectroview.main as main_mod
    original_prewarm = getattr(main_mod.Main, "_prewarm_ai_chat", None)
    main_mod.Main._prewarm_ai_chat = lambda self: None
    window = main_mod.Main()

    # Reconnect to route save_view_options dynamically through current window.settings
    try:
        window.v_spectra_workspace.v_spectra_viewer.allOptionsSyncChanged.disconnect()
        window.v_maps_workspace.v_spectra_viewer.allOptionsSyncChanged.disconnect()
    except (RuntimeError, TypeError):
        pass
    window.v_spectra_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(
        window.v_maps_workspace.v_spectra_viewer.set_options_state
    )
    window.v_maps_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(
        window.v_spectra_workspace.v_spectra_viewer.set_options_state
    )
    window.v_spectra_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(
        lambda opts: window.settings.save_view_options(opts)
    )
    window.v_maps_workspace.v_spectra_viewer.allOptionsSyncChanged.connect(
        lambda opts: window.settings.save_view_options(opts)
    )

    yield window
    window.close()
    if original_prewarm:
        main_mod.Main._prewarm_ai_chat = original_prewarm


@pytest.fixture
def main_window(_shared_main_window):
    # Rebind to current test's isolated QSettings store
    from spectroview.model.m_settings import MSettings
    _shared_main_window.settings = MSettings()
    _shared_main_window.settings.save_view_options({"bestfit": True, "legend_outside": False})
    
    # Reset controls to default state before each test
    _shared_main_window.v_maps_workspace.v_spectra_viewer.btn_bestfit.setChecked(True)
    _shared_main_window.v_spectra_workspace.v_spectra_viewer.btn_bestfit.setChecked(True)
    _shared_main_window.v_maps_workspace.v_spectra_viewer.act_legend_outside.setChecked(False)
    _shared_main_window.v_spectra_workspace.v_spectra_viewer.act_legend_outside.setChecked(False)
    return _shared_main_window


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

    # Toggle back to True from Spectra workspace
    spectra_viewer.btn_bestfit.setChecked(True)
    assert maps_viewer.btn_bestfit.isChecked() is True
    assert spectra_viewer.btn_bestfit.isChecked() is True
    assert settings.load_view_options()["bestfit"] is True


def test_legend_outside_view_options_saved_and_synced(main_window):
    """Confirm changes to 'Move legend box outside' sync between workspaces and persist."""
    maps_viewer = main_window.v_maps_workspace.v_spectra_viewer
    spectra_viewer = main_window.v_spectra_workspace.v_spectra_viewer
    settings = main_window.settings

    # Verify initial default state is False
    assert maps_viewer.act_legend_outside.isChecked() is False
    assert spectra_viewer.act_legend_outside.isChecked() is False
    assert settings.load_view_options()["legend_outside"] is False

    # Toggle in Spectra workspace
    spectra_viewer.act_legend_outside.setChecked(True)

    # Both workspaces should sync and settings should persist True
    assert maps_viewer.act_legend_outside.isChecked() is True
    assert spectra_viewer.act_legend_outside.isChecked() is True
    assert settings.load_view_options()["legend_outside"] is True

    # Toggle back to False from Maps workspace
    maps_viewer.act_legend_outside.setChecked(False)
    assert maps_viewer.act_legend_outside.isChecked() is False
    assert spectra_viewer.act_legend_outside.isChecked() is False
    assert settings.load_view_options()["legend_outside"] is False

