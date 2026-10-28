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
    if hasattr(_shared_main_window, 'theme_mgr'):
        _shared_main_window.theme_mgr._settings = _shared_main_window.settings
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



def test_plot_theme_defaults_to_light_mode(main_window):
    """Confirm plot theme defaults to 'Light Mode' regardless of global app theme."""
    settings = main_window.settings
    settings.set_theme("dark")
    settings.save_view_options({})
    main_window.setup_connections()
    main_window._propagate_theme(main_window.settings.get_theme())

    assert main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.currentText() == "Light Mode"
    assert main_window.v_maps_workspace.v_spectra_viewer.cbb_theme.currentText() == "Light Mode"
    assert main_window.v_maps_workspace.v_map_viewer.current_style == "Light Mode"


def test_plot_theme_persists_across_app_restart(main_window):
    """Confirm user-selected plot theme persists across application startup even when app theme is different."""
    settings = main_window.settings
    settings.set_theme("dark")
    settings.save_view_options({"theme": "Dark Mode"})

    # Simulate application startup sequence
    main_window.setup_connections()
    main_window._propagate_theme(main_window.settings.get_theme())

    assert main_window.settings.get_theme() == "dark"
    assert main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.currentText() == "Dark Mode"
    assert main_window.v_maps_workspace.v_spectra_viewer.cbb_theme.currentText() == "Dark Mode"
    assert main_window.v_maps_workspace.v_map_viewer.current_style == "Dark Mode"
    assert main_window.settings.load_view_options()["theme"] == "Dark Mode"


def test_app_theme_toggle_does_not_change_plot_theme(main_window):
    """Confirm toggling application theme leaves plot theme completely independent."""
    # Initial state: plot theme is Light Mode
    main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.setCurrentText("Light Mode")
    assert main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.currentText() == "Light Mode"

    # Toggle app theme to dark: plot theme must remain Light Mode
    main_window.toggle_theme("dark")
    assert main_window.settings.get_theme() == "dark"
    assert main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.currentText() == "Light Mode"
    assert main_window.v_maps_workspace.v_spectra_viewer.cbb_theme.currentText() == "Light Mode"
    assert main_window.v_maps_workspace.v_map_viewer.current_style == "Light Mode"

    # User explicitly changes plot theme to Dark Mode
    main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.setCurrentText("Dark Mode")
    assert main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.currentText() == "Dark Mode"
    assert main_window.v_maps_workspace.v_spectra_viewer.cbb_theme.currentText() == "Dark Mode"
    assert main_window.v_maps_workspace.v_map_viewer.current_style == "Dark Mode"

    # Toggle app theme to light: plot theme must remain Dark Mode
    main_window.toggle_theme("light")
    assert main_window.settings.get_theme() == "light"
    assert main_window.v_spectra_workspace.v_spectra_viewer.cbb_theme.currentText() == "Dark Mode"
    assert main_window.v_maps_workspace.v_spectra_viewer.cbb_theme.currentText() == "Dark Mode"
    assert main_window.v_maps_workspace.v_map_viewer.current_style == "Dark Mode"


