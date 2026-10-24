from PySide6.QtWidgets import QApplication
from spectroview.model.m_settings import MSettings
from spectroview.model.m_graph import MGraph
from spectroview.view.theme.manager import ThemeManager, _VIEWER_THEME_NAMES, _CLASSIC_THEMES
from spectroview.view.theme.tokens import THEME_TOKENS
from spectroview.view.theme.palettes import _PALETTE_BUILDERS
from spectroview.view.components.v_menubar import VMenuBar, THEME_OPTIONS
from spectroview.view.components.v_spectra_viewer import VSpectraViewer
from spectroview.view.components.v_graph import VGraph
from spectroview.view.components.v_export_dialog import VExportDialog
from spectroview.view.components.customize_graph.customize_more_options import CustomizeMoreOptions


EXPECTED_THEMES = {"dark", "light", "modern_warm", "classic_dark", "classic_light"}


def test_theme_constants_have_exactly_five_themes():
    """Verify that only the 5 canonical themes exist across all theme definitions."""
    assert set(THEME_TOKENS.keys()) == EXPECTED_THEMES
    assert set(_PALETTE_BUILDERS.keys()) == EXPECTED_THEMES
    assert set(_VIEWER_THEME_NAMES.keys()) == EXPECTED_THEMES
    assert set(_CLASSIC_THEMES) == {"classic_dark", "classic_light"}

    menubar_keys = {key for key, _ in THEME_OPTIONS if key != "---"}
    assert menubar_keys == EXPECTED_THEMES
    assert len(menubar_keys) == 5


def test_theme_manager_is_dark():
    settings = MSettings()
    mgr = ThemeManager(settings)

    mgr._current = "dark"
    assert mgr.is_dark is True
    assert mgr.workspace_theme == "dark"
    assert mgr.viewer_theme_name == "Dark Mode"
    assert mgr.icon_color == "#F0F0F0"

    mgr._current = "classic_dark"
    assert mgr.is_dark is True
    assert mgr.workspace_theme == "dark"
    assert mgr.viewer_theme_name == "Dark Mode"

    mgr._current = "light"
    assert mgr.is_dark is False
    assert mgr.workspace_theme == "light"
    assert mgr.viewer_theme_name == "Light Mode"
    assert mgr.icon_color == "#404040"

    mgr._current = "classic_light"
    assert mgr.is_dark is False
    assert mgr.workspace_theme == "light"
    assert mgr.viewer_theme_name == "Light Mode"

    mgr._current = "modern_warm"
    assert mgr.is_dark is False
    assert mgr.workspace_theme == "light"
    assert mgr.viewer_theme_name == "Light Mode"
    assert mgr.icon_color == "#1C1C1E"


def test_theme_manager_migration_from_soft_dark():
    settings = MSettings()
    settings.set_theme("soft_dark")

    mgr = ThemeManager(settings)
    assert mgr.current_theme == "dark"

    mgr.apply("soft_dark")
    assert mgr.current_theme == "dark"
    assert settings.get_theme() == "dark"


def test_theme_manager_apply_all_five_themes(qtbot):
    settings = MSettings()
    mgr = ThemeManager(settings)
    app = QApplication.instance()
    assert app is not None

    for theme_key in EXPECTED_THEMES:
        mgr.apply(theme_key)
        assert mgr.current_theme == theme_key
        assert settings.get_theme() == theme_key
        if theme_key in _CLASSIC_THEMES:
            assert app.styleSheet() == ""
        else:
            assert len(app.styleSheet()) > 0


def test_menubar_theme_selection(qtbot):
    menubar = VMenuBar()
    qtbot.addWidget(menubar)

    for theme_key, _ in THEME_OPTIONS:
        if theme_key == "---":
            continue
        menubar.set_current_theme(theme_key)
        assert menubar._current_theme == theme_key

    # Soft dark migration in menubar
    menubar.set_current_theme("soft_dark")
    assert menubar._current_theme == "dark"

    # Test apply_theme icon switching and restoration
    menubar.apply_theme("modern_warm")
    menubar.apply_theme("dark")


def test_msettings_theme_migration():
    settings = MSettings()
    # Direct write of legacy key to simulate existing user config
    settings.settings.setValue("theme", "soft_dark")
    assert settings.get_theme() == "dark"
    assert settings.settings.value("theme") == "dark"

    # View options migration
    view_opts = {"theme": "Soft Dark Mode", "xaxis": "Raman Shift"}
    settings.save_view_options(view_opts)
    loaded_opts = settings.load_view_options()
    assert loaded_opts.get("theme") == "Dark Mode"

    # Export options migration
    settings.settings.setValue("export_options/theme", "Soft Dark Mode")
    export_opts = settings.load_export_options()
    assert export_opts.get("theme") == "Dark Mode"


def test_mgraph_load_migration():
    mg = MGraph()
    mg.load({"figure_theme": "soft_dark"})
    assert mg.figure_theme == "dark"

    mg2 = MGraph()
    mg2.load({"figure_theme": "light"})
    assert mg2.figure_theme == "light"


def test_customize_more_options_themes(qtbot):
    vg = VGraph(graph_id=1)
    qtbot.addWidget(vg)
    more_tab = CustomizeMoreOptions(vg)
    qtbot.addWidget(more_tab)
    items = [more_tab._combo_figure_theme.itemText(i) for i in range(more_tab._combo_figure_theme.count())]
    assert items == ["Light", "Dark"]


def test_export_dialog_themes(qtbot):
    vg = VGraph(graph_id=1)
    qtbot.addWidget(vg)
    dlg = VExportDialog(vg)
    qtbot.addWidget(dlg)
    items = [dlg.panel.combo_theme.itemText(i) for i in range(dlg.panel.combo_theme.count())]
    assert items == ["Light Mode", "Dark Mode"]


def test_spectra_viewer_themes_and_migration(qtbot):
    viewer = VSpectraViewer()
    qtbot.addWidget(viewer)
    items = [viewer.cbb_theme.itemText(i) for i in range(viewer.cbb_theme.count())]
    assert items == ["Light Mode", "Dark Mode"]

    # Test migration via set_options_state
    viewer.set_options_state({"theme": "Soft Dark Mode"})
    assert viewer.cbb_theme.currentText() == "Dark Mode"
