import os
import re
from PySide6.QtCore import QSize
from spectroview import ICON_DIR
from spectroview.view.theme import svg_icon
from spectroview.view.theme.tokens import DARK_TOKENS, LIGHT_TOKENS
from spectroview.view.theme.qss_template import build_stylesheet
from spectroview.view.components.v_peak_table import VPeakTable, ROW_HEIGHT
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


EXPECTED_THEMES = {"dark", "light", "classic_dark", "classic_light"}


def test_theme_constants_have_exactly_four_themes():
    """Verify that only the 4 canonical themes exist across all theme definitions."""
    assert set(THEME_TOKENS.keys()) == EXPECTED_THEMES
    assert set(_PALETTE_BUILDERS.keys()) == EXPECTED_THEMES
    assert set(_VIEWER_THEME_NAMES.keys()) == EXPECTED_THEMES
    assert set(_CLASSIC_THEMES) == {"classic_dark", "classic_light"}

    menubar_keys = {key for key, _ in THEME_OPTIONS if key != "---"}
    assert menubar_keys == EXPECTED_THEMES
    assert len(menubar_keys) == 4

    assert THEME_TOKENS["dark"].checkbox_checked_image == "check_white.svg"
    assert THEME_TOKENS["light"].checkbox_checked_image == "check_white.svg"


def test_theme_manager_is_dark():
    settings = MSettings()
    mgr = ThemeManager(settings)

    mgr._current = "dark"
    assert mgr.is_dark is True
    assert mgr.workspace_theme == "dark"
    assert mgr.viewer_theme_name == "Dark Mode"
    assert mgr.icon_color == "#EDEDEF"

    mgr._current = "classic_dark"
    assert mgr.is_dark is True
    assert mgr.workspace_theme == "dark"
    assert mgr.viewer_theme_name == "Dark Mode"

    mgr._current = "light"
    assert mgr.is_dark is False
    assert mgr.workspace_theme == "light"
    assert mgr.viewer_theme_name == "Light Mode"
    assert mgr.icon_color == "#141519"

    mgr._current = "classic_light"
    assert mgr.is_dark is False
    assert mgr.workspace_theme == "light"
    assert mgr.viewer_theme_name == "Light Mode"


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
    menubar.apply_theme("light")
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


def test_progressbar_contrast_and_tokens():
    """Verify progress bar text token exists, has high contrast, and is injected into QSS."""
    from spectroview.view.theme.qss_template import build_stylesheet

    for theme_name, tokens in THEME_TOKENS.items():
        assert hasattr(tokens, "progress_text"), f"Missing progress_text on {theme_name}"
        assert tokens.progress_text, f"Empty progress_text on {theme_name}"

    # Specifically for dark theme, progress_text must be high contrast
    dark_tokens = THEME_TOKENS["dark"]
    assert dark_tokens.progress_text == "#EDEDEF"

    # Specifically for light theme, progress_text must be high-contrast white
    light_tokens = THEME_TOKENS["light"]
    assert light_tokens.progress_text == "#FFFFFF"

    # Verify stylesheet contains the progress_text in QProgressBar rule
    qss = build_stylesheet(dark_tokens)
    assert f"color: {dark_tokens.progress_text};" in qss



def test_theme_migration_warm_and_soft_dark(qtbot):
    """Verify automatic migration of modern_warm/warm -> light and soft_dark -> dark."""
    settings = MSettings()
    mgr = ThemeManager(settings)

    # soft_dark -> dark
    settings.settings.setValue("theme", "soft_dark")
    assert settings.get_theme() == "dark"
    mgr.apply("soft_dark")
    assert mgr.current_theme == "dark"

    # modern_warm -> light
    settings.settings.setValue("theme", "modern_warm")
    assert settings.get_theme() == "light"
    mgr.apply("modern_warm")
    assert mgr.current_theme == "light"

    # warm -> light
    settings.settings.setValue("theme", "warm")
    assert settings.get_theme() == "light"
    mgr.apply("warm")
    assert mgr.current_theme == "light"


def test_rgba_integer_alpha_tokens_and_stylesheet():
    """Verify all rgba(...) tokens in DARK_TOKENS, LIGHT_TOKENS, and build_stylesheet use integer alpha 0..255 (never float 0.x)."""
    rgba_pattern = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([^\)]+)\)")
    for tok_obj in (DARK_TOKENS, LIGHT_TOKENS):
        for field, val in tok_obj.__dict__.items():
            if isinstance(val, str) and "rgba" in val:
                for match in rgba_pattern.finditer(val):
                    alpha_str = match.group(4).strip()
                    assert "." not in alpha_str, f"Float alpha found in token {field}: {val}"
                    alpha_int = int(alpha_str)
                    assert 0 <= alpha_int <= 255, f"Alpha out of range 0..255 in token {field}: {val}"

    for tok_obj in (DARK_TOKENS, LIGHT_TOKENS):
        qss = build_stylesheet(tok_obj)
        for match in rgba_pattern.finditer(qss):
            alpha_str = match.group(4).strip()
            assert "." not in alpha_str, f"Float alpha found in generated QSS: {match.group(0)}"
            alpha_int = int(alpha_str)
            assert 0 <= alpha_int <= 255, f"Alpha out of range 0..255 in generated QSS: {match.group(0)}"


def test_v_peak_table_row_height_alignment(qtbot):
    """Verify VPeakTable row height alignment across columns with Modern Dark QSS applied."""
    app = QApplication.instance()
    app.setStyleSheet(build_stylesheet(DARK_TOKENS))
    table = VPeakTable()
    qtbot.addWidget(table)

    fit_model = {
        "peak_models": {
            "0": {
                "PseudoVoigt": {
                    "x0": {"value": 520.0, "vary": True},
                    "ampli": {"value": 100.0, "vary": True},
                    "fwhm": {"value": 10.0, "vary": True},
                    "alpha": {"value": 0.5, "vary": True},
                }
            },
            "1": {
                "Lorentzian": {
                    "x0": {"value": 580.0, "vary": True},
                    "ampli": {"value": 80.0, "vary": True},
                    "fwhm": {"value": 15.0, "vary": True},
                }
            },
        },
        "peak_labels": ["P1", "P2"],
    }
    table.set_spectrum({"fit_model": fit_model})
    assert len(table._row_bindings) == 2
    for row in table._row_bindings:
        assert row["label_le"].height() == ROW_HEIGHT
        assert row["model_cmb"].height() == ROW_HEIGHT
        for p_data in row["param_widgets"].values():
            if p_data.get("val_le"):
                assert p_data["val_le"].height() == ROW_HEIGHT


def test_svg_icon_engine_dynamic_paint_color(qtbot):
    """Verify _SvgIconEngine dynamic paint-time color adaptation when switching ThemeManager."""
    settings = MSettings()
    mgr = ThemeManager(settings)

    mgr.apply("dark")
    icon = svg_icon("settings")
    pix_dark = icon.pixmap(QSize(20, 20))
    assert not pix_dark.isNull()

    mgr.apply("light")
    pix_light = icon.pixmap(QSize(20, 20))
    assert not pix_light.isNull()

    img_dark = pix_dark.toImage()
    img_light = pix_light.toImage()
    assert img_dark.bits() != img_light.bits()


def test_resources_icons_only_contains_active_files():
    """Verify that spectroview/resources/icons contains ONLY the 15 active files."""
    expected = {
        "logo_spectroview.png",
        "logo_spectroview.ico",
        "logo_spectroview.icns",
        "logo_metrocarac.png",
        "dof_scheme.png",
        "penetration_eq.png",
        "check_white.svg",
        "chevron_down_dark.svg",
        "chevron_down_light.svg",
        "chevron_up_dark.svg",
        "chevron_up_light.svg",
        "splitter_grip_h_dark.svg",
        "splitter_grip_h_light.svg",
        "splitter_grip_v_dark.svg",
        "splitter_grip_v_light.svg",
    }
    actual = set(os.listdir(ICON_DIR))
    assert actual == expected, f"Mismatch in active icon files: {actual ^ expected}"



def test_maps_deferred_delete_theme_switch_and_mdi_contrast(qtbot):
    """Verify VWorkspaceMaps.apply_theme survives DeferredDelete of inherited buttons and MdiSubWindow has high contrast in Light mode."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QMdiArea, QWidget
    from spectroview.view.v_workspace_maps import VWorkspaceMaps
    from spectroview.view.v_workspace_graphs import VWorkspaceGraphs, MdiSubWindow

    settings = MSettings()
    mgr = ThemeManager(settings)
    mgr.apply("light")

    ws_maps = VWorkspaceMaps()
    ws_graphs = VWorkspaceGraphs()
    qtbot.addWidget(ws_maps)
    qtbot.addWidget(ws_graphs)
    ws_maps.apply_theme("light")
    ws_graphs.apply_theme("light")

    # Process DeferredDelete so old_right buttons in VWorkspaceMaps are destroyed in C++
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    # Switching to dark must not raise RuntimeError and must update icons to bright foreground
    mgr.apply("dark")
    ws_maps.apply_theme("dark")
    ws_graphs.apply_theme("dark")

    for btn in (
        ws_maps.v_fit_model_builder.btn_undo_corr,
        ws_graphs.btn_minimize_all,
        ws_graphs.btn_view_df,
        ws_graphs.btn_add_plot,
        ws_graphs.v_data_filter.btn_add,
    ):
        img = btn.icon().pixmap(QSize(20, 20)).toImage()
        max_r = max(
            img.pixelColor(x, y).red()
            for x in range(img.width())
            for y in range(img.height())
            if img.pixelColor(x, y).alpha() > 80
        )
        assert max_r >= 220

    # Verify MdiSubWindow active title bar contrast in Light mode
    mgr.apply("light")
    mdi = QMdiArea()
    mdi.resize(320, 220)
    qtbot.addWidget(mdi)
    sub = MdiSubWindow(1, mdi)
    sub.setWindowTitle("Active Graph")
    content = QWidget()
    content.resize(260, 160)
    sub.setWidget(content)
    mdi.addSubWindow(sub)
    sub.show()
    mdi.show()
    mdi.setActiveSubWindow(sub)
    QApplication.processEvents()
    sub_img = sub.grab().toImage()
    title_px = sub_img.pixelColor(20, 12)
    assert title_px.red() < 90 and title_px.blue() > 190


def test_tab_pane_transparent_scroll_rounded_splitter_and_fit_results_buttons(qapp, qtbot, _isolate_qsettings):
    """Tasks 6, 7, 9, 10: transparent tab pane, rounded peak scroll, symmetric splitter, Fit Results buttons."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplitter, QWidget
    from spectroview.view.components.v_fit_model_builder import VFitModelBuilder
    from spectroview.view.components.v_fit_results import VFitResults

    mgr = ThemeManager(MSettings())
    mgr.apply("dark")
    qss = qapp.styleSheet()

    # Task 6: QTabWidget::pane has border + transparent background
    assert "QTabWidget::pane" in qss
    pane_block = qss.split("QTabWidget::pane")[1].split("}")[0]
    assert "border: 1px solid" in pane_block
    assert "border-radius: 8px" in pane_block
    assert "background-color: transparent" in pane_block

    # Task 7: peakScrollArea rounded card styling
    assert "QScrollArea#peakScrollArea" in qss
    fmb = VFitModelBuilder()
    qtbot.addWidget(fmb)
    assert fmb.peak_scroll.objectName() == "peakScrollArea"

    # Task 9: Horizontal and Vertical QSplitter handles have identical 11px span (5px bar + 3px margin each side)
    s_v = QSplitter(Qt.Vertical)
    s_v.addWidget(QWidget())
    s_v.addWidget(QWidget())
    s_v.resize(300, 300)
    qtbot.addWidget(s_v)
    s_v.show()

    s_h = QSplitter(Qt.Horizontal)
    s_h.addWidget(QWidget())
    s_h.addWidget(QWidget())
    s_h.resize(300, 300)
    qtbot.addWidget(s_h)
    s_h.show()
    QApplication.processEvents()

    assert s_v.handle(1).height() == 11
    assert s_h.handle(1).width() == 11

    # Task 10: Fit Results buttons have tooltips and sufficient width for icon + text
    fr = VFitResults()
    fr.resize(900, 400)
    qtbot.addWidget(fr)
    fr.show()
    QApplication.processEvents()

    for name in ("btn_collect", "btn_split", "btn_add", "btn_compute", "btn_send", "btn_save"):
        btn = getattr(fr, name)
        assert btn.toolTip().strip(), f"{name} is missing a tooltip"

    assert fr.btn_compute.width() >= 130
    assert fr.btn_send.width() >= 130

