import os
import re
import pytest
from PySide6.QtCore import QSize
from spectroview import ICON_DIR
from spectroview.view.theme import svg_icon
from spectroview.view.theme.tokens import DARK_TOKENS, GRAY_TOKENS, LIGHT_TOKENS
from spectroview.view.theme.qss_template import build_stylesheet
from spectroview.view.components.lower_tabs import VPeakTable, ROW_HEIGHT
from PySide6.QtWidgets import QApplication
from spectroview.model.m_settings import MSettings
from spectroview.model.graphs.m_graph import MGraph
from spectroview.view.theme.manager import ThemeManager, _VIEWER_THEME_NAMES, _CLASSIC_THEMES
from spectroview.view.theme.tokens import THEME_TOKENS
from spectroview.view.theme.palettes import _PALETTE_BUILDERS
from spectroview.view.components.common.v_menubar import VMenuBar, THEME_OPTIONS
from spectroview.view.components.spectra_viewer import VSpectraViewer
from spectroview.view.components.graphs.v_graph import VGraph
from spectroview.view.components.graphs.v_export_dialog import VExportDialog
from spectroview.view.components.customize_graph.customize_more_options import CustomizeMoreOptions


EXPECTED_THEMES = {"dark", "gray", "light", "classic_dark", "classic_light"}


def test_theme_constants_have_exactly_five_themes():
    """Verify that the 5 canonical themes exist across all theme definitions."""
    assert set(THEME_TOKENS.keys()) == EXPECTED_THEMES
    assert set(_PALETTE_BUILDERS.keys()) == EXPECTED_THEMES
    assert set(_VIEWER_THEME_NAMES.keys()) == EXPECTED_THEMES
    assert set(_CLASSIC_THEMES) == {"classic_dark", "classic_light"}

    menubar_keys = {key for key, _ in THEME_OPTIONS if key != "---"}
    assert menubar_keys == EXPECTED_THEMES
    assert len(menubar_keys) == 5

    assert THEME_TOKENS["dark"].checkbox_checked_image == "check_white.svg"
    assert THEME_TOKENS["gray"].checkbox_checked_image == "check_white.svg"
    assert THEME_TOKENS["light"].checkbox_checked_image == "check_white.svg"


def test_theme_manager_is_dark():
    settings = MSettings()
    mgr = ThemeManager(settings)

    mgr._current = "dark"
    assert mgr.is_dark is True
    assert mgr.workspace_theme == "dark"
    assert mgr.viewer_theme_name == "Dark Mode"
    assert mgr.icon_color == "#EDEDEF"

    mgr._current = "gray"
    assert mgr.is_dark is True
    assert mgr.workspace_theme == "dark"
    assert mgr.viewer_theme_name == "Dark Mode"
    assert mgr.icon_color == "#ECECEE"

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
    assert menubar.actionTheme.icon().name() == "sun"
    menubar.apply_theme("dark")
    assert menubar.actionTheme.icon().name() == "moon"


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

    # Specifically for gray theme, progress_text must be high contrast
    gray_tokens = THEME_TOKENS["gray"]
    assert gray_tokens.progress_text == "#ECECEE"

    # Specifically for light theme, progress_text must be high-contrast white
    light_tokens = THEME_TOKENS["light"]
    assert light_tokens.progress_text == "#FFFFFF"

    # Verify stylesheet contains the progress_text in QProgressBar rule
    qss = build_stylesheet(dark_tokens)
    assert f"color: {dark_tokens.progress_text};" in qss
    gray_qss = build_stylesheet(gray_tokens)
    assert f"color: {gray_tokens.progress_text};" in gray_qss


def test_rgba_integer_alpha_tokens_and_stylesheet():
    """Verify all rgba(...) tokens in DARK_TOKENS, GRAY_TOKENS, LIGHT_TOKENS, and build_stylesheet use integer alpha 0..255."""
    rgba_pattern = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([^\)]+)\)")
    for tok_obj in (DARK_TOKENS, GRAY_TOKENS, LIGHT_TOKENS):
        for field, val in tok_obj.__dict__.items():
            if isinstance(val, str) and "rgba" in val:
                for match in rgba_pattern.finditer(val):
                    alpha_str = match.group(4).strip()
                    assert "." not in alpha_str, f"Float alpha found in token {field}: {val}"
                    alpha_int = int(alpha_str)
                    assert 0 <= alpha_int <= 255, f"Alpha out of range 0..255 in token {field}: {val}"

    for tok_obj in (DARK_TOKENS, GRAY_TOKENS, LIGHT_TOKENS):
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

    mgr.apply("gray")
    pix_gray = icon.pixmap(QSize(20, 20))
    assert not pix_gray.isNull()

    mgr.apply("light")
    pix_light = icon.pixmap(QSize(20, 20))
    assert not pix_light.isNull()

    img_dark = pix_dark.toImage()
    img_gray = pix_gray.toImage()
    img_light = pix_light.toImage()
    assert img_dark.bits() != img_light.bits()
    assert img_gray.bits() != img_light.bits()


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
    actual = {f for f in os.listdir(ICON_DIR) if f != ".DS_Store"}
    assert actual == expected, f"Mismatch in active icon files: {actual ^ expected}"



@pytest.mark.slow
def test_maps_deferred_delete_theme_switch(qtbot):
    """Verify VWorkspaceMaps.apply_theme survives DeferredDelete of inherited buttons."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from spectroview.view.workspaces import VWorkspaceMaps

    settings = MSettings()
    mgr = ThemeManager(settings)
    mgr.apply("light")

    ws_maps = VWorkspaceMaps()
    qtbot.addWidget(ws_maps)
    ws_maps.apply_theme("light")

    # Process DeferredDelete so old_right buttons in VWorkspaceMaps are destroyed in C++
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    # Switching to dark must not raise RuntimeError and must update icons to bright foreground
    mgr.apply("dark")
    ws_maps.apply_theme("dark")

    btn = ws_maps.v_fit_model_builder.btn_undo_corr
    img = btn.icon().pixmap(QSize(20, 20)).toImage()
    max_r = max(
        img.pixelColor(x, y).red()
        for x in range(img.width())
        for y in range(img.height())
        if img.pixelColor(x, y).alpha() > 80
    )
    assert max_r >= 220


def test_mdi_subwindow_contrast(qtbot):
    """Verify MdiSubWindow active title bar contrast in Light mode."""
    from PySide6.QtWidgets import QMdiArea, QWidget
    from spectroview.view.workspaces import MdiSubWindow

    settings = MSettings()
    mgr = ThemeManager(settings)
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
    from spectroview.view.components.lower_tabs import VFitModelBuilder, VFitResults

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

    # Verify Modern Dark uses Neutral Dark / Charcoal gray surfaces & borders (R == G == B)
    from PySide6.QtGui import QColor
    for field in (
        "background",
        "rail_bg",
        "sidebar_bg",
        "surface",
        "surface_sunken",
        "surface_elevated",
        "surface_hover",
        "surface_pressed",
        "input_bg",
        "input_border",
        "border",
        "border_subtle",
        "border_strong",
        "text_secondary",
        "text_muted",
    ):
        col = QColor(getattr(DARK_TOKENS, field))
        assert col.red() == col.green() == col.blue(), f"Token {field}={getattr(DARK_TOKENS, field)} is not neutral gray"


def test_text_browser_styled_with_surface_in_themes():
    """Verify that QTextBrowser uses @surface background and adapts between dark, gray, and light themes."""
    from spectroview.view.theme.qss_template import build_stylesheet
    from spectroview.view.theme.tokens import THEME_TOKENS
    dark_qss = build_stylesheet(THEME_TOKENS["dark"])
    gray_qss = build_stylesheet(THEME_TOKENS["gray"])
    light_qss = build_stylesheet(THEME_TOKENS["light"])

    assert "QTextBrowser" in dark_qss
    assert "QTextBrowser" in gray_qss
    assert "QTextBrowser" in light_qss
    assert THEME_TOKENS["dark"].surface in dark_qss
    assert THEME_TOKENS["gray"].surface in gray_qss
    assert THEME_TOKENS["light"].surface in light_qss


def _rel_luminance(hex_col: str) -> float:
    from PySide6.QtGui import QColor

    c = QColor(hex_col)
    channels = []
    for v in (c.redF(), c.greenF(), c.blueF()):
        channels.append(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(c1: str, c2: str) -> float:
    l1, l2 = _rel_luminance(c1), _rel_luminance(c2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def test_modern_gray_neutral_palette_and_surface_hierarchy():
    """Verify Modern Gray uses neutral gray tones, intermediate brightness, and monotonic surface depth."""
    from PySide6.QtGui import QColor

    for field in (
        "background",
        "rail_bg",
        "sidebar_bg",
        "surface",
        "surface_sunken",
        "surface_elevated",
        "surface_hover",
        "surface_pressed",
        "input_bg",
        "input_border",
        "border",
        "border_subtle",
        "border_strong",
        "text_secondary",
        "text_muted",
    ):
        col = QColor(getattr(GRAY_TOKENS, field))
        assert col.red() == col.green() == col.blue(), f"Token {field}={getattr(GRAY_TOKENS, field)} is not neutral gray"

    # Background brightness sits cleanly between Modern Dark and Modern Light
    assert QColor(DARK_TOKENS.background).lightness() < QColor(GRAY_TOKENS.background).lightness() < QColor(LIGHT_TOKENS.background).lightness()

    # Monotonic 3-tier surface elevation hierarchy
    lums = [
        QColor(GRAY_TOKENS.rail_bg).lightness(),
        QColor(GRAY_TOKENS.sidebar_bg).lightness(),
        QColor(GRAY_TOKENS.background).lightness(),
        QColor(GRAY_TOKENS.surface).lightness(),
        QColor(GRAY_TOKENS.surface_elevated).lightness(),
        QColor(GRAY_TOKENS.surface_hover).lightness(),
        QColor(GRAY_TOKENS.surface_pressed).lightness(),
    ]
    assert lums == sorted(lums) and len(set(lums)) == len(lums)


def test_modern_gray_wcag_contrast_thresholds():
    """Verify Modern Gray text and solid CTA tokens satisfy WCAG contrast thresholds."""
    assert _contrast_ratio(GRAY_TOKENS.text_primary, GRAY_TOKENS.background) >= 7.0
    assert _contrast_ratio(GRAY_TOKENS.text_primary, GRAY_TOKENS.surface) >= 7.0
    assert _contrast_ratio(GRAY_TOKENS.text_primary, GRAY_TOKENS.surface_elevated) >= 4.5
    assert _contrast_ratio(GRAY_TOKENS.text_primary, GRAY_TOKENS.surface_hover) >= 4.5
    assert _contrast_ratio(GRAY_TOKENS.text_secondary, GRAY_TOKENS.surface) >= 4.5
    assert _contrast_ratio(GRAY_TOKENS.on_solid, GRAY_TOKENS.solid) >= 7.0


def test_modern_gray_alias_normalization():
    """Verify 'modern_gray' and 'grey' aliases normalize to canonical 'gray' across layers."""
    from spectroview.view.theme import build_palette, modern_gray_palette, normalize_theme_key

    for alias in ("gray", "modern_gray", "grey", "modern_grey", "Modern_Gray"):
        assert normalize_theme_key(alias) == "gray"

        settings = MSettings()
        settings.set_theme(alias)
        assert settings.get_theme() == "gray"

        mgr = ThemeManager(settings)
        assert mgr.current_theme == "gray"
        assert mgr.is_dark is True

        menubar = VMenuBar()
        menubar.set_current_theme(alias)
        assert menubar._current_theme == "gray"
        menubar.close()

        pal = build_palette(alias)
        assert pal.color(pal.ColorRole.Window) == modern_gray_palette().color(pal.ColorRole.Window)


def test_modern_gray_update_banner_and_live_maps(qtbot):
    """Verify VUpdateBanner and VLiveMaps adapt to Modern Gray cleanly."""
    from spectroview.view.components.common.v_update_banner import VUpdateBanner, _MODERN_GRAY_STYLE
    from spectroview.view.components.rgb_live_maps import VLiveMaps
    from spectroview.view.components.rgb_live_maps.common import _normalize_plot_theme

    settings = MSettings()
    mgr = ThemeManager(settings)
    mgr.apply("gray")

    banner = VUpdateBanner("v99.0.0", "https://example.invalid", lambda _t: None, lambda: None)
    qtbot.addWidget(banner)
    banner.apply_theme("gray")
    assert banner.styleSheet() == _MODERN_GRAY_STYLE

    assert _normalize_plot_theme("gray") == "Dark Mode"
    assert _normalize_plot_theme("modern_gray") == "Dark Mode"

    live_maps = VLiveMaps()
    qtbot.addWidget(live_maps)
    live_maps.apply_theme("gray")
    assert live_maps.get_view_options()["plot_theme"] == "Dark Mode"
    assert GRAY_TOKENS.text_muted in live_maps.lbl_placeholder.styleSheet()


def test_canvas_card_and_graph_coordinate_theme_sync(qtbot):
    """Verify canvas_card border and VGraph coordinates update when switching to Modern Gray."""
    from spectroview.view.components.map_viewer import VMapViewer
    from spectroview.view.workspaces.graphs import VWorkspaceGraphs

    settings = MSettings()
    mgr = ThemeManager(settings)
    mgr.apply("gray")

    sv = VSpectraViewer()
    qtbot.addWidget(sv)
    sv.apply_global_theme("gray")
    assert GRAY_TOKENS.border in sv.canvas_card.styleSheet()
    assert GRAY_TOKENS.text_secondary in sv.lbl_coordinates.styleSheet()

    mv = VMapViewer()
    qtbot.addWidget(mv)
    mv.apply_global_theme("gray")
    assert GRAY_TOKENS.border in mv.canvas_card.styleSheet()
    assert GRAY_TOKENS.text_secondary in mv.lbl_coordinates.styleSheet()

    ws_graphs = VWorkspaceGraphs()
    qtbot.addWidget(ws_graphs)
    vg = VGraph(graph_id=1)
    vg.create_plot_widget(dpi=80)
    qtbot.addWidget(vg)
    ws_graphs.graph_widgets[1] = (vg, None, None)

    mgr.apply("dark")
    ws_graphs.apply_theme("dark")
    assert DARK_TOKENS.text_secondary in vg.lbl_coordinates.styleSheet()

    mgr.apply("gray")
    ws_graphs.apply_theme("gray")
    assert GRAY_TOKENS.text_secondary in vg.lbl_coordinates.styleSheet()


