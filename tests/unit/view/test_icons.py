# tests/unit/view/test_icons.py
"""Unit tests for the new vector SVG icon system in SPECTROview."""

from PySide6.QtGui import QIcon, QPixmap, QPainter, QImage
from PySide6.QtCore import QRect, QSize

from spectroview.view.theme import (
    Icons,
    svg_icon,
    svg_pixmap,
    clear_icon_caches,
)
from spectroview.viewmodel.utils import get_tinted_icon


def test_svg_icon_creation(qapp):
    """Verify that svg_icon produces valid, non-null QIcons."""
    icon = svg_icon("sparkles")
    assert isinstance(icon, QIcon)
    assert not icon.isNull()

    # Pixmap generation from QIcon
    pix = icon.pixmap(QSize(24, 24))
    assert isinstance(pix, QPixmap)
    assert not pix.isNull()
    assert pix.width() == 24
    assert pix.height() == 24


def test_svg_pixmap_high_dpi(qapp):
    """Verify that svg_pixmap renders at 1:1 physical device pixels with DPR."""
    pix = svg_pixmap("settings", size=20, dpr=2.0)
    assert isinstance(pix, QPixmap)
    assert not pix.isNull()
    assert pix.devicePixelRatio() == 2.0
    # 20 logical size * 2.0 DPR = 40 physical pixels
    assert pix.width() == 40
    assert pix.height() == 40


def test_icons_facade_and_legacy_mapping(qapp):
    """Verify Icons convenience methods and LEGACY_ICON_MAP lookups."""
    # Direct methods
    assert not Icons.open().isNull()
    assert not Icons.save().isNull()
    assert not Icons.trash().isNull()
    assert not Icons.check().isNull()
    assert not Icons.close().isNull()
    assert not Icons.settings().isNull()
    assert not Icons.spectrum_1d().isNull()
    assert not Icons.map_2d().isNull()
    assert not Icons.wafer_map().isNull()
    assert not Icons.graph_line().isNull()
    assert not Icons.multipanel().isNull()

    # Legacy mapping lookups via Icons.get()
    assert not Icons.get("open.png").isNull()
    assert not Icons.get("save_color.png").isNull()
    assert not Icons.get("trash.png").isNull()
    assert not Icons.get("select-all.png").isNull()
    assert not Icons.get("calc.png").isNull()
    assert not Icons.get("2Dmap.png").isNull()
    assert not Icons.get("llm_ai.png").isNull()


def test_scientific_and_annotation_icons(qapp):
    """Verify dedicated annotation and scientific vector icons and legacy lookups."""
    # Annotation direct methods
    ann_methods = [
        Icons.ann_vline,
        Icons.ann_hline,
        Icons.ann_text,
        Icons.ann_arrow,
        Icons.ann_vspan,
        Icons.ann_hspan,
        Icons.ann_box,
        Icons.ann_callout,
        Icons.graph_point,
        Icons.eraser,
    ]
    for method in ann_methods:
        icon = method()
        assert isinstance(icon, QIcon)
        assert not icon.isNull()
        pix = icon.pixmap(QSize(20, 20))
        assert not pix.isNull()
        assert pix.width() == 20

    # Legacy filename translation
    legacy_keys = [
        "vline.png", "hline.png", "text.png", "arrow.png",
        "vspan.png", "hspan.png", "box_ann.png", "callout.png",
        "point.png", "eraser.png",
    ]
    for key in legacy_keys:
        icon = Icons.get(key)
        assert not icon.isNull(), f"Icons.get('{key}') returned null icon"


def test_svg_icon_engine_paint(qapp):
    """Verify that _SvgIconEngine paints cleanly into QPainter."""
    icon = svg_icon("spectrum_1d")
    img = QImage(32, 32, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)

    painter = QPainter(img)
    icon.paint(painter, QRect(0, 0, 32, 32))
    painter.end()

    # Verify pixels were painted (not all transparent)
    mv = memoryview(img.bits())
    alpha_slice = mv[3::4]
    assert any(a > 0 for a in alpha_slice)


def test_clear_icon_caches():
    """Verify clear_icon_caches runs cleanly without error."""
    clear_icon_caches()


def test_get_tinted_icon_bridge(qapp):
    """Verify that get_tinted_icon bridges legacy raster icon paths to vector icons."""
    # Standard foreground color should route to vector icon
    icon = get_tinted_icon("c:/dummy/path/save.png", "#EDEDEF")
    assert isinstance(icon, QIcon)
    assert not icon.isNull()

    # Custom tint color
    icon_custom = get_tinted_icon("c:/dummy/path/trash.png", "#EF4444")
    assert isinstance(icon_custom, QIcon)
    assert not icon_custom.isNull()
