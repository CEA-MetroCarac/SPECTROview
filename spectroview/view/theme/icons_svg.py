# spectroview/view/theme/icons_svg.py
"""Vector SVG Icon Engine for SPECTROview.

Provides crystal-clear, theme-adaptive geometric vector icons rendered via QSvgRenderer
using native QIconEngine at 1:1 physical screen pixels with zero UI emojis.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QByteArray, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QIconEngine, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

# ---------------------------------------------------------------------------
# Geometric stroke SVG templates (24x24 viewBox, stroke-width 1.75 / 1.8)
# ---------------------------------------------------------------------------
_SVG_TEMPLATES: dict[str, str] = {
    # ── Common UI & Navigation ──
    "sun": (
        '<circle cx="12" cy="12" r="4"/>'
        '<path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2'
        'M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>'
    ),
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "menu": (
        '<line x1="4" x2="20" y1="12" y2="12"/>'
        '<line x1="4" x2="20" y1="6" y2="6"/>'
        '<line x1="4" x2="20" y1="18" y2="18"/>'
    ),
    "settings": (
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08'
        'a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74'
        'l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 '
        '1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08'
        'a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 '
        '1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25'
        'a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "sidebar": (
        '<rect width="18" height="18" x="3" y="3" rx="2"/>'
        '<path d="M9 3v18"/>'
    ),
    "chevron_left": '<path d="m15 18-6-6 6-6"/>',
    "chevron_right": '<path d="m9 18 6-6-6-6"/>',
    "chevron_up": '<path d="m18 15-6-6-6 6"/>',
    "chevron_down": '<path d="m6 9 6 6 6-6"/>',
    "chevrons_up_down": '<path d="m7 15 5 5 5-5M7 9l5-5 5 5"/>',
    "external_link": (
        '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
        '<polyline points="15 3 21 3 21 9"/>'
        '<line x1="10" x2="21" y1="14" y2="3"/>'
    ),
    "close": (
        '<line x1="18" x2="6" y1="6" y2="18"/>'
        '<line x1="6" x2="18" y1="6" y2="18"/>'
    ),
    "plus": (
        '<line x1="12" x2="12" y1="5" y2="19"/>'
        '<line x1="5" x2="19" y1="12" y2="12"/>'
    ),
    "plus_circle": (
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M8 12h8"/>'
        '<path d="M12 8v8"/>'
    ),
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "check_circle": (
        '<circle cx="12" cy="12" r="10"/>'
        '<polyline points="9 12 12 15 16 9"/>'
    ),
    "more_horizontal": (
        '<circle cx="12" cy="12" r="1.5" fill="currentColor"/>'
        '<circle cx="19" cy="12" r="1.5" fill="currentColor"/>'
        '<circle cx="5" cy="12" r="1.5" fill="currentColor"/>'
    ),
    "more_vertical": (
        '<circle cx="12" cy="12" r="1.5" fill="currentColor"/>'
        '<circle cx="12" cy="5" r="1.5" fill="currentColor"/>'
        '<circle cx="12" cy="19" r="1.5" fill="currentColor"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="8"/>'
        '<path d="m21 21-4.3-4.3"/>'
    ),
    "bell": (
        '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>'
        '<path d="M13.73 21a2 2 0 0 1-3.46 0"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" x2="12" y1="16" y2="12"/>'
        '<line x1="12" x2="12.01" y1="8" y2="8"/>'
    ),
    "help_circle": (
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
        '<line x1="12" x2="12.01" y1="17" y2="17"/>'
    ),
    "minimize_windows": (
        '<rect width="18" height="14" x="3" y="5" rx="2"/>'
        '<line x1="7" x2="13" y1="15" y2="15"/>'
    ),
    "maximize": (
        '<rect width="18" height="18" x="3" y="3" rx="2"/>'
        '<polyline points="9 3 9 9 3 9"/>'
        '<polyline points="15 21 15 15 21 15"/>'
    ),
    "minimize": (
        '<polyline points="4 14 10 14 10 20"/>'
        '<polyline points="20 10 14 10 14 4"/>'
        '<line x1="14" x2="21" y1="10" y2="3"/>'
        '<line x1="3" x2="10" y1="21" y2="14"/>'
    ),
    "sliders": (
        '<line x1="4" x2="4" y1="21" y2="14"/>'
        '<line x1="4" x2="4" y1="10" y2="3"/>'
        '<line x1="12" x2="12" y1="21" y2="12"/>'
        '<line x1="12" x2="12" y1="8" y2="3"/>'
        '<line x1="20" x2="20" y1="21" y2="16"/>'
        '<line x1="20" x2="20" y1="12" y2="3"/>'
        '<line x1="1" x2="7" y1="14" y2="14"/>'
        '<line x1="9" x2="15" y1="8" y2="8"/>'
        '<line x1="17" x2="23" y1="16" y2="16"/>'
    ),
    "pin": (
        '<line x1="12" x2="12" y1="17" y2="22"/>'
        '<path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8'
        'a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24Z"/>'
    ),
    "book": (
        '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/>'
        '<path d="M6 6h10M6 10h10"/>'
    ),
    "github": (
        '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7'
        'A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0'
        'C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7'
        'A3.37 3.37 0 0 0 9 18.13V22"/>'
    ),

    # ── File, Dataset & Clipboard Operations ──
    "folder": (
        '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4'
        'a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'
    ),
    "folder_plus": (
        '<path d="M12 10v6M9 13h6"/>'
        '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4'
        'a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'
    ),
    "open": (
        '<path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4'
        'a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'
        '<polyline points="12 11 12 17"/>'
        '<polyline points="9 14 12 11 15 14"/>'
    ),
    "save": (
        '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>'
        '<polyline points="17 21 17 13 7 13 7 21"/>'
        '<polyline points="7 3 7 8 15 8"/>'
    ),
    "save_as": (
        '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>'
        '<path d="M11 17h6"/>'
        '<path d="M14 14l3 3-3 3"/>'
    ),
    "save_all": (
        '<path d="M16 21H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h10l4 4v10a2 2 0 0 1-2 2z"/>'
        '<path d="M8 3h11l4 4v11a2 2 0 0 1-2 2h-1"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M3 5v14a9 3 0 0 0 18 0V5"/>'
        '<path d="M3 12a9 3 0 0 0 18 0"/>'
    ),
    "file_text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
        '<path d="M10 9H8"/>'
        '<path d="M16 13H8"/>'
        '<path d="M16 17H8"/>'
    ),
    "file_spreadsheet": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
        '<path d="M8 13h8"/>'
        '<path d="M8 17h8"/>'
        '<path d="M12 9v12"/>'
    ),
    "file_pdf": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
        '<path d="M9 13v4M9 13h2a1.5 1.5 0 0 0 0-3H9v3zM15 10v7"/>'
    ),
    "copy": (
        '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
        '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>'
    ),
    "cut": (
        '<circle cx="6" cy="6" r="3"/>'
        '<circle cx="6" cy="18" r="3"/>'
        '<line x1="20" x2="8.12" y1="4" y2="15.88"/>'
        '<line x1="14.47" x2="20" y1="14.48" y2="20"/>'
        '<line x1="8.12" x2="12" y1="8.12" y2="12"/>'
    ),
    "paste": (
        '<rect width="14" height="14" x="5" y="8" rx="2"/>'
        '<path d="M16 8V6a2 2 0 0 0-2-2H10a2 2 0 0 0-2 2v2"/>'
        '<path d="M9 13h6M9 17h4"/>'
    ),
    "duplicate": (
        '<rect width="12" height="12" x="9" y="9" rx="2"/>'
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'
    ),
    "trash": (
        '<path d="M3 6h18"/>'
        '<path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>'
        '<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>'
    ),
    "clear": (
        '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<line x1="10" x2="10" y1="11" y2="17"/>'
        '<line x1="14" x2="14" y1="11" y2="17"/>'
    ),
    "eraser": (
        '<path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/>'
        '<path d="M22 21H7"/>'
        '<path d="m5 11 9 9"/>'
    ),
    "edit": (
        '<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>'
        '<path d="m15 5 4 4"/>'
    ),
    "undo": (
        '<path d="M3 7v6h6"/>'
        '<path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>'
    ),
    "redo": (
        '<path d="M21 7v6h-6"/>'
        '<path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13"/>'
    ),
    "reply": (
        '<polyline points="9 17 4 12 9 7"/>'
        '<path d="M20 18v-2a4 4 0 0 0-4-4H4"/>'
    ),
    "refresh": (
        '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/>'
        '<path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>'
        '<path d="M16 16h5v5"/>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" x2="12" y1="15" y2="3"/>'
    ),
    "upload": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/>'
        '<line x1="12" x2="12" y1="3" y2="15"/>'
    ),
    "export": (
        '<path d="M9 13h6m0 0l-3-3m3 3l-3 3"/>'
        '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v2"/>'
    ),
    "export_all": (
        '<path d="M10 12h6m0 0l-2.5-2.5m2.5 2.5l-2.5 2.5"/>'
        '<rect width="12" height="14" x="4" y="6" rx="2"/>'
        '<path d="M8 2h10a2 2 0 0 1 2 2v12"/>'
    ),
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "convert": (
        '<path d="m16 3 4 4-4 4"/>'
        '<path d="M20 7H4"/>'
        '<path d="m8 21-4-4 4-4"/>'
        '<path d="M4 17h16"/>'
    ),
    "calc": (
        '<rect width="16" height="20" x="4" y="2" rx="2"/>'
        '<line x1="8" x2="16" y1="6" y2="6"/>'
        '<line x1="16" x2="16" y1="14" y2="18"/>'
        '<path d="M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M8 18h.01M12 18h.01"/>'
    ),
    "recipe": (
        '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/>'
        '<path d="m9 10 2 2 4-4"/>'
    ),

    # ── Scientific Spectroscopy & Hyperspectral ──
    "spectrum_1d": '<path d="M2 19h20M3 19l4-3 3-10 3 8 3-4 4 9"/>',
    "map_2d": (
        '<rect width="18" height="18" x="3" y="3" rx="2"/>'
        '<line x1="3" x2="21" y1="9" y2="9"/>'
        '<line x1="3" x2="21" y1="15" y2="15"/>'
        '<line x1="9" x2="9" y1="3" y2="21"/>'
        '<line x1="15" x2="15" y1="3" y2="21"/>'
    ),
    "multipanel": (
        '<rect width="18" height="18" x="3" y="3" rx="2"/>'
        '<line x1="3" x2="21" y1="12" y2="12"/>'
        '<line x1="12" x2="12" y1="3" y2="21"/>'
    ),
    "wafer_map": (
        '<circle cx="12" cy="12" r="9"/>'
        '<line x1="10" x2="14" y1="21" y2="21"/>'
        '<line x1="12" x2="12" y1="6" y2="18"/>'
        '<line x1="6" x2="18" y1="12" y2="12"/>'
    ),
    "baseline": (
        '<path d="M2 16c3-2 6-13 10-13s7 11 10 13"/>'
        '<line x1="2" x2="22" y1="20" y2="20" stroke-dasharray="2.5 2.5"/>'
    ),
    "peak": (
        '<path d="M2 20c4 0 6-17 10-17s6 17 10 17"/>'
        '<line x1="12" x2="12" y1="3" y2="20" stroke-dasharray="2 2"/>'
    ),
    "bestfit": (
        '<path d="M2 19l3-4 3 2 4-13 4 9 3-5 3 11" opacity="0.55"/>'
        '<path d="M2 20c3-2 6-16 10-16s7 14 10 16"/>'
    ),
    "norm": (
        '<rect width="18" height="18" x="3" y="3" rx="2"/>'
        '<polyline points="7 14 12 9 17 14"/>'
        '<line x1="12" x2="12" y1="9" y2="17"/>'
    ),
    "stats": (
        '<path d="M3 19c3-1 6-7 9-7s6 6 9 7"/>'
        '<path d="M3 15c3-1 6-7 9-7s6 6 9 7" stroke-dasharray="2 2" opacity="0.5"/>'
        '<path d="M3 21c3-1 6-7 9-7s6 6 9 7" stroke-dasharray="2 2" opacity="0.5"/>'
    ),

    # ── Graphing & Publication Plotting ──
    "graph_line": (
        '<polyline points="3 3 3 21 21 21"/>'
        '<path d="m7 15 4-5 4 3 5-7"/>'
    ),
    "graph_scatter": (
        '<polyline points="3 3 3 21 21 21"/>'
        '<circle cx="7" cy="14" r="1.5" fill="currentColor"/>'
        '<circle cx="11" cy="9" r="1.5" fill="currentColor"/>'
        '<circle cx="15" cy="13" r="1.5" fill="currentColor"/>'
        '<circle cx="19" cy="6" r="1.5" fill="currentColor"/>'
    ),
    "graph_bar": (
        '<polyline points="3 3 3 21 21 21"/>'
        '<rect width="3" height="8" x="6" y="13"/>'
        '<rect width="3" height="13" x="11" y="8"/>'
        '<rect width="3" height="5" x="16" y="16"/>'
    ),
    "graph_box": (
        '<polyline points="3 3 3 21 21 21"/>'
        '<rect width="6" height="8" x="10" y="8"/>'
        '<line x1="13" x2="13" y1="4" y2="8"/>'
        '<line x1="13" x2="13" y1="16" y2="20"/>'
        '<line x1="10" x2="16" y1="12" y2="12"/>'
        '<line x1="11" x2="15" y1="4" y2="4"/>'
        '<line x1="11" x2="15" y1="20" y2="20"/>'
    ),
    "graph_histogram": (
        '<polyline points="3 3 3 21 21 21"/>'
        '<rect width="3" height="5" x="5" y="16"/>'
        '<rect width="3" height="12" x="9" y="9"/>'
        '<rect width="3" height="15" x="13" y="6"/>'
        '<rect width="3" height="7" x="17" y="14"/>'
    ),
    "graph_trendline": (
        '<polyline points="3 3 3 21 21 21"/>'
        '<line x1="5" x2="20" y1="18" y2="6"/>'
        '<circle cx="8" cy="14" r="1" fill="currentColor"/>'
        '<circle cx="13" cy="12" r="1" fill="currentColor"/>'
        '<circle cx="17" cy="7" r="1" fill="currentColor"/>'
    ),
    "styling": (
        '<circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/>'
        '<circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/>'
        '<circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/>'
        '<circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/>'
        '<path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.93 0 1.5-.66 1.5-1.5 0-.39-.15-.74-.39-1.04'
        '-.24-.31-.38-.69-.38-1.1 0-.97.79-1.75 1.77-1.75H16c3.31 0 6-2.69 6-6 0-5.52-4.48-10-10-10z"/>'
    ),
    "legend": (
        '<rect width="18" height="14" x="3" y="5" rx="2"/>'
        '<line x1="6" x2="9" y1="9" y2="9"/>'
        '<line x1="12" x2="18" y1="9" y2="9"/>'
        '<line x1="6" x2="9" y1="15" y2="15"/>'
        '<line x1="12" x2="18" y1="15" y2="15"/>'
    ),
    "zoom": (
        '<circle cx="11" cy="11" r="7"/>'
        '<line x1="21" x2="16.65" y1="21" y2="16.65"/>'
        '<line x1="11" x2="11" y1="8" y2="14"/>'
        '<line x1="8" x2="14" y1="11" y2="11"/>'
    ),
    "pan": (
        '<polyline points="5 9 2 12 5 15"/>'
        '<polyline points="9 5 12 2 15 5"/>'
        '<polyline points="15 19 12 22 9 19"/>'
        '<polyline points="19 9 22 12 19 15"/>'
        '<line x1="2" x2="22" y1="12" y2="12"/>'
        '<line x1="12" x2="12" y1="2" y2="22"/>'
    ),
    "rescale": (
        '<polyline points="15 3 21 3 21 9"/>'
        '<polyline points="9 21 3 21 3 15"/>'
        '<polyline points="21 15 21 21 15 21"/>'
        '<polyline points="3 9 3 3 9 3"/>'
    ),

    # ── AI Chat & Status ──
    "sparkles": (
        '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8'
        'a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>'
    ),
    "brain": (
        '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 '
        '3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04Z"/>'
        '<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 '
        '3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04Z"/>'
    ),
    "send": (
        '<path d="m22 2-7 20-4-9-9-4Z"/>'
        '<path d="M22 2 11 13"/>'
    ),
    "stop": '<rect width="14" height="14" x="5" y="5" rx="2"/>',
    "play": '<polygon points="6 4 20 12 6 20 6 4"/>',
    "mic": (
        '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>'
        '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/>'
        '<line x1="12" x2="12" y1="19" y2="22"/>'
    ),
    "paperclip": (
        '<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 '
        '8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>'
    ),
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "chat_new": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
        '<line x1="12" x2="12" y1="7" y2="13"/>'
        '<line x1="9" x2="15" y1="10" y2="10"/>'
    ),
    "history": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/>'
        '<path d="M12 7v5l4 2"/>'
    ),

    # ── Graph Annotations & Plots ──
    "ann_vline": (
        '<rect width="18" height="18" x="3" y="3" rx="2" stroke-dasharray="2 2" opacity="0.4"/>'
        '<line x1="12" x2="12" y1="2" y2="22"/>'
    ),
    "ann_hline": (
        '<rect width="18" height="18" x="3" y="3" rx="2" stroke-dasharray="2 2" opacity="0.4"/>'
        '<line x1="2" x2="22" y1="12" y2="12"/>'
    ),
    "ann_text": (
        '<polyline points="4 7 4 4 20 4 20 7"/>'
        '<line x1="9" x2="15" y1="20" y2="20"/>'
        '<line x1="12" x2="12" y1="4" y2="20"/>'
    ),
    "ann_arrow": (
        '<line x1="5" x2="19" y1="19" y2="5"/>'
        '<polyline points="10 5 19 5 19 14"/>'
    ),
    "ann_vspan": (
        '<line x1="7" x2="7" y1="3" y2="21"/>'
        '<line x1="17" x2="17" y1="3" y2="21"/>'
        '<rect width="10" height="18" x="7" y="3" fill="currentColor" fill-opacity="0.2" stroke="none"/>'
    ),
    "ann_hspan": (
        '<line x1="3" x2="21" y1="7" y2="7"/>'
        '<line x1="3" x2="21" y1="17" y2="17"/>'
        '<rect width="18" height="10" x="3" y="7" fill="currentColor" fill-opacity="0.2" stroke="none"/>'
    ),
    "ann_box": '<rect width="16" height="14" x="4" y="5" rx="2" stroke-dasharray="3 3"/>',
    "ann_callout": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
        '<line x1="8" x2="16" y1="9" y2="9"/>'
        '<line x1="8" x2="13" y1="13" y2="13"/>'
    ),
    "graph_point": (
        '<polyline points="3 3 3 21 21 21"/>'
        '<circle cx="8" cy="15" r="2"/>'
        '<circle cx="13" cy="8" r="2"/>'
        '<circle cx="18" cy="12" r="2"/>'
    ),
}

# Aliases
_SVG_TEMPLATES["clock"] = _SVG_TEMPLATES["history"]
_SVG_TEMPLATES["refresh_cw"] = _SVG_TEMPLATES["refresh"]
_SVG_TEMPLATES["palette_color"] = _SVG_TEMPLATES["styling"]
_SVG_TEMPLATES["line"] = _SVG_TEMPLATES["graph_line"]
_SVG_TEMPLATES["scatter"] = _SVG_TEMPLATES["graph_scatter"]
_SVG_TEMPLATES["bar"] = _SVG_TEMPLATES["graph_bar"]
_SVG_TEMPLATES["box"] = _SVG_TEMPLATES["graph_box"]
_SVG_TEMPLATES["histogram"] = _SVG_TEMPLATES["graph_histogram"]
_SVG_TEMPLATES["trendline"] = _SVG_TEMPLATES["graph_trendline"]
_SVG_TEMPLATES["ai_chat"] = _SVG_TEMPLATES["sparkles"]
_SVG_TEMPLATES["point"] = _SVG_TEMPLATES["graph_point"]
_SVG_TEMPLATES["vline"] = _SVG_TEMPLATES["ann_vline"]
_SVG_TEMPLATES["hline"] = _SVG_TEMPLATES["ann_hline"]
_SVG_TEMPLATES["arrow"] = _SVG_TEMPLATES["ann_arrow"]
_SVG_TEMPLATES["vspan"] = _SVG_TEMPLATES["ann_vspan"]
_SVG_TEMPLATES["hspan"] = _SVG_TEMPLATES["ann_hspan"]
_SVG_TEMPLATES["box_ann"] = _SVG_TEMPLATES["ann_box"]
_SVG_TEMPLATES["callout"] = _SVG_TEMPLATES["ann_callout"]

# ---------------------------------------------------------------------------
# Caches & Engine Lifetimes
# ---------------------------------------------------------------------------
_PIXMAP_CACHE: dict[tuple[str, int, str, float], QPixmap] = {}
_IMAGE_CACHE: dict[tuple[str, int, str], QImage] = {}
_ICON_CACHE: dict[tuple[str, str], QIcon] = {}
_ENGINE_REFS: list[_SvgIconEngine] = []


def _resolve_color(color: str | None = None) -> str:
    if color:
        return color
    from spectroview.view.theme import current_tokens
    return current_tokens().text_primary


def clear_icon_caches() -> None:
    """Clear cached pixmaps and rendered images while keeping live QIconEngine references intact."""
    _PIXMAP_CACHE.clear()
    _IMAGE_CACHE.clear()


def _render_image_1to1(name: str, phys_dim: int, color: str) -> QImage:
    """Render 1:1 physical pixel QImage from SVG vector template with optical tuning."""
    key = (name, phys_dim, color)
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached

    inner = _SVG_TEMPLATES.get(name)
    if inner is None:
        inner = _SVG_TEMPLATES.get("sparkles", "")
    stroke_w = "1.8" if phys_dim <= 20 else "1.75"
    svg_xml = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'width="{phys_dim}" height="{phys_dim}" fill="none" stroke="{color}" '
        f'color="{color}" stroke-width="{stroke_w}" stroke-linecap="round" stroke-linejoin="round">'
        f'{inner}</svg>'
    )
    img = QImage(phys_dim, phys_dim, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg_xml.encode("utf-8")))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(p, QRectF(0.0, 0.0, float(phys_dim), float(phys_dim)))
    p.end()

    if phys_dim <= 22:
        # Subtle linear alpha boost for rich contrast on standard DPI displays
        mv = memoryview(img.bits())
        alpha_slice = mv[3::4]
        for i in range(len(alpha_slice)):
            a = alpha_slice[i]
            if 0 < a < 255:
                alpha_slice[i] = min(255, round(a * 1.12))

    img_prem = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    _IMAGE_CACHE[key] = img_prem
    return img_prem


class _SvgIconEngine(QIconEngine):
    """High-DPI crisp vector icon engine rendering directly to physical pixel bounds with dynamic theme colors."""

    def __init__(
        self,
        name: str,
        explicit_color: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._name = name
        if isinstance(explicit_color, str):
            self._explicit_color: str | None = explicit_color
        elif args and isinstance(args[-1], str):
            self._explicit_color = args[-1]
        else:
            self._explicit_color = None
        _ENGINE_REFS.append(self)

    def clone(self) -> QIconEngine:
        return _SvgIconEngine(self._name, self._explicit_color)

    def _active_color(self, mode: QIcon.Mode) -> str:
        from spectroview.view.theme import current_tokens
        tok = current_tokens()
        if mode == QIcon.Mode.Disabled:
            return tok.text_muted
        return self._explicit_color if self._explicit_color else tok.text_primary

    def paint(
        self,
        painter: QPainter,
        rect: QRect,
        mode: QIcon.Mode,
        state: QIcon.State,
    ) -> None:
        dev = painter.device()
        dpr = dev.devicePixelRatioF() if dev is not None and hasattr(dev, "devicePixelRatioF") else 1.0
        if dpr <= 0.0:
            dpr = 1.0
        phys_dim = max(1, min(round(rect.width() * dpr), round(rect.height() * dpr)))
        col = self._active_color(mode)
        img = _render_image_1to1(self._name, phys_dim, col)

        dx = rect.x() + (round((rect.width() * dpr - phys_dim) / 2.0)) / dpr
        dy = rect.y() + (round((rect.height() * dpr - phys_dim) / 2.0)) / dpr
        target_rect = QRectF(dx, dy, float(phys_dim) / dpr, float(phys_dim) / dpr)
        painter.drawImage(target_rect, img)

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(
        self,
        size: QSize,
        mode: QIcon.Mode,
        state: QIcon.State,
        scale: float,
    ) -> QPixmap:
        if scale <= 0.0:
            scale = 1.0
        phys_dim = max(1, min(round(size.width() * scale), round(size.height() * scale)))
        col = self._active_color(mode)
        img = _render_image_1to1(self._name, phys_dim, col)
        pix = QPixmap.fromImage(img)
        pix.setDevicePixelRatio(scale)
        return pix


def svg_pixmap(
    name: str,
    size: int = 16,
    color: str | None = None,
    *,
    dpr: float = 2.0,
) -> QPixmap:
    """Render a named vector SVG icon into a crisp, high-DPI QPixmap."""
    resolved_color = _resolve_color(color)
    key = (name, size, resolved_color, dpr)
    cached = _PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached

    phys_dim = max(1, round(size * dpr))
    img = _render_image_1to1(name, phys_dim, resolved_color)
    pix = QPixmap.fromImage(img)
    pix.setDevicePixelRatio(dpr)
    _PIXMAP_CACHE[key] = pix
    return pix


def svg_icon(
    name: str,
    size: int = 16,
    color: str | None = None,
) -> QIcon:
    """Return a crisp vector QIcon powered by _SvgIconEngine with dynamic theme color resolution."""
    key = (name, color or "")
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached

    engine = _SvgIconEngine(name, explicit_color=color)
    icon = QIcon(engine)
    _ICON_CACHE[key] = icon
    return icon
