# spectroview/view/theme/tokens.py
"""
Design-token definitions for every SPECTROview theme.

Each ``ThemeTokens`` instance is a frozen bag of CSS-ready colour strings
(``rgba(…)`` / ``rgb(…)`` / hex) keyed by *semantic role*, not widget name.
The QSS template in ``qss_template.py`` references these tokens by name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeTokens:
    """Immutable collection of colour tokens for a single theme."""

    # ── Surfaces ─────────────────────────────────────────────
    surface_bg: str              # Main window / QMainWindow background
    surface_card: str            # GroupBox, tab-pane, card backgrounds
    surface_card_border: str     # Card/groupbox border
    surface_input: str           # Input fields, combos, spinboxes
    surface_overlay: str         # Menus, tooltips, dialogs
    surface_list: str            # List/tree/scroll-area backgrounds
    surface_table: str           # Table background
    surface_table_alt: str       # Alternating table row
    surface_toolbar: str         # Top toolbar background

    # ── Text ─────────────────────────────────────────────────
    text_primary: str            # Primary content text
    text_secondary: str          # Subdued labels, titles
    text_disabled: str           # Disabled controls
    text_placeholder: str        # Input placeholder text

    # ── Borders ──────────────────────────────────────────────
    border_subtle: str           # Default resting borders
    border_medium: str           # Hover / focus borders
    border_strong: str           # Active / selected borders
    border_separator: str        # Toolbar separators, HLine/VLine

    # ── Accent ───────────────────────────────────────────────
    accent: str                  # Primary accent (highlight, checked)
    accent_light: str            # Lighter accent variant
    accent_bg: str               # Accent as a translucent background
    accent_bg_strong: str        # Accent background, higher opacity
    accent_border: str           # Accent-coloured border

    # ── Interactive states ───────────────────────────────────
    hover_bg: str                # Generic hover overlay
    hover_bg_strong: str         # Stronger hover (e.g. on buttons)
    pressed_bg: str              # Pressed / active background
    selected_bg: str             # Selected-item background
    selected_border: str         # Selected-item border

    # ── Scrollbar / slider ───────────────────────────────────
    scrollbar_thumb: str
    scrollbar_thumb_hover: str
    slider_groove: str
    slider_filled: str
    slider_handle: str
    slider_handle_border: str

    # ── Progress bar ─────────────────────────────────────────
    progress_bg: str
    progress_chunk_start: str
    progress_chunk_mid: str
    progress_chunk_end: str

    # ── Checkbox indicator ───────────────────────────────────
    checkbox_bg: str
    checkbox_border: str
    checkbox_checked_bg: str
    checkbox_checked_border: str
    checkbox_disabled_bg: str
    checkbox_disabled_border: str
    checkbox_checked_disabled_bg: str
    checkbox_checked_disabled_border: str

    # ── Tab bar ──────────────────────────────────────────────
    tab_bg: str
    tab_border: str
    tab_text: str
    tab_selected_bg: str
    tab_selected_border: str
    tab_selected_bottom: str     # Bottom border of selected tab (pane seam)
    tab_hover_bg: str
    tab_hover_text: str
    tab_pane_bg: str
    tab_pane_border: str

    # ── Miscellaneous ────────────────────────────────────────
    tooltip_bg: str
    tooltip_border: str
    tooltip_text: str
    dialog_bg: str
    menu_bg: str
    menu_border: str
    menu_item_hover: str
    menu_separator: str
    gridline: str
    header_bg: str
    header_border_right: str
    header_border_bottom: str
    header_text: str
    frame_line_color: str

    # ── Icon helpers ─────────────────────────────────────────
    arrow_suffix: str            # "" for dark themes, "-dark" for light
    is_dark: bool                # Convenience flag


# ═══════════════════════════════════════════════════════════════
#  DARK THEME
# ═══════════════════════════════════════════════════════════════
DARK_TOKENS = ThemeTokens(
    # Surfaces
    surface_bg="rgb(30, 30, 32)",
    surface_card="rgba(255, 255, 255, 0.06)",
    surface_card_border="rgba(255, 255, 255, 0.08)",
    surface_input="rgba(255, 255, 255, 0.03)",
    surface_overlay="rgba(38, 38, 40, 250)",
    surface_list="rgba(0, 0, 0, 0.15)",
    surface_table="rgba(0, 0, 0, 0.20)",
    surface_table_alt="rgba(255, 255, 255, 0.04)",
    surface_toolbar="rgba(36, 36, 38, 245)",

    # Text
    text_primary="rgba(255, 255, 255, 0.92)",
    text_secondary="rgba(255, 255, 255, 0.70)",
    text_disabled="rgba(255, 255, 255, 0.30)",
    text_placeholder="rgba(255, 255, 255, 0.35)",

    # Borders
    border_subtle="rgba(255, 255, 255, 0.10)",
    border_medium="rgba(255, 255, 255, 0.18)",
    border_strong="rgba(255, 255, 255, 0.25)",
    border_separator="rgba(255, 255, 255, 0.10)",

    # Accent  (blue)
    accent="rgba(64, 156, 255, 1.0)",
    accent_light="rgba(100, 180, 255, 1.0)",
    accent_bg="rgba(64, 156, 255, 0.18)",
    accent_bg_strong="rgba(64, 156, 255, 0.38)",
    accent_border="rgba(64, 156, 255, 0.65)",

    # Interactive
    hover_bg="rgba(255, 255, 255, 0.07)",
    hover_bg_strong="rgba(255, 255, 255, 0.10)",
    pressed_bg="rgba(64, 156, 255, 0.32)",
    selected_bg="rgba(64, 156, 255, 0.38)",
    selected_border="rgba(64, 156, 255, 0.65)",

    # Scrollbar / slider
    scrollbar_thumb="rgba(255, 255, 255, 0.14)",
    scrollbar_thumb_hover="rgba(255, 255, 255, 0.26)",
    slider_groove="rgba(255, 255, 255, 0.12)",
    slider_filled="rgba(64, 156, 255, 0.80)",
    slider_handle="rgba(255, 255, 255, 0.92)",
    slider_handle_border="rgba(0, 0, 0, 0.30)",

    # Progress bar
    progress_bg="rgba(0, 0, 0, 0.25)",
    progress_chunk_start="rgba(30, 100, 200, 0.75)",
    progress_chunk_mid="rgba(50, 135, 230, 0.80)",
    progress_chunk_end="rgba(64, 156, 255, 0.75)",

    # Checkbox
    checkbox_bg="rgba(0, 0, 0, 0.25)",
    checkbox_border="rgba(255, 255, 255, 0.15)",
    checkbox_checked_bg="rgba(64, 156, 255, 0.90)",
    checkbox_checked_border="rgba(42, 130, 218, 0.90)",
    checkbox_disabled_bg="rgba(255, 255, 255, 0.05)",
    checkbox_disabled_border="rgba(255, 255, 255, 0.05)",
    checkbox_checked_disabled_bg="rgba(255, 255, 255, 0.15)",
    checkbox_checked_disabled_border="rgba(255, 255, 255, 0.15)",

    # Tabs
    tab_bg="rgba(255, 255, 255, 0.04)",
    tab_border="rgba(255, 255, 255, 0.08)",
    tab_text="rgba(255, 255, 255, 0.65)",
    tab_selected_bg="rgba(255, 255, 255, 0.14)",
    tab_selected_border="rgba(255, 255, 255, 0.22)",
    tab_selected_bottom="rgba(30, 30, 32, 255)",
    tab_hover_bg="rgba(255, 255, 255, 0.07)",
    tab_hover_text="rgba(255, 255, 255, 0.88)",
    tab_pane_bg="rgba(34, 34, 36, 235)",
    tab_pane_border="rgba(255, 255, 255, 0.08)",

    # Misc
    tooltip_bg="rgba(48, 48, 50, 248)",
    tooltip_border="rgba(255, 255, 255, 0.14)",
    tooltip_text="white",
    dialog_bg="rgb(38, 38, 40)",
    menu_bg="rgba(40, 40, 42, 252)",
    menu_border="rgba(255, 255, 255, 0.12)",
    menu_item_hover="rgba(255, 255, 255, 0.10)",
    menu_separator="rgba(255, 255, 255, 0.08)",
    gridline="rgba(255, 255, 255, 0.07)",
    header_bg="rgba(255, 255, 255, 0.05)",
    header_border_right="rgba(255, 255, 255, 0.07)",
    header_border_bottom="rgba(255, 255, 255, 0.10)",
    header_text="rgba(255, 255, 255, 0.78)",
    frame_line_color="rgba(255, 255, 255, 0.08)",

    # Icons
    arrow_suffix="",
    is_dark=True,
)


# ═══════════════════════════════════════════════════════════════
#  SOFT DARK THEME
# ═══════════════════════════════════════════════════════════════
SOFT_DARK_TOKENS = ThemeTokens(
    # Surfaces — warmer / lighter grays than DARK
    surface_bg="rgb(50, 50, 52)",
    surface_card="rgba(255, 255, 255, 0.04)",
    surface_card_border="rgba(255, 255, 255, 0.07)",
    surface_input="rgba(0, 0, 0, 0.12)",
    surface_overlay="rgba(58, 58, 60, 250)",
    surface_list="rgba(0, 0, 0, 0.10)",
    surface_table="rgba(0, 0, 0, 0.10)",
    surface_table_alt="rgba(255, 255, 255, 0.04)",
    surface_toolbar="rgba(54, 54, 56, 245)",

    # Text
    text_primary="rgba(255, 255, 255, 0.90)",
    text_secondary="rgba(255, 255, 255, 0.65)",
    text_disabled="rgba(255, 255, 255, 0.28)",
    text_placeholder="rgba(255, 255, 255, 0.32)",

    # Borders
    border_subtle="rgba(255, 255, 255, 0.08)",
    border_medium="rgba(255, 255, 255, 0.16)",
    border_strong="rgba(255, 255, 255, 0.22)",
    border_separator="rgba(255, 255, 255, 0.08)",

    # Accent
    accent="rgba(64, 156, 255, 1.0)",
    accent_light="rgba(100, 180, 255, 1.0)",
    accent_bg="rgba(64, 156, 255, 0.16)",
    accent_bg_strong="rgba(64, 156, 255, 0.35)",
    accent_border="rgba(64, 156, 255, 0.60)",

    # Interactive
    hover_bg="rgba(255, 255, 255, 0.06)",
    hover_bg_strong="rgba(255, 255, 255, 0.09)",
    pressed_bg="rgba(64, 156, 255, 0.30)",
    selected_bg="rgba(64, 156, 255, 0.35)",
    selected_border="rgba(64, 156, 255, 0.60)",

    # Scrollbar / slider
    scrollbar_thumb="rgba(255, 255, 255, 0.12)",
    scrollbar_thumb_hover="rgba(255, 255, 255, 0.24)",
    slider_groove="rgba(255, 255, 255, 0.12)",
    slider_filled="rgba(64, 156, 255, 0.78)",
    slider_handle="rgba(255, 255, 255, 0.90)",
    slider_handle_border="rgba(0, 0, 0, 0.28)",

    # Progress bar
    progress_bg="rgba(0, 0, 0, 0.18)",
    progress_chunk_start="rgba(30, 100, 200, 0.70)",
    progress_chunk_mid="rgba(50, 135, 230, 0.75)",
    progress_chunk_end="rgba(64, 156, 255, 0.70)",

    # Checkbox
    checkbox_bg="rgba(0, 0, 0, 0.20)",
    checkbox_border="rgba(255, 255, 255, 0.14)",
    checkbox_checked_bg="rgba(64, 156, 255, 0.90)",
    checkbox_checked_border="rgba(42, 130, 218, 0.90)",
    checkbox_disabled_bg="rgba(255, 255, 255, 0.05)",
    checkbox_disabled_border="rgba(255, 255, 255, 0.05)",
    checkbox_checked_disabled_bg="rgba(255, 255, 255, 0.15)",
    checkbox_checked_disabled_border="rgba(255, 255, 255, 0.15)",

    # Tabs
    tab_bg="rgba(255, 255, 255, 0.04)",
    tab_border="rgba(255, 255, 255, 0.07)",
    tab_text="rgba(255, 255, 255, 0.60)",
    tab_selected_bg="rgba(255, 255, 255, 0.13)",
    tab_selected_border="rgba(255, 255, 255, 0.20)",
    tab_selected_bottom="rgba(50, 50, 52, 255)",
    tab_hover_bg="rgba(255, 255, 255, 0.06)",
    tab_hover_text="rgba(255, 255, 255, 0.82)",
    tab_pane_bg="rgba(54, 54, 56, 235)",
    tab_pane_border="rgba(255, 255, 255, 0.07)",

    # Misc
    tooltip_bg="rgba(64, 64, 66, 248)",
    tooltip_border="rgba(255, 255, 255, 0.12)",
    tooltip_text="white",
    dialog_bg="rgb(54, 54, 56)",
    menu_bg="rgba(56, 56, 58, 252)",
    menu_border="rgba(255, 255, 255, 0.10)",
    menu_item_hover="rgba(255, 255, 255, 0.09)",
    menu_separator="rgba(255, 255, 255, 0.07)",
    gridline="rgba(255, 255, 255, 0.06)",
    header_bg="rgba(255, 255, 255, 0.05)",
    header_border_right="rgba(255, 255, 255, 0.06)",
    header_border_bottom="rgba(255, 255, 255, 0.08)",
    header_text="rgba(255, 255, 255, 0.75)",
    frame_line_color="rgba(255, 255, 255, 0.07)",

    # Icons
    arrow_suffix="",
    is_dark=True,
)


# ═══════════════════════════════════════════════════════════════
#  LIGHT THEME
# ═══════════════════════════════════════════════════════════════
LIGHT_TOKENS = ThemeTokens(
    # Surfaces
    surface_bg="#E8EAED",
    surface_card="rgba(0, 0, 0, 0.03)",
    surface_card_border="rgba(0, 0, 0, 0.08)",
    surface_input="rgba(255, 255, 255, 220)",
    surface_overlay="rgba(255, 255, 255, 250)",
    surface_list="#EDF0F2",
    surface_table="rgba(255, 255, 255, 200)",
    surface_table_alt="rgba(0, 0, 0, 0.03)",
    surface_toolbar="rgba(245, 245, 245, 248)",

    # Text
    text_primary="rgba(0, 0, 0, 0.88)",
    text_secondary="rgba(0, 0, 0, 0.60)",
    text_disabled="rgba(0, 0, 0, 0.30)",
    text_placeholder="rgba(0, 0, 0, 0.35)",

    # Borders
    border_subtle="rgba(0, 0, 0, 0.08)",
    border_medium="rgba(0, 0, 0, 0.15)",
    border_strong="rgba(0, 0, 0, 0.22)",
    border_separator="rgba(0, 0, 0, 0.08)",

    # Accent
    accent="rgba(42, 130, 218, 1.0)",
    accent_light="rgba(64, 156, 255, 1.0)",
    accent_bg="rgba(42, 130, 218, 0.12)",
    accent_bg_strong="rgba(42, 130, 218, 0.30)",
    accent_border="rgba(42, 130, 218, 0.55)",

    # Interactive
    hover_bg="rgba(0, 0, 0, 0.04)",
    hover_bg_strong="rgba(0, 0, 0, 0.07)",
    pressed_bg="rgba(64, 156, 255, 0.22)",
    selected_bg="rgba(42, 130, 218, 0.15)",
    selected_border="rgba(42, 130, 218, 0.45)",

    # Scrollbar / slider
    scrollbar_thumb="rgba(0, 0, 0, 0.12)",
    scrollbar_thumb_hover="rgba(0, 0, 0, 0.24)",
    slider_groove="rgba(0, 0, 0, 0.10)",
    slider_filled="rgba(42, 130, 218, 0.80)",
    slider_handle="white",
    slider_handle_border="rgba(0, 0, 0, 0.20)",

    # Progress bar
    progress_bg="rgba(0, 0, 0, 0.06)",
    progress_chunk_start="rgba(42, 130, 218, 0.55)",
    progress_chunk_mid="rgba(55, 145, 235, 0.65)",
    progress_chunk_end="rgba(64, 156, 255, 0.60)",

    # Checkbox
    checkbox_bg="rgba(0, 0, 0, 0.06)",
    checkbox_border="rgba(0, 0, 0, 0.15)",
    checkbox_checked_bg="rgba(42, 130, 218, 0.90)",
    checkbox_checked_border="rgba(25, 100, 200, 0.90)",
    checkbox_disabled_bg="rgba(0, 0, 0, 0.05)",
    checkbox_disabled_border="rgba(0, 0, 0, 0.05)",
    checkbox_checked_disabled_bg="rgba(0, 0, 0, 0.15)",
    checkbox_checked_disabled_border="rgba(0, 0, 0, 0.15)",

    # Tabs
    tab_bg="rgba(0, 0, 0, 0.03)",
    tab_border="rgba(0, 0, 0, 0.06)",
    tab_text="rgba(0, 0, 0, 0.50)",
    tab_selected_bg="rgba(255, 255, 255, 0.90)",
    tab_selected_border="rgba(0, 0, 0, 0.18)",
    tab_selected_bottom="#EDF0F2",
    tab_hover_bg="rgba(0, 0, 0, 0.05)",
    tab_hover_text="rgba(0, 0, 0, 0.72)",
    tab_pane_bg="#EDF0F2",
    tab_pane_border="rgba(0, 0, 0, 0.07)",

    # Misc
    tooltip_bg="rgba(255, 255, 255, 248)",
    tooltip_border="rgba(0, 0, 0, 0.10)",
    tooltip_text="rgba(0, 0, 0, 0.85)",
    dialog_bg="rgb(245, 245, 245)",
    menu_bg="rgba(255, 255, 255, 252)",
    menu_border="rgba(0, 0, 0, 0.10)",
    menu_item_hover="rgba(0, 0, 0, 0.06)",
    menu_separator="rgba(0, 0, 0, 0.07)",
    gridline="rgba(0, 0, 0, 0.06)",
    header_bg="rgba(0, 0, 0, 0.03)",
    header_border_right="rgba(0, 0, 0, 0.06)",
    header_border_bottom="rgba(0, 0, 0, 0.08)",
    header_text="rgba(0, 0, 0, 0.65)",
    frame_line_color="rgba(0, 0, 0, 0.07)",

    # Icons
    arrow_suffix="-dark",
    is_dark=False,
)


# ── Lookup table ─────────────────────────────────────────────
THEME_TOKENS: dict[str, ThemeTokens] = {
    "dark": DARK_TOKENS,
    "soft_dark": SOFT_DARK_TOKENS,
    "light": LIGHT_TOKENS,
    # Classic themes reuse the palette-only path; tokens are still
    # available for downstream helpers like icon-color selection.
    "classic_dark": DARK_TOKENS,
    "classic_light": LIGHT_TOKENS,
}
