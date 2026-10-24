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
    progress_text: str
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
    checkbox_checked_image: str
    checkbox_checked_disabled_image: str
    checkbox_indeterminate_image: str

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
    icon_color: str              # Hex colour for tinted GUI icons (e.g. "#F0F0F0")


# ═══════════════════════════════════════════════════════════════
#  DARK THEME
# ═══════════════════════════════════════════════════════════════
DARK_TOKENS = ThemeTokens(
    # Surfaces
    surface_bg="rgb(33, 33, 33)",
    surface_card="rgba(255, 255, 255, 0.08)",
    surface_card_border="rgba(255, 255, 255, 0.08)",
    surface_input="rgba(255, 255, 255, 0.03)",
    surface_overlay="rgba(40, 40, 40, 240)",
    surface_list="rgba(0, 0, 0, 0.15)",
    surface_table="rgba(0, 0, 0, 0.95)",
    surface_table_alt="rgba(255, 255, 255, 0.18)",
    surface_toolbar="rgba(40, 40, 40, 240)",

    # Text
    text_primary="rgba(255, 255, 255, 0.90)",
    text_secondary="rgba(255, 255, 255, 0.80)",
    text_disabled="rgba(255, 255, 255, 0.30)",
    text_placeholder="rgba(255, 255, 255, 0.35)",

    # Borders
    border_subtle="rgba(255, 255, 255, 0.12)",
    border_medium="rgba(255, 255, 255, 0.20)",
    border_strong="rgba(255, 255, 255, 0.25)",
    border_separator="rgba(255, 255, 255, 0.12)",

    # Accent
    accent="rgba(64, 156, 255, 1.0)",
    accent_light="rgba(100, 180, 255, 1.0)",
    accent_bg="rgba(64, 156, 255, 0.16)",
    accent_bg_strong="rgba(64, 156, 255, 0.35)",
    accent_border="rgba(64, 156, 255, 0.60)",

    # Interactive
    hover_bg="rgba(255, 255, 255, 0.10)",
    hover_bg_strong="rgba(255, 255, 255, 0.15)",
    pressed_bg="rgba(64, 156, 255, 0.30)",
    selected_bg="rgba(64, 156, 255, 0.35)",
    selected_border="rgba(64, 156, 255, 0.60)",

    # Scrollbar / slider
    scrollbar_thumb="rgba(255, 255, 255, 0.15)",
    scrollbar_thumb_hover="rgba(255, 255, 255, 0.25)",
    slider_groove="rgba(255, 255, 255, 0.15)",
    slider_filled="rgba(64, 156, 255, 0.78)",
    slider_handle="rgba(255, 255, 255, 0.90)",
    slider_handle_border="rgba(0, 0, 0, 0.30)",

    # Progress bar
    progress_bg="rgba(0, 0, 0, 0.25)",
    progress_text="#FFFFFF",
    progress_chunk_start="rgba(30, 100, 200, 0.70)",
    progress_chunk_mid="rgba(50, 135, 230, 0.75)",
    progress_chunk_end="rgba(64, 156, 255, 0.70)",

    # Checkbox
    checkbox_bg="rgba(0, 0, 0, 0.25)",
    checkbox_border="rgba(255, 255, 255, 0.15)",
    checkbox_checked_bg="rgba(64, 156, 255, 0.90)",
    checkbox_checked_border="rgba(42, 130, 218, 0.90)",
    checkbox_disabled_bg="rgba(255, 255, 255, 0.05)",
    checkbox_disabled_border="rgba(255, 255, 255, 0.05)",
    checkbox_checked_disabled_bg="rgba(255, 255, 255, 0.15)",
    checkbox_checked_disabled_border="rgba(255, 255, 255, 0.15)",
    checkbox_checked_image="check-white.svg",
    checkbox_checked_disabled_image="check-white-disabled.svg",
    checkbox_indeterminate_image="indeterminate-white.svg",

    # Tabs
    tab_bg="rgba(255, 255, 255, 0.04)",
    tab_border="rgba(255, 255, 255, 0.08)",
    tab_text="rgba(255, 255, 255, 0.70)",
    tab_selected_bg="rgba(255, 255, 255, 0.15)",
    tab_selected_border="rgba(255, 255, 255, 0.22)",
    tab_selected_bottom="rgba(38, 38, 38, 230)",
    tab_hover_bg="rgba(255, 255, 255, 0.07)",
    tab_hover_text="rgba(255, 255, 255, 0.90)",
    tab_pane_bg="rgba(38, 38, 38, 230)",
    tab_pane_border="rgba(255, 255, 255, 0.10)",

    # Misc
    tooltip_bg="rgba(55, 55, 55, 240)",
    tooltip_border="rgba(255, 255, 255, 0.20)",
    tooltip_text="rgba(240, 240, 240, 255)",
    dialog_bg="rgb(33, 33, 33)",
    menu_bg="rgba(40, 40, 40, 250)",
    menu_border="rgba(255, 255, 255, 0.15)",
    menu_item_hover="rgba(255, 255, 255, 0.10)",
    menu_separator="rgba(255, 255, 255, 0.10)",
    gridline="rgba(255, 255, 255, 0.08)",
    header_bg="rgba(255, 255, 255, 0.05)",
    header_border_right="rgba(255, 255, 255, 0.08)",
    header_border_bottom="rgba(255, 255, 255, 0.12)",
    header_text="rgba(255, 255, 255, 0.80)",
    frame_line_color="rgba(255, 255, 255, 0.08)",

    # Icons
    arrow_suffix="",
    is_dark=True,
    icon_color="#F0F0F0",
)


# ═══════════════════════════════════════════════════════════════
#  LIGHT THEME
# ═══════════════════════════════════════════════════════════════
LIGHT_TOKENS = ThemeTokens(
    # Surfaces
    surface_bg="#E8EAED",
    surface_card="rgba(0, 0, 0, 0.03)",
    surface_card_border="rgba(0, 0, 0, 0.03)",
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
    progress_bg="rgba(0, 0, 0, 0.20)",
    progress_text="#FFFFFF",
    progress_chunk_start="rgba(25, 118, 210, 0.85)",
    progress_chunk_mid="rgba(33, 150, 243, 0.90)",
    progress_chunk_end="rgba(66, 165, 245, 0.85)",

    # Checkbox
    checkbox_bg="rgba(0, 0, 0, 0.06)",
    checkbox_border="rgba(0, 0, 0, 0.15)",
    checkbox_checked_bg="rgba(42, 130, 218, 0.90)",
    checkbox_checked_border="rgba(25, 100, 200, 0.90)",
    checkbox_disabled_bg="rgba(0, 0, 0, 0.05)",
    checkbox_disabled_border="rgba(0, 0, 0, 0.05)",
    checkbox_checked_disabled_bg="rgba(0, 0, 0, 0.15)",
    checkbox_checked_disabled_border="rgba(0, 0, 0, 0.15)",
    checkbox_checked_image="check-white.svg",
    checkbox_checked_disabled_image="check-white-disabled.svg",
    checkbox_indeterminate_image="indeterminate-white.svg",

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
    icon_color="#404040",
)


# ═══════════════════════════════════════════════════════════════
#  MODERN WARM THEME (Warm Sand / Linen & Charcoal)
# ═══════════════════════════════════════════════════════════════
MODERN_WARM_TOKENS = ThemeTokens(
    # Surfaces
    surface_bg="#EFECE6",
    surface_card="#F7F4EC",
    surface_card_border="#DED8CE",
    surface_input="#FFFFFF",
    surface_overlay="#FAF7F2",
    surface_list="#EAE5DC",
    surface_table="#FFFFFF",
    surface_table_alt="#F5F1E8",
    surface_toolbar="#E8E3DB",

    # Text
    text_primary="#1C1C1E",
    text_secondary="#6E6A63",
    text_disabled="#A8A39A",
    text_placeholder="#9C978E",

    # Borders
    border_subtle="#DED8CE",
    border_medium="#BFB9AD",
    border_strong="#7A756B",
    border_separator="#DCD5CB",

    # Accent (Authentic Jet Black / Deep Charcoal from Image 2)
    accent="#18181A",
    accent_light="#2C2C2E",
    accent_bg="rgba(24, 24, 26, 0.08)",
    accent_bg_strong="rgba(24, 24, 26, 0.16)",
    accent_border="rgba(24, 24, 26, 0.45)",

    # Interactive states
    hover_bg="rgba(24, 24, 26, 0.05)",
    hover_bg_strong="rgba(24, 24, 26, 0.09)",
    pressed_bg="rgba(24, 24, 26, 0.18)",
    selected_bg="rgba(24, 24, 26, 0.12)",
    selected_border="#18181A",

    # Scrollbar / slider
    scrollbar_thumb="rgba(24, 24, 26, 0.16)",
    scrollbar_thumb_hover="rgba(24, 24, 26, 0.30)",
    slider_groove="rgba(24, 24, 26, 0.12)",
    slider_filled="#18181A",
    slider_handle="#FFFFFF",
    slider_handle_border="rgba(24, 24, 26, 0.40)",

    # Progress bar
    progress_bg="rgba(24, 24, 26, 0.25)",
    progress_text="#FAF7F2",
    progress_chunk_start="#38383A",
    progress_chunk_mid="#242426",
    progress_chunk_end="#18181A",

    # Checkbox indicator
    checkbox_bg="#FFFFFF",
    checkbox_border="rgba(24, 24, 26, 0.28)",
    checkbox_checked_bg="#FFFFFF",
    checkbox_checked_border="#18181A",
    checkbox_disabled_bg="rgba(24, 24, 26, 0.04)",
    checkbox_disabled_border="rgba(24, 24, 26, 0.10)",
    checkbox_checked_disabled_bg="rgba(24, 24, 26, 0.04)",
    checkbox_checked_disabled_border="rgba(24, 24, 26, 0.15)",
    checkbox_checked_image="check-dark.svg",
    checkbox_checked_disabled_image="check-dark-disabled.svg",
    checkbox_indeterminate_image="indeterminate-dark.svg",

    # Tab bar
    tab_bg="rgba(24, 24, 26, 0.04)",
    tab_border="rgba(24, 24, 26, 0.08)",
    tab_text="#6E6A63",
    tab_selected_bg="#FAF7F2",
    tab_selected_border="#DED8CE",
    tab_selected_bottom="#FAF7F2",
    tab_hover_bg="rgba(24, 24, 26, 0.06)",
    tab_hover_text="#1C1C1E",
    tab_pane_bg="#FAF7F2",
    tab_pane_border="#DED8CE",

    # Miscellaneous
    tooltip_bg="#18181A",
    tooltip_border="#2C2C2E",
    tooltip_text="#FAF7F2",
    dialog_bg="#EFECE6",
    menu_bg="#FAF7F2",
    menu_border="#DED8CE",
    menu_item_hover="rgba(24, 24, 26, 0.06)",
    menu_separator="#E2DDD4",
    gridline="rgba(24, 24, 26, 0.07)",
    header_bg="#EAE5DC",
    header_border_right="#DED8CE",
    header_border_bottom="#D5CFC3",
    header_text="#4A463F",
    frame_line_color="#DED8CE",

    # Icons
    arrow_suffix="-dark",
    is_dark=False,
    icon_color="#1C1C1E",
)


# ── Lookup table ─────────────────────────────────────────────
THEME_TOKENS: dict[str, ThemeTokens] = {
    "dark": DARK_TOKENS,
    "light": LIGHT_TOKENS,
    "modern_warm": MODERN_WARM_TOKENS,
    # Classic themes reuse the palette-only path; tokens are still
    # available for downstream helpers like icon-color selection.
    "classic_dark": DARK_TOKENS,
    "classic_light": LIGHT_TOKENS,
}
