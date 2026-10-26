# spectroview/view/theme/tokens.py
"""
Design-token definitions for every SPECTROview theme.

Each ``ThemeTokens`` instance is a frozen collection of CSS-ready color strings
(``rgba(r, g, b, a)`` with integer alpha 0..255 / hex) keyed by semantic role,
derived from the PLUME design system.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeTokens:
    """Immutable collection of design tokens for a single theme."""

    # ── Surfaces ─────────────────────────────────────────────
    background: str
    rail_bg: str
    sidebar_bg: str
    surface: str
    surface_sunken: str
    surface_elevated: str
    surface_hover: str
    surface_pressed: str

    # ── Input & Borders ──────────────────────────────────────
    input_bg: str
    input_border: str
    border: str
    border_subtle: str
    border_strong: str

    # ── AI Chat Cards ────────────────────────────────────────
    card_user_bg: str
    card_user_border: str
    card_assistant_bg: str
    card_assistant_border: str

    # ── Typography ───────────────────────────────────────────
    text_primary: str
    text_secondary: str
    text_muted: str

    # ── Monochrome Solid CTAs ────────────────────────────────
    solid: str
    solid_hover: str
    solid_pressed: str
    on_solid: str

    # ── Accents & Selection ──────────────────────────────────
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_tint: str
    selection: str

    # ── Feedback & Semantic States ───────────────────────────
    success: str
    success_tint: str
    warning: str
    warning_tint: str
    error: str
    error_tint: str

    # ── Scrollbars ───────────────────────────────────────────
    scrollbar_thumb: str
    scrollbar_thumb_hover: str

    # ── SPECTROview Compatibility Fields ─────────────────────
    progress_text: str
    checkbox_checked_image: str
    icon_color: str
    is_dark: bool

    @property
    def text(self) -> str:
        """Alias for primary text."""
        return self.text_primary

    @property
    def primary(self) -> str:
        """Alias for accent color."""
        return self.accent

    @property
    def bg(self) -> str:
        """Alias for background."""
        return self.background


# ═══════════════════════════════════════════════════════════════
#  MODERN DARK THEME (Flagship Default)
# ═══════════════════════════════════════════════════════════════
DARK_TOKENS = ThemeTokens(
    background="#181A20",
    rail_bg="#0E0F14",
    sidebar_bg="#13151B",
    surface="#20232C",
    surface_sunken="#12141A",
    surface_elevated="#282C37",
    surface_hover="#323746",
    surface_pressed="#3B4152",
    input_bg="#262A35",
    input_border="#404659",
    border="#353A4A",
    border_subtle="#2C303D",
    border_strong="#4D5468",
    card_user_bg="#1D283B",
    card_user_border="#3E5C94",
    card_assistant_bg="#232732",
    card_assistant_border="#3C4254",
    text_primary="#EDEDEF",
    text_secondary="#B0B4C0",
    text_muted="#8A8F9E",
    solid="#EDEDEF",
    solid_hover="#D8D8DE",
    solid_pressed="#C4C4CC",
    on_solid="#0E0F14",
    accent="#5B8CFF",
    accent_hover="#749DFF",
    accent_pressed="#4476EB",
    accent_tint="rgba(91, 140, 255, 42)",
    selection="rgba(91, 140, 255, 68)",
    success="#3DD68C",
    success_tint="rgba(61, 214, 140, 38)",
    warning="#F5C27A",
    warning_tint="rgba(245, 194, 122, 38)",
    error="#EF4444",
    error_tint="rgba(239, 68, 68, 38)",
    scrollbar_thumb="rgba(255, 255, 255, 56)",
    scrollbar_thumb_hover="rgba(255, 255, 255, 96)",
    progress_text="#EDEDEF",
    checkbox_checked_image="check_white.svg",
    icon_color="#EDEDEF",
    is_dark=True,
)


# ═══════════════════════════════════════════════════════════════
#  MODERN LIGHT THEME (Flagship Daylight)
# ═══════════════════════════════════════════════════════════════
LIGHT_TOKENS = ThemeTokens(
    background="#F3F4F8",
    rail_bg="#E2E5EE",
    sidebar_bg="#EBEEF4",
    surface="#FFFFFF",
    surface_sunken="#E8EBF2",
    surface_elevated="#FFFFFF",
    surface_hover="#E4E8F0",
    surface_pressed="#D8DDE8",
    input_bg="#F7F8FB",
    input_border="#BEC5D4",
    border="#CBD1DF",
    border_subtle="#D8DDE8",
    border_strong="#A6AEBF",
    card_user_bg="#EBF1FF",
    card_user_border="#ADC4F5",
    card_assistant_bg="#FFFFFF",
    card_assistant_border="#C7CEDE",
    text_primary="#141519",
    text_secondary="#525663",
    text_muted="#7B808F",
    solid="#141519",
    solid_hover="#2A2C34",
    solid_pressed="#3A3D47",
    on_solid="#FFFFFF",
    accent="#2F62F0",
    accent_hover="#2452D6",
    accent_pressed="#1B43B8",
    accent_tint="rgba(47, 98, 240, 32)",
    selection="rgba(47, 98, 240, 48)",
    success="#0E8A55",
    success_tint="rgba(14, 138, 85, 32)",
    warning="#A15C00",
    warning_tint="rgba(161, 92, 0, 32)",
    error="#DC2626",
    error_tint="rgba(220, 38, 38, 28)",
    scrollbar_thumb="rgba(0, 0, 0, 52)",
    scrollbar_thumb_hover="rgba(0, 0, 0, 92)",
    progress_text="#FFFFFF",
    checkbox_checked_image="check_white.svg",
    icon_color="#141519",
    is_dark=False,
)


# ═══════════════════════════════════════════════════════════════
#  CLASSIC THEMES (Minimalist / Fusion Palette)
# ═══════════════════════════════════════════════════════════════
CLASSIC_DARK_TOKENS = DARK_TOKENS
CLASSIC_LIGHT_TOKENS = LIGHT_TOKENS


# ── Canonical Dictionary ─────────────────────────────────────
THEME_TOKENS: dict[str, ThemeTokens] = {
    "dark": DARK_TOKENS,
    "light": LIGHT_TOKENS,
    "classic_dark": CLASSIC_DARK_TOKENS,
    "classic_light": CLASSIC_LIGHT_TOKENS,
}
