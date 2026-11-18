"""Backward-compatible facade for legacy ``spectroview.viewmodel.utils`` imports.

Utilities are organized under ``spectroview.viewmodel.common`` and
``spectroview.view.components.v_metadata_tooltip``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectroview.view.components.v_metadata_tooltip import (
        MetadataTooltipManager,
        MetadataTooltipPopup,
    )

from spectroview.viewmodel.common.export_utils import (
    copy_df_to_clb,
    copy_fig_to_clb,
    export_figure_to_file,
    save_df_to_excel,
)
from spectroview.viewmodel.common.spectral_utils import (
    _clean_metadata_value,
    build_clean_fit_model,
    closest_index,
    extract_primary_metadata,
    fano_display_amplitude,
    fano_internal_amplitude,
    format_metadata_tooltip,
    parse_coords_from_fname,
)
from spectroview.viewmodel.common.ui_utils import (
    get_tinted_icon,
    set_spectrum_item_color,
    show_alert,
    show_notification,
    show_toast_notification,
)

__all__ = [
    "MetadataTooltipManager",
    "MetadataTooltipPopup",
    "_clean_metadata_value",
    "build_clean_fit_model",
    "closest_index",
    "copy_df_to_clb",
    "copy_fig_to_clb",
    "export_figure_to_file",
    "extract_primary_metadata",
    "fano_display_amplitude",
    "fano_internal_amplitude",
    "format_metadata_tooltip",
    "get_tinted_icon",
    "parse_coords_from_fname",
    "save_df_to_excel",
    "set_spectrum_item_color",
    "show_alert",
    "show_notification",
    "show_toast_notification",
]


def __getattr__(name: str):
    if name in ("MetadataTooltipManager", "MetadataTooltipPopup"):
        from spectroview.view.components import v_metadata_tooltip

        return getattr(v_metadata_tooltip, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
