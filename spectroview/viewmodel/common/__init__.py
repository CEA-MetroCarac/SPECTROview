"""Shared ViewModel utilities: spectral math, export helpers, and UI bridges."""

from spectroview.viewmodel.common.export_utils import (
    copy_df_to_clb,
    copy_fig_to_clb,
    export_figure_to_file,
    save_df_to_excel,
)
from spectroview.viewmodel.common.spectral_utils import (
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
