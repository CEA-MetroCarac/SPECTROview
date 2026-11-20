"""Export and clipboard utilities for DataFrames and Matplotlib figures."""

from __future__ import annotations

from io import BytesIO

import matplotlib
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from spectroview.i18n import tr
from spectroview.viewmodel.common.ui_utils import show_notification

# Vector formats where embedding real (editable/searchable) text instead of
# outlined paths matters for publication use.
_VECTOR_FORMATS = ("pdf", "eps", "svg")


def save_df_to_excel(save_path, df):
    """Save a DataFrame to an Excel file with colored columns based on prefixes.

    Uses xlsxwriter with direct column writing and prefix cell formats for speed,
    falling back to openpyxl if xlsxwriter is unavailable.
    """
    if not save_path:
        return False, tr("No save path provided.")

    try:
        import pandas as pd

        if df is None or df.empty:
            return False, tr("DataFrame is empty. Nothing to save.")

        palette = ["#bda16d", "#a27ba0", "#cb5b12", "#23993b", "#008281", "#147ce4"]

        try:
            import xlsxwriter

            wb = xlsxwriter.Workbook(save_path, {"nan_inf_to_errors": False})
            ws = wb.add_worksheet("Results")
            header_fmt = wb.add_format({"bold": True})

            prefix_formats = {}
            for col_idx, col_name in enumerate(df.columns):
                prefix = col_name.split("_")[0] if "_" in col_name else col_name
                if prefix not in prefix_formats:
                    color_index = len(prefix_formats) % len(palette)
                    prefix_formats[prefix] = wb.add_format({
                        "bg_color": palette[color_index],
                        "pattern": 1,
                    })
                fmt = prefix_formats[prefix]
                ws.write(0, col_idx, str(col_name), header_fmt)

                col_series = df[col_name]
                if pd.api.types.is_numeric_dtype(col_series):
                    mask_invalid = pd.isna(col_series) | np.isinf(col_series)
                    clean_vals = np.where(mask_invalid, None, col_series).tolist()
                else:
                    clean_vals = np.where(pd.isna(col_series), None, col_series).tolist()

                ws.write_column(1, col_idx, clean_vals, fmt)

            wb.close()
            return True, tr("DataFrame saved successfully.")
        except ImportError:
            pass

        from openpyxl.styles import PatternFill

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
            worksheet = writer.sheets["Results"]
            prefix_colors = {}
            col_fills = []
            for col_name in df.columns:
                prefix = col_name.split("_")[0] if "_" in col_name else col_name
                if prefix not in prefix_colors:
                    color_index = len(prefix_colors) % len(palette)
                    prefix_colors[prefix] = PatternFill(
                        start_color=palette[color_index][1:],
                        end_color=palette[color_index][1:],
                        fill_type="solid",
                    )
                col_fills.append(prefix_colors[prefix])

            for col_cells, fill in zip(
                worksheet.iter_cols(
                    min_row=2,
                    max_row=len(df) + 1,
                    min_col=1,
                    max_col=len(df.columns),
                ),
                col_fills,
            ):
                for cell in col_cells:
                    cell.fill = fill

        return True, tr("DataFrame saved successfully.")

    except Exception as e:
        return False, tr("Error when saving DataFrame: {error}", error=f"{e!s}")


def copy_fig_to_clb(canvas, size_ratio=None):
    """Copy matplotlib figure canvas to clipboard with optional resizing."""
    if not canvas:
        show_notification(tr("No plot to copy."), notification_type="warning")
        return

    figure = canvas.figure
    original_size = figure.get_size_inches()
    original_dpi = figure.dpi

    try:
        if size_ratio:
            figure.set_size_inches(size_ratio, forward=True)
            canvas.draw()
        buf = BytesIO()
        figure.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        qimage = QImage()
        qimage.loadFromData(buf.getvalue())

        clipboard = QApplication.clipboard()
        clipboard.setPixmap(QPixmap.fromImage(qimage))

    finally:
        figure.set_size_inches(original_size, forward=True)
        figure.set_dpi(original_dpi)
        canvas.draw()


def copy_df_to_clb(df) -> bool:
    """Copy a pandas DataFrame to the Qt clipboard as tab-separated text."""
    if df is None or getattr(df, "empty", True):
        return False
    clipboard = QApplication.clipboard()
    if clipboard is None:
        return False
    clipboard.setText(df.to_csv(sep="\t", index=False))
    return True


def export_figure_to_file(
    canvas, filepath, fmt, dpi=300, transparent=False, size_inches=None
):
    """Export a matplotlib figure canvas to a file (PNG/TIFF/SVG/PDF/EPS).

    Mirrors copy_fig_to_clb's temporarily-resize-then-restore pattern so a
    caller can request a specific physical export size without permanently
    resizing the on-screen figure. Vector formats embed editable text via
    ``matplotlib.rc_context``.
    """
    if not canvas:
        show_notification(tr("No plot to export."), notification_type="warning")
        return False

    figure = canvas.figure
    original_size = figure.get_size_inches()
    original_dpi = figure.dpi

    try:
        if size_inches:
            figure.set_size_inches(size_inches, forward=True)
            canvas.draw()

        bbox_inches = None if size_inches else "tight"
        savefig_kwargs = {
            "format": fmt,
            "dpi": dpi,
            "transparent": transparent,
            "bbox_inches": bbox_inches,
        }

        if fmt in _VECTOR_FORMATS:
            with matplotlib.rc_context(
                {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
            ):
                figure.savefig(filepath, **savefig_kwargs)
        else:
            figure.savefig(filepath, **savefig_kwargs)
        return True
    except Exception as e:
        show_notification(tr("Error exporting figure: {error}", error=e), notification_type="error")
        return False
    finally:
        figure.set_size_inches(original_size, forward=True)
        figure.set_dpi(original_dpi)
        canvas.draw()
