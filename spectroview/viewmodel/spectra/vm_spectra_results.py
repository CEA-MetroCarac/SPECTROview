"""Fit results collection, computed columns, and export mixin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog

from spectroview.fit_engine.evaluator import eval_peak_initial
from spectroview.viewmodel.common.export_utils import save_df_to_excel


class SpectraResultsMixin:
    """Collect, enrich, export, and copy fit results and fitted spectrum curves."""

    def copy_spectrum_data_to_clipboard(self):
        """Copy X, Y, and peak model data of the first selected spectrum to clipboard as DataFrame."""
        if not self.selected_fnames:
            return

        fnames = self._get_selected_spectra()
        if not fnames:
            return

        unique_mds = self._get_unique_map_data(fnames)
        if not unique_mds:
            return

        md = unique_mds[0]
        local_idx = md.get_fname_index(fnames[0])
        if local_idx is None:
            return
        x_values = md.x_axis
        y_values = md.y_matrix[local_idx]

        data = {
            "X values": pd.Series(x_values),
            "Y values": pd.Series(y_values),
        }

        if md.fit_model and md.fit_model.get("peak_models"):
            # Create a high-resolution X-axis for smooth curves (at least 3x original resolution)
            n_fine_pts = max(3000, len(x_values) * 5)
            x_fine = np.linspace(x_values.min(), x_values.max(), n_fine_pts)
            data["X_fine"] = pd.Series(x_fine)

            peak_models = list(md.fit_model["peak_models"].values())
            peak_labels = md.fit_model.get("peak_labels", [])
            self._update_fit_model_from_params(md, local_idx)

            y_bestfit_fine = np.zeros_like(x_fine)

            for i, p_model in enumerate(peak_models):
                try:
                    y_peak_fine = eval_peak_initial(x_fine, p_model)
                    y_bestfit_fine += y_peak_fine

                    if i < len(peak_labels):
                        label = peak_labels[i]
                    else:
                        label = f"Peak{i + 1}"

                    data[label] = pd.Series(y_peak_fine)
                except Exception:
                    continue

            if md.baseline_config and getattr(md, "Y_baseline", None) is not None:
                y_base = md.Y_baseline[local_idx]
                if y_base is not None and not np.all(y_base == 0):
                    y_base_fine = np.interp(x_fine, x_values, y_base)
                    y_bestfit_fine += y_base_fine
                    data["Baseline_fine"] = pd.Series(y_base_fine)

            data["BestFit_fine"] = pd.Series(y_bestfit_fine)

        df = pd.DataFrame(data)
        df.to_clipboard(index=False)

    def set_results_decimals(self, decimals: int):
        """Set the decimal precision used to round the collected fit-results
        table. Takes effect on the next collect_fit_results() call."""
        self.results_decimals = max(0, int(decimals))

    def collect_fit_results(self, map_names: list[str] | None = None):
        """Collect best-fit results from target maps and create DataFrame."""
        if map_names is None:
            map_names = self.store.map_names

        if not map_names:
            self.notify.emit("No loaded spectra to collect results from.")
            return

        dfs = []
        for name in map_names:
            md = self.store.get_map_data(name)
            if not md or not md.has_fit_results():
                continue

            peak_labels = None
            if md.fit_model:
                peak_labels = md.fit_model.get("peak_labels", None)

            df = self.store.build_fit_results_df(
                name=name,
                map_type=getattr(self, "map_type", "2Dmap"),
                peak_labels=peak_labels,
                only_converged=False,
                decimals=self.results_decimals,
            )
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            self.notify.emit("No fit results to collect.")
            self.df_fit_results = None
            return

        df_all = pd.concat(dfs, ignore_index=True)

        param_priority = {
            "x0": 0,
            "fwhm": 1,
            "ampli": 2,
            "area": 3,
            "sigma": 4,
            "gamma": 5,
            "fraction": 6,
            "height": 7,
        }

        label_order = {}
        for name in map_names:
            md = self.store.get_map_data(name)
            if md and md.fit_model:
                peak_labels = md.fit_model.get("peak_labels", [])
                for i, lbl in enumerate(peak_labels):
                    if lbl not in label_order:
                        label_order[lbl] = i
                if peak_labels:
                    break

        def sort_key(cname):
            base_cols = ["Filename", "X", "Y", "Quadrant", "Zone"]
            if cname in base_cols:
                return (-1, base_cols.index(cname))
            if "_" in cname:
                ptype, label = cname.split("_", 1)
                return (param_priority.get(ptype, 999), label_order.get(label, 999))
            return (999, 0)

        sorted_cols = sorted(df_all.columns, key=sort_key)
        df_all = df_all[sorted_cols]

        # Drop X and Y columns for Spectra Workspace only
        if type(self).__name__ == "VMWorkspaceSpectra":
            cols_to_drop = [col for col in ["X", "Y"] if col in df_all.columns]
            if cols_to_drop:
                df_all = df_all.drop(columns=cols_to_drop)

        self.df_fit_results = df_all
        self.fit_results_updated.emit(self.df_fit_results)

    def split_filename(self):
        """Split the first filename by underscore and emit parts for combobox."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results available. Collect results first.")
            return

        fname = self.df_fit_results.loc[0, "Filename"]
        parts = fname.split("_")

        self.split_parts_updated.emit(parts)

    def add_column_from_filename(self, col_name: str, part_index: int):
        """Add a new column to fit results by extracting part from filename."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results available.")
            return

        if not col_name:
            self.notify.emit("Please enter a column name.")
            return

        if col_name in self.df_fit_results.columns:
            self.notify.emit(
                f"Column '{col_name}' already exists. Please choose a different name."
            )
            return

        try:
            extracted = self.df_fit_results["Filename"].str.split("_").str[part_index]
            numeric = pd.to_numeric(extracted, errors="coerce")

            if (
                numeric.notna().sum() == extracted.notna().sum()
                and extracted.notna().sum() > 0
            ):
                self.df_fit_results[col_name] = numeric
            else:
                self.df_fit_results[col_name] = extracted.where(
                    numeric.isna(), numeric
                )

            self.fit_results_updated.emit(self.df_fit_results)
            self.notify.emit(f"Added column '{col_name}' in the fit results.")

        except Exception as e:
            self.notify.emit(f"Error adding column: {e}")

    def compute_column_from_expression(self, col_name: str, expression: str):
        """Add a new column to fit results by evaluating a mathematical expression."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results available. Collect results first.")
            return

        if not col_name:
            self.notify.emit("Please enter a column name.")
            return

        if not expression:
            self.notify.emit("Please enter a mathematical expression.")
            return

        if col_name in self.df_fit_results.columns:
            self.notify.emit(f"Column '{col_name}' already exists.")
            return

        try:
            result = self.df_fit_results.eval(expression)

            warnings = []
            nan_count = int(pd.isna(result).sum())
            if nan_count > 0:
                warnings.append(
                    f"Expression resulted in {nan_count} NaN value(s). "
                    "This may be due to division by zero or invalid operations."
                )

            if pd.api.types.is_numeric_dtype(result):
                inf_count = int(np.isinf(result).sum())
                if inf_count > 0:
                    warnings.append(
                        f"Expression resulted in {inf_count} infinite value(s). "
                        "This may be due to division by zero."
                    )

            self.df_fit_results[col_name] = result

            if pd.api.types.is_numeric_dtype(self.df_fit_results[col_name]):
                self.df_fit_results[col_name] = self.df_fit_results[col_name].round(
                    self.results_decimals
                )

            self.fit_results_updated.emit(self.df_fit_results)

            if warnings:
                self.notify.emit(
                    f"Successfully added computed column '{col_name}' (with warnings)."
                )
            else:
                self.notify.emit(f"Successfully added computed column '{col_name}'.")

        except pd.errors.UndefinedVariableError as e:
            self.notify.emit(f"Invalid column name in expression: {e}")
        except SyntaxError as e:
            self.notify.emit(f"Syntax error in expression: {e}")
        except ZeroDivisionError:
            self.notify.emit("Division by zero in expression.")
        except Exception as e:
            self.notify.emit(f"Error evaluating expression: {e}")

    def save_fit_results(self):
        """Save fit results to Excel or CSV file."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results to save.")
            return

        last_dir = self.settings.get_last_directory()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            None,
            "Save Fit Results",
            str(Path(last_dir) / "fit_results.xlsx"),
            "Excel Files (*.xlsx);;CSV Files (*.csv)",
        )

        if file_path:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                ext = Path(file_path).suffix.lower()

                if ext == ".csv" or "CSV" in selected_filter:
                    self.df_fit_results.to_csv(file_path, index=False, sep=";")
                    self.notify.emit(f"Fit results saved: {Path(file_path).name}")
                else:
                    if not ext:
                        file_path += ".xlsx"
                    success, message = save_df_to_excel(file_path, self.df_fit_results)
                    if success:
                        self.notify.emit(f"Fit results saved: {Path(file_path).name}")
                    else:
                        self.notify.emit(f"Error saving fit results: {message}")
                        return

                self.settings.set_last_directory(str(Path(file_path).parent))
            except Exception as e:
                self.notify.emit(f"Error saving fit results: {e}")
            finally:
                QApplication.restoreOverrideCursor()

    def send_results_to_graphs(self, df_name: str, force_replace: bool = False):
        """Send fit results DataFrame to Graphs workspace."""
        if self.df_fit_results is None or self.df_fit_results.empty:
            self.notify.emit("No fit results to send.")
            return

        if not df_name:
            self.notify.emit("Please enter a DataFrame name.")
            return

        self.send_df_to_graphs.emit(df_name, self.df_fit_results, force_replace)
