"""ViewModel for Multivariate Analysis (MVA) — bridges UI signals to model computations."""
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from spectroview.model.m_mva import MMVA, PCAResult, NMFResult


class VMMVA(QObject):
    """ViewModel for Multivariate Analysis (MVA) feature."""

    # ───── ViewModel → View signals ─────
    pca_results_ready = Signal(object)   # PCAResult + metadata dict
    nmf_results_ready = Signal(object)   # NMFResult + metadata dict
    notify = Signal(str)                 # toast notifications
    send_df_to_graphs = Signal(str, object)  # (df_name, pd.DataFrame)

    def __init__(self, m_settings=None):
        super().__init__()
        self.m_settings = m_settings
        self.model = MMVA()

        # Injected reference getter — set by VWorkspaceSpectra after construction
        self._get_spectra = None

        # Cache last results for export
        self._last_pca_result: PCAResult | None = None
        self._last_nmf_result: NMFResult | None = None
        self._last_x_axis: np.ndarray | None = None
        self._last_fnames: list[str] | None = None

    # ── Dependency injection ──────────────────────────────────────────

    def set_spectra(self, spectra_getter):
        """Inject a getter function for the MSpectra list from VMWorkspaceSpectra."""
        self._get_spectra = spectra_getter

    # ── PCA ───────────────────────────────────────────────────────────

    def run_pca(self, n_components: int):
        """Slot: build data matrix from active spectra and run PCA."""
        spectra = self._get_spectra() if self._get_spectra else None
        if spectra is None or len(spectra) == 0:
            self.notify.emit("No spectra loaded.")
            return

        try:
            X, x_axis, fnames = self.model.build_data_matrix(spectra)
        except ValueError as e:
            self.notify.emit(str(e))
            return

        try:
            result = self.model.run_pca(X, n_components)
        except Exception as e:
            self.notify.emit(f"PCA failed: {e}")
            return

        # Cache for export
        self._last_pca_result = result
        self._last_nmf_result = None
        self._last_x_axis = x_axis
        self._last_fnames = fnames

        # Emit to View
        payload = {
            "result": result,
            "x_axis": x_axis,
            "fnames": fnames,
        }
        self.pca_results_ready.emit(payload)
        self.notify.emit(
            f"PCA complete — {n_components} components, "
            f"{result.cumulative_variance[-1]*100:.1f}% variance explained."
        )

    # ── NMF ───────────────────────────────────────────────────────────

    def run_nmf(self, n_components: int, max_iter: int = 500, tol: float = 1e-4):
        """Slot: build data matrix from active spectra and run NMF."""
        spectra = self._get_spectra() if self._get_spectra else None
        if spectra is None or len(spectra) == 0:
            self.notify.emit("No spectra loaded.")
            return

        try:
            X, x_axis, fnames = self.model.build_data_matrix(spectra)
        except ValueError as e:
            self.notify.emit(str(e))
            return

        try:
            result = self.model.run_nmf(X, n_components, max_iter=max_iter, tol=tol)
        except Exception as e:
            self.notify.emit(f"NMF failed: {e}")
            return

        # Cache for export
        self._last_nmf_result = result
        self._last_pca_result = None
        self._last_x_axis = x_axis
        self._last_fnames = fnames

        # Emit to View
        payload = {
            "result": result,
            "x_axis": x_axis,
            "fnames": fnames,
        }
        self.nmf_results_ready.emit(payload)
        self.notify.emit(
            f"NMF complete — {n_components} components, "
            f"{result.n_iterations} iterations, "
            f"reconstruction error = {result.reconstruction_error:.2f}."
        )

    # ── Export to Graphs workspace ────────────────────────────────────

    def send_to_graphs(self, df_name: str):
        """Convert cached results to a DataFrame and emit to Graphs workspace."""
        if self._last_pca_result is not None:
            self._export_pca(df_name)
        elif self._last_nmf_result is not None:
            self._export_nmf(df_name)
        else:
            self.notify.emit("No MVA results to export. Run an analysis first.")

    def _export_pca(self, df_name: str):
        """Export PCA scores as a DataFrame."""
        result = self._last_pca_result
        fnames = self._last_fnames or []

        cols = [f"PC{i+1}" for i in range(result.scores.shape[1])]
        df = pd.DataFrame(result.scores, columns=cols)
        df.insert(0, "Spectrum", fnames)

        # Add explained variance row as metadata
        var_row = {"Spectrum": "Explained Var. (%)"}
        for i, col in enumerate(cols):
            var_row[col] = f"{result.explained_variance_ratio[i]*100:.2f}%"
        
        self.send_df_to_graphs.emit(df_name, df)
        self.notify.emit(f"PCA scores sent to Graphs as '{df_name}'.")

    def _export_nmf(self, df_name: str):
        """Export NMF scores (W matrix) as a DataFrame."""
        result = self._last_nmf_result
        fnames = self._last_fnames or []

        cols = [f"NMF{i+1}" for i in range(result.W.shape[1])]
        df = pd.DataFrame(result.W, columns=cols)
        df.insert(0, "Spectrum", fnames)

        self.send_df_to_graphs.emit(df_name, df)
        self.notify.emit(f"NMF scores sent to Graphs as '{df_name}'.")
