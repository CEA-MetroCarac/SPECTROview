"""ViewModel for Multivariate Analysis (MVA) — bridges UI signals to model computations.

Updated for the SpectraStore tensor-centric architecture.

Supports two modes:
  • Spectra workspace  – each MapData holds 1 spectrum → MVA across all maps.
  • Maps workspace     – a single MapData holds N spectra → MVA within the
                         currently selected map (set via set_current_map_name).
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from PySide6.QtCore import QObject, Signal

from spectroview.model.m_mva import MMVA, PCAResult, NMFResult
from spectroview.model.spectra_store import SpectraStore


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

        # Injected SpectraStore reference — set by VWorkspaceSpectra after construction
        self._store: SpectraStore | None = None

        # Maps workspace: analyse spectra within a single map
        # When set, _build_data_matrix uses all active rows of this map.
        # When None (Spectra workspace), it iterates over all maps (1 spectrum per map).
        self._current_map_name: str | None = None

        # Cache last results for export
        self._last_pca_result: PCAResult | None = None
        self._last_nmf_result: NMFResult | None = None
        self._last_x_axis: np.ndarray | None = None
        self._last_fnames: list[str] | None = None

    # ── Dependency injection ──────────────────────────────────────────

    def set_store(self, store: SpectraStore):
        """Inject the SpectraStore reference from VMWorkspaceSpectra."""
        self._store = store

    def set_current_map_name(self, name: str | None):
        """Set the active map for Maps-workspace MVA.

        When *name* is set, `_build_data_matrix` will pull **all active
        spectra** from that single MapData block instead of iterating
        across maps.
        """
        self._current_map_name = name

    # ── Data matrix construction from SpectraStore ────────────────────

    def _build_data_matrix(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build (X, x_axis, fnames) from the store.

        Behaviour depends on ``_current_map_name``:

        * **None  (Spectra workspace)**  — one spectrum per map, iterate
          over all maps and pick ``Y[0]``.
        * **set   (Maps workspace)**     — many spectra in one map, iterate
          over all *active* rows inside that MapData.

        Returns:
            (X, x_axis, fnames)

        Raises:
            ValueError: if fewer than 2 active spectra are available.
        """
        if self._store is None:
            raise ValueError("No store available.")

        if self._current_map_name is not None:
            return self._build_from_single_map(self._current_map_name)
        else:
            return self._build_from_all_maps()

    # ── Private builders ──────────────────────────────────────────────

    def _build_from_all_maps(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Spectra-workspace mode: 1 spectrum per map, varying x-axes."""
        rows: list[np.ndarray] = []
        fnames: list[str] = []
        x_ref: np.ndarray | None = None

        for name in self._store.map_names:
            md = self._store.get_map_data(name)
            if md is None or not md.is_active[0]:
                continue

            x = (md.x if md.x is not None else md.x0).astype(np.float64)
            y = (md.Y[0] if md.Y is not None else md.Y0[0]).astype(np.float64)

            if x_ref is None:
                x_ref = x.copy()
                rows.append(y.copy())
            elif x.shape == x_ref.shape and np.allclose(x, x_ref, atol=0.01):
                rows.append(y.copy())
            else:
                f = interp1d(x, y, kind="linear", fill_value="extrapolate")
                rows.append(f(x_ref))

            fnames.append(name)

        if len(rows) < 2:
            raise ValueError("At least 2 active spectra are required for MVA.")

        return np.vstack(rows).astype(np.float64), x_ref, fnames

    def _build_from_single_map(self, map_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Maps-workspace mode: all active spectra from one MapData."""
        md = self._store.get_map_data(map_name)
        if md is None:
            raise ValueError(f"Map '{map_name}' not found in store.")

        # Active mask
        active = md.is_active  # bool[N]
        active_indices = np.where(active)[0]

        if len(active_indices) < 2:
            raise ValueError(
                f"At least 2 active spectra are required for MVA "
                f"(map '{map_name}' has {len(active_indices)} active)."
            )

        # Use processed arrays when available, else raw
        x = (md.x if md.x is not None else md.x0).astype(np.float64)
        Y = (md.Y if md.Y is not None else md.Y0)[active_indices].astype(np.float64)

        fnames = [md.fnames[i] for i in active_indices]

        return Y, x, fnames

    # ── PCA ───────────────────────────────────────────────────────────

    def run_pca(self, n_components: int):
        """Slot: build data matrix from active spectra and run PCA."""
        try:
            X, x_axis, fnames = self._build_data_matrix()
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
        try:
            X, x_axis, fnames = self._build_data_matrix()
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
