"""ViewModel for Multivariate Analysis (MVA) — bridges UI signals to model computations.

Updated for the SpectraStore tensor-centric architecture.

Supports two modes:
  • Spectra workspace  – each MapData holds 1 spectrum → MVA across all maps.
  • Maps workspace     – a single MapData holds N spectra → MVA within the
                         currently selected map (set via set_current_map_name).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from spectroview.i18n import tr
from spectroview.model.analysis.m_mva import MMVA, NMFResult, PCAResult
from spectroview.model.spectra.spectra_store import SpectraStore


class VMMVA(QObject):
    """ViewModel for Multivariate Analysis (MVA) feature."""

    # ───── ViewModel → View signals ─────
    pca_results_ready = Signal(object)  # PCAResult + metadata dict
    nmf_results_ready = Signal(object)  # NMFResult + metadata dict
    notify = Signal(str)  # floating notifications
    send_df_to_graphs = Signal(str, object)  # (df_name, pd.DataFrame)

    def __init__(self, m_settings=None):
        super().__init__()
        self.m_settings = m_settings
        self.model = MMVA()

        # Injected SpectraStore reference — set by VWorkspaceSpectra after construction
        self._store: SpectraStore | None = None

        # Maps workspace: analyse spectra within a single map
        self._current_map_name: str | None = None

        # Cache last results for export
        self._last_pca_result: PCAResult | None = None
        self._last_nmf_result: NMFResult | None = None
        self._last_x_axis: np.ndarray | None = None
        self._last_fnames: list[str] | None = None
        self._last_data_matrix: np.ndarray | None = None

    # ── Dependency injection ──────────────────────────────────────────

    def set_store(self, store: SpectraStore):
        """Inject the SpectraStore reference from VMWorkspaceSpectra."""
        self._store = store

    def set_current_map_name(self, name: str | None):
        """Set the active map for Maps-workspace MVA."""
        self._current_map_name = name

    def clear(self):
        """Clear cached results and active map reference."""
        self._current_map_name = None
        self._last_pca_result = None
        self._last_nmf_result = None
        self._last_x_axis = None
        self._last_fnames = None
        self._last_data_matrix = None

    # ── Data matrix construction from SpectraStore ────────────────────

    def _build_data_matrix(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build (X, x_axis, fnames) from the store."""
        if self._store is None:
            raise ValueError(tr("No store available."))

        if self._current_map_name is not None:
            return self._build_from_single_map(self._current_map_name)
        return self._build_from_all_maps()

    # ── Private builders ──────────────────────────────────────────────

    def _build_from_all_maps(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Spectra-workspace mode: 1 spectrum per map, varying x-axes."""
        from scipy.interpolate import interp1d  # deferred: ~0.6 s of import time

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
            raise ValueError(tr("At least 2 active spectra are required for MVA."))

        return np.vstack(rows).astype(np.float64), x_ref, fnames

    def _build_from_single_map(
        self, map_name: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Maps-workspace mode: all active spectra from one MapData."""
        md = self._store.get_map_data(map_name)
        if md is None:
            raise ValueError(tr("Map '{name}' not found in store.", name=map_name))

        active = md.is_active
        active_indices = np.where(active)[0]

        if len(active_indices) < 2:
            raise ValueError(
                tr(
                    "At least 2 active spectra are required for MVA "
                    "(map '{name}' has {count} active).",
                    name=map_name,
                    count=len(active_indices),
                )
            )

        x = (md.x if md.x is not None else md.x0).astype(np.float64)
        y_mat = (md.Y if md.Y is not None else md.Y0)[active_indices].astype(
            np.float64
        )

        fnames = [md.fnames[i] for i in active_indices]

        return y_mat, x, fnames

    # ── PCA ───────────────────────────────────────────────────────────

    def run_pca(self, n_components: int, center: bool = True):
        """Slot: build data matrix from active spectra and run PCA."""
        try:
            x_mat, x_axis, fnames = self._build_data_matrix()
        except ValueError as e:
            self.notify.emit(str(e))
            return

        try:
            result = self.model.run_pca(x_mat, n_components, center=center)
        except Exception as e:
            self.notify.emit(tr("PCA failed: {error}", error=e))
            return

        recon_errors = self.model.reconstruction_error_per_spectrum(
            x_mat,
            result.scores,
            result.loadings,
            mean_spectrum=result.mean_spectrum if center else None,
        )

        self._last_pca_result = result
        self._last_nmf_result = None
        self._last_x_axis = x_axis
        self._last_fnames = fnames
        self._last_data_matrix = x_mat

        payload = {
            "result": result,
            "x_axis": x_axis,
            "fnames": fnames,
            "reconstruction_errors": recon_errors,
            "data_matrix": x_mat,
        }
        self.pca_results_ready.emit(payload)
        self.notify.emit(
            tr(
                "PCA complete — {n_components} components, "
                "{variance}% variance explained.",
                n_components=n_components,
                variance=f"{result.cumulative_variance[-1] * 100:.1f}",
            )
        )

    # ── NMF ───────────────────────────────────────────────────────────

    def run_nmf(
        self, n_components: int, max_iter: int = 500, tol: float = 1e-4, seed: int = 42
    ):
        """Slot: build data matrix from active spectra and run NMF."""
        try:
            x_mat, x_axis, fnames = self._build_data_matrix()
        except ValueError as e:
            self.notify.emit(str(e))
            return

        try:
            result = self.model.run_nmf(
                x_mat, n_components, max_iter=max_iter, tol=tol, seed=seed
            )
        except Exception as e:
            self.notify.emit(tr("NMF failed: {error}", error=e))
            return

        recon_errors = self.model.reconstruction_error_per_spectrum(
            x_mat,
            result.W,
            result.H,
            mean_spectrum=None,
        )

        self._last_nmf_result = result
        self._last_pca_result = None
        self._last_x_axis = x_axis
        self._last_fnames = fnames
        self._last_data_matrix = x_mat

        payload = {
            "result": result,
            "x_axis": x_axis,
            "fnames": fnames,
            "reconstruction_errors": recon_errors,
            "data_matrix": x_mat,
        }
        self.nmf_results_ready.emit(payload)
        self.notify.emit(
            tr(
                "NMF complete — {n_components} components, "
                "{n_iterations} iterations, "
                "reconstruction error = {error}.",
                n_components=n_components,
                n_iterations=result.n_iterations,
                error=f"{result.reconstruction_error:.2f}",
            )
        )

    # ── Export to Graphs workspace ────────────────────────────────────

    def send_to_graphs(self, df_name: str):
        """Convert cached results to a DataFrame and emit to Graphs workspace."""
        if self._last_pca_result is not None:
            self._export_pca(df_name)
        elif self._last_nmf_result is not None:
            self._export_nmf(df_name)
        else:
            self.notify.emit(tr("No MVA results to export. Run an analysis first."))

    def _export_pca(self, df_name: str):
        """Export PCA scores as a DataFrame."""
        result = self._last_pca_result
        fnames = self._last_fnames or []

        cols = [f"PC{i + 1}" for i in range(result.scores.shape[1])]
        df = pd.DataFrame(result.scores, columns=cols)
        df.insert(0, "Spectrum", fnames)

        self.send_df_to_graphs.emit(df_name, df)
        self.notify.emit(tr("PCA scores sent to Graphs as '{name}'.", name=df_name))

    def _export_nmf(self, df_name: str):
        """Export NMF scores (W matrix) as a DataFrame."""
        result = self._last_nmf_result
        fnames = self._last_fnames or []

        cols = [f"NMF{i + 1}" for i in range(result.W.shape[1])]
        df = pd.DataFrame(result.W, columns=cols)
        df.insert(0, "Spectrum", fnames)

        self.send_df_to_graphs.emit(df_name, df)
        self.notify.emit(tr("NMF scores sent to Graphs as '{name}'.", name=df_name))
