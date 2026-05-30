"""Model for Multivariate Analysis (MVA) — pure computation, no Qt dependencies.

Provides PCA (via numpy SVD) and NMF (multiplicative update rules) for
Raman / PL spectroscopic datasets.

The PCA implementation follows the standard covariance-based SVD approach,
which is mathematically equivalent to the sklearn.decomposition.PCA used by
PyFASMA (E. Pavlou & N. Kourkoumelis, *Analyst*, 2025,
DOI: https://doi.org/10.1039/D5AN00452G).

References
----------
.. [1] E. Pavlou and N. Kourkoumelis, "PyFasma: an open-source, modular
       spectroscopy data", *Analyst*, 2025.
       DOI: 10.1039/D5AN00452G
.. [2] D. D. Lee and H. S. Seung, "Learning the parts of objects by
       non-negative matrix factorization," *Nature*, vol. 401, no. 6755,
       pp. 788-791, 1999.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d


# ═══════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PCAResult:
    """Container for PCA results.

    Follows the same conventions as sklearn.decomposition.PCA and
    PyFASMA's ``modeling.PCA`` class [1]_.
    """
    scores: np.ndarray              # (n_spectra, n_components)
    loadings: np.ndarray            # (n_components, n_wavenumbers)
    explained_variance: np.ndarray  # (n_components,)
    explained_variance_ratio: np.ndarray  # (n_components,) — fraction 0..1
    cumulative_variance: np.ndarray       # (n_components,) — cumulative sum
    mean_spectrum: np.ndarray       # (n_wavenumbers,) — subtracted mean
    eigenvalues: np.ndarray         # (min(n,p),) — all singular values squared / (n-1)
    n_components: int               # number of components retained


@dataclass
class NMFResult:
    """Container for NMF results."""
    W: np.ndarray                   # (n_spectra, n_components) — scores / concentrations
    H: np.ndarray                   # (n_components, n_wavenumbers) — loadings / endmembers
    reconstruction_error: float     # Frobenius norm of (X - W·H)
    n_iterations: int               # actual iterations performed


# ═══════════════════════════════════════════════════════════════════════
# Core MVA engine
# ═══════════════════════════════════════════════════════════════════════

class MMVA:
    """Model for Multivariate Analysis (MVA) feature.

    The PCA method is based on SVD decomposition of mean-centred data,
    equivalent to the approach used in PyFASMA [1]_ (which wraps
    ``sklearn.decomposition.PCA``).

    References
    ----------
    .. [1] E. Pavlou and N. Kourkoumelis, *Analyst*, 2025,
           DOI: 10.1039/D5AN00452G
    """

    # ── Data matrix construction ──────────────────────────────────────

    @staticmethod
    def build_data_matrix(spectra) -> tuple:
        """Stack active spectra into an (n_spectra × n_wavenumbers) matrix.

        If spectra have different x-axis lengths they are interpolated onto
        the x-axis of the first spectrum.

        Args:
            spectra: iterable of MSpectrum objects (must have .x and .y).

        Returns:
            (X, x_axis, fnames):
                X       — (n, p) float64 data matrix
                x_axis  — (p,) common wavenumber axis
                fnames  — list[str] of spectrum identifiers

        Raises:
            ValueError: if fewer than 2 spectra are provided.
        """
        active = [s for s in spectra if getattr(s, "is_active", True)]
        if len(active) < 2:
            raise ValueError("At least 2 active spectra are required for MVA.")

        # Reference x-axis (first active spectrum)
        x_ref = np.asarray(active[0].x, dtype=np.float64)
        rows = []
        fnames = []

        for s in active:
            x_s = np.asarray(s.x, dtype=np.float64)
            y_s = np.asarray(s.y, dtype=np.float64)

            if x_s.shape == x_ref.shape and np.allclose(x_s, x_ref, atol=0.01):
                rows.append(y_s.copy())
            else:
                # Interpolate onto reference grid
                f = interp1d(x_s, y_s, kind="linear", fill_value="extrapolate")
                rows.append(f(x_ref))

            fnames.append(getattr(s, "fname", f"spectrum_{len(fnames)}"))

        X = np.vstack(rows).astype(np.float64)
        return X, x_ref, fnames

    # ── PCA ───────────────────────────────────────────────────────────

    @staticmethod
    def run_pca(X: np.ndarray, n_components: int, center: bool = True) -> PCAResult:
        """Principal Component Analysis via truncated SVD on mean-centred data.

        This implementation is mathematically equivalent to
        ``sklearn.decomposition.PCA`` used by PyFASMA [1]_.

        Args:
            X: (n, p) data matrix — rows are spectra, columns are wavenumbers.
            n_components: number of components to retain (≤ min(n, p)).
            center: if True (default), subtract the mean spectrum before SVD.
                    This is standard practice for PCA on spectral data.

        Returns:
            PCAResult dataclass.

        References
        ----------
        .. [1] E. Pavlou and N. Kourkoumelis, *Analyst*, 2025,
               DOI: 10.1039/D5AN00452G
        """
        n, p = X.shape
        n_components = min(n_components, n, p)

        # Mean-centre (standard PCA preprocessing, as in PyFASMA/sklearn)
        if center:
            mean_spectrum = X.mean(axis=0)
            X_centered = X - mean_spectrum
        else:
            mean_spectrum = np.zeros(p, dtype=np.float64)
            X_centered = X.copy()

        # Full SVD (thin)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Truncate
        U_k = U[:, :n_components]
        S_k = S[:n_components]
        Vt_k = Vt[:n_components, :]

        # Explained variance (proportional to singular values squared)
        all_eigenvalues = (S ** 2) / (n - 1)
        total_var = np.sum(all_eigenvalues)
        explained_var = (S_k ** 2) / (n - 1)
        explained_var_ratio = explained_var / total_var if total_var > 0 else np.zeros_like(explained_var)
        cumulative = np.cumsum(explained_var_ratio)

        scores = U_k * S_k  # (n, k)
        loadings = Vt_k      # (k, p)

        return PCAResult(
            scores=scores,
            loadings=loadings,
            explained_variance=explained_var,
            explained_variance_ratio=explained_var_ratio,
            cumulative_variance=cumulative,
            mean_spectrum=mean_spectrum,
            eigenvalues=all_eigenvalues,
            n_components=n_components,
        )

    # ── NMF ───────────────────────────────────────────────────────────

    @staticmethod
    def run_nmf(
        X: np.ndarray,
        n_components: int,
        max_iter: int = 500,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> NMFResult:
        """Non-negative Matrix Factorisation via multiplicative update rules
        (Lee & Seung, 2001).

        Args:
            X: (n, p) **non-negative** data matrix.
            n_components: number of components.
            max_iter: maximum iterations.
            tol: convergence tolerance on relative change of reconstruction error.
            seed: random seed for reproducibility.

        Returns:
            NMFResult dataclass.
        """
        # Ensure non-negativity — clip small negatives (e.g. from baseline noise)
        X = np.maximum(X, 0.0)
        n, p = X.shape
        n_components = min(n_components, n, p)

        eps = 1e-12  # numerical stability

        # Random initialisation (NNDSVD-style seed for stability)
        rng = np.random.default_rng(seed)
        W = np.abs(rng.normal(scale=np.sqrt(X.mean() / n_components), size=(n, n_components))) + eps
        H = np.abs(rng.normal(scale=np.sqrt(X.mean() / n_components), size=(n_components, p))) + eps

        prev_error = np.inf
        actual_iter = 0

        for i in range(max_iter):
            # Update H
            numerator_H = W.T @ X
            denominator_H = W.T @ W @ H + eps
            H *= numerator_H / denominator_H

            # Update W
            numerator_W = X @ H.T
            denominator_W = W @ H @ H.T + eps
            W *= numerator_W / denominator_W

            # Check convergence every 10 iterations
            actual_iter = i + 1
            if (i + 1) % 10 == 0:
                error = np.linalg.norm(X - W @ H, "fro")
                if prev_error < np.inf:
                    rel_change = abs(prev_error - error) / (prev_error + eps)
                    if rel_change < tol:
                        break
                prev_error = error

        reconstruction_error = float(np.linalg.norm(X - W @ H, "fro"))

        return NMFResult(
            W=W,
            H=H,
            reconstruction_error=reconstruction_error,
            n_iterations=actual_iter,
        )

    # ── Reconstruction helpers ────────────────────────────────────────

    @staticmethod
    def reconstruct(scores: np.ndarray, loadings: np.ndarray) -> np.ndarray:
        """Reconstruct data matrix from scores × loadings.

        Works for both PCA (after adding back mean) and NMF (W @ H).
        """
        return scores @ loadings

    @staticmethod
    def reconstruction_error_per_spectrum(
        X: np.ndarray,
        scores: np.ndarray,
        loadings: np.ndarray,
        mean_spectrum: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute per-spectrum reconstruction error (L2 norm).

        Args:
            X: (n, p) original data matrix.
            scores: (n, k) score matrix.
            loadings: (k, p) loading matrix.
            mean_spectrum: (p,) mean to add back for PCA reconstruction.
                           Pass None for NMF.

        Returns:
            (n,) array of per-spectrum L2 reconstruction errors.
        """
        X_reconstructed = scores @ loadings
        if mean_spectrum is not None:
            X_reconstructed += mean_spectrum
        residuals = X - X_reconstructed
        return np.linalg.norm(residuals, axis=1)
