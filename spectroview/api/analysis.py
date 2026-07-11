"""Multivariate analysis functions (PCA, NMF)."""
import numpy as np
from spectroview.model.m_mva import MMVA, PCAResult, NMFResult

def pca(X: np.ndarray, n_components: int = 5, center: bool = True) -> PCAResult:
    """
    Run Principal Component Analysis (PCA) on a spectral data matrix.
    
    Args:
        X: (n_spectra, n_wavenumbers) intensity matrix.
        n_components: number of components to retain.
        center: subtract mean spectrum before SVD (default True).
    
    Returns:
        PCAResult object with scores, loadings, explained_variance, etc.
    """
    return MMVA.run_pca(X, n_components, center)

def nmf(X: np.ndarray, n_components: int = 3, max_iter: int = 500) -> NMFResult:
    """
    Run Non-negative Matrix Factorization (NMF) on a spectral data matrix.
    
    Args:
        X: (n_spectra, n_wavenumbers) non-negative intensity matrix.
        n_components: number of components to retain.
        max_iter: maximum iterations.
        
    Returns:
        NMFResult object with W (scores) and H (loadings).
    """
    return MMVA.run_nmf(X, n_components, max_iter)
