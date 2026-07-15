"""Tests for spectroview.api.analysis -- thin PCA/NMF/reconstruction_error
wrapper contract (math itself belongs to spectroview.model.m_mva)."""
import numpy as np

from spectroview.api import analysis


class TestPCA:
    def test_shapes_and_variance_ordering(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 50))
        result = analysis.pca(X, n_components=4)
        assert result.scores.shape == (20, 4)
        assert result.loadings.shape == (4, 50)
        assert np.all(np.diff(result.explained_variance) <= 1e-9)  # non-increasing


class TestNMF:
    def test_shapes_and_nonnegativity(self):
        rng = np.random.default_rng(0)
        X = np.abs(rng.normal(size=(15, 30)))
        result = analysis.nmf(X, n_components=3, max_iter=100)
        assert result.W.shape == (15, 3)
        assert result.H.shape == (3, 30)
        assert (result.W >= 0).all()
        assert (result.H >= 0).all()


class TestReconstructionError:
    def test_perfect_reconstruction_is_near_zero(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 20))
        result = analysis.pca(X, n_components=10, center=True)
        errors = analysis.reconstruction_error(X, result.scores, result.loadings, result.mean_spectrum)
        assert errors.shape == (10,)
        assert np.allclose(errors, 0.0, atol=1e-8)
