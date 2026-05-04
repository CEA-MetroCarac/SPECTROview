"""Tensor Fitting Engine — public API.

Fits all spectra in a hyperspectral map simultaneously using a custom
batched Levenberg-Marquardt optimizer.

Usage:
    engine = TensorFittingEngine()
    results = engine.fit_spectra(spectra, fit_model, ...)
"""

import time
import numpy as np
from fitspy.core.utils import eval_noise_amplitude

from spectroview.fit_engine.evaluator import TensorEvaluator
from spectroview.fit_engine.optimizer import batched_levenberg_marquardt
from spectroview.fit_engine.scalar_models import FitResult
from spectroview.viewmodel.utils import apply_custom_fit_model


class TensorFittingEngine:
    """High-performance tensor fitting engine for hyperspectral data."""
    
    def __init__(self):
        self.timings = {}

    def fit_spectra(
        self,
        spectra,
        fit_model: dict,
        fit_params: dict = None,
        progress_callback=None,
        cancel_check=None,
        apply_model_to_spectra: bool = True,
    ):
        """Fit all spectra simultaneously using the tensor engine.

        Args:
            spectra: list of MSpectrum objects (already preprocessed)
            fit_model: fit model dict (from spectrum.save() or JSON)
            fit_params: dict with 'method', 'xtol', 'max_ite', etc.
            progress_callback: callable(current, total)
            cancel_check: callable() → bool
            apply_model_to_spectra: if True, apply fit model to spectra first

        Returns:
            list of FitResult objects
        """
        n_spectra = len(spectra)
        t_total = time.perf_counter()

        # ─── 1. Apply fit model to spectra ───
        if apply_model_to_spectra:
            t0 = time.perf_counter()
            self._apply_model_to_all(spectra, fit_model)
            self.timings["Step 1 - apply_model"] = f"{time.perf_counter()-t0:.3f}s"

        # ─── 2. Build evaluator ───
        evaluator = TensorEvaluator.from_fit_model(fit_model)

        if evaluator.n_params_free == 0:
            if progress_callback:
                progress_callback(n_spectra, n_spectra)
            return [FitResult(True, {}, np.array([])) for _ in spectra]

        # ─── 3. Preprocess all spectra (only when needed) ───
        t0 = time.perf_counter()
        for spectrum in spectra:
            if not getattr(spectrum, 'is_preprocessed', False):
                spectrum.preprocess()
        self.timings["Step 2 - preprocess"] = f"{time.perf_counter()-t0:.3f}s"

        # ─── 4. Extract data matrix ───
        max_M = max((len(s.x) if s.x is not None else 0) for s in spectra)
        if max_M == 0:
            return [FitResult(False, {}, np.array([])) for _ in spectra]

        X_matrix = np.zeros((n_spectra, max_M), dtype=np.float64)
        Y_matrix = np.zeros((n_spectra, max_M), dtype=np.float64)
        for i, s in enumerate(spectra):
            if s.x is not None and s.y is not None:
                M_s = len(s.x)
                X_matrix[i, :M_s] = s.x
                Y_matrix[i, :M_s] = s.y

        # Build weights matrix to match lmfit masking behavior
        weights_matrix = self._build_fit_weights(spectra, fit_params, max_M)

        # ─── 5. Build initial parameter matrix ───
        t0 = time.perf_counter()
        if not apply_model_to_spectra:
            # Re-fitting: extract from existing fitted values
            p0 = np.empty((n_spectra, evaluator.n_params_free))
            for i, s in enumerate(spectra):
                p0[i] = evaluator.extract_p0_from_spectrum(s)
            
            # Perturb warm-start p0 slightly to escape premature convergence.
            # Without this, the optimizer sees the previous optimum and
            # immediately declares convergence on the first iteration,
            # making repeated Fit clicks useless.
            rng = np.random.default_rng(42)
            noise = rng.uniform(-1e-3, 1e-3, size=p0.shape)
            p0 *= (1.0 + noise)
            # Re-clip to bounds
            p0 = np.clip(p0, evaluator.lower_bounds, evaluator.upper_bounds)
        else:
            # First fit: scale amplitudes per spectrum
            p0 = evaluator.build_p0_matrix(spectra)
            
        evaluator.apply_noise_threshold(spectra, p0, fit_params)
        self.timings["Step 3 - build p0"] = f"{time.perf_counter()-t0:.3f}s"

        # ─── 6. Parse fit parameters ───
        if fit_params is None:
            fit_params = {}
        xtol = float(fit_params.get("xtol", 1e-4))
        ftol = float(fit_params.get("ftol", 1e-4))
        max_ite = int(fit_params.get("max_ite", 200))

        # ─── 7. TENSOR FIT ───
        t0 = time.perf_counter()
        p_opt, success, cost = batched_levenberg_marquardt(
            x=X_matrix,
            Y_data=Y_matrix,
            evaluate_fn=evaluator.evaluate,
            jacobian_fn=evaluator.jacobian,
            p0=p0,
            lower_bounds=evaluator.lower_bounds,
            upper_bounds=evaluator.upper_bounds,
            weights=weights_matrix,
            max_iter=max_ite,
            xtol=xtol,
            ftol=ftol,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        fit_time = time.perf_counter() - t0
        self.timings["Step 4 - tensor fit"] = f"{fit_time:.3f}s ({fit_time/n_spectra*1000:.1f} ms/spectrum, {success.sum()}/{n_spectra} converged)"

        evaluator.apply_noise_threshold(spectra, p_opt, fit_params)
        
        # ─── 8. Write back results ───
        t0 = time.perf_counter()
        fit_results = []
        for i, spectrum in enumerate(spectra):
            M_s = len(spectrum.x) if spectrum.x is not None else 0
            w = weights_matrix[i, :M_s] if weights_matrix is not None else None
            fr = evaluator.build_result(p_opt[i], spectrum.x, spectrum.y, bool(success[i]), weights=w)
            if weights_matrix is not None and w is not None:
                fr.best_fit = fr.best_fit.copy()
                fr.best_fit[w == 0] = 0.0
            evaluator.write_back_to_spectrum(spectrum, fr)
            fit_results.append(fr)
        self.timings["Step 5 - write_back"] = f"{time.perf_counter()-t0:.3f}s"

        return fit_results

    def _apply_model_to_all(self, spectra, fit_model):
        """Apply fit model dict to all spectra (set peak_models, baseline, etc.)."""
        for spectrum in spectra:
            apply_custom_fit_model(spectrum, fit_model, spectrum.fname)

    def _build_fit_weights(self, spectra, fit_params, max_M):
        """Build a weights matrix that mimics lmfit's masking behavior."""
        

        weights = []
        if fit_params is None:
            fit_params = {}
        fit_negative = bool(fit_params.get("fit_negative", False))
        fit_outliers = bool(fit_params.get("fit_outliers", False))
        coef_noise = float(fit_params.get("coef_noise", 0))

        for spectrum in spectra:
            if spectrum.y is None:
                weights.append(np.zeros(max_M, dtype=np.float64))
                continue

            w = np.ones_like(spectrum.y, dtype=np.float64)
            if not fit_negative:
                w[spectrum.y < 0] = 0.0

            if not fit_outliers:
                x_outliers, _ = spectrum.calculate_outliers()
                if x_outliers is not None:
                    w[np.isin(spectrum.x, x_outliers)] = 0.0

            if coef_noise > 0:
                ampli_noise = eval_noise_amplitude(spectrum.y)
                noise_level = coef_noise * ampli_noise
                ymean = np.convolve(spectrum.y, np.ones(5, dtype=np.float64) / 5.0, mode='same')
                w[ymean < noise_level] = 0.0

            if spectrum.weights is not None:
                w = w * spectrum.weights

            w_padded = np.zeros(max_M, dtype=np.float64)
            w_padded[:len(w)] = w
            weights.append(w_padded)

        return np.vstack(weights)
