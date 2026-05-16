"""Tensor Fitting Engine — public API.

Fits all spectra in a hyperspectral map simultaneously using a custom
batched Levenberg-Marquardt optimizer.

Usage:
    engine = TensorFittingEngine()
    results = engine.fit_spectra(spectra, fit_model, ...)
"""

import os
import time
import itertools
import numpy as np
from copy import deepcopy

from fitspy.core.utils import eval_noise_amplitude
from fitspy.core.baseline import BaseLine

from spectroview.fit_engine.evaluator import TensorEvaluator
from spectroview.fit_engine.optimizer import batched_levenberg_marquardt
from spectroview.fit_engine.scalar_models import FitResult
from spectroview.viewmodel.utils import apply_custom_fit_model

# Import fitspy model infrastructure for fast model creation
import fitspy
from fitspy.core.spectrum import create_model


# ═══════════════════════════════════════════════════════════════════════════
# Per-spectrum attributes that must be preserved during batch model apply
# ═══════════════════════════════════════════════════════════════════════════
_SPECTRUM_OWN_ATTRS = (
    "xcorrection_value",
    "intensity_norm_factor",
    "label",
    "color",
    "metadata",
)


def _prepare_fit_model_template(fit_model):
    """Pre-compute a reusable template from a fit_model dict.

    This avoids per-spectrum deepcopy, migrate_model_dict, and create_model calls.
    Returns a template dict containing pre-built model objects and pre-cleaned
    parameter hints.
    """
    # Sanitise fwhm/sigma once (same logic as apply_custom_fit_model)
    peak_models_dict = fit_model.get("peak_models", {})
    cleaned_peak_hints = {}
    for p_idx, p_model in peak_models_dict.items():
        for model_name, params in p_model.items():
            hints = deepcopy(params)
            for pn in ("fwhm", "sigma"):
                if pn in hints:
                    if hints[pn].get("value", 1) <= 0:
                        hints[pn]["value"] = 1e-6
                    if hints[pn].get("min", 0) <= 0:
                        hints[pn]["min"] = 1e-6
            cleaned_peak_hints[p_idx] = (model_name, hints)

    # Pre-build one set of lmfit Model objects as templates
    peak_model_templates = []
    peak_counter = itertools.count(start=1)
    for p_idx in sorted(cleaned_peak_hints.keys(), key=lambda k: int(k)):
        model_name, hints = cleaned_peak_hints[p_idx]
        index = next(peak_counter)
        prefix = f"m{index:02d}_"
        model_obj = create_model(fitspy.PEAK_MODELS[model_name], model_name, prefix)
        # We'll clone param_hints per spectrum, but the model object itself can be shallow-copied
        peak_model_templates.append((model_obj, hints))

    # Pre-process background model
    bkg_template = None
    bkg_dict = fit_model.get("bkg_model")
    if bkg_dict:
        bkg_model_name, bkg_hints = list(bkg_dict.items())[0]
        bkg_model = create_model(fitspy.BKG_MODELS[bkg_model_name], bkg_model_name)
        bkg_model.name2 = bkg_model_name
        bkg_template = (bkg_model, deepcopy(bkg_hints))

    # Pre-extract scalar attributes from fit_model
    scalar_attrs = {}
    skip_keys = {"peak_models", "bkg_model", "baseline", "fname", "x0", "y0",
                 "weights0", "result_fit_success", "schema_version", "peak_labels"}
    skip_keys.update(_SPECTRUM_OWN_ATTRS)
    for key, val in fit_model.items():
        if key not in skip_keys:
            scalar_attrs[key] = val

    # Baseline config
    baseline_config = fit_model.get("baseline", {})

    # Peak labels
    peak_labels = fit_model.get("peak_labels", [])
    if not peak_labels:
        peak_labels = [str(i + 1) for i in range(len(peak_model_templates))]

    return {
        "peak_templates": peak_model_templates,
        "bkg_template": bkg_template,
        "scalar_attrs": scalar_attrs,
        "baseline_config": baseline_config,
        "peak_labels": list(peak_labels),
    }


def _apply_template_to_spectrum(spectrum, template):
    """Apply a pre-computed template to a single spectrum (fast path).

    This replaces apply_custom_fit_model + set_attributes with minimal work:
    - No deepcopy of the full model dict
    - No migrate_model_dict
    - No create_model / inspect.signature per spectrum
    """
    # Set scalar attributes (range_min, range_max, normalize, etc.)
    for key, val in template["scalar_attrs"].items():
        setattr(spectrum, key, val)

    # Clone peak models: reuse the pre-built Model objects, deepcopy only param_hints
    spectrum.peak_index = itertools.count(start=len(template["peak_templates"]) + 1)
    spectrum.peak_models = []
    for model_obj, hints in template["peak_templates"]:
        # Shallow copy the model, deepcopy only the mutable param_hints
        cloned = model_obj.__class__.__new__(model_obj.__class__)
        cloned.__dict__.update(model_obj.__dict__)
        cloned.param_hints = deepcopy(hints)
        spectrum.peak_models.append(cloned)

    # Background model
    if template["bkg_template"] is not None:
        bkg_obj, bkg_hints = template["bkg_template"]
        cloned_bkg = bkg_obj.__class__.__new__(bkg_obj.__class__)
        cloned_bkg.__dict__.update(bkg_obj.__dict__)
        cloned_bkg.param_hints = deepcopy(bkg_hints)
        spectrum.bkg_model = cloned_bkg
    else:
        spectrum.bkg_model = None

    # Peak labels
    spectrum.peak_labels = list(template["peak_labels"])

    # Baseline configuration
    bl_config = template["baseline_config"]
    if bl_config:
        bl = spectrum.baseline
        for key in ("mode", "coef", "order_max", "sigma", "attached"):
            if key in bl_config:
                setattr(bl, key, bl_config[key])
        if "points" in bl_config:
            bl.points = deepcopy(bl_config["points"])
        # Always reset: baseline will be recomputed from raw data by preprocess
        bl.is_subtracted = False
        bl.y_eval = None

    # Ensure x0/y0 are numpy arrays (usually already are, avoid redundant conversion)
    if spectrum.x0 is not None and not isinstance(spectrum.x0, np.ndarray):
        spectrum.x0 = np.asarray(spectrum.x0)
    if spectrum.y0 is not None and not isinstance(spectrum.y0, np.ndarray):
        spectrum.y0 = np.asarray(spectrum.y0)

    spectrum.fname = os.path.normpath(spectrum.fname) if spectrum.fname else spectrum.fname
    spectrum.is_preprocessed = False
    spectrum.result_fit = lambda: None


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
        print_benchmark: bool = True, #To print the benchmark fitting times.
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
            t_step1 = time.perf_counter() - t0
            self.timings["Step 1 - apply_model"] = f"{t_step1:.3f}s"
            if print_benchmark:
                print(f"[TensorEngine] Step 1 - apply_model: {t_step1:.3f}s")

        # ─── 2. Build evaluator ───
        evaluator = TensorEvaluator.from_fit_model(fit_model)

        if evaluator.n_params_free == 0:
            if progress_callback:
                progress_callback(n_spectra, n_spectra)
            return [FitResult(True, {}, np.array([])) for _ in spectra]

        # ─── 3. Preprocess all spectra (batch-optimized) ───
        t0 = time.perf_counter()
        self._batch_preprocess(spectra, fit_model)
        t_step2 = time.perf_counter() - t0
        self.timings["Step 2 - preprocess"] = f"{t_step2:.3f}s"
        if print_benchmark:
            print(f"[TensorEngine] Step 2 - preprocess: {t_step2:.3f}s")

        # ─── 4. Extract data matrix ───
        x_ref = spectra[0].x
        if x_ref is None or len(x_ref) == 0:
            return [FitResult(False, {}, np.array([])) for _ in spectra]

        # Detect shared x-axis (common case for maps) → use fast 1D path
        shared_x = all(
            s.x is not None and len(s.x) == len(x_ref) and np.array_equal(s.x, x_ref)
            for s in spectra
        )

        if shared_x:
            # Fast path: all spectra share the same x-axis
            x_input = x_ref  # 1D (M,)
            M = len(x_ref)
            Y_matrix = np.empty((n_spectra, M), dtype=np.float64)
            for i, s in enumerate(spectra):
                if s.y is not None and len(s.y) == M:
                    Y_matrix[i] = s.y
                else:
                    Y_matrix[i] = 0.0
        else:
            # Fallback: variable-length spectra, pad to 2D
            max_M = max((len(s.x) if s.x is not None else 0) for s in spectra)
            x_input = np.zeros((n_spectra, max_M), dtype=np.float64)
            Y_matrix = np.zeros((n_spectra, max_M), dtype=np.float64)
            for i, s in enumerate(spectra):
                if s.x is not None and s.y is not None:
                    M_s = len(s.x)
                    x_input[i, :M_s] = s.x
                    Y_matrix[i, :M_s] = s.y

        # Build weights matrix to match lmfit masking behavior
        weights_matrix = self._build_fit_weights(spectra, fit_params, shared_x)

        # ─── 5. Build initial parameter matrix ───
        t0 = time.perf_counter()
        if not apply_model_to_spectra:
            # Re-fitting: extract from existing fitted values
            p0 = np.empty((n_spectra, evaluator.n_params_free))
            for i, s in enumerate(spectra):
                p0[i] = evaluator.extract_p0_from_spectrum(s)
        else:
            # First fit: scale amplitudes per spectrum
            p0 = evaluator.build_p0_matrix(spectra)
            
        evaluator.apply_noise_threshold(spectra, p0, fit_params)
        t_step3 = time.perf_counter() - t0
        self.timings["Step 3 - build p0"] = f"{t_step3:.3f}s"
        if print_benchmark:
            print(f"[TensorEngine] Step 3 - build p0: {t_step3:.3f}s")

        # ─── 6. Parse fit parameters ───
        if fit_params is None:
            fit_params = {}
        xtol = float(fit_params.get("xtol", 1e-4))
        ftol = float(fit_params.get("ftol", 1e-4))
        max_ite = int(fit_params.get("max_ite", 200))

        # ─── 7. TENSOR FIT ───
        t0 = time.perf_counter()
        p_opt, success, cost = batched_levenberg_marquardt(
            x=x_input,
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
        self.timings["Step 4 - tensor fit"] = f"{fit_time:.3f}s ({fit_time/n_spectra*1000:.3f} ms/spectrum, {success.sum()}/{n_spectra} converged)"
        if print_benchmark:
            print(f"[TensorEngine] Step 4 - tensor fit: {fit_time:.3f}s ({fit_time/n_spectra*1000:.1f} ms/spectrum, {success.sum()}/{n_spectra} converged)")

        evaluator.apply_noise_threshold(spectra, p_opt, fit_params, p0_matrix=p0)
        
        # ─── 8. Write back results (batch-optimized) ───
        t0 = time.perf_counter()
        fit_results = evaluator.build_results_batch(
            p_opt, x_input, Y_matrix, success, weights_matrix, shared_x, spectra
        )
        t_step5 = time.perf_counter() - t0
        self.timings["Step 5 - write_back"] = f"{t_step5:.3f}s"
        if print_benchmark:
            print(f"[TensorEngine] Step 5 - write_back: {t_step5:.3f}s")
            print(f"[TensorEngine] TOTAL: {time.perf_counter()-t_total:.3f}s")
        
        # Cleanup cached noise parameters
        for s in spectra:
            if hasattr(s, '_fit_ampli_noise'):
                del s._fit_ampli_noise
            if hasattr(s, '_fit_ymean'):
                del s._fit_ymean

        return fit_results

    def _apply_model_to_all(self, spectra, fit_model):
        """Apply fit model dict to all spectra (optimized batch path).
        
        Pre-computes a template once and applies it to each spectrum,
        avoiding per-spectrum deepcopy/migrate/create_model overhead.
        """
        template = _prepare_fit_model_template(fit_model)
        for spectrum in spectra:
            _apply_template_to_spectrum(spectrum, template)

    def _batch_preprocess(self, spectra, fit_model):
        """Batch-optimized preprocessing for map spectra sharing the same config.
        
        When all spectra share the same x0 length, range, baseline config,
        and have no outliers, we pre-compute shared work once (range mask,
        baseline point indices) and apply efficiently to all spectra.
        Falls back to per-spectrum preprocess() otherwise.
        """
        if not spectra:
            return

        # Check if batch preprocessing is possible
        first = spectra[0]
        can_batch = (
            first.x0 is not None
            and not first.normalize  # No normalization (rare in maps)
            and (first.outliers_limit is None)
            and len(first.outliers_inds) == 0
        )

        if not can_batch:
            # Fallback to per-spectrum
            for s in spectra:
                if not getattr(s, 'is_preprocessed', False):
                    s.preprocess()
            return

        # ── Compute range mask once ──
        x0 = first.x0
        range_min = first.range_min
        range_max = first.range_max
        
        if range_min is not None or range_max is not None:
            mask = np.logical_and(
                x0 >= (range_min if range_min is not None else -np.inf),
                x0 <= (range_max if range_max is not None else np.inf)
            )
            x = x0[mask].copy()
            if len(x) == 0:
                # Range mask produces empty x — fall back to per-spectrum
                for s in spectra:
                    if not getattr(s, 'is_preprocessed', False):
                        s.preprocess()
                return
        else:
            mask = None
            x = x0.copy()

        # ── Prepare baseline computation ──
        bl = first.baseline
        baseline_mode = bl.mode
        baseline_attached = bl.attached
        
        # Pre-compute baseline strategy
        baseline_strategy = None  # 'none', 'static', 'per_spectrum_linear', 'per_spectrum_poly', 'fallback'
        baseline_static = None  # shared baseline array (for non-attached modes)
        bl_point_indices = None  # indices of baseline points in x array
        bl_points_x = None
        bl_sigma = bl.sigma

        if baseline_mode is None:
            baseline_strategy = 'none'
        elif not baseline_attached:
            # Non-attached baseline: same for all spectra → compute once
            baseline_strategy = 'static'
            y_first = first.y0[mask].copy() if mask is not None else first.y0.copy()
            baseline_static = bl.eval(x, y_first, attached=False)
        elif baseline_attached and baseline_mode == 'Linear' and len(bl.points[0]) >= 1:
            # Attached Linear baseline: points are projected onto each spectrum's y
            # Pre-compute the indices of baseline points in the x array
            baseline_strategy = 'per_spectrum_linear'
            bl_points_x = np.array(bl.points[0])
            bl_point_indices = np.array([np.argmin(np.abs(x - xp)) for xp in bl_points_x])
        elif baseline_attached and baseline_mode == 'Polynomial' and len(bl.points[0]) >= 2:
            baseline_strategy = 'per_spectrum_poly'
            bl_points_x = np.array(bl.points[0])
            bl_point_indices = np.array([np.argmin(np.abs(x - xp)) for xp in bl_points_x])
        else:
            # Complex baseline modes (arpls, pybaselines, etc.) — fall back
            baseline_strategy = 'fallback'

        if baseline_strategy == 'fallback':
            # Can't batch-optimize this baseline mode — fall back
            for s in spectra:
                if not getattr(s, 'is_preprocessed', False):
                    s.preprocess()
            return

        # ── Apply to all spectra ──
        from scipy.interpolate import interp1d
        from scipy.ndimage import gaussian_filter1d

        for s in spectra:
            if getattr(s, 'is_preprocessed', False):
                continue

            # Clear cached noise parameters so they are re-evaluated on the fresh y data
            if hasattr(s, '_fit_ampli_noise'):
                delattr(s, '_fit_ampli_noise')
            if hasattr(s, '_fit_ymean'):
                delattr(s, '_fit_ymean')

            # load_profile + apply_range equivalent
            if mask is not None:
                s.x = x.copy()
                s.y = s.y0[mask].copy()
                if s.weights0 is not None:
                    s.weights = s.weights0[mask].copy()
            else:
                s.x = s.x0.copy()
                s.y = s.y0.copy()
                if s.weights0 is not None:
                    s.weights = s.weights0.copy()

            # Compute and subtract baseline
            if baseline_strategy == 'static':
                s.baseline.y_eval = baseline_static
                s.y = s.y - baseline_static
                s.baseline.is_subtracted = True

            elif baseline_strategy == 'per_spectrum_linear':
                # Attached linear baseline: get y-values at baseline point positions
                y_at_points = s.y[bl_point_indices]
                if bl_sigma > 0:
                    y_smooth = gaussian_filter1d(s.y, sigma=bl_sigma)
                    y_at_points = y_smooth[bl_point_indices]
                    
                if len(bl_point_indices) == 1:
                    s.baseline.y_eval = y_at_points[0] * np.ones_like(x)
                else:
                    pts_x = x[bl_point_indices]
                    # Check if baseline x-coords match spectrum x-coords exactly
                    if set(pts_x.tolist()).issubset(set(x.tolist())) and len(pts_x) == len(x):
                        d = dict(zip(pts_x, y_at_points))
                        s.baseline.y_eval = np.array([d[xi] for xi in x])
                    else:
                        func_interp = interp1d(pts_x, y_at_points, fill_value="extrapolate")
                        s.baseline.y_eval = func_interp(x)
                s.y = s.y - s.baseline.y_eval
                s.baseline.is_subtracted = True

            elif baseline_strategy == 'per_spectrum_poly':
                y_at_points = s.y[bl_point_indices]
                if bl_sigma > 0:
                    y_smooth = gaussian_filter1d(s.y, sigma=bl_sigma)
                    y_at_points = y_smooth[bl_point_indices]
                pts_x = x[bl_point_indices]
                order = min(bl.order_max, len(pts_x) - 1)
                coefs = np.polyfit(pts_x, y_at_points, order)
                s.baseline.y_eval = np.polyval(coefs, x)
                s.y = s.y - s.baseline.y_eval
                s.baseline.is_subtracted = True

            # 'none' strategy: nothing to do

            s.is_preprocessed = True

    def _build_fit_weights(self, spectra, fit_params, shared_x):
        """Build a weights matrix that mimics lmfit's masking behavior."""

        weights = []
        if fit_params is None:
            fit_params = {}
        fit_negative = bool(fit_params.get("fit_negative", False))
        fit_outliers = bool(fit_params.get("fit_outliers", False))
        coef_noise = float(fit_params.get("coef_noise", 1))

        for spectrum in spectra:
            if spectrum.y is None:
                M = len(spectra[0].x) if spectra[0].x is not None else 0
                weights.append(np.zeros(M, dtype=np.float64))
                continue

            w = np.ones_like(spectrum.y, dtype=np.float64)
            if not fit_negative:
                w[spectrum.y < 0] = 0.0

            if not fit_outliers:
                x_outliers, _ = spectrum.calculate_outliers()
                if x_outliers is not None:
                    w[np.isin(spectrum.x, x_outliers)] = 0.0

            if coef_noise > 0:
                if not hasattr(spectrum, '_fit_ampli_noise'):
                    spectrum._fit_ampli_noise = eval_noise_amplitude(spectrum.y)
                    spectrum._fit_ymean = np.convolve(spectrum.y, np.ones(5, dtype=np.float64) / 5.0, mode='same')
                
                ampli_noise = spectrum._fit_ampli_noise
                ymean = spectrum._fit_ymean
                noise_level = coef_noise * ampli_noise
                w[ymean < noise_level] = 0.0

            if spectrum.weights is not None:
                w = w * spectrum.weights

            if shared_x:
                weights.append(w)
            else:
                max_M = max((len(s.x) if s.x is not None else 0) for s in spectra)
                w_padded = np.zeros(max_M, dtype=np.float64)
                w_padded[:len(w)] = w
                weights.append(w_padded)

        return np.vstack(weights)
