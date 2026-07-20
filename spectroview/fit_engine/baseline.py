import numpy as np

_INTERNAL_METHODS = {
    None: {'label': 'None', 'use_points': False}, 
    'Linear': {'label': 'Linear Interpolation', 'use_points': True, 'sigma_kwarg': 'sigma', 'category': 'Manual'}, 
    'Polynomial': {'label': 'Polynomial Fit', 'use_points': True, 'order_kwarg': 'order_max', 'sigma_kwarg': 'sigma', 'category': 'Manual'},
    'arpls': {'label': 'arPLS ⭐', 'coef_kwarg': 'smoothing_factor', 'use_points': False, 'category': 'Whittaker', 'help': 'Asymmetrically Reweighted PLS.'}
}

_PYBASELINES_WHITELIST = {
    'airpls': {'label': 'airPLS', 'category': 'Whittaker', 'coef_kwarg': 'lam', 'help': 'Adaptive Iterative PLS. Excellent for varying noise levels.'}, 
    'asls': {'label': 'AsLS', 'category': 'Whittaker', 'coef_kwarg': 'lam', 'help': 'Asymmetric Least Squares. The classic algorithm.'}, 
    'modpoly': {'label': 'ModPoly', 'category': 'Polynomial', 'order_kwarg': 'poly_order', 'help': 'Modified Polynomial. Good for simple baselines.'}
}

def get_baseline_method_meta(mode: str) -> dict:
    if mode in _INTERNAL_METHODS:
        return _INTERNAL_METHODS[mode]
    if mode in _PYBASELINES_WHITELIST:
        return _PYBASELINES_WHITELIST[mode]
    return {}

def eval_baseline(x: np.ndarray, y: np.ndarray, config: dict) -> np.ndarray:
    """Evaluate baseline curve for a single spectrum."""
    mode = config.get("mode")
    if not mode:
        return np.zeros_like(x)
        
    attached = config.get("attached", False)
    points = config.get("points", [[], []])
    
    if mode == "Linear":
        bl_points_x = np.array(points[0])
        if len(bl_points_x) == 0:
            return np.zeros_like(x)
            
        bl_point_indices = np.array([np.argmin(np.abs(x - xp)) for xp in bl_points_x])
        
        if attached:
            y_at_points = y[bl_point_indices]
            sigma = config.get("sigma", 4)
            if sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                y_smooth = gaussian_filter1d(y, sigma=sigma)
                y_at_points = y_smooth[bl_point_indices]
        else:
            y_at_points = np.array(points[1])
            
        if len(bl_point_indices) == 1:
            return y_at_points[0] * np.ones_like(x)
        else:
            pts_x = x[bl_point_indices]
            if set(pts_x.tolist()).issubset(set(x.tolist())) and len(pts_x) == len(x):
                d = dict(zip(pts_x, y_at_points))
                return np.array([d[xi] for xi in x])
            else:
                from scipy.interpolate import interp1d
                func_interp = interp1d(pts_x, y_at_points, fill_value="extrapolate")
                return func_interp(x)
                
    elif mode == "Polynomial":
        bl_points_x = np.array(points[0])
        if len(bl_points_x) == 0:
            return np.zeros_like(x)
            
        bl_point_indices = np.array([np.argmin(np.abs(x - xp)) for xp in bl_points_x])
        
        if attached:
            y_at_points = y[bl_point_indices]
            sigma = config.get("sigma", 4)
            if sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                y_smooth = gaussian_filter1d(y, sigma=sigma)
                y_at_points = y_smooth[bl_point_indices]
        else:
            y_at_points = np.array(points[1])
            
        pts_x = x[bl_point_indices]
        order_max = config.get("order_max", 3)
        order = min(order_max, len(pts_x) - 1)
        if order < 0:
            return np.zeros_like(x)
        coefs = np.polyfit(pts_x, y_at_points, order)
        return np.polyval(coefs, x)
        
    elif mode == 'arpls':
        try:
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x)
            lam = 10 ** config.get("coef", 5.0)
            b, _ = baseline_fitter.arpls(y, lam=lam)
            return b
        except Exception:
            return np.zeros_like(x)
    elif mode == 'sonneveld_vesser':
        try:
            from pybaselines.classification import Classification
            baseline_fitter = Classification(x_data=x)
            niter = config.get("coef", 100)
            b, _ = baseline_fitter.dietrich(y, num_iter=int(niter)) # Just an approximation for Sonneveld-Vesser
            return b
        except Exception:
            return np.zeros_like(x)
    else:
        try:
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x)
            meta = get_baseline_method_meta(mode)
            kwargs = {}
            if meta.get("coef_kwarg"):
                val = config.get("coef", 5.0)
                if meta.get("coef_kwarg") == "lam":
                    val = 10 ** val
                kwargs[meta["coef_kwarg"]] = val
            if meta.get("order_kwarg"):
                kwargs[meta["order_kwarg"]] = config.get("order_max", 3)
            if meta.get("sigma_kwarg"):
                kwargs[meta["sigma_kwarg"]] = config.get("sigma", 3)
                
            func = getattr(baseline_fitter, mode, None)
            if func:
                b, _ = func(y, **kwargs)
                return b
            else:
                return np.zeros_like(x)
        except Exception:
            return np.zeros_like(x)

def eval_baseline_batch(x: np.ndarray, Y: np.ndarray, config: dict) -> np.ndarray:
    """Evaluate baseline for a batch of spectra (N, M)."""
    mode = config.get("mode")
    if not mode:
        return np.zeros_like(Y)

    attached = config.get("attached", False)
    points = config.get("points", [[], []])

    # Fully vectorize Linear and Polynomial modes across the entire map
    if mode in ["Linear", "Polynomial"]:
        bl_points_x = np.array(points[0])
        if len(bl_points_x) == 0:
            return np.zeros_like(Y)

        bl_point_indices = np.array([np.argmin(np.abs(x - xp)) for xp in bl_points_x])

        if attached:
            sigma = config.get("sigma", 4)
            if sigma > 0:
                # gaussian_filter1d vectorizes natively along axis=-1 (M dimension)
                from scipy.ndimage import gaussian_filter1d
                y_smooth = gaussian_filter1d(Y, sigma=sigma, axis=-1)
                y_at_points = y_smooth[:, bl_point_indices]
            else:
                y_at_points = Y[:, bl_point_indices]
        else:
            y_at_points = np.array(points[1])
            # Broadcast to match batch shape (N, K)
            y_at_points = np.tile(y_at_points, (Y.shape[0], 1))

        if mode == "Linear":
            if len(bl_point_indices) == 1:
                return np.tile(y_at_points[:, 0:1], (1, len(x)))
            else:
                pts_x = x[bl_point_indices]
                if set(pts_x.tolist()).issubset(set(x.tolist())) and len(pts_x) == len(x):
                    return y_at_points
                else:
                    from scipy.interpolate import interp1d
                    # interp1d vectorizes naturally if provided a 2D array, interpolating along axis=-1
                    func_interp = interp1d(pts_x, y_at_points, axis=-1, fill_value="extrapolate")
                    return func_interp(x).astype(np.float32)

        elif mode == "Polynomial":
            pts_x = x[bl_point_indices]
            order_max = config.get("order_max", 3)
            order = min(order_max, len(pts_x) - 1)
            if order < 0:
                return np.zeros_like(Y)
            
            # np.polyfit can fit N polynomials simultaneously if y is shape (K, N)
            coefs = np.polyfit(pts_x, y_at_points.T, order) # shape: (order + 1, N)
            
            # Evaluate optimally via Vandermonde matrix dot product
            V = np.vander(x, order + 1) # shape: (M, order + 1)
            Y_baseline = (V @ coefs).T # output shape: (N, M)
            return Y_baseline.astype(np.float32)

    # For iterative solvers like pybaselines (arpls, airpls) which strictly require 1D arrays,
    # we must fall back to evaluating them sequentially per spectrum.
    N = Y.shape[0]
    Y_baseline = np.empty_like(Y)
    for i in range(N):
        Y_baseline[i] = eval_baseline(x, Y[i], config)
    return Y_baseline
