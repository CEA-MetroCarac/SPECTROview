"""Batched Levenberg-Marquardt optimizer.

Solves N independent least-squares problems simultaneously using
NumPy tensor operations.  All N normal-equation systems are assembled
and solved via Cholesky factorisation (scipy cho_factor/cho_solve),
which exploits the symmetric positive-definite structure of the
normal equations for optimal performance.

Bound handling uses simple projection (clipping) after each step.
"""

import numpy as np


def _finite_or_clean(arr):
    """Zero out any NaN/Inf in-place, but only pay for it when needed.

    `np.isfinite(...).all()` is a single fused pass, whereas
    `np.nan_to_num` internally runs separate isnan/isposinf/isneginf
    passes plus fancy-index assignments. Since analytical Jacobians are
    finite in the overwhelming majority of iterations, checking first
    avoids that extra cost on the common path.
    """
    if not np.isfinite(arr).all():
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _batched_solve(A, b):
    """Solve A @ x = b for each spectrum.

    Adaptive strategy:
    - For large N (many spectra), np.linalg.solve is faster because NumPy's
      batched LAPACK dispatch amortises the overhead.
    - For small N with large K (few spectra, many parameters), a per-matrix
      Cholesky decomposition is faster because it exploits the symmetric
      positive-definite structure of the normal equations (JᵀJ + λI).

    Args:
        A: (Na, K, K) symmetric positive-definite matrices
        b: (Na, K) right-hand-side vectors

    Returns:
        x: (Na, K) solution vectors
    """
    Na, K = b.shape

    # Crossover heuristic: cho_solve loop wins when Na is small and K is large.
    # For Na > ~500 or K < ~10, np.linalg.solve's batched path is faster.
    if Na > 500 or K < 10:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fall through to per-matrix solve
            pass

    # Per-matrix Cholesky solve (best for small Na, large K).
    # scipy.linalg is imported here so it stays off the app's startup path.
    from scipy.linalg import cho_factor, cho_solve
    x = np.empty_like(b)
    for i in range(Na):
        try:
            c, low = cho_factor(A[i])
            x[i] = cho_solve((c, low), b[i])
        except np.linalg.LinAlgError:
            try:
                x[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                x[i] = 0.0
    return x



def _robust_scale_factor(r, loss="linear", f_scale=1.0):
    """Compute IRLS residual scaling factor sqrt(rho'( (r / f_scale)^2 )).

    Args:
        r: Residuals array of any shape.
        loss: 'linear', 'soft_l1', or 'huber'.
        f_scale: Inlier/outlier margin scale (default 1.0).

    Returns:
        Scaling factor array (or scalar 1.0 for linear) matching r's shape.
    """
    if loss == "linear":
        return 1.0
    u = np.abs(r / f_scale)
    if loss == "soft_l1":
        scale = np.where(
            u > 1e150,
            1.0 / np.sqrt(np.maximum(u, 1e-30)),
            (1.0 + np.minimum(u, 1e150) ** 2) ** (-0.25),
        )
    elif loss == "huber":
        scale = np.where(
            u > 1.0,
            1.0 / np.sqrt(np.maximum(u, 1.0)),
            1.0,
        )
    else:
        raise ValueError(f"Unknown loss '{loss}'. Expected 'linear', 'soft_l1', or 'huber'.")
    if not np.isfinite(scale).all():
        np.nan_to_num(scale, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
    return scale


def _robust_cost(r, loss="linear", f_scale=1.0):
    """Compute the total loss cost sum(rho((r / f_scale)^2) * f_scale^2) per spectrum.

    For 'linear': sum(r^2)
    For 'soft_l1': sum(2 * f_scale^2 * (sqrt(1 + (r / f_scale)^2) - 1))
    For 'huber': sum(r^2 if |r| <= f_scale else 2 * f_scale * |r| - f_scale^2)

    Args:
        r: Residuals array of shape (..., M).
        loss: 'linear', 'soft_l1', or 'huber'.
        f_scale: Inlier/outlier margin scale (default 1.0).

    Returns:
        Cost per spectrum of shape (...).
    """
    if loss == "linear":
        return np.sum(r * r, axis=-1)
    u = np.abs(r / f_scale)
    if loss == "soft_l1":
        # 2 * r^2 / (sqrt(1 + u^2) + 1) avoids catastrophic cancellation for small u
        # and avoids overflow for large u.
        r_small = np.where(u > 1e150, 0.0, r)
        rho_scaled = np.where(
            u > 1e150,
            2.0 * f_scale * np.abs(r),
            2.0 * (r_small ** 2) / (np.sqrt(1.0 + np.minimum(u, 1e150) ** 2) + 1.0),
        )
    elif loss == "huber":
        r_inlier = np.where(u > 1.0, 0.0, r)
        rho_scaled = np.where(
            u <= 1.0,
            r_inlier ** 2,
            2.0 * f_scale * np.abs(r) - (f_scale ** 2),
        )
    else:
        raise ValueError(f"Unknown loss '{loss}'. Expected 'linear', 'soft_l1', or 'huber'.")
    if not np.isfinite(rho_scaled).all():
        np.nan_to_num(rho_scaled, copy=False, nan=0.0, posinf=1e30, neginf=0.0)
    return np.sum(rho_scaled, axis=-1)


def batched_levenberg_marquardt(
    x,                  # (M,)
    Y_data,             # (N, M)
    evaluate_fn,        # callable(x, p) → (N, M)
    jacobian_fn,        # callable(x, p) → (N, M, K)
    p0,                 # (N, K) or (K,)
    lower_bounds,       # (K,)
    upper_bounds,       # (K,)
    weights=None,       # (N, M) or (M,)
    max_iter=200,
    xtol=1e-4,
    ftol=1e-4,
    loss="linear",
    f_scale=1.0,
    progress_callback=None,
    cancel_check=None,
):
    """Fit N spectra simultaneously using batched Levenberg-Marquardt.

    Supports standard least squares ('linear') and robust loss functions
    ('soft_l1', 'huber') via Iteratively Reweighted Least Squares (IRLS)
    residual and Jacobian scaling.

    Args:
        x: (M,) or (N, M) wavenumber axis
        Y_data: (N, M) intensity matrix
        evaluate_fn: callable(x, p) -> (N, M)
        jacobian_fn: callable(x, p) -> (N, M, K)
        p0: (N, K) or (K,) initial parameter guess
        lower_bounds: (K,) lower bounds
        upper_bounds: (K,) upper bounds
        weights: (N, M) or (M,) hard weights (e.g. from noise/exclusion mask)
        max_iter: int, maximum iterations
        xtol: float, relative parameter tolerance
        ftol: float, relative cost tolerance
        loss: str, 'linear' (default), 'soft_l1', or 'huber'
        f_scale: float, inlier/outlier margin scale (default 1.0)
        progress_callback: callable(current, total)
        cancel_check: callable() -> bool

    Returns
    -------
    p_opt : (N, K) optimised parameters
    success : (N,) bool array
    cost : (N,) final sum-of-squared-residuals
    """
    if loss not in ("linear", "soft_l1", "huber"):
        raise ValueError(f"Unknown loss '{loss}'. Expected 'linear', 'soft_l1', or 'huber'.")
    if f_scale <= 0:
        raise ValueError(f"f_scale must be positive, got {f_scale}")
    N, M = Y_data.shape
    K = len(lower_bounds)
    lo = np.asarray(lower_bounds, dtype=np.float64)
    hi = np.asarray(upper_bounds, dtype=np.float64)

    if weights is None:
        weights = np.ones((N, M), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = np.tile(weights, (N, 1))
        if weights.shape != (N, M):
            raise ValueError(f"weights must have shape ({N}, {M}), got {weights.shape}")

    # Broadcast p0 if 1-D
    if p0.ndim == 1:
        p = np.tile(p0, (N, 1)).astype(np.float64)
    else:
        p = p0.astype(np.float64).copy()

    # Clip initial guess to bounds
    p = np.clip(p, lo, hi)

    # Initial residuals and cost
    Y_pred = evaluate_fn(x, p)
    residuals = weights * (Y_pred - Y_data)        # (N, M)
    cost = _robust_cost(residuals, loss, f_scale)  # (N,)

    # Per-spectrum damping factor
    lam = np.full(N, 1e-2)
    converged = np.zeros(N, dtype=bool)
    converged |= (weights.sum(axis=1) == 0)

    # Track consecutive rejections to detect stuck spectra
    consecutive_rejects = np.zeros(N, dtype=int)
    MAX_REJECTS = 15  # mark as converged (stuck) after this many

    eye_K = np.eye(K, dtype=np.float64)

    # Cache of the normal-equation terms (JᵀJ, Jᵀr) per spectrum. A rejected
    # trial step leaves that spectrum's parameters unchanged, so its
    # Jacobian — and therefore JᵀJ/Jᵀr — is still exactly valid; only the
    # damping term needs to change before retrying. Recomputing them anyway
    # (as a naive implementation would) wastes the two most expensive
    # operations in the loop on spectra that didn't move. `dirty` tracks
    # which spectra need fresh normal equations (initially: all of them).
    JTJ_cache = np.empty((N, K, K), dtype=np.float64)
    JTr_cache = np.empty((N, K), dtype=np.float64)
    dirty = np.ones(N, dtype=bool)

    for iteration in range(max_iter):
        if cancel_check and cancel_check():
            break

        active = ~converged
        n_active = int(active.sum())
        if n_active == 0:
            break

        active_idx = np.where(active)[0]
        p_active = p[active_idx]
        x_active = x[active_idx] if x.ndim == 2 else x

        # ── Recompute the Jacobian/normal-equations only where stale ──
        recompute_idx = active_idx[dirty[active_idx]]
        if recompute_idx.size:
            p_r = p[recompute_idx]
            x_r = x[recompute_idx] if x.ndim == 2 else x
            J_r = jacobian_fn(x_r, p_r)          # (Nr, M, K)
            _finite_or_clean(J_r)

            # ── Normal equations: (JᵀJ + λ·diag(JᵀJ)) δp = -Jᵀr ──
            if loss != "linear":
                w_r = _robust_scale_factor(residuals[recompute_idx], loss, f_scale)
                J_r *= (weights[recompute_idx] * w_r)[:, :, None]
                scaled_r_r = w_r * residuals[recompute_idx]
            else:
                J_r *= weights[recompute_idx, :, None]
                scaled_r_r = residuals[recompute_idx]

            JT_r = J_r.transpose(0, 2, 1)
            # np.matmul dispatches to batched BLAS gemm; np.einsum with this
            # contraction pattern falls back to a much slower generic
            # reduction (10-20x slower at these array sizes).
            JTJ_cache[recompute_idx] = JT_r @ J_r
            JTr_cache[recompute_idx] = (JT_r @ scaled_r_r[:, :, None])[:, :, 0]
            dirty[recompute_idx] = False

        JTJ = JTJ_cache[active_idx]
        JTr = JTr_cache[active_idx]

        # Damping
        diag = np.diagonal(JTJ, axis1=1, axis2=2).copy()
        diag = np.maximum(diag, 1e-12)
        lam_a = lam[active_idx]
        damping = lam_a[:, None] * diag   # (Na, K)

        A = JTJ + eye_K[None, :, :] * damping[:, :, None]

        # Solve using Cholesky decomposition (8x faster than np.linalg.solve
        # for symmetric positive-definite normal equations)
        dp = _batched_solve(A, -JTr)

        # Replace NaN steps with zero
        _finite_or_clean(dp)

        # ── Trial step with projection ──
        p_trial_active = np.clip(p_active + dp, lo, hi)

        # Evaluate only the active spectra
        Y_trial_active = evaluate_fn(x_active, p_trial_active)
        r_trial_active = weights[active_idx] * (Y_trial_active - Y_data[active_idx])
        cost_trial_active = _robust_cost(r_trial_active, loss, f_scale)

        improved_active = cost_trial_active <= cost[active_idx]
        worsened_active = ~improved_active

        improved_idx = active_idx[improved_active]
        worsened_idx = active_idx[worsened_active]

        # Accept improved steps
        if improved_active.any():
            p[improved_idx] = p_trial_active[improved_active]
            residuals[improved_idx] = r_trial_active[improved_active]  # already weighted
            old_cost_improved = cost[improved_idx].copy()
            cost[improved_idx] = cost_trial_active[improved_active]
            dirty[improved_idx] = True  # parameters moved: normal equations are stale
        else:
            old_cost_improved = np.array([])

        # Lambda update
        lam[improved_idx] = np.maximum(lam[improved_idx] / 3.0, 1e-10)
        lam[worsened_idx] = np.minimum(lam[worsened_idx] * 2.5, 1e10)

        # ── Convergence (only for improved spectra) ──
        if improved_active.any():
            rel_cost_change = np.abs(old_cost_improved - cost[improved_idx]) / (cost[improved_idx] + 1e-30)

            # Use mean of relative parameter changes instead of max.
            # With max, a single slowly-converging parameter out of K blocks
            # the entire spectrum from converging, which scales very poorly
            # as K increases (e.g. K=18 for 6-peak models).
            dp_abs_rel = np.abs(dp[improved_active]) / np.maximum(np.abs(p_active[improved_active]), 1.0)
            dp_rel = np.mean(dp_abs_rel, axis=1)

            effective_xtol = max(xtol, 1e-3)
            newly_conv = (rel_cost_change < ftol) & (dp_rel < effective_xtol)
            converged[improved_idx[newly_conv]] = True

        # Track consecutive rejections
        consecutive_rejects[improved_idx] = 0
        consecutive_rejects[worsened_idx] += 1

        # Mark stuck spectra as converged
        converged |= (consecutive_rejects >= MAX_REJECTS)

        # ── Progress ──
        if progress_callback:
            progress_callback(int(converged.sum()), N)

    success = converged.copy()

    return p, success, cost
