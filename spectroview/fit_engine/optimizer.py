"""Batched Levenberg-Marquardt optimizer.

Solves N independent least-squares problems simultaneously using
NumPy tensor operations.  Normal-equation systems are assembled via
batched matrix multiplication and solved with batched LAPACK
(``np.linalg.solve``), falling back to per-matrix Cholesky/solve on
singular cases.

Bound handling uses simple projection (clipping) after each step.
"""

import inspect

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

    Uses NumPy's batched ``np.linalg.solve`` (vectorized LAPACK across all
    ``Na`` systems) as the primary path for every batch size and parameter
    count, falling back to per-matrix SciPy Cholesky / ``np.linalg.solve`` /
    zeros only when a matrix in the batch is singular.

    Args:
        A: (Na, K, K) symmetric positive-definite matrices
        b: (Na, K) right-hand-side vectors

    Returns:
        x: (Na, K) solution vectors
    """
    Na, K = b.shape
    if Na == 0 or K == 0:
        return np.empty_like(b)

    # Batched C/LAPACK solve outperforms the Python cho_factor/cho_solve loop
    # across all realistic Na and K ranges (including small Na tails).
    try:
        return np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        pass

    # Per-matrix fallback when one or more systems in the batch are singular.
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


def _lm_batch_core(
    x,
    Y_data,
    evaluate_fn,
    jacobian_fn,
    p,
    lo,
    hi,
    weights,
    weights_is_ones,
    max_iter,
    xtol,
    ftol,
    loss,
    f_scale,
    eval_buf,
    J_buf_base,
    progress_callback=None,
    cancel_check=None,
    progress_offset=0,
    progress_total=None,
):
    """Run batched Levenberg-Marquardt on a single batch/chunk in-place on `p`."""
    N, M = Y_data.shape
    K = len(lo)
    if progress_total is None:
        progress_total = N

    # Initial residuals and cost
    Y_pred = evaluate_fn(x, p, out=eval_buf[:N]) if eval_buf is not None else evaluate_fn(x, p)
    if weights_is_ones:
        residuals = Y_pred - Y_data            # (N, M), distinct buffer from eval_buf
    else:
        residuals = weights * (Y_pred - Y_data)  # (N, M)
    cost = _robust_cost(residuals, loss, f_scale)  # (N,)

    # Per-spectrum damping factor
    lam = np.full(N, 1e-2)
    converged = np.zeros(N, dtype=bool)
    if weights_is_ones:
        if M == 0:
            converged[:] = True
    else:
        converged |= (weights.sum(axis=1) == 0)

    # Track consecutive rejections to detect stuck spectra
    consecutive_rejects = np.zeros(N, dtype=int)
    MAX_REJECTS = 15  # mark as converged (stuck) after this many

    diag_idx = np.arange(K)
    effective_xtol = max(xtol, 1e-3)

    # Cache of the normal-equation terms (JᵀJ, Jᵀr) per spectrum.
    JTJ_cache = np.empty((N, K, K), dtype=np.float64)
    JTr_cache = np.empty((N, K), dtype=np.float64)
    dirty = np.ones(N, dtype=bool)
    cancelled = False

    for iteration in range(max_iter):
        if cancel_check and cancel_check():
            cancelled = True
            break

        active = ~converged
        n_active = int(active.sum())
        if n_active == 0:
            break

        all_active = (n_active == N)
        if all_active:
            active_idx = None
            p_active = p
            x_active = x
            all_recompute = bool(dirty.all())
            recompute_idx = None if all_recompute else np.where(dirty)[0]
            n_recompute = N if all_recompute else recompute_idx.size
        else:
            active_idx = np.where(active)[0]
            p_active = p[active_idx]
            x_active = x[active_idx] if x.ndim == 2 else x
            all_recompute = False
            recompute_idx = active_idx[dirty[active_idx]]
            n_recompute = recompute_idx.size

        # ── Recompute the Jacobian/normal-equations only where stale ──
        if n_recompute:
            if all_recompute:
                p_r = p
                x_r = x
                res_r = residuals
                w_r_hard = None if weights_is_ones else weights
            else:
                p_r = p[recompute_idx]
                x_r = x[recompute_idx] if x.ndim == 2 else x
                res_r = residuals[recompute_idx]
                w_r_hard = None if weights_is_ones else weights[recompute_idx]

            if J_buf_base is not None:
                J_r_buf = J_buf_base[:, :n_recompute, :].transpose(1, 2, 0)
                J_r = jacobian_fn(x_r, p_r, out=J_r_buf)
            else:
                J_r = jacobian_fn(x_r, p_r)          # (Nr, M, K)
            _finite_or_clean(J_r)

            # ── Normal equations: (JᵀJ + λ·diag(JᵀJ)) δp = -Jᵀr ──
            if loss != "linear":
                w_r = _robust_scale_factor(res_r, loss, f_scale)
                scale_r = w_r if weights_is_ones else (w_r_hard * w_r)
                J_r *= scale_r[:, :, None]
                scaled_r_r = w_r * res_r
            else:
                if not weights_is_ones:
                    J_r *= w_r_hard[:, :, None]
                scaled_r_r = res_r

            JT_r = J_r.transpose(0, 2, 1)
            # np.matmul dispatches to batched BLAS gemm; when J_r is backed by
            # (K, Nr, M), the contraction axis M is stride-8 contiguous.
            if all_recompute:
                np.matmul(JT_r, J_r, out=JTJ_cache)
                np.matmul(JT_r, scaled_r_r[:, :, None], out=JTr_cache[:, :, None])
                dirty.fill(False)
            else:
                JTJ_cache[recompute_idx] = JT_r @ J_r
                JTr_cache[recompute_idx] = (JT_r @ scaled_r_r[:, :, None])[:, :, 0]
                dirty[recompute_idx] = False

        if all_active:
            JTJ = JTJ_cache
            JTr = JTr_cache
            lam_a = lam
        else:
            JTJ = JTJ_cache[active_idx]
            JTr = JTr_cache[active_idx]
            lam_a = lam[active_idx]

        # Damping: add λ·max(diag(JᵀJ), 1e-12) directly to the diagonal of A
        diag = np.maximum(np.diagonal(JTJ, axis1=1, axis2=2), 1e-12)
        A = JTJ.copy()
        if K > 0:
            A[:, diag_idx, diag_idx] += lam_a[:, None] * diag

        # Solve normal equations in batched C/LAPACK
        dp = _batched_solve(A, -JTr)

        # Replace NaN steps with zero
        _finite_or_clean(dp)

        # ── Trial step with projection ──
        p_trial_active = np.clip(p_active + dp, lo, hi)

        # Evaluate only the active spectra
        Y_active = Y_data if all_active else Y_data[active_idx]
        w_active = None if weights_is_ones else (weights if all_active else weights[active_idx])
        if eval_buf is not None:
            r_trial_active = evaluate_fn(x_active, p_trial_active, out=eval_buf[:n_active])
            r_trial_active -= Y_active
            if not weights_is_ones:
                r_trial_active *= w_active
        else:
            Y_trial_active = evaluate_fn(x_active, p_trial_active)
            r_trial_active = (
                Y_trial_active - Y_active
                if weights_is_ones
                else w_active * (Y_trial_active - Y_active)
            )
        cost_trial_active = _robust_cost(r_trial_active, loss, f_scale)

        cost_active = cost if all_active else cost[active_idx]
        improved_active = cost_trial_active <= cost_active
        worsened_active = ~improved_active

        if all_active:
            if improved_active.any():
                if improved_active.all():
                    old_cost_improved = cost.copy()
                    dp_actual = p_trial_active - p
                    dp_abs_rel = np.abs(dp_actual) / np.maximum(np.abs(p), 1.0)
                    dp_rel = np.mean(dp_abs_rel, axis=1)

                    p[:] = p_trial_active
                    residuals[:] = r_trial_active
                    cost[:] = cost_trial_active
                    dirty.fill(True)

                    rel_cost_change = np.abs(old_cost_improved - cost) / (cost + 1e-30)
                    newly_conv = (rel_cost_change < ftol) & (dp_rel < effective_xtol)
                    converged |= newly_conv
                    lam[:] = np.maximum(lam / 3.0, 1e-10)
                    consecutive_rejects.fill(0)
                else:
                    improved_idx = np.where(improved_active)[0]
                    worsened_idx = np.where(worsened_active)[0]
                    old_cost_improved = cost[improved_idx].copy()
                    p_improved_old = p[improved_idx]
                    p_improved_new = p_trial_active[improved_idx]
                    dp_actual = p_improved_new - p_improved_old
                    dp_abs_rel = np.abs(dp_actual) / np.maximum(np.abs(p_improved_old), 1.0)
                    dp_rel = np.mean(dp_abs_rel, axis=1)

                    p[improved_idx] = p_improved_new
                    residuals[improved_idx] = r_trial_active[improved_idx]
                    cost[improved_idx] = cost_trial_active[improved_idx]
                    dirty[improved_idx] = True

                    rel_cost_change = np.abs(old_cost_improved - cost[improved_idx]) / (cost[improved_idx] + 1e-30)
                    newly_conv = (rel_cost_change < ftol) & (dp_rel < effective_xtol)
                    converged[improved_idx[newly_conv]] = True

                    lam[improved_idx] = np.maximum(lam[improved_idx] / 3.0, 1e-10)
                    lam[worsened_idx] = np.minimum(lam[worsened_idx] * 2.5, 1e10)
                    consecutive_rejects[improved_idx] = 0
                    consecutive_rejects[worsened_idx] += 1
            else:
                lam[:] = np.minimum(lam * 2.5, 1e10)
                consecutive_rejects += 1
        else:
            improved_idx = active_idx[improved_active]
            worsened_idx = active_idx[worsened_active]

            if improved_active.any():
                old_cost_improved = cost[improved_idx].copy()
                p_improved_old = p_active[improved_active]
                p_improved_new = p_trial_active[improved_active]
                dp_actual = p_improved_new - p_improved_old
                dp_abs_rel = np.abs(dp_actual) / np.maximum(np.abs(p_improved_old), 1.0)
                dp_rel = np.mean(dp_abs_rel, axis=1)

                p[improved_idx] = p_improved_new
                residuals[improved_idx] = r_trial_active[improved_active]
                cost[improved_idx] = cost_trial_active[improved_active]
                dirty[improved_idx] = True

                rel_cost_change = np.abs(old_cost_improved - cost[improved_idx]) / (cost[improved_idx] + 1e-30)
                newly_conv = (rel_cost_change < ftol) & (dp_rel < effective_xtol)
                converged[improved_idx[newly_conv]] = True

            lam[improved_idx] = np.maximum(lam[improved_idx] / 3.0, 1e-10)
            lam[worsened_idx] = np.minimum(lam[worsened_idx] * 2.5, 1e10)
            consecutive_rejects[improved_idx] = 0
            consecutive_rejects[worsened_idx] += 1

        # Mark stuck spectra as converged
        converged |= (consecutive_rejects >= MAX_REJECTS)

        # ── Progress ──
        if progress_callback:
            progress_callback(progress_offset + int(converged.sum()), progress_total)

    return p, converged.copy(), cost, cancelled


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

    weights_is_ones = False
    if weights is None:
        weights_is_ones = True
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = np.tile(weights, (N, 1))
        if weights.shape != (N, M):
            raise ValueError(f"weights must have shape ({N}, {M}), got {weights.shape}")
        weights_is_ones = bool(np.all(weights == 1.0))

    # Broadcast p0 if 1-D
    if p0.ndim == 1:
        p = np.tile(p0, (N, 1)).astype(np.float64)
    else:
        p = p0.astype(np.float64).copy()

    # Clip initial guess to bounds
    p = np.clip(p, lo, hi)

    try:
        _eval_supports_out = "out" in inspect.signature(evaluate_fn).parameters
    except (TypeError, ValueError):
        _eval_supports_out = False
    try:
        _jac_supports_out = "out" in inspect.signature(jacobian_fn).parameters
    except (TypeError, ValueError):
        _jac_supports_out = False

    # Adaptive L3-cache chunking for large batches: keeps per-chunk Jacobian,
    # residual, and evaluation buffers inside CPU L2/L3 cache while isolating
    # slow-converging tail spectra to their own chunk.
    mk = max(M * max(K, 1), 1)
    if N * mk > 2_097_152 or (N >= 2048 and N * mk > 1_048_576):
        raw_chunk = max(512, min(1024, 524_288 // mk))
        chunk_size = ((raw_chunk + 255) // 256) * 256
    else:
        chunk_size = N

    buf_n = min(N, max(chunk_size, 1))
    eval_buf = np.empty((buf_n, M), dtype=np.float64) if _eval_supports_out and N > 0 else None
    J_buf_base = np.empty((K, buf_n, M), dtype=np.float64) if _jac_supports_out and N > 0 and K > 0 else None

    if chunk_size >= N:
        p_opt, success, cost, _ = _lm_batch_core(
            x=x,
            Y_data=Y_data,
            evaluate_fn=evaluate_fn,
            jacobian_fn=jacobian_fn,
            p=p,
            lo=lo,
            hi=hi,
            weights=weights,
            weights_is_ones=weights_is_ones,
            max_iter=max_iter,
            xtol=xtol,
            ftol=ftol,
            loss=loss,
            f_scale=f_scale,
            eval_buf=eval_buf,
            J_buf_base=J_buf_base,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            progress_offset=0,
            progress_total=N,
        )
        return p_opt, success, cost

    success = np.zeros(N, dtype=bool)
    cost = np.empty(N, dtype=np.float64)
    completed_conv = 0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_c = x[start:end] if x.ndim == 2 else x
        Y_c = Y_data[start:end]
        w_c = None if weights_is_ones else weights[start:end]
        w_c_ones = True if weights_is_ones else bool(np.all(w_c == 1.0))

        _, s_c, c_c, cancelled = _lm_batch_core(
            x=x_c,
            Y_data=Y_c,
            evaluate_fn=evaluate_fn,
            jacobian_fn=jacobian_fn,
            p=p[start:end],
            lo=lo,
            hi=hi,
            weights=w_c,
            weights_is_ones=w_c_ones,
            max_iter=max_iter,
            xtol=xtol,
            ftol=ftol,
            loss=loss,
            f_scale=f_scale,
            eval_buf=eval_buf,
            J_buf_base=J_buf_base,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            progress_offset=completed_conv,
            progress_total=N,
        )
        success[start:end] = s_c
        cost[start:end] = c_c
        completed_conv += int(s_c.sum())
        if cancelled:
            # Fill remaining unvisited chunks' initial cost if cancelled early
            if end < N:
                x_rem = x[end:] if x.ndim == 2 else x
                Y_rem_pred = evaluate_fn(x_rem, p[end:])
                r_rem = (
                    Y_rem_pred - Y_data[end:]
                    if weights_is_ones
                    else weights[end:] * (Y_rem_pred - Y_data[end:])
                )
                cost[end:] = _robust_cost(r_rem, loss, f_scale)
            break

    return p, success, cost
