# spectroview/core2/optimizer.py
"""Batched Levenberg-Marquardt optimizer.

Solves N independent least-squares problems simultaneously using
NumPy tensor operations.  All N normal-equation systems are assembled
and solved in a single call to np.linalg.solve, which dispatches to
LAPACK and runs at C speed.

Bound handling uses simple projection (clipping) after each step.
"""

import numpy as np


def batched_levenberg_marquardt(
    x,                  # (M,)
    Y_data,             # (N, M)
    evaluate_fn,        # callable(x, p) → (N, M)
    jacobian_fn,        # callable(x, p) → (N, M, K)
    p0,                 # (N, K) or (K,)
    lower_bounds,       # (K,)
    upper_bounds,       # (K,)
    max_iter=200,
    xtol=1e-3,
    ftol=1e-3,
    progress_callback=None,
    cancel_check=None,
):
    """Fit N spectra simultaneously using batched Levenberg-Marquardt.

    Returns
    -------
    p_opt : (N, K) optimised parameters
    success : (N,) bool array
    cost : (N,) final sum-of-squared-residuals
    """
    N, M = Y_data.shape
    K = len(lower_bounds)
    lo = np.asarray(lower_bounds, dtype=np.float64)
    hi = np.asarray(upper_bounds, dtype=np.float64)

    # Broadcast p0 if 1-D
    if p0.ndim == 1:
        p = np.tile(p0, (N, 1)).astype(np.float64)
    else:
        p = p0.astype(np.float64).copy()

    # Clip initial guess to bounds
    p = np.clip(p, lo, hi)

    # Initial residuals and cost
    Y_pred = evaluate_fn(x, p)
    residuals = Y_pred - Y_data        # (N, M)
    cost = np.sum(residuals * residuals, axis=1)  # (N,)

    # Per-spectrum damping factor
    lam = np.full(N, 1e-2)
    converged = np.zeros(N, dtype=bool)

    # Track consecutive rejections to detect stuck spectra
    consecutive_rejects = np.zeros(N, dtype=int)
    MAX_REJECTS = 15  # mark as converged (stuck) after this many

    eye_K = np.eye(K, dtype=np.float64)

    for iteration in range(max_iter):
        if cancel_check and cancel_check():
            break

        active = ~converged
        n_active = int(active.sum())
        if n_active == 0:
            break

        # ── Compute Jacobian ONLY for active spectra ──
        # Extract active subset
        p_active = p[active]                # (Na, K)
        r_active = residuals[active]        # (Na, M)

        J_active = jacobian_fn(x, p_active)  # (Na, M, K)

        # Replace any NaN/Inf in Jacobian with zero
        np.nan_to_num(J_active, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Normal equations: (JᵀJ + λ·diag(JᵀJ)) δp = -Jᵀr ──
        JTJ = np.einsum('nmk,nml->nkl', J_active, J_active)  # (Na, K, K)
        JTr = np.einsum('nmk,nm->nk', J_active, r_active)    # (Na, K)

        # Damping
        diag = np.diagonal(JTJ, axis1=1, axis2=2).copy()
        diag = np.maximum(diag, 1e-12)
        lam_a = lam[active]
        damping = lam_a[:, None] * diag   # (Na, K)

        A = JTJ + eye_K[None, :, :] * damping[:, :, None]

        # Solve
        try:
            dp = np.linalg.solve(A, -JTr)  # (Na, K)
        except np.linalg.LinAlgError:
            # Add strong regularization and retry
            A += eye_K[None, :, :] * 1e-2
            try:
                dp = np.linalg.solve(A, -JTr)
            except np.linalg.LinAlgError:
                # Give up on this iteration
                lam[active] *= 10.0
                consecutive_rejects[active] += 1
                converged |= (consecutive_rejects >= MAX_REJECTS)
                continue

        # Replace NaN steps with zero
        np.nan_to_num(dp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Trial step with projection ──
        p_trial_active = np.clip(p_active + dp, lo, hi)

        # Evaluate only the active spectra
        Y_trial_active = evaluate_fn(x, p_trial_active)
        r_trial_active = Y_trial_active - Y_data[active]
        cost_trial_active = np.sum(r_trial_active * r_trial_active, axis=1)

        improved_active = cost_trial_active < cost[active]
        worsened_active = ~improved_active

        active_idx = np.where(active)[0]
        improved_idx = active_idx[improved_active]
        worsened_idx = active_idx[worsened_active]

        # Accept improved steps
        if improved_active.any():
            p[improved_idx] = p_trial_active[improved_active]
            residuals[improved_idx] = r_trial_active[improved_active]
            old_cost_improved = cost[improved_idx].copy()
            cost[improved_idx] = cost_trial_active[improved_active]
        else:
            old_cost_improved = np.array([])

        # Lambda update
        lam[improved_idx] = np.maximum(lam[improved_idx] / 3.0, 1e-10)
        lam[worsened_idx] = np.minimum(lam[worsened_idx] * 2.5, 1e10)

        # ── Convergence (only for improved spectra) ──
        if improved_active.any():
            rel_cost_change = np.abs(old_cost_improved - cost[improved_idx]) / (cost[improved_idx] + 1e-30)
            dp_for_improved = np.max(np.abs(dp[improved_active]), axis=1)
            newly_conv = (rel_cost_change < ftol) & (dp_for_improved < xtol)
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
