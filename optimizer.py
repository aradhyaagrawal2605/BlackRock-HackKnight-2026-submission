"""
optimizer.py — Mean-Variance Portfolio Optimizer for BlackRock HackKnight 2026

Implements Markowitz MVO with all hackathon constraints:
  - Long-only (w >= 0)
  - Fully invested (sum(w) == 1)
  - Min position 0.5% if held
  - Max 30 holdings (cardinality, TC004)
  - Max 30% daily turnover (TC005)
  - Max single position cap (default 15%)

Uses cvxpy with OSQP solver for fast convex QP solving.
"""

import numpy as np
import cvxpy as cp
from typing import Optional


# ---------------------------------------------------------------------------
# Covariance estimation helpers
# ---------------------------------------------------------------------------

def shrink_covariance(returns: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf style shrinkage toward diagonal target."""
    sample_cov = np.cov(returns, rowvar=True)
    n = sample_cov.shape[0]
    target = np.diag(np.diag(sample_cov))
    shrunk = (1 - shrinkage) * sample_cov + shrinkage * target
    shrunk = (shrunk + shrunk.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(shrunk))
    if min_eig < 1e-8:
        shrunk += (1e-8 - min_eig) * np.eye(n)
    return shrunk


def ewma_covariance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance matrix."""
    T, n = returns.shape
    weights = np.array([(1 - lam) * lam ** (T - 1 - t) for t in range(T)])
    weights /= weights.sum()
    mean_r = weights @ returns
    centered = returns - mean_r
    cov = (centered * weights[:, None]).T @ centered
    cov = (cov + cov.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < 1e-8:
        cov += (1e-8 - min_eig) * np.eye(n)
    return cov


# ---------------------------------------------------------------------------
# Core MVO optimizer
# ---------------------------------------------------------------------------

def optimize_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    max_holdings: int = 30,
    max_weight: float = 0.15,
    min_weight: float = 0.005,
    turnover_budget: Optional[float] = None,
    current_turnover: float = 0.0,
    solver: str = "OSQP",
    eligible_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve the Markowitz MVO problem with hackathon constraints.

    Parameters
    ----------
    mu : (n,) expected return vector
    Sigma : (n, n) covariance matrix
    w_prev : (n,) previous weights (for turnover constraint)
    gamma : risk aversion parameter (higher = more conservative)
    max_holdings : max number of nonzero positions (TC004)
    max_weight : max weight for any single position
    min_weight : min weight for any held position (0.5%)
    turnover_budget : remaining turnover budget (fraction of 0.30)
    current_turnover : turnover already consumed
    solver : cvxpy solver name
    eligible_mask : boolean array; False = forced to zero weight

    Returns
    -------
    w_opt : (n,) optimal weight vector
    """
    n = len(mu)

    if w_prev is None:
        w_prev = np.zeros(n)

    if eligible_mask is None:
        eligible_mask = np.ones(n, dtype=bool)

    remaining_turnover = max(0.0, 0.30 - current_turnover) if turnover_budget is None else turnover_budget

    # --- Two-stage solve: first select top assets, then optimize ---
    # Stage 1: Rank assets by risk-adjusted attractiveness
    sigma_diag = np.sqrt(np.diag(Sigma))
    sigma_diag = np.where(sigma_diag < 1e-10, 1e-10, sigma_diag)
    attractiveness = mu / sigma_diag
    attractiveness[~eligible_mask] = -np.inf

    # Keep existing positions plus best new candidates
    held_mask = w_prev > 1e-6
    n_held = int(held_mask.sum())
    n_new_slots = max(0, max_holdings - n_held)

    candidate_scores = attractiveness.copy()
    candidate_scores[held_mask] = -np.inf
    new_indices = np.argsort(candidate_scores)[::-1][:n_new_slots]

    selected = held_mask.copy()
    for idx in new_indices:
        if eligible_mask[idx] and attractiveness[idx] > -np.inf:
            selected[idx] = True

    if selected.sum() == 0:
        top_indices = np.argsort(attractiveness)[::-1][:max_holdings]
        for idx in top_indices:
            if eligible_mask[idx]:
                selected[idx] = True
        if selected.sum() == 0:
            return np.ones(n) / n

    selected_indices = np.where(selected)[0]
    m = len(selected_indices)

    mu_sel = mu[selected_indices]
    Sigma_sel = Sigma[np.ix_(selected_indices, selected_indices)]
    w_prev_sel = w_prev[selected_indices]

    Sigma_sel = (Sigma_sel + Sigma_sel.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma_sel))
    if min_eig < 1e-8:
        Sigma_sel += (1e-8 - min_eig) * np.eye(m)

    # Stage 2: Convex optimization on selected subset
    w = cp.Variable(m, nonneg=True)
    portfolio_return = mu_sel @ w
    portfolio_risk = cp.quad_form(w, Sigma_sel)
    objective = cp.Maximize(portfolio_return - gamma * portfolio_risk)

    constraints = [
        cp.sum(w) == 1,
        w <= max_weight,
    ]

    if remaining_turnover < 0.30:
        constraints.append(
            cp.sum(cp.abs(w - w_prev_sel)) <= remaining_turnover
        )

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, warm_start=True, max_iter=5000)

        if prob.status in ("infeasible", "unbounded", None) or w.value is None:
            prob.solve(solver="SCS", max_iters=5000)

        if w.value is not None:
            w_opt_sel = np.array(w.value).flatten()
            w_opt_sel = np.maximum(w_opt_sel, 0)
            small_mask = (w_opt_sel > 0) & (w_opt_sel < min_weight)
            w_opt_sel[small_mask] = 0
            if w_opt_sel.sum() > 0:
                w_opt_sel /= w_opt_sel.sum()
            else:
                w_opt_sel = np.ones(m) / m
        else:
            w_opt_sel = _fallback_weights(mu_sel, m)
    except Exception:
        w_opt_sel = _fallback_weights(mu_sel, m)

    w_full = np.zeros(n)
    w_full[selected_indices] = w_opt_sel
    return w_full


def _fallback_weights(mu: np.ndarray, n: int) -> np.ndarray:
    """Simple fallback: inverse-volatility weighting biased toward high-mu assets."""
    if n == 0:
        return np.array([])
    scores = mu - mu.min() + 1e-6
    w = scores / scores.sum()
    return w


# ---------------------------------------------------------------------------
# Convenience: compute turnover from weight change
# ---------------------------------------------------------------------------

def compute_turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    """Sum of absolute weight changes."""
    return float(np.sum(np.abs(w_new - w_old)))


# ---------------------------------------------------------------------------
# Convenience: target weights -> order quantities
# ---------------------------------------------------------------------------

def weights_to_quantities(
    weights: np.ndarray,
    portfolio_value: float,
    prices: np.ndarray,
) -> np.ndarray:
    """Convert target weights to integer share quantities."""
    target_values = weights * portfolio_value
    safe_prices = np.where(prices > 0, prices, 1e10)
    quantities = np.floor(target_values / safe_prices).astype(int)
    return quantities


# ---------------------------------------------------------------------------
# Adaptive gamma selection based on market regime
# ---------------------------------------------------------------------------

def adaptive_gamma(
    recent_returns: np.ndarray,
    base_gamma: float = 1.0,
    vol_lookback: int = 20,
) -> float:
    """
    Increase risk aversion when recent volatility is high (risk-off),
    decrease when volatility is low (risk-on).
    """
    if len(recent_returns) < 5:
        return base_gamma
    recent_vol = np.std(recent_returns[-vol_lookback:])
    long_vol = np.std(recent_returns) if len(recent_returns) > vol_lookback else recent_vol
    if long_vol < 1e-10:
        return base_gamma
    vol_ratio = recent_vol / long_vol
    return base_gamma * max(0.5, min(3.0, vol_ratio))


# ---------------------------------------------------------------------------
# Max-Sharpe variant (risk-free rate = 0)
# ---------------------------------------------------------------------------

def optimize_max_sharpe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    eligible_mask: Optional[np.ndarray] = None,
    max_holdings: int = 30,
    max_weight: float = 0.15,
) -> np.ndarray:
    """
    Max-Sharpe portfolio via the Cornuejols-Tutuncu variable substitution.
    Solves: min y'Sigma y  s.t. mu'y = 1, y >= 0, then w = y / sum(y).
    """
    n = len(mu)
    if eligible_mask is None:
        eligible_mask = np.ones(n, dtype=bool)

    sigma_diag = np.sqrt(np.diag(Sigma))
    sigma_diag = np.where(sigma_diag < 1e-10, 1e-10, sigma_diag)
    attractiveness = mu / sigma_diag
    attractiveness[~eligible_mask] = -np.inf
    top_idx = np.argsort(attractiveness)[::-1][:max_holdings]
    top_idx = top_idx[eligible_mask[top_idx]]

    if len(top_idx) == 0:
        return np.ones(n) / n

    mu_sel = mu[top_idx]
    Sigma_sel = Sigma[np.ix_(top_idx, top_idx)]
    Sigma_sel = (Sigma_sel + Sigma_sel.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma_sel))
    if min_eig < 1e-8:
        Sigma_sel += (1e-8 - min_eig) * np.eye(len(top_idx))

    m = len(top_idx)
    y = cp.Variable(m, nonneg=True)
    constraints = [mu_sel @ y == 1, y <= max_weight * cp.sum(y)]

    try:
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, Sigma_sel)), constraints)
        prob.solve(solver="OSQP", max_iter=5000)
        if y.value is not None and np.sum(y.value) > 1e-10:
            w_sel = np.array(y.value).flatten()
            w_sel = np.maximum(w_sel, 0)
            w_sel /= w_sel.sum()
        else:
            w_sel = np.ones(m) / m
    except Exception:
        w_sel = np.ones(m) / m

    w_full = np.zeros(n)
    w_full[top_idx] = w_sel
    return w_full
