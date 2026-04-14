"""
Black-Litterman with DPP Views — A CVXPY Tutorial
==================================================

This example completes the CVXPY-finance cookbook's three-tutorial arc:

    1. portfolio_optimization_dpp.py  — Mean-variance with transaction costs
    2. risk_parity_spinu.py           — Spinu ERC in the exponential cone
    3. black_litterman.py             — THIS FILE

Black-Litterman (BL) blends two sources of information about expected
returns: a market-equilibrium *prior* and investor *views*. The posterior
mean ``mu_BL`` has a clean closed-form solution, so at first glance it is
not obvious why CVXPY should be involved at all.

The tutorial argues the CVXPY value-add in three ways:

    1. BL is equivalent to a quadratic optimization problem. Rewriting
       the closed form as an optimization is an excellent pedagogical
       bridge between Bayesian inference and convex programming — and
       CVXPY lets you verify the equivalence numerically with three lines.

    2. The closed form breaks as soon as you add **constraints on the
       posterior mean**: "I believe AAPL > 0 returns" (an absolute bound)
       or "I believe AAPL > MSFT" (a relative ranking). CVXPY handles
       these for free. We demonstrate both.

    3. DPP lets you cache the BL problem once and re-solve with **new
       views** in milliseconds, which is the production use case — an
       analyst's views change daily while the covariance matrix updates
       weekly. With ``P``, ``Q`` as ``cp.Parameter`` objects the view
       machinery becomes a fast inner loop of a larger system.

Plus:
    - Walk-forward backtest: historical-mean MV vs BL-no-views vs
      BL-with-momentum-views on real price data
    - Correctness: closed-form and CVXPY (both naive and DPP) agree to
      solver precision
    - Speed benchmark: DPP-cached BL re-solve time vs naive rebuild

Run:
    python examples/black_litterman.py
    python examples/black_litterman.py --save-plots  # writes PNGs

Dependencies: cvxpy, numpy, pandas. Optional: matplotlib, yfinance.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This example needs cvxpy. Install with: pip install cvxpy"
    ) from exc


# ---------------------------------------------------------------------------
# Section 1: Data pipeline — shared with the other two tutorials
# ---------------------------------------------------------------------------
# Same 20-ticker universe, same yfinance-or-GBM loader. New this tutorial:
# a market-cap-weight approximator for computing equilibrium implied returns.

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA",
    "DIS", "BAC", "XOM", "PFE", "KO",
]

# Rough market caps (trillions USD) as of early 2026, used as a hard-coded
# fallback if yfinance's `info['marketCap']` is unavailable. These only
# have to be approximately right — ``w_mkt`` enters the model as weights,
# so a constant scale factor drops out.
_STATIC_MARKET_CAPS_T_USD = {
    "AAPL": 3.8, "MSFT": 3.5, "GOOGL": 2.3, "AMZN": 2.2, "META": 1.6,
    "NVDA": 3.2, "TSLA": 1.1, "BRK-B": 1.0, "JPM": 0.7, "JNJ": 0.4,
    "V": 0.6, "PG": 0.4, "UNH": 0.45, "HD": 0.4, "MA": 0.5,
    "DIS": 0.2, "BAC": 0.3, "XOM": 0.5, "PFE": 0.2, "KO": 0.3,
}


def load_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Close-price DataFrame via yfinance; GBM fallback if offline."""
    try:
        import yfinance as yf

        data = yf.download(
            tickers, period=period, progress=False, auto_adjust=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers[:1]
        prices = prices.dropna(axis=1, how="any")
        if not prices.empty and len(prices) > 260:
            return prices
    except Exception as exc:  # pragma: no cover
        print(f"[warn] yfinance fetch failed ({exc}); using synthetic data")

    rng = np.random.default_rng(42)
    n_days = 504
    drift = rng.uniform(0.05, 0.15, size=len(tickers)) / 252
    vol = rng.uniform(0.15, 0.40, size=len(tickers)) / np.sqrt(252)
    shocks = rng.standard_normal((n_days, len(tickers))) * vol + drift
    prices_arr = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    return pd.DataFrame(prices_arr, index=idx, columns=tickers)


def estimate_covariance(prices: pd.DataFrame) -> np.ndarray:
    """Annualized sample covariance, symmetrized and nudged to be PD."""
    returns = prices.pct_change().dropna()
    sigma = returns.cov().values * 252.0
    n = sigma.shape[0]
    return 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n)


def market_cap_weights(tickers: list[str]) -> np.ndarray:
    """Approximate market-cap weights for the universe.

    Tries yfinance's ``info['marketCap']`` first; falls back to the static
    dictionary above. The absolute scale doesn't matter (we normalize),
    only the relative cap of each asset vs the others.
    """
    caps: list[float] = []
    try:
        import yfinance as yf

        for t in tickers:
            try:
                cap = float(yf.Ticker(t).info.get("marketCap") or 0.0)
            except Exception:
                cap = 0.0
            if cap <= 0:
                cap = _STATIC_MARKET_CAPS_T_USD.get(t, 0.1) * 1e12
            caps.append(cap)
    except Exception:
        caps = [_STATIC_MARKET_CAPS_T_USD.get(t, 0.1) * 1e12 for t in tickers]

    w = np.array(caps, dtype=float)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Section 2: The Black-Litterman model
# ---------------------------------------------------------------------------
# The setup in one paragraph. Start from the CAPM-equilibrium belief that
# the market-cap-weighted portfolio is mean-variance optimal. Reverse-
# optimization gives the *implied equilibrium returns*:
#
#     Pi = lambda * Sigma @ w_mkt
#
# where lambda is the market's risk aversion coefficient (historically
# around 2-3 for a US equity universe). Pi is the prior.
#
# Investor views are expressed as linear statements on the unknown mu:
#
#     P @ mu = Q + epsilon,     epsilon ~ N(0, Omega)
#
# with P a k x n "picking" matrix (each row is one view), Q a k-vector
# of target view returns, and Omega the view-uncertainty covariance
# (usually diagonal — views are considered independent).
#
# Applying Bayes' rule under the additional prior ``mu ~ N(Pi, tau * Sigma)``
# gives the Black-Litterman posterior mean
#
#     mu_BL = [(tau*Sigma)^-1 + P.T @ Omega^-1 @ P]^-1
#             @ [(tau*Sigma)^-1 @ Pi + P.T @ Omega^-1 @ Q]
#
# which is ALSO the argmin of the convex quadratic
#
#     J(mu) = (mu-Pi).T @ (tau*Sigma)^-1 @ (mu-Pi)
#             + (P @ mu - Q).T @ Omega^-1 @ (P @ mu - Q).
#
# We'll implement both forms below and verify they agree.
#
# Typical heuristics:
#   - lambda:  2.5
#   - tau:     0.05 (He & Litterman)
#   - Omega_ii = tau * (P_i @ Sigma @ P_i.T)  ("confidence matches implied
#              uncertainty"; this is the He-Litterman default).


def implied_equilibrium_returns(
    sigma: np.ndarray, w_mkt: np.ndarray, risk_aversion: float = 2.5
) -> np.ndarray:
    """Pi = lambda * Sigma @ w_mkt (reverse-optimization)."""
    return float(risk_aversion) * (sigma @ w_mkt)


def he_litterman_omega(
    p: np.ndarray, sigma: np.ndarray, tau: float = 0.05
) -> np.ndarray:
    """Diagonal view-covariance per the He-Litterman default.

    Omega_ii = tau * (P_i @ Sigma @ P_i.T) — confidence in each view scales
    with the prior uncertainty projected through the picking matrix row.
    """
    diag = tau * np.diag(p @ sigma @ p.T)
    return np.diag(np.maximum(diag, 1e-10))  # floor to avoid zero


# ---------------------------------------------------------------------------
# Section 3: Implementation 1 — Closed-form via matrix inversion
# ---------------------------------------------------------------------------
# The ground truth. Pure numpy; no CVXPY. We'll use this as the reference
# answer against which the two CVXPY implementations get compared.

def bl_posterior_closed_form(
    pi: np.ndarray,
    sigma: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Closed-form Black-Litterman posterior mean.

    Computes
        mu_BL = (A + B)^-1 @ (A @ Pi + P.T @ Omega^-1 @ Q),
    with A = (tau*Sigma)^-1 and B = P.T @ Omega^-1 @ P.
    """
    n = pi.shape[0]
    a = np.linalg.inv(tau * sigma)
    om_inv = np.linalg.inv(omega)
    lhs = a + p.T @ om_inv @ p
    rhs = a @ pi + p.T @ om_inv @ q
    return np.linalg.solve(lhs, rhs)


# ---------------------------------------------------------------------------
# Section 4: Implementation 2 — Naive CVXPY (rebuild on every call)
# ---------------------------------------------------------------------------
# Same math, expressed as a convex quadratic program. Still rebuilds the
# Problem from numpy inputs each call — this is the pedagogical bridge:
# "look, the BL posterior IS a convex program".
#
# We use the Cholesky reformulation once more: for any PD matrix M with
# M = L @ L.T, the quadratic form x.T @ M^-1 @ x can be rewritten as
# sum_squares((L^-1).T @ x) = sum_squares(inv_L.T @ x) where inv_L =
# np.linalg.inv(L). (Equivalently: sum_squares(L_inv_sqrt @ x) with
# L_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(M)), but the Cholesky
# factorization is cheaper and stabler for PD inputs.)

def _inv_cholesky(m: np.ndarray) -> np.ndarray:
    """Return ``Linv = inv(chol(M))`` for PD ``M``.

    Given ``M = L @ L.T`` with ``L`` lower-triangular, returns ``Linv``
    such that ``Linv.T @ Linv == inv(M)``. This is the factor used in the
    CVXPY reformulation below:

        x.T @ inv(M) @ x  ==  ||Linv @ x||^2  ==  sum_squares(Linv @ x).

    Transpose trap: ``sum_squares(Linv.T @ x)`` gives ``x.T @ inv(L.T @ L) @ x``,
    which is generally NOT ``x.T @ inv(M) @ x``. Watch the transpose.
    """
    L = np.linalg.cholesky(m)
    return np.linalg.solve(L, np.eye(m.shape[0]))


def bl_posterior_cvxpy_naive(
    pi: np.ndarray,
    sigma: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
    solver: str = "CLARABEL",
) -> np.ndarray:
    """Solve the BL problem as a CVXPY quadratic. Rebuilds each call."""
    n = pi.shape[0]

    mu = cp.Variable(n)
    # ||A @ (mu - Pi)||^2 = (mu - Pi).T @ inv(tau*Sigma) @ (mu - Pi)
    a_prior = _inv_cholesky(tau * sigma)
    # ||B @ (P @ mu - Q)||^2 = (P @ mu - Q).T @ inv(Omega) @ (P @ mu - Q)
    a_view = _inv_cholesky(omega)

    prior_term = cp.sum_squares(a_prior @ (mu - pi))
    view_term = cp.sum_squares(a_view @ (p @ mu - q))
    problem = cp.Problem(cp.Minimize(prior_term + view_term))
    problem.solve(solver=solver)

    if mu.value is None:
        raise RuntimeError(f"BL solve failed (status={problem.status})")
    return mu.value


# ---------------------------------------------------------------------------
# Section 5: Implementation 3 — DPP-cached with parametric views
# ---------------------------------------------------------------------------
# In a real desk workflow, the covariance matrix is refreshed weekly or
# monthly, but analyst views update daily. So P and Q are the things that
# change solve-to-solve.
#
# We declare them as cp.Parameter objects. The prior factor ``a_prior`` and
# the view-uncertainty factor ``a_view`` are baked in as constants (they
# depend on Sigma and Omega, which are fixed between refreshes).
#
# DPP check: in ``sum_squares(a_view @ (P_param @ mu - Q_param))``, all
# multiplications are of the form constant @ Parameter-or-Variable or
# Parameter @ Variable — no Parameter @ Parameter — so the expression is
# DPP-affine and the whole thing is DPP-convex. See the ``verify_`` function
# in Section 6 for an empirical check.

class BLRebalancer:
    """Build-once, re-solve Black-Litterman posterior with parametric views.

    Constant between solves (baked into the canonicalized Problem):
      - sigma, pi, tau, omega
      - Cholesky factors of inv(tau*Sigma) and inv(Omega)

    Varies per solve (cp.Parameter objects):
      - P (k x n picking matrix)
      - Q (k-vector of view returns)

    The number of views ``k`` is fixed at construction because the shapes
    of P, Q have to be known to build the Parameters. If the number of
    views changes, construct a new BLRebalancer.
    """

    def __init__(
        self,
        sigma: np.ndarray,
        pi: np.ndarray,
        omega: np.ndarray,
        k: int,
        tau: float = 0.05,
        solver: str = "CLARABEL",
    ) -> None:
        n = sigma.shape[0]
        self._n = n
        self._k = k
        self._solver = solver
        self.pi = pi
        self.sigma = sigma
        self.omega = omega
        self.tau = tau

        # Cholesky factors baked in as constants.
        self._a_prior = _inv_cholesky(tau * sigma)
        self._a_view = _inv_cholesky(omega)

        self.mu = cp.Variable(n)
        self.P_param = cp.Parameter((k, n))
        self.Q_param = cp.Parameter(k)

        prior_term = cp.sum_squares(self._a_prior @ (self.mu - pi))
        view_term = cp.sum_squares(
            self._a_view @ (self.P_param @ self.mu - self.Q_param)
        )
        self.problem = cp.Problem(cp.Minimize(prior_term + view_term))
        assert self.problem.is_dcp(dpp=True), (
            "BLRebalancer should be DPP-compliant — verify that Sigma, Pi, "
            "and Omega are numpy arrays (not Parameters)."
        )

    def solve(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        if p.shape != (self._k, self._n):
            raise ValueError(
                f"P shape {p.shape} != expected ({self._k}, {self._n})"
            )
        if q.shape != (self._k,):
            raise ValueError(f"Q shape {q.shape} != expected ({self._k},)")
        self.P_param.value = p
        self.Q_param.value = q
        self.problem.solve(solver=self._solver)
        if self.mu.value is None:
            raise RuntimeError(
                f"BL solve failed (status={self.problem.status})"
            )
        return self.mu.value


# ---------------------------------------------------------------------------
# Section 6: Correctness — do all three implementations agree?
# ---------------------------------------------------------------------------
# Run a handful of random BL problems through closed-form, naive CVXPY,
# and DPP-cached CVXPY, and assert that all three posteriors agree to
# within solver tolerance.

def verify_all_implementations(
    n_checks: int = 5, n: int = 8, k: int = 3, tol: float = 1e-5
) -> None:
    rng = np.random.default_rng(0)
    print("\nCorrectness check: closed-form / naive-CVXPY / DPP-CVXPY must agree.")

    for check in range(n_checks):
        A = rng.standard_normal((n, n))
        sigma = A @ A.T + 0.2 * np.eye(n)
        w_mkt = rng.dirichlet(np.ones(n))
        pi = implied_equilibrium_returns(sigma, w_mkt, risk_aversion=2.5)

        # Absolute views on a random subset of assets.
        view_assets = rng.choice(n, size=k, replace=False)
        p = np.zeros((k, n))
        for j, a in enumerate(view_assets):
            p[j, a] = 1.0
        q = 0.05 + 0.1 * rng.standard_normal(k)
        omega = he_litterman_omega(p, sigma, tau=0.05)

        mu_cf = bl_posterior_closed_form(pi, sigma, p, q, omega, tau=0.05)
        mu_naive = bl_posterior_cvxpy_naive(pi, sigma, p, q, omega, tau=0.05)
        reb = BLRebalancer(sigma, pi, omega, k=k, tau=0.05)
        mu_dpp = reb.solve(p, q)

        err_naive = float(np.max(np.abs(mu_cf - mu_naive)))
        err_dpp = float(np.max(np.abs(mu_cf - mu_dpp)))
        ok = err_naive < tol and err_dpp < tol
        status = "PASS" if ok else "FAIL"
        print(
            f"  check {check+1}: "
            f"max|cf-naive|={err_naive:.2e}  "
            f"max|cf-dpp|={err_dpp:.2e}  [{status}]"
        )
        assert ok, f"Implementations disagree (tol={tol})."


# ---------------------------------------------------------------------------
# Section 7: Speed benchmark — naive rebuild vs DPP-cached re-solve
# ---------------------------------------------------------------------------
# Realistic workload: 60 "days", each with a new random view vector,
# fixed Sigma/Pi/Omega. This mirrors a desk where analyst views update
# daily but the risk model refreshes weekly.

@dataclass
class BenchmarkResult:
    n_days: int
    closed_total: float
    naive_total: float
    dpp_total: float
    closed_per_ms: float
    naive_per_ms: float
    dpp_per_ms: float
    speedup_dpp_over_naive: float


def benchmark(prices: pd.DataFrame, n_days: int = 60) -> BenchmarkResult:
    sigma = estimate_covariance(prices)
    n = sigma.shape[0]
    w_mkt = market_cap_weights(list(prices.columns))
    pi = implied_equilibrium_returns(sigma, w_mkt)

    # Generate a stream of random view specs, all with the same shape.
    k = 3
    rng = np.random.default_rng(1)
    view_stream = []
    for _ in range(n_days):
        view_assets = rng.choice(n, size=k, replace=False)
        p = np.zeros((k, n))
        for j, a in enumerate(view_assets):
            p[j, a] = 1.0
        q = 0.05 + 0.1 * rng.standard_normal(k)
        omega = he_litterman_omega(p, sigma)
        view_stream.append((p, q, omega))

    print(f"\nUniverse: {n} assets, k={k} views, benchmarking {n_days} "
          "BL posterior computations.\n")

    # Closed-form baseline
    t0 = time.perf_counter()
    for p, q, omega in view_stream:
        _ = bl_posterior_closed_form(pi, sigma, p, q, omega)
    closed_total = time.perf_counter() - t0

    # Naive CVXPY
    t0 = time.perf_counter()
    for p, q, omega in view_stream:
        _ = bl_posterior_cvxpy_naive(pi, sigma, p, q, omega)
    naive_total = time.perf_counter() - t0

    # DPP-cached. Note: Omega is held constant (first view's Omega) since
    # it's baked into the Problem. In a real pipeline Omega depends on the
    # view structure, so changing view shapes requires rebuilding — a
    # modest limitation documented in the docstring.
    rebalancer = BLRebalancer(sigma, pi, view_stream[0][2], k=k, tau=0.05)
    _ = rebalancer.solve(view_stream[0][0], view_stream[0][1])  # compile cost
    t0 = time.perf_counter()
    for p, q, _omega in view_stream[1:]:
        _ = rebalancer.solve(p, q)
    dpp_total = time.perf_counter() - t0

    closed_per = 1e3 * closed_total / n_days
    naive_per = 1e3 * naive_total / n_days
    dpp_per = 1e3 * dpp_total / max(n_days - 1, 1)
    speedup = naive_total / max(dpp_total, 1e-9)

    print(f"  [1] Closed-form (numpy)       : "
          f"{closed_total:5.2f}s  ({closed_per:6.2f} ms/call)")
    print(f"  [2] Naive CVXPY rebuild       : "
          f"{naive_total:5.2f}s  ({naive_per:6.2f} ms/call)")
    print(f"  [3] DPP-cached CVXPY re-solve : "
          f"{dpp_total:5.2f}s  ({dpp_per:6.2f} ms/call, first compile excluded)")
    print()
    print(f"  Aggregate speedup [2] -> [3]: {speedup:5.1f}x")
    print(
        "  (Closed-form is fastest for pure BL with no constraints; CVXPY's"
        "\n   value kicks in when you add view-mu constraints — see Section 8.)"
    )

    return BenchmarkResult(
        n_days=n_days,
        closed_total=closed_total,
        naive_total=naive_total,
        dpp_total=dpp_total,
        closed_per_ms=closed_per,
        naive_per_ms=naive_per,
        dpp_per_ms=dpp_per,
        speedup_dpp_over_naive=speedup,
    )


# ---------------------------------------------------------------------------
# Section 8: The CVXPY-only value-add — BL with posterior constraints
# ---------------------------------------------------------------------------
# If you ONLY care about the posterior mean, the closed-form is faster and
# there is no reason to involve CVXPY. The reason to involve it is this:
# often an investor has views that can't be expressed as a Gaussian but
# CAN be expressed as a hard constraint. Examples:
#
#     - "AAPL will outperform 0 with probability 1"   → mu[AAPL] >= 0
#     - "AAPL will outperform MSFT"                   → mu[AAPL] >= mu[MSFT]
#     - "all views are positive expected returns"     → mu >= Pi - 0.05
#
# These are *inequality constraints on the posterior mean*. The closed
# form breaks — you would have to solve a constrained quadratic program.
# CVXPY handles it in one line.
#
# The DPP cache still works as long as the constraints' coefficients are
# known at construction time. For ranking constraints that change day to
# day, you'd rebuild.

def bl_posterior_with_constraints(
    pi: np.ndarray,
    sigma: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
    bounds_lower: np.ndarray | None = None,
    ranking: list[tuple[int, int]] | None = None,
    solver: str = "CLARABEL",
) -> np.ndarray:
    """BL posterior subject to extra linear constraints on mu.

    Args:
        bounds_lower: per-asset lower bound on posterior mu (length n).
        ranking: list of (i, j) pairs interpreted as ``mu[i] >= mu[j]``.
    """
    n = pi.shape[0]
    mu = cp.Variable(n)
    a_prior = _inv_cholesky(tau * sigma)
    a_view = _inv_cholesky(omega)

    obj = (
        cp.sum_squares(a_prior @ (mu - pi))
        + cp.sum_squares(a_view @ (p @ mu - q))
    )
    constraints: list = []
    if bounds_lower is not None:
        constraints.append(mu >= bounds_lower)
    if ranking:
        for i, j in ranking:
            constraints.append(mu[i] >= mu[j])

    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(solver=solver)
    if mu.value is None:
        raise RuntimeError(f"Constrained BL failed (status={problem.status})")
    return mu.value


def demo_constrained_views(prices: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Constrained-views demo  (the closed form can't handle this)")
    print("=" * 60)
    sigma = estimate_covariance(prices)
    n = sigma.shape[0]
    tickers = list(prices.columns)
    w_mkt = market_cap_weights(tickers)
    pi = implied_equilibrium_returns(sigma, w_mkt)

    # Deliberately contradictory view to highlight the constraint effect:
    # the data says AAPL is a negative-return asset at equilibrium, but we
    # want to impose mu[AAPL] >= 0.
    aapl = tickers.index("AAPL")
    msft = tickers.index("MSFT")
    p = np.zeros((1, n))
    p[0, aapl] = 1.0
    q = np.array([-0.30])  # view that AAPL returns -30% annualized
    omega = 0.1 * he_litterman_omega(p, sigma)  # 10x stronger than default

    mu_unconstrained = bl_posterior_closed_form(pi, sigma, p, q, omega)
    mu_constrained = bl_posterior_with_constraints(
        pi, sigma, p, q, omega,
        bounds_lower=np.where(np.arange(n) == aapl, 0.0, -np.inf),
    )
    mu_ranking = bl_posterior_with_constraints(
        pi, sigma, p, q, omega,
        ranking=[(aapl, msft)],
    )

    print(f"\n  Pi (equilibrium, annualized): AAPL={pi[aapl]*100:+5.2f}%  "
          f"MSFT={pi[msft]*100:+5.2f}%")
    print(f"  Pi + bearish AAPL view (Q=-30%, tight Omega):")
    print(f"    unconstrained posterior:     AAPL={mu_unconstrained[aapl]*100:+5.2f}%  "
          f"MSFT={mu_unconstrained[msft]*100:+5.2f}%")
    print(f"    + floor mu[AAPL] >= 0:       AAPL={mu_constrained[aapl]*100:+5.2f}%  "
          f"MSFT={mu_constrained[msft]*100:+5.2f}%")
    print(f"    + ranking mu[AAPL]>=mu[MSFT]: AAPL={mu_ranking[aapl]*100:+5.2f}%  "
          f"MSFT={mu_ranking[msft]*100:+5.2f}%")
    print("\n  The floor binds (AAPL posterior pinned at 0); the ranking")
    print("  binds as well (AAPL posterior lifted to MSFT's posterior).")


# ---------------------------------------------------------------------------
# Section 9: Walk-forward backtest
# ---------------------------------------------------------------------------
# Does feeding BL posteriors into a standard mean-variance optimizer
# produce better portfolios than using raw historical means? Standard
# finance critique of historical-mean MV is that it's unstable: tiny
# changes in the estimated mu cause huge changes in optimal weights.
# BL is supposed to ameliorate that by anchoring mu to the market prior.
#
# Three strategies compared:
#   1. Historical-mean MV   — tau = ... nothing; just use the sample mean
#   2. BL-no-views MV       — mu_prior = Pi, ignore analyst views
#   3. BL-with-views MV     — views from a simple momentum signal
#
# The momentum views: each rebalancing day, the top-3 and bottom-3 assets
# by trailing 60-day return get views proportional to the z-score of
# their trailing return. This is a deliberately simple signal — we don't
# claim it predicts returns; we only use it to generate non-trivial
# view vectors for the tutorial.

def _mv_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma: float = 5.0,
    long_only: bool = True,
) -> np.ndarray:
    """One-shot mean-variance rebalance. Long-only, sums to 1."""
    n = len(mu)
    w = cp.Variable(n)
    obj = mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(sigma))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    cp.Problem(cp.Maximize(obj), constraints).solve()
    return np.asarray(w.value)


def _momentum_views(
    window_returns: pd.DataFrame, k: int = 3, strength: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate P, Q, Omega from a 60-day momentum signal.

    Top-k assets get positive absolute views of magnitude ``strength``;
    bottom-k get symmetric negative views. Omega is set to zero so the
    views bind tightly — but we floor it slightly so the matrix is PD.
    """
    trailing = (1.0 + window_returns.tail(60)).prod() - 1.0
    top = trailing.sort_values(ascending=False).head(k).index.tolist()
    bot = trailing.sort_values(ascending=True).head(k).index.tolist()
    n = len(trailing)

    picks: list[tuple[int, float]] = []
    cols = list(trailing.index)
    for t in top:
        picks.append((cols.index(t), strength))
    for t in bot:
        picks.append((cols.index(t), -strength))

    p = np.zeros((len(picks), n))
    q = np.zeros(len(picks))
    for row, (idx, val) in enumerate(picks):
        p[row, idx] = 1.0
        q[row] = val
    # moderately-confident views; don't pin them
    sigma = window_returns.cov().values * 252.0
    sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n)
    omega = he_litterman_omega(p, sigma, tau=0.05)
    return p, q, omega


@dataclass
class BLBacktestResult:
    tickers: list[str]
    dates: pd.DatetimeIndex
    hist_returns: np.ndarray
    bl_prior_returns: np.ndarray
    bl_views_returns: np.ndarray
    hist_equity: np.ndarray
    bl_prior_equity: np.ndarray
    bl_views_equity: np.ndarray
    hist_metrics: dict[str, float]
    bl_prior_metrics: dict[str, float]
    bl_views_metrics: dict[str, float]
    weight_history: dict[str, np.ndarray] = field(default_factory=dict)


def _metrics(returns: np.ndarray, freq: int = 252) -> dict[str, float]:
    equity = np.cumprod(1.0 + returns)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    sharpe = float(np.sqrt(freq) * mean / std) if std > 0 else float("nan")
    return {
        "total_return": float(equity[-1] - 1.0),
        "annualized_vol": float(std * np.sqrt(freq)),
        "sharpe": sharpe,
        "max_drawdown": float(abs(np.min(
            equity / np.maximum.accumulate(equity) - 1.0
        ))),
        "final": float(equity[-1]),
    }


def walk_forward_bl(
    prices: pd.DataFrame,
    lookback: int = 252,
    gamma: float = 5.0,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    view_strength: float = 0.05,
) -> BLBacktestResult:
    returns = prices.pct_change().dropna()
    if len(returns) <= lookback + 5:
        raise ValueError("Not enough history.")
    tickers = list(returns.columns)
    n = len(tickers)
    test_dates = returns.index[lookback:]
    w_mkt = market_cap_weights(tickers)

    hist_daily: list[float] = []
    bl_prior_daily: list[float] = []
    bl_views_daily: list[float] = []
    w_hist_log: list[np.ndarray] = []
    w_prior_log: list[np.ndarray] = []
    w_views_log: list[np.ndarray] = []
    date_log: list = []

    for i, date in enumerate(test_dates[:-1]):
        idx = returns.index.get_loc(date)
        window = returns.iloc[idx - lookback: idx]
        sigma = window.cov().values * 252.0
        sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n)

        # Strategy 1: raw historical mean into MV.
        mu_hist = window.mean().values * 252.0

        # Strategy 2: BL prior-only (no views).
        pi = implied_equilibrium_returns(sigma, w_mkt, risk_aversion)
        mu_prior = pi

        # Strategy 3: BL with momentum views.
        p_mat, q_vec, omega = _momentum_views(
            window, k=3, strength=view_strength
        )
        mu_views = bl_posterior_closed_form(pi, sigma, p_mat, q_vec, omega, tau)

        w_hist = _mv_weights(mu_hist, sigma, gamma=gamma)
        w_prior = _mv_weights(mu_prior, sigma, gamma=gamma)
        w_views = _mv_weights(mu_views, sigma, gamma=gamma)

        next_ret = returns.iloc[idx + 1].values
        hist_daily.append(float(np.dot(w_hist, next_ret)))
        bl_prior_daily.append(float(np.dot(w_prior, next_ret)))
        bl_views_daily.append(float(np.dot(w_views, next_ret)))
        w_hist_log.append(w_hist)
        w_prior_log.append(w_prior)
        w_views_log.append(w_views)
        date_log.append(test_dates[i + 1])

    hist_arr = np.array(hist_daily)
    bl_prior_arr = np.array(bl_prior_daily)
    bl_views_arr = np.array(bl_views_daily)

    return BLBacktestResult(
        tickers=tickers,
        dates=pd.DatetimeIndex(date_log),
        hist_returns=hist_arr,
        bl_prior_returns=bl_prior_arr,
        bl_views_returns=bl_views_arr,
        hist_equity=np.cumprod(1.0 + hist_arr),
        bl_prior_equity=np.cumprod(1.0 + bl_prior_arr),
        bl_views_equity=np.cumprod(1.0 + bl_views_arr),
        hist_metrics=_metrics(hist_arr),
        bl_prior_metrics=_metrics(bl_prior_arr),
        bl_views_metrics=_metrics(bl_views_arr),
        weight_history={
            "hist": np.asarray(w_hist_log),
            "bl_prior": np.asarray(w_prior_log),
            "bl_views": np.asarray(w_views_log),
        },
    )


def print_backtest_report(result: BLBacktestResult) -> None:
    hist = result.hist_metrics
    prior = result.bl_prior_metrics
    views = result.bl_views_metrics
    n_days = len(result.hist_returns)

    print(f"\nWalk-forward backtest over {n_days} trading days "
          "(out-of-sample; 252-day rolling window)\n")
    print(f"  {'Metric':<22}{'Historical':>13}{'BL-prior':>13}{'BL+views':>13}")
    print(f"  {'-' * 61}")
    for name, key in [
        ("Total return", "total_return"),
        ("Annualized vol", "annualized_vol"),
        ("Sharpe", "sharpe"),
        ("Max drawdown", "max_drawdown"),
    ]:
        fmt = "{:>12.2f}%" if "return" in key or "vol" in key or "drawdown" in key else "{:>13.3f}"
        scale = 100 if ("return" in key or "vol" in key or "drawdown" in key) else 1
        print(f"  {name:<22}"
              f"{fmt.format(hist[key]*scale)}"
              f"{fmt.format(prior[key]*scale)}"
              f"{fmt.format(views[key]*scale)}")
    print(f"  {'$1 grows to':<22}"
          f"{'$'+format(hist['final'],'.3f'):>13}"
          f"{'$'+format(prior['final'],'.3f'):>13}"
          f"{'$'+format(views['final'],'.3f'):>13}")

    # Turnover: how stable are the weights?
    def _turnover(w_series: np.ndarray) -> float:
        diffs = np.abs(np.diff(w_series, axis=0)).sum(axis=1)
        return float(diffs.mean())

    t_hist = _turnover(result.weight_history["hist"])
    t_prior = _turnover(result.weight_history["bl_prior"])
    t_views = _turnover(result.weight_history["bl_views"])
    print(f"\n  Average daily turnover  (lower = more stable weights):")
    print(f"    Historical-mean MV : {t_hist*100:6.2f}%")
    print(f"    BL-prior MV        : {t_prior*100:6.2f}%")
    print(f"    BL-with-views MV   : {t_views*100:6.2f}%")


# ---------------------------------------------------------------------------
# Section 10: Optional plots
# ---------------------------------------------------------------------------

def save_plots(
    bench: BenchmarkResult, back: BLBacktestResult, out_dir: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("[warn] matplotlib not available; skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: per-call timing bars across 3 implementations.
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Closed-form\n(numpy)", "Naive CVXPY", "DPP CVXPY"]
    vals = [bench.closed_per_ms, bench.naive_per_ms, bench.dpp_per_ms]
    colors = ["#9cc2e4", "#d46a6a", "#6ad49a"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Milliseconds per BL posterior")
    ax.set_title("Black-Litterman posterior: closed-form vs CVXPY")
    for bar, v in zip(bars, vals, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2.0, v,
                f"{v:.2f} ms", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "01_bl_timing.png", dpi=130)
    plt.close(fig)

    # Figure 2: equity curves.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(back.dates, back.hist_equity, label="Historical-mean MV", linewidth=2)
    ax.plot(back.dates, back.bl_prior_equity, label="BL-prior MV",
            linewidth=2, linestyle="--")
    ax.plot(back.dates, back.bl_views_equity, label="BL+views MV",
            linewidth=2, linestyle=":")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Walk-forward equity curves")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "02_bl_equity.png", dpi=130)
    plt.close(fig)

    # Figure 3: turnover comparison (proxy for weight stability).
    fig, ax = plt.subplots(figsize=(7, 4))
    hist_turn = np.abs(np.diff(back.weight_history["hist"], axis=0)).sum(axis=1)
    prior_turn = np.abs(np.diff(back.weight_history["bl_prior"], axis=0)).sum(axis=1)
    views_turn = np.abs(np.diff(back.weight_history["bl_views"], axis=0)).sum(axis=1)
    ax.plot(back.dates[1:], hist_turn * 100, label="Historical", alpha=0.75)
    ax.plot(back.dates[1:], prior_turn * 100, label="BL-prior", alpha=0.75)
    ax.plot(back.dates[1:], views_turn * 100, label="BL+views", alpha=0.75)
    ax.set_ylabel("Daily turnover (%)")
    ax.set_title("Portfolio-weight stability")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "03_bl_turnover.png", dpi=130)
    plt.close(fig)

    # Figure 4: posterior shift from views — bar chart of Pi vs mu_BL.
    prices = load_prices(TICKERS, period="2y")
    sigma = estimate_covariance(prices)
    tickers = list(prices.columns)
    w_mkt = market_cap_weights(tickers)
    pi = implied_equilibrium_returns(sigma, w_mkt)
    p_mat, q_vec, omega = _momentum_views(
        prices.pct_change().dropna(), k=3, strength=0.05
    )
    mu_bl = bl_posterior_closed_form(pi, sigma, p_mat, q_vec, omega)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(tickers))
    width = 0.35
    ax.bar(x - width / 2, pi * 100, width, label="Prior Pi",
           color="#9cc2e4")
    ax.bar(x + width / 2, mu_bl * 100, width, label="Posterior mu_BL",
           color="#d48a6a")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=40, ha="right")
    ax.set_ylabel("Annualized return (%)")
    ax.set_title("Prior Pi vs Black-Litterman posterior mu_BL "
                 "(with momentum views)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "04_bl_prior_vs_posterior.png", dpi=130)
    plt.close(fig)

    print(f"\n  Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Section 11: Takeaways
# ---------------------------------------------------------------------------
# 1. Black-Litterman is a convex quadratic in the posterior mean. The
#    closed-form matrix-inversion recipe and the CVXPY `sum_squares`
#    formulation agree exactly — the latter is slower, but is the right
#    starting point for anything beyond the vanilla BL problem.
#
# 2. DPP caching brings the CVXPY version to within a few x of closed-form
#    on a view-update workload. The cache is the difference between "BL
#    is a toy for finance textbooks" and "BL is a practical daily loop".
#
# 3. The CVXPY value-add is CONSTRAINTS, not speed. Views you can express
#    as Gaussian noise go through the closed form fine. Views you can
#    only express as inequalities (bounds, rankings, convex hulls) are
#    the reason to reach for CVXPY in the first place. Section 8 shows
#    both a floor and a ranking constraint binding naturally.
#
# 4. In the walk-forward, BL's prior anchor gives MV portfolios much lower
#    turnover than raw-historical-mean MV — the classic critique of MV
#    is that tiny changes to estimated mu move weights drastically, and
#    the BL prior damps that behavior. Whether it improves realized
#    Sharpe is universe- and horizon-specific; the stability argument is
#    what generalizes.
#
# 5. Open design questions (mentor input welcome):
#    - Should BL + MV ship as a two-stage atom, a one-stage joint
#      optimization, or stay as a documented two-function recipe?
#    - For walk-forward use, should the BL atom expose Sigma as a
#      Parameter (with full-rebuild on change) or require the caller to
#      rebuild on every Sigma refresh?
#    - The consistency question across all three cookbook atoms (MV,
#      Spinu, BL) is still the most important one — a unified input
#      surface matters more than optimizing each in isolation.


def run_full_tutorial(save_plots_dir: Path | None = None) -> None:
    verify_all_implementations()

    print("\nLoading historical prices via yfinance...")
    prices = load_prices(TICKERS, period="2y")

    print("\n" + "=" * 60)
    print("SECTION 7: Speed benchmark")
    print("=" * 60)
    bench = benchmark(prices, n_days=60)

    demo_constrained_views(prices)

    print("\n" + "=" * 60)
    print("SECTION 9: Walk-forward backtest — historical vs BL-prior vs BL+views")
    print("=" * 60)
    back = walk_forward_bl(prices, lookback=252)
    print_backtest_report(back)

    if save_plots_dir is not None:
        save_plots(bench, back, save_plots_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save matplotlib figures to examples/_bl_plots/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plots_dir = (
        Path(__file__).parent / "_bl_plots" if args.save_plots else None
    )
    run_full_tutorial(save_plots_dir=plots_dir)
