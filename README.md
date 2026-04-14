# CVXPY Finance Cookbook

Tutorial examples for portfolio optimization with [CVXPY](https://github.com/cvxpy/cvxpy),
focused on the **Disciplined Parametrized Programming (DPP)** patterns that
make repeated re-solves fast, plus the cone-level subtleties that come up
once you move past pure quadratic programs.

## Tutorials

### 1. Mean-variance with transaction costs — the DPP walkthrough

- **Source:** [`examples/portfolio_optimization_dpp.py`](examples/portfolio_optimization_dpp.py)
- **Notebook:** [`examples/portfolio_optimization_dpp.ipynb`](examples/portfolio_optimization_dpp.ipynb)
  (pre-executed, renders inline on GitHub with plots)

Three implementations of the same mean-variance rebalancer:

1. **Naive** — rebuilds `cp.Problem` on every call.
2. **DPP with constant Σ** — `mu` and `w_prev` as `cp.Parameter`, Σ baked in.
3. **DPP with Cholesky Σ** — the subtle one. `quad_form(w, Sigma_param)`
   is **not** DPP-compliant; the fix is to use the Cholesky factor:
   `sum_squares(L.T @ w) == w.T @ Sigma @ w`. Watch the transpose.

Plus: a correctness check across all three, a ~4x speed benchmark, and a
246-day out-of-sample walk-forward on real price data showing the
DPP-cached optimizer beats equal-weight on total return, Sharpe, and max
drawdown.

### 2. Spinu (2013) risk parity — the exponential-cone cousin

- **Source:** [`examples/risk_parity_spinu.py`](examples/risk_parity_spinu.py)
- **Notebook:** [`examples/risk_parity_spinu.ipynb`](examples/risk_parity_spinu.ipynb)

The non-convex ERC condition `w_i · (Σw)_i = const` becomes a **strictly
convex program** under Spinu's change of variables:

```
minimize   0.5 · y.T @ Σ @ y − sum_i b_i · log(y_i)     over y > 0
```

The log barrier puts the canonicalized problem in the **exponential cone**,
so solver choice matters — OSQP is out, CLARABEL (bundled with CVXPY) works.
The same Cholesky trick from MV extends cleanly to DPP-cache Σ, plus there
is a sign-attribute gotcha specific to concave atoms that the tutorial
exhibits explicitly.

Also includes numerical-stability probes up to condition number ~10⁹ and a
walk-forward backtest showing ERC realizes its budget within machine zero
while equal-weight disperses ~3% daily (largest asset: 13.7% of risk vs
the 5% target).

### 3. Black-Litterman with DPP views — Bayesian posterior in CVXPY

- **Source:** [`examples/black_litterman.py`](examples/black_litterman.py)
- **Notebook:** [`examples/black_litterman.ipynb`](examples/black_litterman.ipynb)

Black-Litterman blends market-equilibrium implied returns with investor
views. The posterior has a clean closed-form solution, so at first glance
CVXPY adds nothing. The tutorial argues the CVXPY value-add in three ways:

1. BL is equivalent to a convex quadratic; all three implementations
   (closed-form, naive CVXPY, DPP-cached CVXPY) agree to floating-point
   precision.
2. **The CVXPY value-add is constraints, not speed** — inequality views
   on the posterior (`mu[AAPL] >= 0`, `mu[AAPL] >= mu[MSFT]`) are where
   the closed form breaks. Section 8 shows both binding non-trivially.
3. DPP with `P`, `Q` as `cp.Parameter` gives ~5x speedup over naive
   rebuild — the inner loop of a daily view-update pipeline.

Walk-forward result: BL-prior MV has **~7x lower daily turnover** than
historical-mean MV (1.7% vs 11.9%), at a comparable Sharpe. The classic
"tiny changes in μ flip the optimal weights" critique of MV is
empirically corroborated and empirically solved by the BL prior.

## Why this repo

I am preparing for GSoC 2026 with CVXPY. The scope I am discussing with
the mentors is **cookbook-style examples + DPP documentation contributions**,
not new API surface. All three tutorials above are first-draft deliverables
for that scope. A contribution to the performance-tips docs is planned
next.

## Running

```bash
pip install -r requirements.txt

# Runnable scripts
python examples/portfolio_optimization_dpp.py
python examples/risk_parity_spinu.py
python examples/black_litterman.py

# With matplotlib figures written to examples/_{dpp,spinu,bl}_plots/
python examples/portfolio_optimization_dpp.py --save-plots
python examples/risk_parity_spinu.py --save-plots
python examples/black_litterman.py --save-plots
```

Tested with Python 3.12+, CVXPY 1.5+.

## Open questions for mentors

The tutorials are designed to raise as much as they teach. Spots where I
would genuinely want mentor input before expanding further:

- **Atom vs recipe** — should patterns like `SpinuRebalancerCholesky` or
  `BLRebalancer` ship as first-class atoms (`cp.finance.*`?), or stay as
  documented recipes? A consistent answer across all three cookbook
  examples is probably more valuable than optimizing each in isolation.
- **Input surface** — if they become atoms, does the caller pass Σ (and
  we do the Cholesky internally), or L (so they control the factorization
  and avoid re-factoring when Σ hasn't changed)?
- **Failure modes** — should the atoms raise on non-PD Σ, or fall back
  gracefully (diagonal jitter, nearest-PD projection)?
- **Interaction with [PR #3142](https://github.com/cvxpy/cvxpy/pull/3142)** —
  the tutorials currently teach `quad_form(x, Sigma_param)` as the
  canonical DPP gotcha, with Cholesky as the workaround. Once #3142
  lands, the workaround becomes optional (for QP solvers in the
  objective). I'd want to update the tutorials' framing to reflect the
  new landscape — Cholesky stays relevant for the EXP-cone case
  (Spinu), for constraint-side `quad_form`, and for factor-model
  structures not currently supported by param-affine coefficient
  extraction.

## License

MIT — see [LICENSE](LICENSE).
