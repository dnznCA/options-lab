"""
monte_carlo.py — Monte Carlo European option pricer under GBM.

Conventions match black_scholes.py:
  - sigma, r, q are annualized decimals (0.20 = 20%, not 20).
  - T is in years.
  - Returns RAW price (no trader scaling — there's nothing to scale here).

Under risk-neutral GBM, the terminal spot has a closed-form distribution, so
pricing a European option doesn't require stepping through time at all —
one draw per path lands exactly on S_T:

    S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),   Z ~ N(0,1)

That's the whole simulation for pricing. simulate_gbm_paths() below stepped
through time instead (same exact-transition formula, repeated per step) —
that's for visualizing path shapes, not needed for the price itself.

Needs black_scholes.py: bs_price(), for the self-test comparison.
"""

import numpy as np
from black_scholes import bs_price


# --- terminal spot sampling ---------------------------------------------

def simulate_terminal_spot(S, T, r, sigma, q=0.0, n_paths=100_000,
                            antithetic=True, seed=None):
    """
    Draw n_paths samples of S_T under risk-neutral GBM.

    antithetic=True pairs each Z with -Z (antithetic variates): halves the
    effective random draws but cuts variance substantially for free, since
    the payoff is monotonic-ish in Z. n_paths is still the number of prices
    returned (n_paths/2 independent Z's, mirrored).
    """
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)

    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)

    return S * np.exp(drift + diffusion * z)


# --- pricing -----------------------------------------------------------

def mc_price(S, K, T, r, sigma, option_type="call", q=0.0,
             n_paths=100_000, antithetic=True, seed=None):
    """
    Price a European option by averaging discounted payoffs over simulated
    terminal spots.

    Returns dict: price, std_error (of the price estimate), ci_low/ci_high
    (95% CI), n_paths. std_error lets a caller judge whether the estimate
    is trustworthy — a lone point estimate from MC means nothing without it.
    """
    S_T = simulate_terminal_spot(S, T, r, sigma, q, n_paths, antithetic, seed)

    if option_type.lower() in ("c", "call"):
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    disc = np.exp(-r * T)
    discounted = disc * payoff

    price = discounted.mean()
    # sample std of the discounted payoff, / sqrt(n) for the std error of the mean
    std_error = discounted.std(ddof=1) / np.sqrt(n_paths)

    return {
        "price": price,
        "std_error": std_error,
        "ci_low": price - 1.96 * std_error,
        "ci_high": price + 1.96 * std_error,
        "n_paths": n_paths,
    }


# --- full paths, for visualization only (not used in pricing above) ----

def simulate_gbm_paths(S, T, r, sigma, q=0.0, n_paths=40, n_steps=252, seed=None):
    """
    Step through time using the same exact GBM transition at each step
    (not an Euler approximation). Returns (t_grid, paths) where paths has
    shape (n_paths, n_steps+1). Only for plotting path shapes — pricing
    doesn't need intermediate steps since European payoffs only look at S_T.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))

    t_grid = np.linspace(0, T, n_steps + 1)
    return t_grid, paths


# --- self-test: convergence to the closed-form Black-Scholes price -----

if __name__ == "__main__":
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    print(f"Base case: S={S} K={K} T={T} r={r} sigma={sigma} q={q}\n")

    for ot in ("call", "put"):
        bs = bs_price(S, K, T, r, sigma, ot, q)
        print(f"{ot.upper()} — closed-form Black-Scholes price: {bs:.4f}\n")

        print(f"{'n_paths':>10} {'MC price':>10} {'std err':>9} "
              f"{'95% CI':>21} {'abs err':>9}")
        for n in (1_000, 10_000, 100_000, 1_000_000):
            result = mc_price(S, K, T, r, sigma, ot, q, n_paths=n, seed=42)
            err = abs(result["price"] - bs)
            ci = f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
            print(f"{n:>10,} {result['price']:>10.4f} {result['std_error']:>9.4f} "
                  f"{ci:>21} {err:>9.4f}")
        print()
