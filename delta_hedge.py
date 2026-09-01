"""
delta_hedge.py — discrete delta-hedging simulator. Project 5, the capstone.

Sell an option at t=0, delta-hedge it with the underlying through expiry
along a simulated GBM path, and track cash/stock like an actual trading book.
If hedging were continuous and the vol you hedged with matched what actually
happened, the hedge would replicate the option's payoff exactly and P&L would
be zero — that's the whole content of the Black-Scholes replication argument.
This file makes the leak in that argument concrete: discrete (not continuous)
rebalancing, and any gap between the vol you hedged with and the vol that
actually realized, both show up directly as P&L.

Conventions match the rest of the repo: T in years, r/q/sigma as decimals.
Needs black_scholes.py (bs_price, delta, gamma) from this repo.

--- where the P&L attribution formula comes from ---

Let V(S,t) be the Black-Scholes value using sigma_hedge, and Pi_t = cash_t +
delta_t*S_t the hedge portfolio (self-financing, rebalanced continuously).
Pi_0 = V_0 by construction. Ito's lemma on V, plus the Black-Scholes PDE
(theta + (r-q)S*delta + 0.5*sigma_hedge^2*S^2*gamma = r*V), gives the
instantaneous hedge error:

    d(Pi - V) = -q*S*delta*dt + 0.5*gamma*S^2*(sigma_hedge^2*dt - (dS/S)^2)

With q=0, that's just 0.5*gamma*S^2*(sigma_hedge^2*dt - realized_return^2)
per step: positive when the stock moved less than sigma_hedge priced for
that instant, negative when it moved more. Since Pi_0=V_0 and V_T=payoff,
summing this over every step should reconstruct simulate_hedge()'s final
P&L (final cash+stock, minus the payoff owed) — see the self-test below,
which checks the two agree.
"""

import numpy as np
from black_scholes import bs_price, delta as bs_delta, gamma as bs_gamma
from monte_carlo import simulate_gbm_paths


def simulate_hedge(S0, K, T, r, sigma_hedge, sigma_realized, option_type="call",
                    q=0.0, n_steps=252, n_paths=1, seed=None, track_history=False):
    """
    Sell one option at t=0 (priced at sigma_hedge), delta-hedge it through
    expiry along a GBM path simulated at sigma_realized, and return the final
    hedging P&L: (cash + stock) at expiry, minus what's owed on the option.

    sigma_hedge is what you believe/quote — it prices the option and drives
    every delta you hedge with. sigma_realized is what the path actually
    does. Equal vols + frequent rebalancing -> P&L clusters near zero.
    A vol gap, or infrequent rebalancing, pushes it away from zero.

    Vectorized across n_paths independent simulations. Set track_history=True
    (sensible only for a handful of paths) to get the full time series back
    instead of just the final P&L — used for plotting one illustrative path.
    """
    dt = T / n_steps
    is_call = option_type.lower() in ("c", "call")

    # the realized path is exactly Project 3's simulator — reused rather than
    # re-deriving the same GBM stepping logic a second time in this file
    _, S_paths = simulate_gbm_paths(S0, T, r, sigma_realized, q, n_paths, n_steps, seed)

    premium = bs_price(S0, K, T, r, sigma_hedge, option_type, q)
    shares = bs_delta(S0, K, T, r, sigma_hedge, option_type, q) * np.ones(n_paths)
    cash = premium - shares * S_paths[:, 0]

    if track_history:
        delta_hist = np.empty((n_steps + 1, n_paths)); delta_hist[0] = shares
        portfolio_hist = np.empty((n_steps + 1, n_paths)); portfolio_hist[0] = cash + shares * S_paths[:, 0]

    for i in range(1, n_steps + 1):
        S_prev, S_now = S_paths[:, i - 1], S_paths[:, i]
        cash = cash * np.exp(r * dt) + shares * S_prev * q * dt  # interest, plus dividends on shares held over the interval

        if i < n_steps:  # no rebalance needed right at expiry — nothing left to hedge
            t_remaining = T - i * dt
            new_delta = bs_delta(S_now, K, t_remaining, r, sigma_hedge, option_type, q)
            cash -= (new_delta - shares) * S_now
            shares = new_delta

        if track_history:
            delta_hist[i] = shares; portfolio_hist[i] = cash + shares * S_now

    S = S_paths[:, -1]

    payoff = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    hedge_pnl = (cash + shares * S) - payoff

    result = {"hedge_pnl": hedge_pnl, "premium": premium, "final_spot": S, "payoff": payoff}
    if track_history:
        result.update(S=S_paths.T, delta=delta_hist, portfolio=portfolio_hist)
    return result


def gamma_pnl_attribution(S_path, K, T, r, sigma_hedge, option_type="call", q=0.0):
    """
    Per-step theoretical hedging P&L along one realized path S_path (length
    n_steps+1), from the derivation above: 0.5*gamma*S^2*(sigma_hedge^2*dt -
    realized_return^2) each step. Summing this should approximate the same
    total simulate_hedge() computes by direct cash/stock bookkeeping —
    that agreement is what "P&L comes from gamma and the vol gap" means
    concretely, rather than as an assertion.
    """
    n_steps = len(S_path) - 1
    dt = T / n_steps
    step_pnl = np.zeros(n_steps)
    for i in range(n_steps):
        t_remaining = T - i * dt
        if t_remaining <= 0:
            break
        g = bs_gamma(S_path[i], K, t_remaining, r, sigma_hedge, q)
        realized_return = np.log(S_path[i + 1] / S_path[i])
        step_pnl[i] = 0.5 * g * S_path[i] ** 2 * (sigma_hedge ** 2 * dt - realized_return ** 2)
    return step_pnl


if __name__ == "__main__":
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    SIGMA_HEDGE = 0.20

    # --- sanity check: does the gamma/vol-gap formula actually reconstruct
    #     the simulated hedge P&L for one path? ---
    one = simulate_hedge(S0, K, T, r, SIGMA_HEDGE, sigma_realized=0.20, n_steps=252,
                          n_paths=1, seed=7, track_history=True)
    S_path = one["S"][:, 0]
    attributed = gamma_pnl_attribution(S_path, K, T, r, SIGMA_HEDGE).sum()
    simulated = one["hedge_pnl"][0]
    print(f"Single path check (sigma_hedge=sigma_realized={SIGMA_HEDGE}):")
    print(f"  simulated hedge P&L (cash+stock-payoff): {simulated:+.4f}")
    print(f"  sum of per-step gamma/vol-gap attribution: {attributed:+.4f}")
    print(f"  difference (discretization error): {simulated - attributed:+.4f}\n")

    # --- does realized vol vs. hedging vol drive the sign and size of P&L? ---
    print(f"Vol-gap study — hedged at sigma={SIGMA_HEDGE}, n_steps=252, 4000 paths each:")
    print(f"{'realized sigma':>15} {'mean P&L':>10} {'std P&L':>10}")
    for sigma_real in (0.10, 0.15, 0.20, 0.25, 0.30):
        res = simulate_hedge(S0, K, T, r, SIGMA_HEDGE, sigma_real, n_steps=252,
                              n_paths=4000, seed=42)
        pnl = res["hedge_pnl"]
        print(f"{sigma_real:>15.2f} {pnl.mean():>10.4f} {pnl.std():>10.4f}")

    # --- does rebalancing more often shrink the noise (not the bias)? ---
    print(f"\nRebalancing-frequency study — hedged at {SIGMA_HEDGE}, realized at 0.30, 4000 paths each:")
    print(f"{'n_steps':>8} {'mean P&L':>10} {'std P&L':>10}")
    for n_steps in (12, 52, 252, 1008):
        res = simulate_hedge(S0, K, T, r, SIGMA_HEDGE, 0.30, n_steps=n_steps,
                              n_paths=4000, seed=42)
        pnl = res["hedge_pnl"]
        print(f"{n_steps:>8} {pnl.mean():>10.4f} {pnl.std():>10.4f}")
