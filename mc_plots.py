"""
mc_plots.py — visualize the Monte Carlo pricer from monte_carlo.py.

Running `python mc_plots.py` writes:
  fig4_mc_paths.png         a handful of simulated GBM paths, terminal payoffs
  fig5_mc_convergence.png   MC price vs n_paths converging to the closed-form
                             Black-Scholes price, with shrinking 95% CI band
"""

import numpy as np
import matplotlib.pyplot as plt
import black_scholes as bs
import monte_carlo as mc

# base case — same scenario used throughout the repo
S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0

# ---------------------------------------------------------------- figure 4
# a sample of simulated paths, colored by whether they finish ITM for a call
t_grid, paths = mc.simulate_gbm_paths(S, T, r, sigma, q, n_paths=60, n_steps=252, seed=7)
finishes_itm = paths[:, -1] > K

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})
for path, itm in zip(paths, finishes_itm):
    ax1.plot(t_grid, path, lw=0.8, alpha=0.6, color="C0" if itm else "C3")
ax1.axhline(K, color="k", alpha=0.3, lw=1, ls="--", label=f"strike K={K:.0f}")
ax1.set_title(f"60 simulated GBM paths  (S0={S:.0f}, T={T:.0f}y, r={r:.0%}, σ={sigma:.0%})")
ax1.set_xlabel("time (years)"); ax1.set_ylabel("spot")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# terminal spot distribution, big sample, with the call payoff region shaded
S_T_big = mc.simulate_terminal_spot(S, T, r, sigma, q, n_paths=50_000, seed=7)
ax2.hist(S_T_big, bins=80, color="C0", alpha=0.7, orientation="horizontal")
ax2.axhline(K, color="k", alpha=0.3, lw=1, ls="--")
ax2.set_title("terminal spot distribution")
ax2.set_xlabel("count"); ax2.grid(alpha=0.3)
fig4.tight_layout(); fig4.savefig("fig4_mc_paths.png", dpi=130)

# ---------------------------------------------------------------- figure 5
# MC price estimate vs path count, converging to the closed-form BS price
bs_call = bs.bs_price(S, K, T, r, sigma, "call", q)
path_counts = np.unique(np.logspace(2, 6, 25).astype(int))

prices, ci_lo, ci_hi = [], [], []
for n in path_counts:
    result = mc.mc_price(S, K, T, r, sigma, "call", q, n_paths=int(n), seed=42)
    prices.append(result["price"])
    ci_lo.append(result["ci_low"])
    ci_hi.append(result["ci_high"])

fig5, (b1, b2) = plt.subplots(1, 2, figsize=(13, 5))

b1.plot(path_counts, prices, lw=1.5, color="C0", label="MC price")
b1.fill_between(path_counts, ci_lo, ci_hi, color="C0", alpha=0.2, label="95% CI")
b1.axhline(bs_call, color="k", lw=1, ls="--", label=f"Black-Scholes = {bs_call:.4f}")
b1.set_xscale("log")
b1.set_title("MC call price vs path count")
b1.set_xlabel("n_paths (log scale)"); b1.set_ylabel("price")
b1.legend(fontsize=8); b1.grid(alpha=0.3)

# error should shrink like 1/sqrt(N) — check that on a log-log plot
abs_err = np.abs(np.array(prices) - bs_call)
b2.loglog(path_counts, abs_err, "o-", lw=1.2, ms=4, color="C2", label="|MC price - BS price|")
ref = abs_err[0] * np.sqrt(path_counts[0] / path_counts)
b2.loglog(path_counts, ref, "--", color="gray", lw=1, label=r"$\propto 1/\sqrt{N}$ reference")
b2.set_title("convergence rate")
b2.set_xlabel("n_paths (log scale)"); b2.set_ylabel("absolute error (log scale)")
b2.legend(fontsize=8); b2.grid(alpha=0.3, which="both")

fig5.suptitle(f"Monte Carlo convergence to closed-form price  (call, K={K:.0f})", y=1.02)
fig5.tight_layout(); fig5.savefig("fig5_mc_convergence.png", dpi=130, bbox_inches="tight")

print("saved fig4_mc_paths.png, fig5_mc_convergence.png")
