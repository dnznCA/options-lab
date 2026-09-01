"""
delta_hedge_plots.py — visualize the delta-hedging simulator from delta_hedge.py.

Running `python delta_hedge_plots.py` writes:
  fig8_hedge_path.png       one illustrative path: spot, hedge ratio, and
                             cumulative P&L over time
  fig9_hedge_pnl_dist.png   final P&L distributions across realized vols
  fig10_rebalancing.png     P&L mean/std vs. rebalancing frequency
"""

import numpy as np
import matplotlib.pyplot as plt

from delta_hedge import simulate_hedge

S0, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
SIGMA_HEDGE = 0.20

# ---------------------------------------------------------------- figure 8
# one path: spot, the hedge ratio tracking it, and cumulative hedge P&L —
# a "day in the life" of a delta-hedged book, realized vol matching hedge vol
one = simulate_hedge(S0, K, T, r, SIGMA_HEDGE, sigma_realized=SIGMA_HEDGE, n_steps=252,
                      n_paths=1, seed=3, track_history=True)
t_grid = np.linspace(0, T, 253)
S_path = one["S"][:, 0]
delta_path = one["delta"][:, 0]
pnl_path = one["portfolio"][:, 0] - one["premium"]  # running P&L vs. the option's book value at inception

fig8, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axes[0].plot(t_grid, S_path, color="C0")
axes[0].axhline(K, color="k", ls="--", alpha=0.3, lw=1)
axes[0].set_ylabel("spot"); axes[0].set_title("Delta-hedging one simulated path (realized vol = hedging vol)")
axes[0].grid(alpha=0.3)

axes[1].plot(t_grid, delta_path, color="C2")
axes[1].set_ylabel("hedge ratio (Δ)")
axes[1].grid(alpha=0.3)

axes[2].plot(t_grid, pnl_path, color="C3")
axes[2].axhline(0, color="k", lw=1, alpha=0.3)
axes[2].set_ylabel("cumulative hedge P&L"); axes[2].set_xlabel("time (years)")
axes[2].grid(alpha=0.3)

fig8.tight_layout()
fig8.savefig("fig8_hedge_path.png", dpi=130)

# ---------------------------------------------------------------- figure 9
# distribution of final P&L across a few realized vols — the vol seller's
# classic pattern: profit when realized < hedged, loss when realized > hedged
fig9, ax9 = plt.subplots(figsize=(8.5, 5.5))
scenarios = [0.10, 0.20, 0.30]
colors = ["C2", "C0", "C3"]
for sigma_real, color in zip(scenarios, colors):
    res = simulate_hedge(S0, K, T, r, SIGMA_HEDGE, sigma_real, n_steps=252,
                          n_paths=5000, seed=42)
    pnl = res["hedge_pnl"]
    ax9.hist(pnl, bins=60, alpha=0.55, color=color,
             label=f"realized σ={sigma_real:.2f}  (mean={pnl.mean():+.2f})")
ax9.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
ax9.set_title(f"Hedge P&L distribution — hedged at σ={SIGMA_HEDGE:.2f}, daily rebalancing, 5000 paths")
ax9.set_xlabel("final hedge P&L"); ax9.set_ylabel("count")
ax9.legend(fontsize=9)
ax9.grid(alpha=0.3)
fig9.tight_layout()
fig9.savefig("fig9_hedge_pnl_dist.png", dpi=130)

# --------------------------------------------------------------- figure 10
# rebalancing frequency: variance shrinks, the vol-gap bias doesn't
n_steps_grid = [12, 26, 52, 104, 252, 504, 1008]
means, stds = [], []
for n_steps in n_steps_grid:
    res = simulate_hedge(S0, K, T, r, SIGMA_HEDGE, sigma_realized=0.30, n_steps=n_steps,
                          n_paths=5000, seed=42)
    pnl = res["hedge_pnl"]
    means.append(pnl.mean()); stds.append(pnl.std())
means, stds = np.array(means), np.array(stds)

fig10, ax10 = plt.subplots(figsize=(8, 5.5))
ax10.plot(n_steps_grid, means, "o-", color="C3", label="mean P&L (the vol-gap bias)")
ax10.fill_between(n_steps_grid, means - stds, means + stds, color="C0", alpha=0.2,
                    label="±1 std (discrete-hedging noise)")
ax10.set_xscale("log")
ax10.set_title("Rebalancing frequency: noise shrinks, the vol-gap bias doesn't\n"
                f"(hedged at σ={SIGMA_HEDGE:.2f}, realized at 0.30)")
ax10.set_xlabel("rebalances per year (log scale)"); ax10.set_ylabel("hedge P&L")
ax10.legend(fontsize=9)
ax10.grid(alpha=0.3, which="both")
fig10.tight_layout()
fig10.savefig("fig10_rebalancing.png", dpi=130)

print("saved fig8_hedge_path.png, fig9_hedge_pnl_dist.png, fig10_rebalancing.png")
