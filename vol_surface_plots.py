"""
vol_surface_plots.py — visualize the smile/surface from vol_surface.py.

Running `python vol_surface_plots.py` writes:
  fig6_vol_smile.png     implied vol vs moneyness, one line per expiry
  fig7_vol_surface.png   the full surface: implied vol vs moneyness and time
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection
import numpy as np

from vol_surface import fetch_option_chain, clean_quotes, solve_ivs, RISK_FREE_RATE

TICKER = "AAPL"

print(f"Fetching option chain for {TICKER}...")
spot, q, raw = fetch_option_chain(TICKER, max_expiries=6)
cleaned = clean_quotes(raw, spot)
surface = solve_ivs(cleaned, spot, RISK_FREE_RATE, q)

# ---------------------------------------------------------------- figure 6
# vol smile: IV vs moneyness, one colored line per expiry — the classic
# "smile" or "skew" shape that a constant-sigma Black-Scholes can't produce
fig6, ax = plt.subplots(figsize=(8, 5.5))
expiries = list(surface["expiry"].unique())
colors = plt.cm.viridis(np.linspace(0, 0.85, len(expiries)))

for expiry, color in zip(expiries, colors):
    grp = surface[surface["expiry"] == expiry].sort_values("moneyness")
    T = grp["T"].iloc[0]
    ax.plot(grp["moneyness"], grp["iv"], "o-", color=color, ms=4, lw=1.3,
             label=f"{expiry}  (T={T:.3f}y)")

ax.axvline(1.0, color="k", alpha=0.3, lw=1, ls="--", label="ATM (K=S)")
ax.set_title(f"{TICKER} implied vol smile — spot={spot:.2f}, {len(surface)} quotes across {len(expiries)} expiries")
ax.set_xlabel("moneyness (K / S)")
ax.set_ylabel("implied volatility")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig6.tight_layout()
fig6.savefig("fig6_vol_smile.png", dpi=130)

# ---------------------------------------------------------------- figure 7
# vol surface: same data, now across moneyness AND time. Strikes differ by
# expiry so the (moneyness, T) points are scattered, not a regular grid —
# plot_trisurf handles that natively via Delaunay triangulation, no
# interpolation library needed.
fig7 = plt.figure(figsize=(9, 7))
ax3d = fig7.add_subplot(111, projection="3d")

surf = ax3d.plot_trisurf(surface["moneyness"], surface["T"], surface["iv"],
                          cmap="viridis", edgecolor="none", alpha=0.9)
ax3d.scatter(surface["moneyness"], surface["T"], surface["iv"],
             color="k", s=6, alpha=0.4)
ax3d.set_xlabel("moneyness (K / S)")
ax3d.set_ylabel("T (years)")
ax3d.set_zlabel("implied volatility")
ax3d.set_title(f"{TICKER} implied volatility surface")
fig7.colorbar(surf, shrink=0.6, aspect=12, label="implied vol")
fig7.tight_layout()
fig7.savefig("fig7_vol_surface.png", dpi=130)

print("saved fig6_vol_smile.png, fig7_vol_surface.png")
