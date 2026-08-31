"""
greeks_plots.py — visualize price and the six Greeks from black_scholes.py.

No new math here — just calls the pricing/Greek functions at a range of
spot prices and plots the shapes. See black_scholes.py for what everything
means.

Running `python greeks_plots.py` writes:
  fig1_price.png          call/put price vs spot, with intrinsic value overlaid
  fig2_greeks.png          all six Greeks vs spot, 2x3 grid
  fig3_term_structure.png  gamma/vega/theta across a few different maturities
"""

import numpy as np
import matplotlib.pyplot as plt
import black_scholes as bs

# base case scenario every chart is drawn against
K, r, sigma, q = 100.0, 0.05, 0.20, 0.0
T = 1.0
S = np.linspace(40, 160, 400)   # spot range to sweep across

# ---------------------------------------------------------------- figure 1
fig1, ax = plt.subplots(figsize=(8, 5))
call = bs.bs_price(S, K, T, r, sigma, "call", q)
put = bs.bs_price(S, K, T, r, sigma, "put", q)
ax.plot(S, call, label="call", lw=2)
ax.plot(S, put, label="put", lw=2)
# intrinsic = payoff if it expired right now — gap vs actual price is time value
ax.plot(S, np.maximum(S - K, 0), "--", color="gray", lw=1, label="call intrinsic")
ax.plot(S, np.maximum(K - S, 0), ":", color="gray", lw=1, label="put intrinsic")
ax.axvline(K, color="k", alpha=0.2, lw=1)
ax.set_title(f"Option price vs spot  (K={K:.0f}, T={T:.0f}y, r={r:.0%}, σ={sigma:.0%})")
ax.set_xlabel("spot S"); ax.set_ylabel("price")
ax.legend(); ax.grid(alpha=0.3)
fig1.tight_layout(); fig1.savefig("fig1_price.png", dpi=130)

# ---------------------------------------------------------------- figure 2
# (name, fn, split) — split=True means call/put differ and need two lines
greek_specs = [
    ("delta", lambda ot: bs.delta(S, K, T, r, sigma, ot, q), True),
    ("gamma", lambda ot: bs.gamma(S, K, T, r, sigma, q),     False),
    ("vega",  lambda ot: bs.vega(S, K, T, r, sigma, q) / 100, False),   # per vol-point
    ("theta", lambda ot: bs.theta(S, K, T, r, sigma, ot, q) / 365, True),  # per day
    ("rho",   lambda ot: bs.rho(S, K, T, r, sigma, ot, q) / 100,  True),   # per 1%
    ("vanna", lambda ot: bs.vanna(S, K, T, r, sigma, q),     False),
]

fig2, axes = plt.subplots(2, 3, figsize=(13, 7))
for axg, (name, fn, split) in zip(axes.flat, greek_specs):
    if split:
        axg.plot(S, fn("call"), lw=2, label="call")
        axg.plot(S, fn("put"), lw=2, label="put")
        axg.legend(fontsize=8)
    else:
        axg.plot(S, fn("call"), lw=2, color="C2")
    axg.axvline(K, color="k", alpha=0.2, lw=1)   # strike
    axg.axhline(0, color="k", alpha=0.2, lw=1)   # zero line
    axg.set_title(name); axg.set_xlabel("spot S"); axg.grid(alpha=0.3)
fig2.suptitle("The six Greeks vs spot  "
              "(vega /100, theta /365, rho /100 — trader scaling)", y=1.02)
fig2.tight_layout(); fig2.savefig("fig2_greeks.png", dpi=130, bbox_inches="tight")

# ---------------------------------------------------------------- figure 3
# term structure: same Greeks, overlaid across a few different maturities
maturities = [0.05, 0.25, 1.0, 2.0]   # ~18d, 3mo, 1y, 2y
fig3, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.2))
for Tm in maturities:
    lbl = f"T={Tm:g}y"
    a1.plot(S, bs.gamma(S, K, Tm, r, sigma, q), label=lbl)
    a2.plot(S, bs.vega(S, K, Tm, r, sigma, q) / 100, label=lbl)
    a3.plot(S, bs.theta(S, K, Tm, r, sigma, "call", q) / 365, label=lbl)
for axx, ttl in zip((a1, a2, a3), ("gamma", "vega (/vol-pt)", "theta (/day, call)")):
    axx.axvline(K, color="k", alpha=0.2, lw=1)
    axx.set_title(ttl); axx.set_xlabel("spot S"); axx.grid(alpha=0.3); axx.legend(fontsize=8)
fig3.suptitle("Term structure: short-dated options are spikier (gamma/theta), "
              "long-dated carry more vega", y=1.03)
fig3.tight_layout(); fig3.savefig("fig3_term_structure.png", dpi=130, bbox_inches="tight")

print("saved fig1_price.png, fig2_greeks.png, fig3_term_structure.png")
