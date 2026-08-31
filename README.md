# options-lab

A self-directed sequence of quant finance projects, each building on the last.
Working toward a solid grasp of options pricing, volatility, and hedging — not
just running code, but understanding why it's built this way.

**Dependency chain:**

```
1. Black-Scholes pricer + Greeks   ──┬──▶ 2. Implied vol solver ──▶ 4. Vol surface plotter
                                      └──▶ 3. Monte Carlo pricer  ──▶ 5. Delta-hedging P&L sim
```

Everything lives in one repo and one Python environment, since later projects
import directly from earlier ones.

---

## Project 1 — Black-Scholes pricer + Greeks visualizer

Closed-form European option pricing, hand-rolled (no `scipy`, no external
pricing libraries — the normal CDF is built from `math.erf`).

**Files:**
- `black_scholes.py` — the pricer and six Greeks (delta, gamma, vega, theta,
  rho, vanna), plus a put-call parity self-test.
- `greeks_plots.py` — static PNG plots of price and all six Greeks vs. spot.
- `greeks_visualizer.jsx` — an interactive React version of the same math,
  with live sliders and a tangent-line view of delta. Built to make the name
  "visualizer" actually mean something — see [below](#why-two-visualizers).

**Run it:**
```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python black_scholes.py          # self-test: prices, Greeks, parity check
python greeks_plots.py           # writes fig1/fig2/fig3 PNGs
```

Base case (S=100, K=100, T=1y, r=5%, σ=20%, q=0) gives a call price of
**10.4506** and a put-call parity residual of **0.00e+00**.

### Design decisions

A few choices made deliberately, because later projects depend on them:

- **Greeks are returned raw** (per unit of underlying variable — per $1 of
  vol, per year, per 1.00 of rate), not pre-scaled to trader conventions.
  Trader-facing scaling (vega/100, theta/365, rho/100) is applied only at the
  call site. This keeps the core math textbook-comparable and keeps the
  scaling decision visible rather than buried.
- **`bs_price` collapses to discounted intrinsic value** when `T <= 0` or
  `σ <= 0`, rather than erroring. This matters later: Project 5's hedging
  loop marches time forward to expiry, and the pricer needs to behave
  sensibly at that boundary.
- **Dividend yield `q` is plumbed through every function**, defaulting to
  zero. Not needed yet, but Project 4 pulls real option chains, where
  dividends matter — better to have the parameter in place from the start
  than retrofit it.
- **`norm_cdf` returns a plain float for scalar input, vectorizes over
  arrays.** The scalar path keeps Project 2's Newton-Raphson loop clean;
  the array path drives the plots.

### Why two visualizers

`greeks_plots.py` produces static PNGs — useful for the README and for
committing a fixed reference image, but not much of a "visualizer" in the
interactive sense the name implies. `greeks_visualizer.jsx` is the real one:
drag any of the six inputs (spot, strike, time, vol, rate, dividend) and
watch the price curve, the tangent line (= delta), and all six Greek curves
update live, with plain-English interpretation of what's on screen.

### Self-test

Put-call parity: `C - P = S·e^(-qT) - K·e^(-rT)`, checked to machine
precision on every run of `black_scholes.py`.

---

## Project 2 — Implied volatility solver (next)

Newton-Raphson root-finding on `bs_price`, using `vega` (already built in
Project 1) as the derivative. Given a market price, solves for the σ that
reproduces it.

## Project 3 — Monte Carlo pricer

Simulates GBM stock paths, averages discounted payoffs, and checks
convergence against the closed-form Project 1 price.

## Project 4 — Volatility smile / surface plotter

Applies Project 2's solver across a real option chain (strikes × maturities)
pulled from market data. The math is done by this point — the difficulty is
data hygiene: filtering illiquid strikes, handling bid/ask spreads.

## Project 5 — Delta-hedging P&L simulator

The capstone. Combines Project 3's simulated paths with a discrete
delta-hedging loop, tracks stock/cash/option positions over time, and
attributes P&L to gamma and the realized-vs-implied volatility gap.

---

## Setup

```bash
git clone <this-repo>
cd options-lab
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: `numpy`, `matplotlib`.