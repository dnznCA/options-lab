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

New to quant finance? [NOTES.md](NOTES.md) walks through the concepts behind
every file here — what an option is, why risk-neutral pricing works, and a
full reference for every variable name used in the code.

**Interactive dashboard:** [dashboard.html](dashboard.html) is a single
self-contained HTML file — no build step, no server, just open it in a
browser — with live sliders over the same math as `black_scholes.py`,
`implied_vol.py`, and `monte_carlo.py` (all reimplemented in vanilla JS so
there's nothing to install). Three tabs: price/Greeks vs. spot with a
tangent-line view of delta, the implied vol solver with its Newton/bisection
trace, and Monte Carlo convergence with sample paths and a terminal-price
histogram.

---

## Project 1 — Black-Scholes pricer + Greeks visualizer

Closed-form European option pricing, hand-rolled (no `scipy`, no external
pricing libraries — the normal CDF is built from `math.erf`).

**Files:**
- `black_scholes.py` — the pricer and six Greeks (delta, gamma, vega, theta,
  rho, vanna), plus a put-call parity self-test.
- `greeks_plots.py` — static PNG plots of price and all six Greeks vs. spot.

For an interactive version — live sliders, a tangent-line view of delta,
all six Greeks updating in real time — see [dashboard.html](dashboard.html)
at the repo root, which covers this project plus Projects 2 and 3 in one
page.

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

### Self-test

Put-call parity: `C - P = S·e^(-qT) - K·e^(-rT)`, checked to machine
precision on every run of `black_scholes.py`.

---

## Project 2 — Implied volatility solver

Newton-Raphson root-finding on `bs_price`, using `vega` (already built in
Project 1) as the derivative. Given a market price, solves for the σ that
reproduces it, falling back to bisection when vega is too flat to trust
(deep ITM/OTM, short-dated) for Newton's step to mean anything.

**Files:** `implied_vol.py` — solver + self-test that round-trips a known σ
through `bs_price` and checks it comes back, across ATM, deep ITM/OTM, and
short-dated cases.

**Run it:** `python implied_vol.py`

---

## Project 3 — Monte Carlo pricer

Prices European options by simulating terminal stock prices under
risk-neutral GBM and averaging discounted payoffs, then checks convergence
against the closed-form Project 1 price.

**Files:**
- `monte_carlo.py` — `simulate_terminal_spot()` draws `S_T` directly from
  its closed-form lognormal distribution (a European payoff only depends on
  the terminal spot, so there's no need to step through time to price one).
  `mc_price()` wraps that into a price + standard error + 95% CI.
  `simulate_gbm_paths()` steps through time instead (same exact GBM
  transition, repeated per step) — only used for the path plot below, not
  for pricing.
- `mc_plots.py` — writes `fig4_mc_paths.png` (a sample of simulated paths
  and the terminal spot distribution) and `fig5_mc_convergence.png` (MC
  price vs. path count converging to the Black-Scholes price, with a
  log-log check that the error shrinks like 1/√N).

**Run it:**
```bash
python monte_carlo.py    # self-test: MC price/CI vs closed-form, at increasing path counts
python mc_plots.py       # writes fig4/fig5 PNGs
```

At 1,000,000 paths, the base case (S=100, K=100, T=1y, r=5%, σ=20%) MC call
price lands within ~0.006 of the closed-form **10.4506**, comfortably inside
its 95% CI.

### Design decisions

- **Antithetic variates by default** (`antithetic=True`): each standard
  normal draw `Z` is paired with `-Z`. Free variance reduction since the
  payoff is roughly monotonic in `Z` — cuts the standard error without
  extra random draws.
- **Terminal spot is sampled directly, not simulated path-by-path.** GBM's
  transition density is closed-form, so `S_T` can be drawn in one shot; a
  European payoff never looks at the path in between. Path stepping
  (`simulate_gbm_paths`) exists only for the visualization, and Project 5's
  hedging loop — which *does* need the path, since it rebalances along the
  way.
- **`mc_price` returns a standard error and 95% CI, not just a point
  estimate.** A Monte Carlo price without an error bar doesn't say whether
  it's trustworthy; every result here reports both.

---

## Project 4 — Volatility smile / surface plotter

Applies Project 2's solver across a real option chain (strikes × maturities)
pulled live from Yahoo Finance, instead of the hand-picked scenarios used
everywhere else in this repo. The math was already built — the actual work
here is data hygiene: which of ~400 raw quotes can the solver even trust?

**Files:**
- `vol_surface.py` — `fetch_option_chain()` pulls spot, dividend yield, and
  every strike/expiry from `yfinance`, keeping only the OTM side of each
  strike (calls above spot, puts below — the liquid side). `clean_quotes()`
  drops quotes with no real bid/ask, spreads too wide to trust, strikes too
  far from the money, or no volume today — and prints a before/after count
  so the filtering isn't a silent black box. `solve_ivs()` runs every
  surviving quote through `implied_vol.py`'s solver.
- `vol_surface_plots.py` — writes `fig6_vol_smile.png` (implied vol vs.
  moneyness, one line per expiry) and `fig7_vol_surface.png` (the same data
  as a 3D surface across moneyness and time).

**Run it:**
```bash
python vol_surface.py          # fetches AAPL, cleans, solves, prints a per-expiry IV summary
python vol_surface_plots.py    # writes fig6/fig7 PNGs
```

A typical run pulls ~385 raw OTM quotes across 6 expiries and keeps around
130 after cleaning — the rest get dropped for having no real market, a
spread too wide to trust, or too little volume to mean anything.

### Design decisions

- **Only the OTM side of each strike is kept**, before any bid/ask filtering
  even runs. OTM options are the actively-traded side of a real chain — deep
  ITM options are thin, and (unlike this repo's synthetic examples) their
  call and put IVs don't agree perfectly in practice, thanks to American-style
  exercise and dividends. Picking one side per strike sidesteps that
  disagreement entirely instead of averaging over it.
- **Mid price `(bid+ask)/2` is used as "the" market price**, never last
  trade. Last trade can be stale — for a thin contract it might reflect a
  fill from hours or days ago, while bid/ask reflect what's tradable right
  now.
- **The risk-free rate is a single flat constant**, not a real yield curve.
  `yfinance` doesn't expose one, and this repo doesn't have a bootstrapping
  pipeline — a real desk would use a proper term structure here, especially
  for the longer-dated expiries where the difference actually matters.
- **Filtering prints what it dropped and why**, matching this repo's habit
  (see Project 2's `implied_vol.py` warnings) of surfacing uncertainty
  instead of quietly discarding it.

## Project 5 — Delta-hedging P&L simulator

The capstone. Sell an option, delta-hedge it with the underlying through
expiry along a simulated path, and track cash/stock like an actual trading
book. If hedging were continuous and the vol you hedged with matched what
actually happened, the hedge would replicate the payoff exactly and P&L would
be zero — that's the whole content of the Black-Scholes replication argument.
This project makes the leak in that argument concrete.

**Files:**
- `delta_hedge.py` — `simulate_hedge()` sells one option at `sigma_hedge`,
  then rebalances a stock position to match `delta()` at every step along a
  path simulated at `sigma_realized` (which can differ), vectorized across
  paths. `gamma_pnl_attribution()` is the theoretical side: a per-step
  formula, derived from the Black-Scholes PDE, that predicts each step's
  hedging P&L from gamma and the gap between `sigma_hedge` and the
  realized move — see the file's docstring for the derivation. The
  self-test checks that summing this formula over a path reproduces the
  same P&L `simulate_hedge()` computes by direct bookkeeping.
- `delta_hedge_plots.py` — writes `fig8_hedge_path.png` (spot, hedge ratio,
  and cumulative P&L over one path), `fig9_hedge_pnl_dist.png` (P&L
  distributions at three realized vols), and `fig10_rebalancing.png` (P&L
  mean/std vs. rebalancing frequency).

**Run it:**
```bash
python delta_hedge.py          # self-test: attribution check, vol-gap study, rebalancing-frequency study
python delta_hedge_plots.py    # writes fig8/fig9/fig10 PNGs
```

At the base case (hedged at σ=20%, daily rebalancing), realized vol below
20% nets the seller a profit, above 20% a loss, and right at 20% the mean
P&L across 4,000 simulated paths is **-0.0007** — indistinguishable from
zero. Rebalancing more often shrinks the spread of outcomes without moving
that mean at all.

### Design decisions

- **Two independent ways to compute the same P&L, checked against each
  other.** `simulate_hedge()` gets its answer from cash/stock bookkeeping —
  the same way a real trading book would. `gamma_pnl_attribution()` gets
  its answer from a closed-form formula. They agree to within discretization
  error on every path tested, which is what actually justifies the "P&L
  comes from gamma and the vol gap" claim, instead of just asserting it.
- **The realized path and the hedging vol are separate parameters on
  purpose.** Every other project in this repo has one `sigma`; this one
  has two (`sigma_hedge`, `sigma_realized`) because the entire lesson is
  in the gap between them. Setting them equal recovers "textbook" hedging.
- **Vectorized across paths, not just steps.** `simulate_hedge()` loops
  over time steps in Python but vectorizes every step across `n_paths` with
  numpy, so running the 4,000-path studies in the self-test takes a fraction
  of a second instead of looping in pure Python per path.

---

## Setup

```bash
git clone <this-repo>
cd options-lab
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: `numpy`, `matplotlib`, `pandas`, `yfinance`
(the last two are only needed for Project 4, which pulls a live option chain).