# Notes — the quant basics behind this repo

Written for someone who knows how to code but is new to quant finance.
Goal: after reading this, you should be able to open any file in this repo
and understand not just *what* the code does, but *why* it's built that way.

---

## 1. What even is an option

A **call option** gives you the right (not obligation) to buy a stock at a
fixed price (the **strike**, `K`) on/by some future date (**expiry**, `T`).
A **put** is the same but the right to *sell*.

At expiry, the payoff is mechanical:

- Call: `max(S_T - K, 0)` — worth the difference if the stock (`S_T`) ended
  above the strike, worthless otherwise. Nobody exercises a call to buy at
  $100 when the market price is $90.
- Put: `max(K - S_T, 0)` — the mirror image.

That `max(..., 0)` is the entire reason options math is more interesting
than "average outcome" math: the payoff is **kinked**, not linear. That
kink is what creates everything downstream — why options have curvature
(gamma), why volatility has value (vega), why the whole field exists.

The hard question is: **before expiry**, what should this option cost
*today*? That's what every file in this repo answers, in increasingly
sophisticated ways.

---

## 2. The key trick: risk-neutral pricing

Naive idea: simulate the stock's future, average the discounted payoff,
done. But whose probabilities do you use? The stock's *real* expected
return depends on the market's risk appetite, which is unobservable and
argued about forever.

The insight that makes options pricing tractable (Black-Scholes-Merton,
1973) is that you don't need the real probabilities at all. If you can
**replicate** the option's payoff exactly using a dynamically-adjusted
position in the stock and cash (this is what delta-hedging is — see §6 and
Project 5), then the option's price is just whatever that replicating
portfolio costs, by no-arbitrage. That replication argument turns out to be
equivalent to pricing under a fictional "**risk-neutral**" world where every
asset drifts at the risk-free rate `r`, regardless of how investors actually
feel about risk.

This is why every formula and simulation in this repo drifts the stock at
`r` (or `r - q` with dividends), not at some estimated "real" expected
return. It's not that stocks actually grow at the risk-free rate — it's
that pricing this way is exactly equivalent to a no-arbitrage hedge, and
the real-world drift cancels out of the replication argument entirely.

---

## 3. The Black-Scholes model — [black_scholes.py](black_scholes.py)

**The assumption:** the stock follows **Geometric Brownian Motion (GBM)**
under the risk-neutral measure:

```
dS = (r - q) S dt + σ S dW
```

Plain English: over an instant `dt`, the stock drifts up at rate `r - q`
(risk-free rate minus dividend yield) and jiggles randomly with size
proportional to `σ` (volatility) and the current price `S`. `dW` is
Brownian motion — the continuous-time version of "flip a coin every
instant."

Because this SDE has a known closed-form solution, you can integrate the
discounted expected payoff analytically and get a formula instead of a
simulation. That formula is `bs_price()`:

```python
bs_price(S, K, T, r, sigma, option_type="call", q=0.0)
```

**Where the formula actually comes from, intuitively:** the call price
`disc_S * N(d1) - disc_K * N(d2)` is "PV of the stock you'd receive if
exercised" minus "PV of the strike you'd pay if exercised," each weighted
by a *risk-neutral probability of finishing in the money* (`N(d1)` and
`N(d2)` — not quite the same probability, but close enough for now).
`N(·)` here is the standard normal CDF, hand-rolled in this repo via
`norm_cdf()` from `math.erf` — no scipy dependency, so you can see every
piece of the machinery.

**Why this repo hand-rolls `norm_cdf`/`norm_pdf`:** it's a deliberate
choice to keep the whole pricing stack visible and dependency-free,
[black_scholes.py:20-37](black_scholes.py#L20-L37). `norm_cdf` returns a
plain float for scalar input but vectorizes over numpy arrays — the scalar
path keeps Project 2's root-finding loop clean (see §4), the array path
is what lets `greeks_plots.py` sweep a whole range of spot prices in one
call.

**The Greeks** are just partial derivatives of the price formula — "if I
nudge one input, how much does the price move?" All six live in this file:

| Greek | ∂Price/∂ | Intuition |
|---|---|---|
| delta | spot | how much the option price moves per $1 of stock move — also ≈ the hedge ratio |
| gamma | delta (2nd deriv of spot) | how fast delta itself changes — curvature; peaks ATM near expiry |
| vega | volatility | how much the price moves per 1% change in implied vol |
| theta | time | time decay — how much value bleeds away per day, all else equal |
| rho | interest rate | sensitivity to rates — usually the least important Greek for short-dated equity options |
| vanna, vomma | 2nd-order (spot×vol, vol×vol) | how the other Greeks *themselves* shift as vol changes — bonus, not core |

**Design choice worth noting:** Greeks are returned **raw** (per $1 of
vol, per year, per 1.00 of rate) rather than pre-scaled to trader
convention (vega/100 = per vol-*point*, theta/365 = per *day*, rho/100 =
per 1%). The scaling is applied only at the call site
([greeks_plots.py:43-45](greeks_plots.py#L43-L45)), so the core formulas
stay textbook-comparable and the scaling decision stays visible instead of
silently baked in.

**Self-test:** put-call parity, `C - P = S·e^(-qT) - K·e^(-rT)`. This is a
pure no-arbitrage identity — true regardless of which pricing model you
use — so checking it to machine precision on every run is a cheap sanity
check that the formulas weren't fat-fingered.

**Visualized in:** `greeks_plots.py` — sweeps spot price across a range and
plots price + all six Greeks (`fig1`, `fig2`), plus how gamma/vega/theta
change across different maturities (`fig3`) — short-dated options are
"spikier" (more gamma/theta risk concentrated near expiry), long-dated ones
carry more vega.

---

## 4. Implied volatility — [implied_vol.py](implied_vol.py)

Flip the question around. `bs_price()` takes `σ` as an input and gives you
a price. But in the real world, you *observe* a price (the market quote)
and don't know what `σ` the market is implicitly using. **Implied
volatility** is that reverse-engineered `σ`: the number you'd have to feed
`bs_price()` to reproduce the quote you're looking at.

This is a **root-finding problem**: find `σ` such that
`bs_price(S, K, T, r, σ, ...) - market_price = 0`. Two methods are used
here, chained together:

- **Newton-Raphson** ([implied_vol.py:76-96](implied_vol.py#L76-L96)):
  fast, and it doesn't need a new derivative — `vega` **is** literally
  `∂Price/∂σ`, already built in Project 1. Each step is
  `σ_new = σ - (price_error) / vega`, i.e. "how far off is the price, and
  how steeply does price move with vol here — take a step of that size."
- **Bisection fallback** ([implied_vol.py:98-138](implied_vol.py#L98-L138)):
  Newton's step size depends on dividing by vega, which breaks down when
  vega is near zero — deep ITM/OTM or very short-dated options, where the
  price is nearly flat in `σ` (lots of different vols would produce about
  the same price). Bisection is slower but can't diverge, so it's the
  safety net when Newton's slope isn't trustworthy.

**Why this matters conceptually:** implied vol is the market's own forecast
of future volatility, extracted from what people are actually willing to
pay — not a historical average. It's the single number traders actually
quote and trade (options desks quote in vol, not dollars). Project 4 will
compute this across many strikes/maturities at once to build the **vol
smile/surface** — the pattern of implied vols isn't flat across strikes in
real markets, which is itself evidence that the Black-Scholes constant-vol
assumption is an approximation, not reality.

**Self-test:** round-trip — pick a known `σ_true`, price it with
`bs_price`, feed that price back into `implied_vol()`, and check you get
`σ_true` back out. Run across ATM, deep ITM/OTM, and short-dated cases to
exercise both the Newton and bisection paths.

---

## 5. Monte Carlo pricing — [monte_carlo.py](monte_carlo.py)

Black-Scholes gives a closed-form answer *because* GBM has a solvable
integral for a simple payoff like `max(S_T - K, 0)`. Most real-world
payoffs (path-dependent options, exotic structures, the P&L of a hedging
strategy) don't have a clean closed form. Monte Carlo is the general-purpose
fallback: simulate the random process many times, average the outcome.

**The core idea, concretely:**

1. Draw a large number of samples of the terminal stock price `S_T` under
   the same risk-neutral GBM as Black-Scholes assumes.
2. Compute each payoff, e.g. `max(S_T - K, 0)`.
3. Average them, then discount back to today at `e^{-rT}`.
4. By the law of large numbers, this average converges to the true
   risk-neutral expected discounted payoff — which is exactly what
   Black-Scholes computes analytically. Same answer, two roads.

**Why `simulate_terminal_spot()` doesn't step through time**
([monte_carlo.py:24-42](monte_carlo.py#L24-L42)): GBM has a closed-form
transition — you can draw `S_T` directly in one shot from

```
S_T = S0 * exp((r - q - 0.5σ²)T + σ√T · Z),   Z ~ N(0,1)
```

instead of simulating day-by-day. A European option's payoff only depends
on where the stock ends up, not the path it took to get there, so there's
no reason to pay for the extra steps. (`simulate_gbm_paths()` *does* step
through time — same exact formula, applied repeatedly — but that's only
used for the path plot, and will matter for real in Project 5, where the
hedging loop needs to rebalance *along* the path, not just look at the
endpoint.)

**Two things every MC estimate needs that a closed-form price doesn't:**

- **A standard error.** A single MC price is a noisy estimate — a number
  without an error bar is unfalsifiable. `mc_price()` returns
  `std_error = std(discounted payoffs) / sqrt(n_paths)` and a 95% CI, so
  you can see how much to trust it. The self-test in `monte_carlo.py`
  prints this at increasing path counts — watch the CI visibly tighten
  around the true Black-Scholes price as `n_paths` grows.
- **Variance reduction**, if you care about efficiency. `antithetic=True`
  (the default) pairs every random draw `Z` with its mirror `-Z`. This is
  free — no extra random numbers needed — and cuts the standard error
  because the option payoff is roughly monotonic in `Z`, so the pair's
  errors partially cancel instead of stacking.

**Why convergence looks noisy, not smooth**
(`fig5_mc_convergence.png`, right panel): each point is drawn with a fixed
seed, but Monte Carlo error doesn't shrink monotonically — it shrinks *on
average* like `1/√N`, with real sample-to-sample noise on top. That's not
a bug in the plot; it's the actual statistical behavior, and it's the
reason the standard error matters more than any single point estimate.

**Visualized in:** `mc_plots.py` — `fig4` shows a sample of simulated paths
(colored by whether they finish above/below the strike) plus the terminal
price distribution; `fig5` shows the MC price converging to the
closed-form value as `n_paths` grows, plus a log-log check that the error
actually tracks the theoretical `1/√N` rate.

---

## 6. Where this is headed (Projects 4 & 5)

- **Project 4 (vol smile/surface):** run `implied_vol.py`'s solver across a
  real option chain — many strikes and maturities at once. The math is
  already built; the new difficulty is *data*: real quotes have bid/ask
  spreads, illiquid strikes with unreliable prices, and dividends that need
  handling correctly (this is why `q` is threaded through every function
  in `black_scholes.py` even though it's always been zero so far).
- **Project 5 (delta-hedging P&L simulator):** this is where "replication"
  from §2 stops being a theoretical argument and becomes actual simulated
  P&L. Take a simulated path from `simulate_gbm_paths()`, and at each step,
  hold `delta()` shares of stock against a short option position,
  rebalancing as delta changes. In a perfect continuous-time world with the
  realized vol exactly matching the vol you hedged with, this replication
  is exact and the hedged P&L is zero — that's the whole content of the
  Black-Scholes argument. The simulator makes it concrete: discrete
  rebalancing (not continuous) and realized vol ≠ implied vol both leak
  P&L, and that leak decomposes cleanly into gamma P&L and a vega/vol-gap
  term. This is the project that turns "delta is the hedge ratio" from a
  definition into something you've watched happen.

---

## Variable reference

Every symbol and parameter name used across the codebase, in one place.
Grouped by role, not by file, since most of these appear in multiple files.

### Core market/contract inputs (appear almost everywhere)

| Name | Meaning | Units / notes |
|---|---|---|
| `S` | Spot — current stock price | dollars |
| `K` | Strike — fixed price in the option contract | dollars |
| `T` | Time to expiry ("maturity") | **years** (0.5 = 6 months). `bs_price` treats `T <= 0` as expired: collapses to discounted intrinsic value instead of dividing by zero. |
| `r` | Risk-free rate | annualized decimal (0.05 = 5%), continuously compounded |
| `q` | Dividend yield | annualized decimal, continuous. Defaults to 0 everywhere; reduces the risk-neutral drift because a shareholder captures dividends an option holder doesn't. |
| `sigma` / `σ` | Volatility — annualized standard deviation of the stock's *returns* | decimal (0.20 = 20%), not directly observable — either estimated from history or backed out from prices (that's the whole point of `implied_vol.py`) |
| `option_type` | `"call"` or `"put"` (also accepts `"c"`/`"p"`) | string, case-insensitive throughout |
| `S_T` | The stock price **at expiry** — the random outcome everything is priced against | dollars; this is what `S` evolves *into* under GBM |

### Black-Scholes internals — [black_scholes.py](black_scholes.py)

| Name | Meaning |
|---|---|
| `d1`, `d2` | The two standardized "distance to strike, in vol-adjusted units" terms the whole formula is built from. `d1 = (ln(S/K) + (r - q + 0.5σ²)T) / (σ√T)`, `d2 = d1 - σ√T`. Loosely: `d2` is the risk-neutral probability driver for the option finishing ITM; `d1` is the same idea but weighted toward the stock leg. Computed once in `_d1_d2()` and reused by every Greek. |
| `vol_sqrt_T` | `σ√T` — "total volatility over the life of the option." Shows up constantly because it's the standard deviation of the *log* stock return over `[0, T]`. |
| `N(x)` / `norm_cdf(x)` | Standard normal CDF, `Φ(x)` — probability a standard normal variable is ≤ `x`. Built from `math.erf`, not scipy. Returns a scalar for scalar input, vectorizes over numpy arrays. |
| `φ(x)` / `norm_pdf(x)` | Standard normal density — the bell curve itself, used inside gamma/vega/theta. |
| `disc_S` | `S · e^{-qT}` — present value of "the stock leg" (spot discounted for dividends the holder forgoes) |
| `disc_K` | `K · e^{-rT}` — present value of "the strike leg" (the cash you'd pay, discounted at the risk-free rate) |
| `is_call` | Boolean derived from `option_type`, used to pick the call vs. put branch of each formula |
| `fwd`, `intrinsic` | Used only in the `T<=0`/`σ<=0` fallback path: `fwd` is the forward value `S·e^{-qT} - K·e^{-rT}`, `intrinsic` is `max(fwd, 0)` (call) or `max(-fwd, 0)` (put) — what the option is worth if it's already expired |
| `_SQRT2`, `_SQRT2PI` | Precomputed constants (`√2`, `√(2π)`) so they aren't recomputed on every call |

### The Greeks — outputs of black_scholes.py

| Name | ∂Price/∂ | Plain-English read |
|---|---|---|
| `delta` | spot | price change per $1 stock move; also ≈ hedge ratio (shares to hold per option) |
| `gamma` | delta (2nd deriv of spot) | how fast delta itself changes — curvature, peaks ATM near expiry |
| `vega` | volatility | price change per 1.00 (100 vol points) of `σ`; divide by 100 for "per vol-point" |
| `theta` | time | time decay; raw = per year, divide by 365 for "per day" |
| `rho` | interest rate | price change per 1.00 (100%) of `r`; divide by 100 for "per 1%" |
| `vanna` | spot then vol (or vice versa) | how delta shifts as vol changes (= how vega shifts as spot changes) |
| `vomma` | vol, twice | "vol convexity" — how vega itself changes as vol changes |

All six are returned **raw** (per unit of the underlying variable), not
trader-scaled — the `/100`, `/365` scaling above is applied by the caller
(e.g. [greeks_plots.py:43-45](greeks_plots.py#L43-L45)), not baked into the
formula.

### Implied vol solver — [implied_vol.py](implied_vol.py)

| Name | Meaning |
|---|---|
| `market_price` | The observed option price you're trying to explain — the thing you're solving *for* the `σ` that reproduces |
| `tol` | Convergence tolerance — stop once the price error is smaller than this |
| `max_iter` | Cap on iterations for each phase (Newton, then bisection), so a pathological input can't loop forever |
| `vega_floor` | Minimum vega required to trust a Newton step. Below this, `diff / vega` is dividing by ~0 — the price is nearly flat in `σ` (deep ITM/OTM, short-dated), so the code bails to bisection instead |
| `bisect_lo`, `bisect_hi` | The `[lo, hi]` bracket bisection searches within (defaults `1e-6` to `5.0`, i.e. ~0% to 500% vol) |
| `lower`, `upper` | No-arbitrage price bounds for the given inputs (from `_intrinsic_bounds`) — if `market_price` falls outside these, no real `σ` can reproduce it, and the solver fails fast instead of chasing a nonexistent root |
| `diff` | `price - market_price` at the current guess — the residual Newton is driving to zero |
| `v` | Vega at the current `σ` guess, used both as Newton's slope and as the trust check against `vega_floor` |
| `sigma_new` | Newton's proposed next guess, `sigma - diff / v` |
| `lo`, `hi`, `f_lo`, `f_hi`, `mid`, `f_mid` | Standard bisection bookkeeping: current bracket `[lo, hi]`, the price error at each end, and at the midpoint each iteration |
| `iv` | The solved implied volatility (or `None` on failure) — the function's main return value |
| `iterations` | How many steps the solver actually took |
| `method` | Which phase produced the answer: `'newton'` or `'bisection'` |
| `converged` | Whether the solver actually met tolerance, vs. hit `max_iter` |
| `error` | Human-readable failure reason (out-of-bounds price, bad bracket, etc.), only set on failure |
| `warning` | Set when vega *at the solution* is tiny — the price matched, but many different `σ` values would have matched about as well, so the number is poorly determined even though it "converged" |

### Monte Carlo — [monte_carlo.py](monte_carlo.py)

| Name | Meaning |
|---|---|
| `n_paths` | Number of simulated terminal-price draws (or full paths) to average over. More paths = tighter estimate, at `1/√n_paths` cost/benefit |
| `n_steps` | Number of time steps *within* one path — only relevant to `simulate_gbm_paths()` (for plotting); terminal-only pricing doesn't need this since it draws `S_T` in one shot |
| `antithetic` | If `True` (default), pairs each random draw `Z` with `-Z` — free variance reduction, since the payoff is roughly monotonic in `Z` |
| `seed` | Seeds the random generator for reproducibility — same seed, same simulated paths, same price, useful for comparing settings apples-to-apples |
| `rng` | The numpy random generator instance (`np.random.default_rng(seed)`) |
| `Z` | A draw from the standard normal distribution — the actual source of randomness in GBM. Every simulated path is ultimately "some deterministic drift, plus `σ√dt · Z`" |
| `drift` | The deterministic part of the log-return over the interval: `(r - q - 0.5σ²) · T` (or `· dt` per step) |
| `diffusion` | The random part's scale: `σ√T` (or `σ√dt` per step) — multiplies `Z` |
| `dt` | One time step's length in years, `T / n_steps` |
| `payoff` | `max(S_T - K, 0)` or `max(K - S_T, 0)`, computed per simulated path, before discounting |
| `disc` | The discount factor `e^{-rT}` applied to each payoff |
| `discounted` | Per-path discounted payoff — the quantity that gets averaged (→ `price`) and whose sample standard deviation gives the standard error |
| `price` | The Monte Carlo price estimate — mean of `discounted` across all paths |
| `std_error` | Standard error of `price` — `std(discounted) / √n_paths`. This is what tells you *how much to trust* a single MC price; shrinks like `1/√n_paths` |
| `ci_low`, `ci_high` | 95% confidence interval, `price ± 1.96 · std_error` |
| `t_grid` | The array of time points `[0, dt, 2dt, ..., T]` a simulated path is plotted against |
| `paths` | Full simulated path array from `simulate_gbm_paths()`, shape `(n_paths, n_steps+1)` — used only for visualization |

### Conceptual terms

- **ATM / ITM / OTM** — at/in/out-of-the-money: whether the strike is
  near, favorable, or unfavorable relative to the current spot.
- **GBM** — Geometric Brownian Motion, the stochastic process this whole
  repo assumes the stock follows: `dS = (r-q)S dt + σS dW`.
- **`dW`** — an infinitesimal increment of Brownian motion (Wiener
  process) — the continuous-time formalization of "random noise
  accumulating over time." You never see `dW` directly in the code; its
  discrete-time stand-in is `√dt · Z`.
- **No-arbitrage** — the principle that two portfolios with identical
  future payoffs must have identical prices today, or someone could make
  riskless profit. This is the bedrock every formula here rests on.
- **Risk-neutral measure** — the "fictional pricing world" from §2 where
  every asset drifts at `r` (not its real-world expected return); pricing
  under this measure is mathematically equivalent to the no-arbitrage
  replication argument.
