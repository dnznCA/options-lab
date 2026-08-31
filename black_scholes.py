"""
black_scholes.py — closed-form European option pricer + Greeks.

Conventions:
  - sigma, r, q are annualized decimals (0.20 = 20%, not 20).
  - T is in years (0.5 = 6 months).
  - Greeks are RAW (per unit move), not trader-scaled. Trader scaling
    (vega/100, theta/365, rho/100) is the caller's job, noted per-function
    below — not baked into the formulas.
"""

from math import erf, log, sqrt, exp, pi
import numpy as np

# avoid recomputing these every call
_SQRT2 = sqrt(2.0)
_SQRT2PI = sqrt(2.0 * pi)


# --- normal distribution, hand-rolled (no scipy dependency) ------------------

def norm_pdf(x):
    """Standard normal density phi(x). Vectorized over numpy arrays."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT2PI


def norm_cdf(x):
    """
    Standard normal CDF Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
    erf-based, so exact rather than a rational approximation.
    """
    # scalar path uses math.erf directly; array path vectorizes it since
    # math.erf only takes one number at a time
    if np.ndim(x) == 0:
        return 0.5 * (1.0 + erf(float(x) / _SQRT2))
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x, dtype=float) / _SQRT2))


# --- d1 / d2 -------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma, q=0.0):
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


# --- price -----------------------------------------------------------------

def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    European option price under Black-Scholes-Merton (q = continuous div yield).

    T<=0 or sigma<=0 collapses to discounted intrinsic value instead of
    dividing by zero — keeps this well-behaved as expiry approaches.
    """
    is_call = option_type.lower() in ("c", "call")

    # only branches when T/sigma are scalars — can't do this per-element on arrays
    if np.ndim(T) == 0 and np.ndim(sigma) == 0 and (T <= 0 or sigma <= 0):
        fwd = S * exp(-q * T) - K * exp(-r * T)
        intrinsic = max(fwd, 0.0) if is_call else max(-fwd, 0.0)
        return intrinsic

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_S = S * np.exp(-q * T)   # PV of the stock leg
    disc_K = K * np.exp(-r * T)   # PV of the strike leg

    if is_call:
        return disc_S * norm_cdf(d1) - disc_K * norm_cdf(d2)
    return disc_K * norm_cdf(-d2) - disc_S * norm_cdf(-d1)


# --- the six Greeks ----------------------------------------------------------
# core five (delta/gamma/vega/theta/rho) + vanna/vomma as second-order bonus
# Greeks — how the core Greeks shift as vol changes.

def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    dPrice/dSpot. Call in [0, e^-qT], put in [-e^-qT, 0]
    (just [0,1]/[-1,0] when q=0).
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc = np.exp(-q * T)
    if option_type.lower() in ("c", "call"):
        return disc * norm_cdf(d1)
    return disc * (norm_cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """d2Price/dSpot2. Same call/put. Peaks ATM, blows up as T->0."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm_pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """dPrice/dSigma. Same call/put. RAW = per 1.00 vol move. Trader vega = /100."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm_pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dt. RAW = per year. Trader theta (per-day decay) = /365."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    term_decay = -disc_S * norm_pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type.lower() in ("c", "call"):
        return (term_decay
                - r * disc_K * norm_cdf(d2)
                + q * disc_S * norm_cdf(d1))
    return (term_decay
            + r * disc_K * norm_cdf(-d2)
            - q * disc_S * norm_cdf(-d1))


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dr. RAW = per 1.00 rate move. Trader rho (per 1%) = /100."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_K = K * T * np.exp(-r * T)
    if option_type.lower() in ("c", "call"):
        return disc_K * norm_cdf(d2)
    return -disc_K * norm_cdf(-d2)


def vanna(S, K, T, r, sigma, q=0.0):
    """d2Price/(dSpot dSigma) = dVega/dSpot = dDelta/dSigma. Same call/put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return -np.exp(-q * T) * norm_pdf(d1) * d2 / sigma


def vomma(S, K, T, r, sigma, q=0.0):
    """d2Price/dSigma2 ("vol convexity"). Same call/put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return vega(S, K, T, r, sigma, q) * d1 * d2 / sigma


def greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Bundle: price + all six Greeks in one dict."""
    return {
        "price": bs_price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega":  vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho":   rho(S, K, T, r, sigma, option_type, q),
        "vanna": vanna(S, K, T, r, sigma, q),
        "vomma": vomma(S, K, T, r, sigma, q),
    }


# --- self-test: put-call parity ---------------------------------------------
# C - P = S e^-qT - K e^-rT must hold to machine precision, regardless of the
# pricing model (pure no-arbitrage). If bs_price is right, this residual is
# ~0; if not, something in the pricing formulas is broken.

def put_call_parity_residual(S, K, T, r, sigma, q=0.0):
    c = bs_price(S, K, T, r, sigma, "call", q)
    p = bs_price(S, K, T, r, sigma, "put", q)
    return (c - p) - (S * exp(-q * T) - K * exp(-r * T))


if __name__ == "__main__":
    # base case: ATM, 1y, 5% rates, 20% vol, no div
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    print(f"Base case: S={S} K={K} T={T} r={r} sigma={sigma} q={q}\n")

    for ot in ("call", "put"):
        g = greeks(S, K, T, r, sigma, ot, q)
        print(f"{ot.upper():>4}  price={g['price']:.4f}  "
              f"delta={g['delta']:+.4f}  gamma={g['gamma']:.4f}  "
              f"vega={g['vega']/100:+.4f}/%  "
              f"theta={g['theta']/365:+.4f}/day  rho={g['rho']/100:+.4f}/%")

    resid = put_call_parity_residual(S, K, T, r, sigma, q)
    print(f"\nput-call parity residual: {resid:.2e}  (should be ~0)")
