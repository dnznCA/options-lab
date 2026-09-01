"""
vol_surface.py — implied volatility smile/surface from a real option chain.

Project 4. The math is already built (Project 2's implied_vol.py, run across a
grid instead of one quote) — the actual difficulty here is data. This file
pulls a live option chain via yfinance and turns it into something the solver
can trust: dropping quotes with no real market, throwing out wide bid/ask
spreads, and preferring the OTM side of each strike (more liquid than ITM).

Conventions match the rest of the repo: T in years, r/q/sigma as decimals.

Needs: black_scholes.py, implied_vol.py (this repo) and yfinance (pip).
"""

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from implied_vol import implied_vol

# yfinance doesn't expose a risk-free rate; a real desk would bootstrap one
# from the Treasury curve. This repo doesn't have that pipeline, so a single
# flat short-term rate stands in — good enough for a teaching smile, not for
# a trading desk.
RISK_FREE_RATE = 0.045


def fetch_option_chain(ticker_symbol, max_expiries=6):
    """
    Pull spot, dividend yield, and a combined calls+puts DataFrame across the
    first `max_expiries` expiries, with time-to-expiry (T, years) attached.

    Keeps only the OTM side of each strike (calls above spot, puts below) —
    OTM options are the actively-traded side of the chain, so this is a free
    liquidity win before any bid/ask filtering even happens. It also sidesteps
    call/put IV disagreeing on the same strike (American exercise, dividends,
    and thin ITM liquidity all nudge them apart in real quotes).
    """
    ticker = yf.Ticker(ticker_symbol)
    spot = ticker.fast_info["lastPrice"]
    q = ticker.info.get("trailingAnnualDividendYield") or 0.0

    today = dt.date.today()
    expiries = ticker.options[:max_expiries]

    frames = []
    for exp_str in expiries:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        T = (exp_date - today).days / 365.0
        if T <= 0:
            continue

        chain = ticker.option_chain(exp_str)
        calls_otm = chain.calls[chain.calls["strike"] >= spot].copy()
        calls_otm["option_type"] = "call"
        puts_otm = chain.puts[chain.puts["strike"] < spot].copy()
        puts_otm["option_type"] = "put"

        combined = pd.concat([calls_otm, puts_otm], ignore_index=True)
        combined["expiry"] = exp_str
        combined["T"] = T
        frames.append(combined)

    raw = pd.concat(frames, ignore_index=True)
    return spot, q, raw


def clean_quotes(raw, spot, max_rel_spread=0.5, moneyness_range=(0.7, 1.3), min_volume=1):
    """
    Data hygiene pass, each step documented because "just filter bad data" is
    doing a lot of unstated work:

      - drop quotes with no bid or no ask — not a real two-sided market, just
        a stale listing.
      - use mid price = (bid+ask)/2 as "the" market price, not last trade
        (which can be stale for anything that hasn't traded today).
      - drop quotes whose spread exceeds max_rel_spread of the mid — a spread
        that wide means the "price" is closer to a guess than a market.
      - drop strikes outside moneyness_range — deep OTM options are thin and
        their IV is poorly determined anyway (implied_vol.py's own vega-floor
        warning would flag most of these regardless).
      - drop quotes with less than min_volume contracts traded today.

    Prints a before/after count so the filtering isn't a silent black box.
    """
    df = raw.copy()
    n_start = len(df)

    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[df["rel_spread"] <= max_rel_spread]

    df["moneyness"] = df["strike"] / spot
    lo, hi = moneyness_range
    df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]

    if min_volume is not None:
        df = df[df["volume"].fillna(0) >= min_volume]

    n_end = len(df)
    print(f"cleaned {n_start} quotes -> {n_end} kept "
          f"({n_start - n_end} dropped: no market / wide spread / far OTM / no volume)")
    return df.reset_index(drop=True)


def solve_ivs(df, spot, r, q):
    """Run implied_vol() over every cleaned quote, attaching iv/converged/method/warning."""
    records = []
    for row in df.itertuples():
        result = implied_vol(row.mid, spot, row.strike, row.T, r, q, row.option_type)
        records.append({
            "expiry": row.expiry, "T": row.T, "strike": row.strike,
            "moneyness": row.moneyness, "option_type": row.option_type,
            "mid": row.mid, "volume": row.volume,
            "iv": result["iv"], "converged": result["converged"],
            "method": result["method"], "warning": result.get("warning"),
        })
    out = pd.DataFrame(records)
    out = out[out["converged"] & out["iv"].notna()].reset_index(drop=True)
    return out


if __name__ == "__main__":
    TICKER = "AAPL"
    print(f"Fetching option chain for {TICKER}...")
    spot, q, raw = fetch_option_chain(TICKER, max_expiries=6)
    print(f"spot={spot:.2f}  dividend yield q={q:.4f}  risk-free rate r={RISK_FREE_RATE:.4f}")
    print(f"raw OTM quotes across {raw['expiry'].nunique()} expiries: {len(raw)}\n")

    cleaned = clean_quotes(raw, spot)
    surface = solve_ivs(cleaned, spot, RISK_FREE_RATE, q)
    print(f"solved {len(surface)} implied vols\n")

    print(f"{'expiry':<12} {'T':>6} {'n':>4} {'min IV':>8} {'ATM IV':>8} {'max IV':>8}")
    print("-" * 52)
    for expiry, grp in surface.groupby("expiry", sort=False):
        atm_row = grp.iloc[(grp["moneyness"] - 1.0).abs().argsort().iloc[0]]
        print(f"{expiry:<12} {atm_row['T']:>6.3f} {len(grp):>4} "
              f"{grp['iv'].min():>8.3f} {atm_row['iv']:>8.3f} {grp['iv'].max():>8.3f}")
