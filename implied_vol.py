"""
implied_vol.py - implied vol solver

BS goes sigma -> price. market gives us price, we need sigma. inverts
bs_price() with Newton-Raphson (uses vega as slope, fast) and falls back
to bisection when Newton's unreliable (vega ~0: deep ITM/OTM, short-dated).

needs black_scholes.py: bs_price(), vega()
using kwargs for option_type/q on every call below - positional args here
are an easy way to feed a float into the wrong slot.
"""

import math
from black_scholes import bs_price, vega


def _intrinsic_bounds(S, K, T, r, q, option_type):
    """
    no-arb price bounds. price outside this range = no real IV exists,
    not a solver bug (bad price, or wrong r/q/T). check before root-finding,
    don't wait for Newton to fail first.
    """
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option_type == 'call':
        lower = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r
    return lower, upper


def _initial_guess(S, T, market_price):
    # brenner-subrahmanyam ATM approx: sigma ~ sqrt(2pi/T) * price/S
    # bad far from ATM but cheap, just needs to land Newton in the neighborhood
    if T <= 0:
        return 0.2
    guess = math.sqrt(2 * math.pi / T) * market_price / S
    return min(max(guess, 0.05), 2.0)  # clamp to sane range


def implied_vol(market_price, S, K, T, r, q=0.0, option_type='call',
                 tol=1e-8, max_iter=100,
                 vega_floor=1e-8, bisect_lo=1e-6, bisect_hi=5.0):
    """
    solves for sigma st bs_price(S, K, T, r, sigma, option_type, q) == market_price

    returns dict: iv, iterations, method ('newton'/'bisection'), converged,
    error (only set if something went wrong)
    """
    lower, upper = _intrinsic_bounds(S, K, T, r, q, option_type)
    if market_price < lower - 1e-10 or market_price > upper + 1e-10:
        return {
            'iv': None, 'iterations': 0, 'method': None, 'converged': False,
            'error': (f'Price {market_price:.6f} outside no-arbitrage bounds '
                      f'[{lower:.6f}, {upper:.6f}]')
        }

    sigma = _initial_guess(S, T, market_price)

    def _with_vega_warning(result, sigma_val):
        # matching price doesn't mean sigma is pinned down - if vega's tiny
        # here, tons of sigmas would've priced about the same. warn instead
        # of pretending this is a clean confident answer
        v_at_sol = vega(S, K, T, r, sigma_val, q)
        if v_at_sol < 1e-4:
            result['warning'] = (
                f'vega at solution is {v_at_sol:.2e} — price is nearly flat in sigma here '
                f'(deep ITM/OTM or very short-dated). IV is poorly determined by this price; '
                f'treat the number with caution.'
            )
        return result

    # --- newton phase ---
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type=option_type, q=q)
        diff = price - market_price
        v = vega(S, K, T, r, sigma, q)

        # need both: price matches AND vega big enough that the match means
        # something (tiny diff next to a ~0 price is trivial, proves nothing)
        if abs(diff) < tol and v > vega_floor:
            return _with_vega_warning(
                {'iv': sigma, 'iterations': i + 1, 'method': 'newton', 'converged': True},
                sigma
            )

        if v < vega_floor:
            break  # too flat to trust the slope, bail to bisection

        sigma_new = sigma - diff / v
        if sigma_new <= 0 or sigma_new > 10:
            break  # step went somewhere dumb, bail to bisection

        sigma = sigma_new

    # --- bisection fallback ---
    lo, hi = bisect_lo, bisect_hi
    f_lo = bs_price(S, K, T, r, lo, option_type=option_type, q=q) - market_price
    f_hi = bs_price(S, K, T, r, hi, option_type=option_type, q=q) - market_price

    # root sitting right at lo/hi happens a lot for deep OTM (price rounds
    # to 0 at the low end) - sign check breaks at exactly 0 so handle first
    if abs(f_lo) < tol:
        return _with_vega_warning(
            {'iv': lo, 'iterations': 0, 'method': 'bisection', 'converged': True}, lo
        )
    if abs(f_hi) < tol:
        return _with_vega_warning(
            {'iv': hi, 'iterations': 0, 'method': 'bisection', 'converged': True}, hi
        )

    if f_lo * f_hi > 0:
        return {
            'iv': None, 'iterations': 0, 'method': 'bisection', 'converged': False,
            'error': 'Bisection bracket does not contain a sign change — check inputs.'
        }

    for i in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = bs_price(S, K, T, r, mid, option_type=option_type, q=q) - market_price
        # trust interval width over |f_mid| < tol - same flat-price issue as
        # newton can make f_mid look converged when it isn't
        if (hi - lo) < tol or abs(f_mid) < tol:
            return _with_vega_warning(
                {'iv': mid, 'iterations': i + 1, 'method': 'bisection', 'converged': True},
                mid
            )
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid

    return {
        'iv': (lo + hi) / 2, 'iterations': max_iter, 'method': 'bisection', 'converged': False,
        'error': 'Max iterations reached without full convergence.'
    }


# --- self-test: round-trip a known sigma through bs_price -> implied_vol, check it comes back
if __name__ == '__main__':
    test_cases = [
        # label, S, K, T, r, sigma_true, q, option_type
        ('ATM call, 1yr',            100, 100, 1.0, 0.05, 0.20, 0.00, 'call'),
        ('ATM put, 1yr',             100, 100, 1.0, 0.05, 0.20, 0.00, 'put'),
        ('10% OTM call, 3mo',        100, 110, 0.25, 0.05, 0.25, 0.00, 'call'),
        ('Deep OTM call, low vega',  100, 180, 0.10, 0.05, 0.30, 0.00, 'call'),
        ('Deep OTM put, low vega',   100, 20,  0.10, 0.05, 0.30, 0.00, 'put'),
        ('Deep ITM put, low vega',   100, 180, 0.10, 0.05, 0.30, 0.00, 'put'),
        ('With dividend yield',      100, 100, 1.0, 0.05, 0.20, 0.03, 'call'),
        ('Short-dated, near-flat vega', 100, 95, 0.02, 0.05, 0.40, 0.00, 'call'),
    ]

    print(f"{'Case':<32} {'true σ':>8} {'solved σ':>10} {'err':>10} {'method':>10} {'iters':>6}")
    print('-' * 82)
    for label, S, K, T, r, sigma_true, q, opt in test_cases:
        market_price = bs_price(S, K, T, r, sigma_true, option_type=opt, q=q)
        result = implied_vol(market_price, S, K, T, r, q, opt)
        if result['iv'] is None:
            print(f"{label:<32} {sigma_true:>8.4f} {'FAILED':>10} {'-':>10} {str(result['method']):>10}")
            continue
        err = abs(result['iv'] - sigma_true)
        flag = '  <-- ' + result['warning'] if 'warning' in result else ''
        print(f"{label:<32} {sigma_true:>8.4f} {result['iv']:>10.6f} {err:>10.2e} "
              f"{result['method']:>10} {result['iterations']:>6}{flag}")

    # price above upper bound should just fail cleanly, not blow up
    print()
    bad_price_result = implied_vol(market_price=200, S=100, K=100, T=1.0, r=0.05)
    print(f"Out-of-bounds price test -> converged={bad_price_result['converged']}, "
          f"error='{bad_price_result.get('error')}'")
