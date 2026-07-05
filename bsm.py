"""
Black-Scholes-Merton (BSM) European Call Option Pricer
=======================================================
Derived from the BSM PDE via Ito's Lemma and no-arbitrage conditions.

    C = S0 * N(d1) - K * exp(-r*T) * N(d2)

    d1 = [ln(S0/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
"""

import numpy as np
from scipy.stats import norm


def bsm_call_price(S0, K, r, T, sigma):
    """
    Compute European Call price under Black-Scholes-Merton model.

    Parameters
    ----------
    S0    : float  - Current stock price
    K     : float  - Strike price
    r     : float  - Risk-free interest rate (annualised)
    T     : float  - Time to maturity (in years)
    sigma : float  - Volatility (annualised)

    Returns
    -------
    float : BSM call price
    """
    if T <= 0:
        return max(S0 - K, 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call


def bsm_call_vector(S0, K_array, r, T, sigma):
    """
    Vectorised BSM call prices across a range of strikes.

    Parameters
    ----------
    S0      : float      - Current stock price
    K_array : np.ndarray - Array of strike prices
    r       : float      - Risk-free rate
    T       : float      - Time to maturity
    sigma   : float      - Volatility

    Returns
    -------
    np.ndarray : BSM call prices for each strike
    """
    K_array = np.asarray(K_array)
    d1 = (np.log(S0 / K_array) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K_array * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(C_market, S0, K, r, T, tol=1e-6, max_iter=200):
    """
    Compute implied volatility via bisection method.

    Parameters
    ----------
    C_market : float - Observed market call price
    S0       : float - Current stock price
    K        : float - Strike price
    r        : float - Risk-free rate
    T        : float - Time to maturity

    Returns
    -------
    float : Implied volatility (or np.nan if not found)
    """
    low, high = 1e-6, 10.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        price = bsm_call_price(S0, K, r, T, mid)
        if abs(price - C_market) < tol:
            return mid
        if price < C_market:
            low = mid
        else:
            high = mid
    return np.nan


if __name__ == "__main__":
    # Example
    S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.2
    price = bsm_call_price(S0, K, r, T, sigma)
    print(f"BSM Call Price: {price:.4f}")
