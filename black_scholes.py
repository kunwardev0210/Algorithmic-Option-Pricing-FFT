"""
Closed-form Black-Scholes-Merton (BSM) option pricing.
"""
import numpy as np
from scipy.stats import norm


def bsm_price(S0: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call", q: float = 0.0) -> float:
    """
    Price a European option using the Black-Scholes-Merton closed-form formula.

    Parameters
    ----------
    S0 : float   Current underlying asset price
    K  : float   Strike price
    T  : float   Time to maturity (in years)
    r  : float   Risk-free interest rate
    sigma : float  Volatility of the underlying asset
    option_type : str  "call" or "put"
    q  : float   Continuous dividend yield (default 0)

    Returns
    -------
    float : theoretical option price
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price
