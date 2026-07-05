"""
FFT-based option pricing using the Carr-Madan (1999) framework.

Prices a whole grid of strikes in a single Fast Fourier Transform pass,
using the characteristic function of the log-asset-price under the
risk-neutral measure. This gives an order-of-magnitude speedup over
naive numerical integration when pricing many complex derivatives.
"""
import numpy as np


def _bsm_characteristic_function(u, S0, T, r, sigma, q=0.0):
    """Characteristic function of log-price under BSM dynamics."""
    mu = np.log(S0) + (r - q - 0.5 * sigma ** 2) * T
    return np.exp(1j * u * mu - 0.5 * (sigma ** 2) * T * u ** 2)


def fft_option_price(S0: float, K: float, T: float, r: float, sigma: float,
                      option_type: str = "call", alpha: float = 1.5,
                      n: int = 12, B: float = 500.0):
    """
    Price a European option via the Carr-Madan FFT method.

    Parameters
    ----------
    S0, K, T, r, sigma : as in black_scholes.bsm_price
    option_type : "call" or "put"
    alpha : float   Damping factor for the modified call price (typical: 1.0-2.0)
    n : int         log2 of number of FFT grid points (2^n grid points)
    B : float       Upper bound for the integration domain

    Returns
    -------
    float : option price at strike K
    """
    N = 2 ** n
    eta = B / N
    lambda_ = (2 * np.pi) / (N * eta)
    b = (N * lambda_) / 2

    u = np.arange(N) * eta
    ku = -b + lambda_ * np.arange(N)  # log-strike grid

    # Carr-Madan integrand for the damped call price
    v = u - (alpha + 1) * 1j
    cf = _bsm_characteristic_function(v, S0, T, r, sigma)
    denom = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
    psi = np.exp(-r * T) * cf / denom

    # Simpson's rule weighting for better accuracy
    simpson = (3 + (-1) ** (np.arange(N) + 1))
    simpson[0] = 1
    x = np.exp(1j * b * u) * psi * eta * simpson / 3

    fft_prices = np.fft.fft(x).real
    call_prices = (np.exp(-alpha * ku) / np.pi) * fft_prices

    strikes = np.exp(ku)
    call_price = np.interp(K, strikes, call_prices)

    if option_type == "call":
        return call_price
    elif option_type == "put":
        # Put-call parity
        return call_price - S0 + K * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
