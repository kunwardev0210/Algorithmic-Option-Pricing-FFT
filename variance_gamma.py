"""
Variance Gamma (VG) Characteristic Function
============================================
The VG process models asset returns as a Brownian motion with drift,
time-changed by a Gamma process. It captures:

    - Skewness  (theta): asymmetric return distributions
    - Kurtosis  (nu)   : fat tails and jump-like behaviour
    - Volatility (sigma): diffusion component

Characteristic Function:
    phi_T(u) = exp(i*u*omega*T) * [1 - i*u*theta*nu + sigma^2*nu*u^2/2]^(-T/nu)

where omega = (1/nu) * log(1 - theta*nu - sigma^2*nu/2)  [martingale correction]
"""

import numpy as np


def vg_char_func(u, T, sigma, nu, theta, r):
    """
    Characteristic function of the log-price under the Variance Gamma model.

    Parameters
    ----------
    u     : np.ndarray - Frequency domain variable
    T     : float      - Time to maturity
    sigma : float      - VG volatility parameter
    nu    : float      - VG kurtosis (variance rate) parameter
    theta : float      - VG skewness (drift) parameter
    r     : float      - Risk-free rate

    Returns
    -------
    np.ndarray : Complex characteristic function values
    """
    # Martingale correction term (ensures no-arbitrage)
    omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * sigma ** 2 * nu)

    # VG characteristic function
    phi = np.exp(1j * u * (r + omega) * T) * \
          (1.0 - 1j * u * theta * nu + 0.5 * sigma ** 2 * nu * u ** 2) ** (-T / nu)

    return phi


if __name__ == "__main__":
    # Quick sanity check: phi(0) should equal 1
    u_test = np.array([0.0, 1.0, 2.0])
    phi = vg_char_func(u_test, T=1.0, sigma=0.2, nu=0.1, theta=-0.1, r=0.05)
    print("phi(0) =", phi[0].real, "(should be 1.0)")
    print("phi(1) =", phi[1])
