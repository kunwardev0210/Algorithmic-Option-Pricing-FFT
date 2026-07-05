"""
Carr-Madan FFT Option Pricing Engine
=====================================
Implements the Carr-Madan (1999) Fast Fourier Transform methodology for
pricing European Call Options under any model with an analytical
characteristic function (BSM, Variance Gamma, Heston, etc.)

Methodology:
------------
1. The raw call payoff e^{-rT}(S_T - K)+ is not square-integrable.
   Apply exponential damping: z_T(k) = e^{alpha*k} * C_T(k)

2. Take the Fourier transform of z_T(k) to obtain psi_T(v):
   psi_T(v) = e^{-rT} * phi_T(v - (alpha+1)i) / [alpha^2 + alpha - v^2 + i(2*alpha+1)*v]

3. Invert via FFT using Simpson's rule discretisation with N points,
   integration step eta, and log-strike grid spacing lambda = 2*pi / (N*eta)

4. Recover call prices: C_T(k) = e^{-alpha*k} / pi * Re[FFT(psi)]

Reference: Carr, P., Madan, D.B. (1999). "Option Valuation using the Fast Fourier Transform."
           Journal of Computational Finance, 2(4), 61-73.
"""

import numpy as np
from variance_gamma import vg_char_func


def carr_madan_fft(S0, r, T, sigma, nu, theta,
                   alpha=1.5, N=4096, eta=0.25,
                   char_func=None):
    """
    Price European Call options via Carr-Madan FFT.

    Parameters
    ----------
    S0     : float  - Current stock price
    r      : float  - Risk-free rate
    T      : float  - Time to maturity
    sigma  : float  - VG volatility parameter
    nu     : float  - VG kurtosis parameter
    theta  : float  - VG skewness parameter
    alpha  : float  - Damping factor (must satisfy: alpha > 0, alpha+1 < max u)
    N      : int    - Number of FFT points (power of 2 for efficiency)
    eta    : float  - Integration step size
    char_func : callable or None - Custom characteristic function

    Returns
    -------
    strikes : np.ndarray - Array of strike prices
    prices  : np.ndarray - Array of corresponding call prices
    """
    # --- Grid Setup ---
    # Nyquist condition: lambda * eta = 2*pi / N
    lam = (2 * np.pi) / (N * eta)

    # Log-strike grid centred at log(S0)
    k0 = np.log(S0) - (N / 2) * lam
    k_array = k0 + lam * np.arange(N)       # log-strikes
    strikes = np.exp(k_array)               # actual strikes

    # --- Integration Grid ---
    v = eta * np.arange(N)                  # frequency grid

    # --- Characteristic Function ---
    if char_func is None:
        phi = vg_char_func(v - (alpha + 1) * 1j, T, sigma, nu, theta, r)
    else:
        phi = char_func(v - (alpha + 1) * 1j)

    # --- Carr-Madan Integrand ---
    numerator   = np.exp(-r * T) * phi
    denominator = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v

    # Avoid division by zero at v=0
    denominator = np.where(np.abs(denominator) < 1e-10,
                           1e-10 * np.ones_like(denominator),
                           denominator)

    psi = numerator / denominator

    # --- Simpson's Rule Weights ---
    weights = eta / 3.0 * (3.0 + (-1.0) ** np.arange(N) - (np.arange(N) == 0))

    # Phase correction for non-zero log-strike grid start
    phase = np.exp(1j * v * k0)

    # --- FFT ---
    fft_input = psi * weights * phase
    fft_output = np.fft.fft(fft_input)

    # --- Recover Call Prices ---
    call_prices = (np.exp(-alpha * k_array) / np.pi) * np.real(fft_output)

    # Enforce non-negativity (numerical noise can produce tiny negatives)
    call_prices = np.maximum(call_prices, 0.0)

    return strikes, call_prices


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bsm import bsm_call_vector

    # Parameters
    S0    = 100.0
    r     = 0.05
    T     = 1.0
    sigma = 0.2
    nu    = 0.1
    theta = -0.1

    strikes, vg_prices = carr_madan_fft(S0, r, T, sigma, nu, theta)

    # Compare with BSM using same sigma
    bsm_prices = bsm_call_vector(S0, strikes, r, T, sigma)

    # Filter to reasonable strike range
    mask = (strikes > 60) & (strikes < 160)

    plt.figure(figsize=(10, 5))
    plt.plot(strikes[mask], vg_prices[mask], label='Variance Gamma (FFT)', color='steelblue', lw=2)
    plt.plot(strikes[mask], bsm_prices[mask], label='Black-Scholes-Merton', color='tomato', lw=2, linestyle='--')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Call Option Price')
    plt.title('European Call: VG (FFT) vs BSM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/vg_vs_bsm.png', dpi=150)
    plt.show()
    print("Plot saved to results/vg_vs_bsm.png")
