"""
Volatility Smile Analysis
==========================
Extracts implied volatilities from VG FFT prices and plots the
volatility smile — demonstrating where BSM systematically fails
to match more realistic pricing models.

The "smile" arises because BSM assumes constant sigma, but real
markets price OTM options at higher implied vol than ATM options.
"""

import numpy as np
import matplotlib.pyplot as plt
from bsm import bsm_call_vector, implied_volatility
from fft_pricer import carr_madan_fft

def compute_vol_smile(S0, r, T, sigma, nu, theta, strike_range=(70, 140)):
    """
    Compute implied volatility smile from VG FFT prices.

    Parameters
    ----------
    S0           : float  - Spot price
    r            : float  - Risk-free rate
    T            : float  - Time to maturity
    sigma        : float  - VG sigma
    nu           : float  - VG kurtosis
    theta        : float  - VG skewness
    strike_range : tuple  - (min_strike, max_strike) for filtering

    Returns
    -------
    strikes_filt : np.ndarray - Filtered strikes
    ivols        : np.ndarray - Implied volatilities
    """
    strikes, vg_prices = carr_madan_fft(S0, r, T, sigma, nu, theta)

    mask = (strikes >= strike_range[0]) & (strikes <= strike_range[1])
    strikes_filt = strikes[mask]
    vg_filt      = vg_prices[mask]

    ivols = np.array([
        implied_volatility(c, S0, k, r, T)
        for c, k in zip(vg_filt, strikes_filt)
    ])

    return strikes_filt, ivols


def plot_vol_smile(S0=100, r=0.05, T=1.0, sigma=0.2, nu=0.1, theta=-0.14):
    strikes, ivols = compute_vol_smile(S0, r, T, sigma, nu, theta)

    # Remove NaN values
    valid = ~np.isnan(ivols)
    strikes, ivols = strikes[valid], ivols[valid]

    moneyness = strikes / S0  # K/S0

    plt.figure(figsize=(10, 5))
    plt.plot(moneyness, ivols * 100, color='steelblue', lw=2.5, label='Implied Vol (VG)')
    plt.axhline(y=sigma * 100, color='tomato', linestyle='--', lw=1.8,
                label=f'BSM Flat Vol ({sigma*100:.0f}%)')
    plt.axvline(x=1.0, color='grey', linestyle=':', lw=1.2, label='ATM (K=S0)')
    plt.xlabel('Moneyness (K / S0)')
    plt.ylabel('Implied Volatility (%)')
    plt.title('Volatility Smile: VG Model vs BSM Flat Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/volatility_smile.png', dpi=150)
    plt.show()
    print("Volatility smile saved to results/volatility_smile.png")


def plot_parameter_sensitivity(S0=100, r=0.05, T=1.0, sigma=0.2):
    """
    Show how nu (kurtosis) and theta (skewness) shape the vol smile.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Effect of kurtosis (nu)
    ax = axes[0]
    for nu in [0.05, 0.10, 0.20, 0.40]:
        strikes, ivols = compute_vol_smile(S0, r, T, sigma, nu, theta=-0.1)
        valid = ~np.isnan(ivols)
        ax.plot(strikes[valid] / S0, ivols[valid] * 100, lw=2, label=f'ν = {nu}')
    ax.axhline(y=sigma * 100, color='black', linestyle='--', lw=1.2, label='BSM flat')
    ax.set_xlabel('Moneyness (K/S0)')
    ax.set_ylabel('Implied Vol (%)')
    ax.set_title('Effect of Kurtosis (ν) on Vol Smile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Effect of skewness (theta)
    ax = axes[1]
    for theta in [0.05, 0.0, -0.10, -0.20]:
        strikes, ivols = compute_vol_smile(S0, r, T, sigma, nu=0.1, theta=theta)
        valid = ~np.isnan(ivols)
        ax.plot(strikes[valid] / S0, ivols[valid] * 100, lw=2, label=f'θ = {theta}')
    ax.axhline(y=sigma * 100, color='black', linestyle='--', lw=1.2, label='BSM flat')
    ax.set_xlabel('Moneyness (K/S0)')
    ax.set_ylabel('Implied Vol (%)')
    ax.set_title('Effect of Skewness (θ) on Vol Smile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/parameter_sensitivity.png', dpi=150)
    plt.show()
    print("Parameter sensitivity saved to results/parameter_sensitivity.png")


if __name__ == "__main__":
    plot_vol_smile()
    plot_parameter_sensitivity()
