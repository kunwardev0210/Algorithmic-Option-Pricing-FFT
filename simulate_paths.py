"""
Simulate underlying asset price paths under geometric Brownian motion,
derived via Itô's Lemma, for Monte Carlo risk evaluation.
"""
import numpy as np


def simulate_gbm_paths(S0: float, T: float, r: float, sigma: float,
                        n_paths: int = 10_000, n_steps: int = 252,
                        seed: int | None = None) -> np.ndarray:
    """
    Simulate asset price paths under geometric Brownian motion:
        dS_t = r * S_t * dt + sigma * S_t * dW_t

    Parameters
    ----------
    S0 : float       Initial asset price
    T  : float       Time horizon (years)
    r  : float       Risk-free / drift rate
    sigma : float    Volatility
    n_paths : int    Number of simulated trajectories
    n_steps : int    Number of time steps per path
    seed : int       Optional random seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps + 1) containing simulated price paths
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))

    log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])

    paths = S0 * np.exp(log_paths)
    return paths
