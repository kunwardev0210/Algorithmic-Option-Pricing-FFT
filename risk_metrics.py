"""
Risk evaluation utilities for simulated asset price paths / portfolios.
"""
import numpy as np


def value_at_risk(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Value at Risk (VaR) at the given confidence level."""
    return -np.percentile(pnl, 100 * (1 - confidence))


def conditional_value_at_risk(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) at the given confidence level."""
    var = value_at_risk(pnl, confidence)
    tail_losses = pnl[pnl <= -var]
    return -tail_losses.mean() if len(tail_losses) > 0 else var


def mean_absolute_error(model_prices: np.ndarray, market_prices: np.ndarray) -> float:
    """Mean Absolute Error between model and market prices (as a fraction)."""
    return np.mean(np.abs(model_prices - market_prices) / market_prices)
