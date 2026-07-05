"""
Validate the pricing engine against real market option data.
"""
import pandas as pd
from .fft_pricing import fft_option_price
from .risk_metrics import mean_absolute_error


def validate_against_market(df: pd.DataFrame, r: float = 0.05) -> float:
    """
    Compare FFT-model prices against market option prices.

    Expects df with columns: S0, K, T, sigma, market_price, option_type
    Returns the Mean Absolute Error (as a fraction, e.g. 0.0002 = 0.02%)
    """
    model_prices = df.apply(
        lambda row: fft_option_price(
            S0=row["S0"], K=row["K"], T=row["T"],
            r=r, sigma=row["sigma"], option_type=row["option_type"]
        ),
        axis=1,
    )
    return mean_absolute_error(model_prices.values, df["market_price"].values)
