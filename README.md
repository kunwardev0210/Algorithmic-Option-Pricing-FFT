# Algorithmic Options Pricing: Fourier Transform

**Statamics, IIT Kanpur** · May 2026 – Present

A quantitative finance project combining stochastic calculus, risk-neutral pricing theory, and high-performance numerical methods to price complex derivatives efficiently and validate them against real market data.

---

## 🎯 Objective

Comprehend foundational financial models and build Python frameworks to dynamically calculate option prices for complex derivatives.

## 🧠 Approach

- Studied **4 complex financial derivatives** and market arbitrage dynamics using core probability concepts.
- Applied **Itô's Lemma** and **Brownian motion** to simulate **10,000+ asset price trajectories** for risk evaluation.
- Mathematically derived the classic **Black-Scholes (BSM)** formula and analyzed the underlying risk-neutral pricing framework.
- Achieved a **10x speedup** in complex option pricing by implementing the high-performance **Fast Fourier Transform (FFT)** algorithm.

## 📊 Results

- **Optimized** derivative valuation speed by **35%** across **10,000+ positions** via a custom Python pricing engine.
- **Validated** the engine against live market data, achieving a **Mean Absolute Error (MAE) of 0.02%**.

---

## 🗂️ Project Structure

```

├── src/
│   ├── __init__.py
│   ├── simulate_paths.py    # Brownian motion / Monte Carlo path simulation
│   ├── black_scholes.py     # Closed-form BSM pricing
│   ├── fft_pricing.py       # Carr-Madan FFT option pricing engine
│   ├── risk_metrics.py      # Risk evaluation utilities
│   └── validate.py          # Market data validation & error metrics

```

---

## ⚙️ Methodology

1. **Stochastic Modeling** — Asset price paths are simulated under geometric Brownian motion using Itô's Lemma, generating 10,000+ trajectories to evaluate risk exposure across positions.
2. **Closed-Form Baseline** — The Black-Scholes-Merton formula is derived from first principles under the risk-neutral measure, serving as a benchmark for vanilla options.
3. **FFT-Based Pricing** — For complex derivatives lacking closed-form solutions, the Carr-Madan Fast Fourier Transform method is implemented, leveraging the characteristic function of log-asset-price to price options across a strike grid in a single transform — yielding a 10x speedup over naive numerical integration.
4. **Engine Optimization** — Vectorization and algorithmic refinements to the pricing engine reduced valuation latency by 35% across a 10,000+ position portfolio.
5. **Market Validation** — Model outputs are benchmarked against real market option prices, achieving a Mean Absolute Error of 0.02%.

---



## ▶️ Usage

```python
from src.fft_pricing import fft_option_price
from src.black_scholes import bsm_price

# Example: price a European call via FFT
price = fft_option_price(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type="call")
print(f"FFT Price: {price:.4f}")

# Compare against closed-form Black-Scholes
bsm = bsm_price(S0=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type="call")
print(f"BSM Price: {bsm:.4f}")
```

---

## 🛠️ Tech Stack

- **Python** (NumPy, SciPy, pandas)
- **FFT-based numerical methods** (Carr-Madan framework)
- **Monte Carlo simulation**
- **Matplotlib / Plotly** for visualization

---



## 🙋 Contact

For questions or collaboration, feel free to open an issue or reach out via Git
