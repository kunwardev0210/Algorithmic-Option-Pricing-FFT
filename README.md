# Algorithmic Option Pricing: Bridging Stochastic Calculus & Fourier Transforms

> A quantitative finance pricing engine implementing the **Carr-Madan (1999)** methodology to price European Call Options under the **Variance Gamma stochastic process**, benchmarked against the analytical **Black-Scholes-Merton** model — using the Fast Fourier Transform for O(N log N) computational efficiency.
## 📌 Motivation

Traditional option pricing (Black-Scholes-Merton) assumes constant volatility and normally distributed returns — assumptions that break down in real financial markets, which exhibit **skewness**, **excess kurtosis**, and **volatility smiles**.

Meanwhile, pricing options under more realistic models like the **Variance Gamma process** requires integrating a non-analytical Probability Density Function — computationally expensive when done naively.

This project addresses both problems:
1. **Better model:** Replace log-normal assumptions with the Variance Gamma process to capture real-world jump-diffusion behavior
2. **Faster computation:** Shift the pricing problem from the time domain to the **frequency domain** using FFT, enabling simultaneous pricing of an entire chain of option strikes in **O(N log N)** time

> *"We are not just predicting stock prices — we are calculating the cost of insurance for the entire financial system using the same mathematics we use to study heat and signal processing."*

---

## 🧠 Mathematical Foundation

### 1. From Stochastic Calculus to Option Pricing

The journey to the pricing engine starts with the mathematical foundations of how asset prices move:

| Concept | Description |
|---|---|
| **Brownian Motion** | Continuous-time stochastic process modeling random asset price movements |
| **Itô's Integral** | Stochastic integral differing from the classical Riemann integral due to the non-differentiability of Brownian paths |
| **Itô's Lemma** | The stochastic calculus chain rule — foundation of the BSM derivation |
| **Martingales** | Fair-game processes; asset prices under risk-neutral measure are martingales |
| **Quadratic Variation** | Key property of Brownian motion: $[W, W]_t = t$ |

### 2. Black-Scholes-Merton (BSM) Model

Derived from Itô's Lemma and the no-arbitrage condition, the **BSM PDE** is:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

The closed-form solution for a European Call Option:

$$C = S_0 N(d_1) - Ke^{-rT} N(d_2)$$

where:

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Limitation:** BSM assumes constant volatility $\sigma$ — this creates systematic mispricing for deep in/out-of-the-money options (the "volatility smile" problem).

### 3. Variance Gamma (VG) Process

The VG process models asset returns as a **Brownian motion with drift, time-changed by a Gamma process** — capturing:
- **Skewness** ($\theta$): asymmetric return distributions
- **Kurtosis** ($\nu$): fat tails and jump-like behavior
- More accurate pricing of out-of-the-money options

The **characteristic function** of the VG process (closed-form and explicit) is:

$$\phi_T^{VG}(u) = \left(1 - iu\theta\nu + \frac{\sigma^2\nu u^2}{2}\right)^{-T/\nu}$$

### 4. Carr-Madan FFT Methodology

Since the VG PDF is non-analytical, we cannot integrate it directly. Instead, we use the **characteristic function** $\phi_T(u)$ and price in the frequency domain:

**Step 1 — Damping:** The raw call payoff is not square-integrable. We apply exponential damping $\alpha$ to derive:

$$\psi_T(v) = \frac{e^{-rT}\phi_T(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$

**Step 2 — Discretization:** Using Simpson's Rule with integration step $\eta$ and log-strike grid spacing $\lambda$:

$$C_T(k_u) \approx \frac{e^{-\alpha k_u}}{\pi}\sum_{j=1}^{N}e^{-i\frac{2\pi}{N}(j-1)(u-1)}e^{ibv_j}\psi_T(v_j)\frac{\eta}{3}\left[3+(-1)^j-\delta_{j-1}\right]$$

**Step 3 — FFT Constraint:** For valid DFT alignment, grid parameters must satisfy the Nyquist condition:

$$\lambda\eta = \frac{2\pi}{N}$$

This structure maps perfectly onto a **Discrete Fourier Transform**, enabling the **FFT algorithm** to compute prices for all strikes simultaneously in $O(N \log N)$ time.

---

## ⚙️ Key Features

- **Variance Gamma Pricing Engine** — Captures real-world skewness ($\theta$) and kurtosis ($\nu$), neutralizing BSM's constant-volatility bias
- **Carr-Madan Damping** — Exponential damping factor $\alpha$ to handle non-square-integrability of raw call payoff
- **Hyperbolic Smoothing** — Modified $\sinh$ damping to eliminate high-frequency oscillations in short-maturity options ($T \to 0$)
- **Simpson's Rule Discretization** — Accurate numerical integration matching Nyquist grid constraints
- **BSM Benchmark** — Side-by-side comparison revealing the volatility smile discrepancy

---

## 🔑 BSM vs Variance Gamma — Key Differences

| Property | Black-Scholes-Merton | Variance Gamma |
|---|---|---|
| **Volatility** | Constant | Stochastic via Gamma time-change |
| **Return Distribution** | Log-normal | Skewed + fat-tailed |
| **Volatility Smile** | Cannot explain | Captures naturally |
| **Jump Behavior** | No jumps | Jump-like via time-change |
| **Computation** | Closed-form (instant) | FFT — O(N log N) |
| **Pricing Accuracy (OTM)** | Systematically underprices | More accurate |

---

## 📐 Risk Neutral Pricing

Beyond BSM, this project also covers the **Risk Neutral Pricing** framework as an alternative derivation:

- Pricing derivatives as discounted expected payoffs under the **risk-neutral measure** $\mathbb{Q}$
- **Sub and super hedging** bounds and their relationship to no-arbitrage pricing
- Why BSM limitations in real-world scenarios motivate more sophisticated processes like VG

---

## 🛠️ Tech Stack

| Component | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Core Numerics** | NumPy, SciPy (`scipy.fft.fft`, `scipy.stats.norm`) |
| **Data / Market** | Pandas, yfinance |
| **Visualization** | Matplotlib, Streamlit |
| **Environment** | Jupyter Notebook |

---

## 📁 Repository Structure

```
├── Pricing_Engine.py          # Main FFT-based VG pricing engine
├── notebooks/
│   ├── bsm_derivation.ipynb   # BSM PDE derivation from Itô's Lemma
│   ├── vg_process.ipynb       # Variance Gamma characteristic function
│   └── fft_pricing.ipynb      # Carr-Madan FFT implementation
├── results/
│   └── volatility_smile.png   # BSM vs VG comparison plot
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Princekaga/Pricing-Engine
cd Pricing-Engine

# Install dependencies
pip install numpy scipy pandas matplotlib yfinance streamlit

# Run the pricing engine
python Pricing_Engine.py

# Or launch the interactive dashboard
streamlit run Pricing_Engine.py
```

---

## 📊 Sample Results

By adjusting **Kurtosis ($\nu$)** and **Skewness ($\theta$)** parameters:

- Setting $\nu \to 0$, $\theta \to 0$ aligns VG back toward log-normal BSM behavior
- Setting $\theta < 0$ tilts the pricing curve, highlighting where BSM **systematically underprices** out-of-the-money options
- The resulting **volatility smile** plot directly visualizes the real-world pricing gap BSM cannot explain

---

## 📚 Learning Progression

This project followed a structured 10-week curriculum:

| Weeks | Topic |
|---|---|
| 1 | Probability foundations: sigma-algebra, random variables, martingales |
| 2 | Derivatives: types, arbitrage, intrinsic/time value of European options |
| 3–5 | Stochastic Calculus: Brownian motion, Itô's integral, Itô's Lemma |
| 6–7 | Black-Scholes-Merton PDE: derivation, terminal conditions, limitations |
| 8 | Risk Neutral Pricing: sub/super hedging, BSM via martingale approach |
| 9–10 | FFT in Finance: Fourier Transforms, Carr-Madan method, Python implementation |

---


---

*This project was conducted under Stamatics, IIT Kanpur as part of the Summer Project Program.*
