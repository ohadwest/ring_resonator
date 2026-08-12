# ⭕ Ring Resonator Simulation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, interactive simulation and analysis engine for **Integrated Silicon Photonics Ring Resonators**, built with Python and Streamlit.

This tool provides a rapid numerical solver for extracting critical ring resonator metrics such as the Quality Factor ($Q_L$, $Q_i$, $Q_c$), Extinction Ratio ($E_r$), Free Spectral Range (FSR), and broad-spectrum transmission characteristics.

---

## ✨ Key Features

### 1. Single Resonance Analysis & Sweeps
* **Flexible Loss Inputs:** Define internal losses via Intrinsic $Q_i$, linear loss $\alpha$ (dB/cm), or loss per roundtrip (%). The engine converts these automatically.
* **Parameter Sweeps:** Sweep the coupling quality factor ($Q_c$) or internal quality factor ($Q_i$) to instantly visualize changes in the transmission spectrum.
* **Critical Coupling Identification:** Dynamically calculates and plots the Extinction Ratio ($E_r$) and Loaded Quality Factor ($Q_L$) to pinpoint the exact critical coupling condition ($Q_i = Q_c$).
* **Physics & Math Module:** Built-in LaTeX documentation explaining the analytical All-Pass configuration equations.

### 2. Full Spectrum Analysis
* **Broadband Transmission:** Simulates the transmission spectrum over a wide wavelength range ($\lambda_{\min}$ to $\lambda_{\max}$).
* **Dispersion-Aware Coupling:** Supports wavelength-dependent coupling coefficients $\kappa(\lambda)$ via a user-defined dispersion slope $d\kappa/d\lambda$.
* **Realistic Noise Modeling:** Option to inject orthogonal polarization noise (unresonant background) to mimic real-world measurement floors.
* **FSR Extraction:** Automatically calculates the theoretical Free Spectral Range (FSR) based on the group index ($n_g$) and cavity geometry.

---

## 🛠️ Installation & Local Setup

To run this simulation engine on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ring-resonator-simulator.git](https://github.com/your-username/ring-resonator-simulator.git)
   cd ring-resonator-simulator
