# ChipChainReaction : Full-Stack Quantitative Trading System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![Finance](https://img.shields.io/badge/Finance-Quantitative-lightgrey.svg)

**ChipChainReaction** is an end-to-end algorithmic trading system exploring causal transmission across the semiconductor supply chain:

**Commodities → Foundries → Fabless Designers → Tech Integrators**

The project combines classical econometrics, Machine Learning (HMM), Deep Learning (LSTM), and mathematical portfolio optimisation (Markowitz / Risk Parity) into an End-to-end quantitative research and portfolio optimisation system, distributed via a REST API and an interactive dashboard.

> **Predecessor of [ChipDiffusion](https://github.com/PierreRollin/chipdiffusion)** - which extends this work into continuous time using options pricing, stochastic calculus, and volatility arbitrage.

---

## Key Results

| Metric | Without LSTM | With LSTM Shield |
|---|---|---|
| Max Drawdown (Markowitz L/S) | **−32.00%** | **−17.28%** |
| Sharpe Ratio (Markowitz L/S) | 1.238 | **1.174** |

The LSTM is not an alpha generator, it is a **crash shield**. In strongly trending bull markets it creates an opportunity cost. But in high-risk strategies (long/short with 30% short exposure), it becomes critical: it blocks counter-trend short positions during drawdowns, cutting losses by more than half.

---

## Architecture : Decision pipeline

The decision system relies on the synergy of three distinct models:

### 1. Layer 1 - Macro Regime Detection (HMM)
A **Hidden Markov Model** (Baum-Welch / Forward algorithm) analyses global volatility (VIX) and interest rates (TNX) to dynamically detect the current market regime: *Bull, Crab, or Bear*.

- **Walk-Forward expanding window**: the model is retrained at each step on all available past data, no Viterbi global decoding to prevent look-ahead bias.
- **3-state Gaussian HMM**: each state has its own covariance matrix, used directly by the portfolio optimiser.

### 2. Layer 2 - Sector Crash Signal (LSTM)
A **CNN1D + LSTM** neural network reads 3D tensors (21-day rolling windows) of supply chain data to anticipate a crash in the "Fabless" sector (NVDA, AMD).

- **LSTM V5** selected after systematic comparison of 7 architectures (V1–V7) via Walk-Forward Validation on 4 rolling folds.
- V7 (Double LSTM + Multi-Head Attention) collapsed during the 2022 crash due to overfitting. V5 survived.
- Trained with strict **class weights** to penalise missed downturns, the costliest prediction error in practice.
- **Chip Fear Index**: proprietary engineered feature combining supply chain signals (SMIC spread, ASML/TSM ratio, VIX momentum).

### 3. Layer 3 - Constrained Portfolio Optimisation (Markowitz / Risk Parity)
A constrained optimiser (`scipy.optimize`, SLSQP) allocates capital using the covariance matrix of the current HMM regime, with LSTM "emergency brakes" applied when the model detects elevated crash risk.

Four strategies available:
- **Markowitz Long-Only** (cap 30% per asset)
- **Markowitz Long/Short** (cap ±30%, max 30% short exposure)
- **Risk Parity Long-Only**
- **Risk Parity Long/Short**

---

## Scientific Methodology

### Statistical validation (Notebook 02)
Before any modelling, the raw price series are validated:
- **ADF + KPSS tests**: non-stationarity confirmed on prices, stationarity on log-returns.
- **Ljung-Box test**: residual autocorrelation detected - motivating the use of temporal models (HMM, LSTM).
- **Log-returns** used throughout (not simple returns) for additivity and symmetry.

### LSTM model selection (Notebooks 07–12)
Seven architectures tested (V1–V7). Key finding: complexity does not imply robustness.

| Version | Architecture | Walk-Forward result |
|---|---|---|
| V1–V3 | Basic LSTM | Directional prediction fails |
| V4 | CNN1D + LSTM | First robust version |
| **V5** | CNN1D + 8-unit LSTM + class weights | **Selected - survives 2022 crash** |
| V6 | CNN1D + LSTM + attention | Marginal improvement, more fragile |
| V7 | Double LSTM + Multi-Head Attention | Collapses in 2022 - overfitting |

### Ablation test (Notebook 13)
The final Master Pipeline runs a systematic ablation: every strategy backtested **with** and **without** the LSTM shield, over the full period and on 4 Walk-Forward folds.

---

## Ticker Universe

```python
commodities_and_macro    = ['USO', 'CGW', 'PICK', '^VIX', '^TNX']
foundries_and_equipment  = ['ASML', 'TSM', '0981.HK']   # SMIC as China proxy
fabless_designers        = ['NVDA', 'AMD', 'INTC', 'AVGO']
integrators_and_ai       = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
inverse_hedge            = ['GLD', 'TLT']
```

18 tickers total. Data sourced from Yahoo Finance via `yfinance`, from 2010 to present. Missing values handled using the US market calendar (NVDA-anchored) with forward-fill for Asian market closures (0981.HK).

---

## Project Context

ChipChainReaction was born from reading *Chip War* (Chris Miller, 2022), which traces the geopolitical and industrial architecture of the global semiconductor supply chain. The book made the causal cascade between commodity prices, foundries, chip designers, and tech integrators both obvious and underexplored in quantitative finance.

The name reflects this: each link in the chain triggers a reaction in the next, a cascade that creates predictable, tradeable signals if observed with enough lead time.

---

## Stack

| Layer | Tools |
|---|---|
| Data Engineering | `pandas`, `numpy`, `yfinance` |
| Statistical Tests | `statsmodels` (ADF, KPSS, Ljung-Box) |
| ML / DL | `hmmlearn`, `TensorFlow/Keras`, `scikit-learn`, `keras-tuner` |
| Portfolio Optimisation | `scipy.optimize` (SLSQP) |
| API | `FastAPI`, `Uvicorn` |
| Dashboard | `Streamlit`, `plotly` |

---

## Installation & Launch

```bash
git clone https://github.com/[handle]/ChipChainReaction.git
cd ChipChainReaction
pip install -r requirements.txt
```

**Terminal 1 : FastAPI engine:**
```bash
uvicorn api.main:app --reload --port 8000
# Swagger docs: http://127.0.0.1:8000/docs
```

**Terminal 2 : Streamlit dashboard:**
```bash
streamlit run app.py
# http://localhost:8501
```

---

## Project Structure

```
ChipChainReaction/
├── api/
│   └── main.py               FastAPI routes + quantitative logic
├── data/
│   ├── api_data/             Pre-computed equity curves (for API)
│   ├── processed/            Cleaned data, features, indicators
│   ├── raw/                  Raw price data
│   ├── results/              LSTM version comparison results
│   └── tensors/              Keras models (.keras) and NumPy tensors (.npz)
├── notebooks/
│   ├── 01_data_harvesting.ipynb
│   ├── 02_statistical_tests.ipynb
│   ├── 03_custom_indicators.ipynb
│   ├── 04_market_regime_hmm.ipynb
│   ├── 05_adaptive_markowitz.ipynb
│   ├── 06_advanced_portfolio.ipynb
│   ├── 07_lstm_data_engineering.ipynb
│   ├── 08_lstm_training.ipynb
│   ├── 09_target_smoothing.ipynb
│   ├── 10_lstm_v2_fast_features.ipynb
│   ├── 11_lstm_v3_others_propositions.ipynb
│   ├── 12_lstm_v4_final_v.ipynb
│   └── 13_master_pipeline.ipynb
├── app.py                    Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Known Limitations

- **Look-ahead bias eliminated**: HMM uses expanding window Forward algorithm only (no Viterbi global decoding). LSTM uses strict Walk-Forward Validation.
- **No transaction costs**: the backtest does not model bid/ask spreads or brokerage fees. Real-world performance would be lower.
- **Data source**: yfinance data may have gaps and is not tick-level. Results are indicative, not production-ready.
- **LSTM signal is static**: the model requires re-running Notebook 13 to update predictions on new data.
- **0981.HK (SMIC)**: used as a China semiconductor proxy. Liquidity and data quality are lower than US-listed tickers.

---

*Personal project developed alongside an engineering degree at ISEP Paris (2022–2027),  
specialising in Business Intelligence & Data Science.*