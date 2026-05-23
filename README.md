# Bull & Bear Market Detection Using Hidden Markov Models

**Bachelor's Thesis – Statistics Program, Data Analysis & Business Intelligence**
**Örebro University, 2026**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Questions](#research-questions)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Results](#results)
8. [Key Findings](#key-findings)
9. [Limitations](#limitations)
10. [Technologies Used](#technologies-used)

---

## Project Overview

This thesis investigates how well **Hidden Markov Models (HMMs)** can automatically detect **bull and bear market regimes** in the **S&P 500** index. The study evaluates HMM as an unsupervised statistical approach for regime detection and compares four model configurations to identify which combination of features and state space best captures market dynamics.

Bull and bear markets represent structurally different economic environments — detecting them accurately has implications for investment strategy, risk management, and portfolio allocation. Rather than relying on manual observation or lagging rule-based indicators, this study explores whether a probabilistic latent-variable model can reliably identify these regimes in real time.

The analysis spans approximately **26 years of daily price data (2000–2025)**, sourced from Yahoo Finance.

---

## Research Questions

### Main Question

> *Hur väl kan Hidden Markov Models identifiera bull- och bear-marknader i S&P 500?*
>
> *(How well can Hidden Markov Models identify bull and bear markets in the S&P 500?)*

### Sub-Questions

| # | Research Question | Focus |
|---|-------------------|-------|
| **RQ1** | How does the number of states (2 vs 3) affect HMM performance? | Model complexity |
| **RQ2** | Do engineered features (volatility, volume) improve HMM regime detection over raw returns alone? | Feature engineering |

---

## Dataset

### Source

| Asset | Ticker | Source | Period | Trading Days |
|-------|--------|--------|--------|--------------|
| S&P 500 | `^GSPC` | Yahoo Finance | 2000-03-29 → 2025-12-31 | 6,479 |

### Bull/Bear Labeling (Ground Truth)

Ground truth labels are generated using the **Lunde-Timmermann (2004) algorithm** — a rule-based, non-parametric approach that identifies local peaks and troughs:

| Threshold | Value |
|-----------|-------|
| Bear (drop from peak) | −10% |
| Bull (rise from trough) | +15% |

**Regime distribution:**

| Class | Days | Percentage |
|-------|------|------------|
| Bull | 4,770 | 73.6% |
| Bear | 1,709 | 26.4% |

Empirical mean daily return per regime:
- Bull-days: **+0.056%**
- Bear-days: **−0.040%**

---

## Methodology

### Features

Two feature sets are evaluated:

| Set | Features | Count |
|-----|----------|-------|
| **RAW** | `Return` | 1 |
| **ENG** | `Return`, `Volatility_60`, `Volume_ratio` | 3 |

- **Volatility_60** — 60-day rolling standard deviation of daily returns
- **Volume_ratio** — daily volume relative to its 50-day moving average

All features are standardized with `StandardScaler` before HMM training.

### HMM Configurations

Four Gaussian HMM models are trained:

| Version | Features | States |
|---------|----------|--------|
| RAW-2 | Return | 2 |
| RAW-3 | Return | 3 |
| ENG-2 | Return, Volatility_60, Volume_ratio | 2 |
| ENG-3 | Return, Volatility_60, Volume_ratio | 3 |

Training uses the **Baum-Welch algorithm** (EM variant) via `hmmlearn`.

### Inference — Filtering (no look-ahead bias)

We use the **forward algorithm** for filtered state inference:

```
P(s_t | y_1:t) = α_t / Σ_j α_t(j)
```

This uses only historical data up to time t — unlike Viterbi/smoothing which uses the entire sequence. Filtering reflects realistic real-time deployment of the regime detector.

### Online Persistence Filter

To reduce noise from daily probability fluctuations, an **online persistence filter** is applied: a regime change is only confirmed if the new state has been the argmax for at least 30 consecutive days. This is still online (no look-ahead).

### State Mapping — Rank-based

States are mapped to regime labels by rank of estimated mean return:

| Model | Mapping |
|---|---|
| 2-state | Lowest μ → Bear, Highest μ → Bull |
| 3-state | Lowest μ → Bear, Middle μ → Neutral, Highest μ → Bull |

For 3-state models, **Neutral is kept as its own category** (not force-mapped to Bull) and excluded from accuracy/F1 evaluation. Coverage is reported separately.

### Evaluation

- **Coverage** — share of days with definitive classification (Bull or Bear)
- **Accuracy, F1, Bear Recall** — computed on days with classification
- **Log-likelihood, AIC, BIC** — objective unsupervised model comparison

---

## Project Structure

```
bull-bear-thesis/
├── data/                              # Raw and processed data
│   ├── sp500.csv                      # Raw OHLCV from Yahoo Finance
│   ├── sp500_labeled.csv              # With LT bull/bear labels
│   ├── sp500_features.csv             # With engineered features
│   └── sp500_hmm_v2.csv               # With HMM predictions
├── notebooks/
│   ├── 01_data_collection.ipynb       # Yahoo Finance data download
│   ├── 02_labeling.ipynb              # Lunde-Timmermann labeling
│   ├── 03_feature_engineering.ipynb   # Volatility, volume, returns
│   ├── 04_hmm_filtered.ipynb          # HMM training, filtering, evaluation
│   └── 05_results.ipynb               # Final results & comparison
├── results/                           # Plots and summary tables
└── README.md
```

---

## Installation & Setup

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn hmmlearn scipy yfinance jupyter
```

### Run Order

Notebooks should be executed in numerical order:

```bash
jupyter notebook notebooks/01_data_collection.ipynb
jupyter notebook notebooks/02_labeling.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
jupyter notebook notebooks/04_hmm_filtered.ipynb
jupyter notebook notebooks/05_results.ipynb
```

Each notebook reads its input from `data/` and writes its output to `data/` and `results/`.

---

## Results

### Main Performance Table

| Model | Coverage | Accuracy | F1 | Bear Recall |
|-------|----------|----------|-----|-------------|
| HMM RAW-2 | 100.0% | 0.780 | 0.782 | 0.616 |
| HMM RAW-3 | 54.7% | 0.889 | 0.884 | 0.585 |
| HMM ENG-2 | 100.0% | 0.783 | 0.793 | **0.819** |
| HMM ENG-3 | 46.3% | 0.501 | 0.474 | 0.723 |

### Information Criteria

| Model | log L | AIC | BIC |
|-------|-------|-----|-----|
| HMM RAW-2 | −7,849.2 | 15,712.3 | 15,759.8 |
| HMM RAW-3 | −7,556.8 | **15,141.6** | **15,236.4** |
| HMM ENG-2 | −13,228.5 | 26,482.9 | 26,571.0 |
| HMM ENG-3 | −10,824.2 | 21,694.4 | 21,850.3 |

### Best Model — Two Perspectives

| Perspective | Winner | Reason |
|---|---|---|
| **Practical** (matching LT labels) | ENG-2 | Highest F1 × Coverage = 0.793; highest Bear Recall (0.819) |
| **Statistical** (explaining observed data) | RAW-3 | Lowest AIC and BIC — most parsimonious model fit |

---

## Key Findings

1. **HMM successfully identifies bull/bear regimes without ground truth labels** — the model finds economically meaningful regimes using only price data.

2. **Engineered features improve Bear Recall significantly** — ENG-2 achieves 0.819 Bear Recall versus 0.616 for RAW-2, demonstrating that volatility and volume help the model identify downturns.

3. **Leverage effect confirmed** — the bear regime in HMM RAW-2 has approximately **2.8× higher volatility** than the bull regime, matching the well-documented finance phenomenon (Black, 1976).

4. **Statistical vs supervised divergence** — RAW-3 wins on AIC/BIC (statistically optimal) but ENG-2 wins on F1 × Coverage (best match to LT labels). This is a classic unsupervised-vs-supervised tension.

5. **Persistent regimes** — diagonal values of 0.96–0.99 in transition matrices confirm that markets remain in the same regime with >95% probability the next day. Bull and Bear states never transition directly to each other in 3-state models — always through Neutral.

---

## Limitations

- **Only S&P 500 is analyzed** — generalization to other asset classes (cryptocurrencies, commodities, emerging markets) is not evaluated.
- **Ground truth depends on Lunde-Timmermann thresholds** — different threshold choices would yield different labels.
- **Gaussian emission assumption** — HMM assumes returns within each regime follow a multivariate normal distribution. Real return distributions exhibit fat tails.
- **No transaction costs or portfolio implementation** — this is a regime detection study, not a trading strategy backtest.

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `hmmlearn` | Gaussian Hidden Markov Model |
| `scikit-learn` | StandardScaler, metrics |
| `scipy.stats` | Multivariate normal log-pdf for forward algorithm |
| `matplotlib`, `seaborn` | Visualization |
| `yfinance` | Yahoo Finance data |
| `jupyter` | Notebook environment |

---

## License

Academic project — Örebro University, 2026.
