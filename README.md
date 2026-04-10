# Bull & Bear Market Detection Using Statistical Models and Machine Learning

**Bachelor's Thesis – Statistics Program, Data Analysis & Business Intelligence**  
**Örebro University, 2026**  
**Authors: Viktor Emilsson & Loran Ali**

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

This project investigates the ability of statistical models and machine learning algorithms to automatically detect **bull and bear market regimes** across three distinct asset classes: **S&P 500**, **Bitcoin**, and **Gold**. The study combines classical statistical time series methods with modern supervised machine learning and deep learning in a multi-layered comparative framework.

Bull and bear markets represent structurally different economic environments — detecting them accurately has significant implications for investment strategy, risk management, and portfolio allocation. Rather than relying on manual observation or lagging indicators, this study explores whether data-driven models can reliably identify these regimes.

The analysis spans approximately **12 years of daily price data (2014–2025)**, sourced from Yahoo Finance, and evaluates five distinct modeling approaches ranging from unsupervised statistical models to deep learning architectures.

---

## Research Questions

### Main Question
> *"Hur väl kan statistiska modeller och maskininlärning identifiera bull- och bear-marknader i S&P 500, Bitcoin och Guld?"*
> 
> *(How well can statistical models and machine learning identify bull and bear markets in S&P 500, Bitcoin, and Gold?)*

### Sub-Questions

| # | Research Question | Focus |
|---|-------------------|-------|
| **RQ1** | How well does the Hidden Markov Model (HMM) identify bull/bear regimes compared to the 20% rule baseline? | Statistical vs. rule-based |
| **RQ2** | Does boosting (XGBoost) outperform bagging (Random Forest) for market regime classification? | Bagging vs. boosting |
| **RQ3** | Does the best-performing model trained on S&P 500 generalize to Bitcoin and Gold? | Cross-asset generalization |

---

## Dataset

### Sources & Assets

| Asset | Ticker | Source | Period | Trading Days |
|-------|--------|--------|--------|--------------|
| S&P 500 | `^GSPC` | Yahoo Finance | 2014–2025 | ~2,764 |
| Bitcoin | `BTC-USD` | Yahoo Finance | 2014–2025 | ~4,124 |
| Gold | `GC=F` | Yahoo Finance | 2014–2025 | ~2,764 |

### Bull/Bear Labeling (Ground Truth)

The 20% rule with asset-specific thresholds and minimum duration requirements is used to generate ground truth labels:

| Asset | Bear Threshold | Bull Threshold | Min. Duration |
|-------|---------------|---------------|---------------|
| S&P 500 | -20% from peak | +20% from trough | 60 days |
| Bitcoin | -60% from peak | +60% from trough | 90 days |
| Gold | -20% from peak | +20% from trough | 60 days |

Bitcoin uses a higher threshold because 20% swings are common noise in cryptocurrency markets — historical bear markets have seen drawdowns of 70–86%.

**Regime distribution after labeling:**

| Asset | Bull Days | Bear Days | Bull % |
|-------|-----------|-----------|--------|
| S&P 500 | 2,475 | 289 | 89.5% |
| Bitcoin | 2,085 | 580 | 78.2% |
| Gold | 2,489 | 275 | 90.1% |

---

## Methodology

### Feature Engineering

A total of **30 features per asset** were engineered across five categories:

| Category | Features | Count |
|----------|----------|-------|
| **Trend** | Price/MA20, Price/MA50, Price/MA200, Golden Cross, 52-week position | 5 |
| **Momentum** | RSI14, RSI7, MACD, MACD Signal, MACD Hist, Return lags (1/2/3/5/10d), Cumulative returns (5/10/20d) | 13 |
| **Volatility** | Vol 10/20/60d, Bollinger Band width & position, ATR14%, Volatility ratio | 7 |
| **Volume** | Relative volume, OBV MA20 | 2 |
| **Cross-asset Correlation** | Rolling 60d correlation: S&P500↔BTC, S&P500↔Gold, BTC↔Gold | 3 |

The cross-asset correlation features are a **unique contribution** of this study — motivated by the finding that all pairwise correlations increase significantly during bear markets (correlation breakdown phenomenon):

| Pair | Full Period | Bull Market | Bear Market |
|------|------------|-------------|-------------|
| S&P 500 vs Bitcoin | 0.23 | 0.16 | **0.53** |
| S&P 500 vs Gold | 0.03 | -0.05 | **0.30** |
| Bitcoin vs Gold | 0.09 | 0.05 | **0.29** |

### Models

#### 1. Hidden Markov Model (HMM)
- **Type:** Unsupervised statistical model
- **Library:** `hmmlearn`
- **Input features:** Daily return, 20-day volatility, Price/MA200 ratio
- **Variants:** 2-state (bull/bear) and 3-state (bull/neutral/bear)
- **Key aspect:** No labels used during training — purely data-driven regime discovery

#### 2. Random Forest (Bagging)
- **Type:** Supervised ensemble, bagging
- **Library:** `scikit-learn`
- **Parameters:** n_estimators=200, max_depth=6, class_weight='balanced'
- **n_estimators** justified via OOB (Out-of-Bag) analysis showing convergence at 200 trees

#### 3. XGBoost (Boosting)
- **Type:** Supervised ensemble, boosting
- **Library:** `xgboost`
- **Parameters:** n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight adjusted per asset
- **Key advantage:** Sequential learning from misclassified examples

#### 4. LSTM (Long Short-Term Memory)
- **Type:** Deep learning, recurrent neural network
- **Library:** `TensorFlow / Keras`
- **Architecture:** LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(16) → Dense(1, sigmoid)
- **Sequence length:** 60 trading days
- **Training:** EarlyStopping (patience=15), ReduceLROnPlateau, class_weight balancing

### Train/Test Split

All supervised models use a **temporal 70/30 split** to prevent data leakage:

| Asset | Training Period | Test Period |
|-------|----------------|-------------|
| S&P 500 | 2014–2022 | Sep 2022–Dec 2025 |
| Bitcoin | 2015–2022 | Oct 2022–Dec 2025 |
| Gold | 2014–2022 | Sep 2022–Dec 2025 |

---

## Project Structure

```
bull_bear_thesis/
│
├── notebooks/
│   ├── 01_data_collection.ipynb        # Download OHLCV data from Yahoo Finance
│   ├── 02_labeling.ipynb               # Apply 20% rule to generate bull/bear labels
│   ├── 03_feature_engineering.ipynb    # Compute 30 features per asset
│   ├── 04_hmm.ipynb                    # Hidden Markov Model (2 and 3 states)
│   ├── 05_random_forest_xgboost.ipynb  # RF vs XGBoost + OOB analysis + generalization
│   ├── 06_lstm.ipynb                   # LSTM deep learning model
│   └── 07_results.ipynb               # Full comparison + RQ answers
│
├── data/
│   ├── sp500.csv                       # Raw price data
│   ├── bitcoin.csv
│   ├── gold.csv
│   ├── sp500_labeled.csv               # With bull/bear regime labels
│   ├── bitcoin_labeled.csv
│   ├── gold_labeled.csv
│   ├── sp500_features.csv              # With 30 engineered features
│   ├── bitcoin_features.csv
│   ├── gold_features.csv
│   ├── sp500_hmm.csv                   # With HMM predictions
│   ├── bitcoin_hmm.csv
│   ├── gold_hmm.csv
│   ├── sp500_ml.csv                    # With RF/XGBoost predictions
│   ├── bitcoin_ml.csv
│   ├── gold_ml.csv
│   ├── sp500_lstm.csv                  # With LSTM predictions
│   ├── bitcoin_lstm.csv
│   ├── gold_lstm.csv
│   └── combined_features.csv
│
├── results/
│   ├── 03_correlations.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_hmm_best.png
│   ├── 04_hmm_transitions.png
│   ├── 05_oob_analysis.png
│   ├── 05_feature_importance.png
│   ├── 05_predictions.png
│   ├── 06_lstm_history.png
│   ├── 06_lstm_predictions.png
│   ├── 07_accuracy_comparison.png
│   ├── 07_f1_auc_heatmap.png
│   └── 07_summary_table.csv
│
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.11
- Conda (recommended)

### Environment Setup

```bash
# Create conda environment
conda create -n thesis python=3.11
conda activate thesis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
pip install yfinance xgboost hmmlearn tensorflow
pip install jupyterlab ipykernel

# Register kernel
python -m ipykernel install --user --name thesis --display-name "Thesis (Python 3.11)"
```

### Running the Notebooks

Run notebooks in order:

```bash
jupyter lab
```

1. `01_data_collection.ipynb` — downloads and saves raw data
2. `02_labeling.ipynb` — generates bull/bear labels
3. `03_feature_engineering.ipynb` — computes all features
4. `04_hmm.ipynb` — trains and evaluates HMM
5. `05_random_forest_xgboost.ipynb` — trains RF and XGBoost
6. `06_lstm.ipynb` — trains LSTM
7. `07_results.ipynb` — aggregates and compares all results

---

## Results

### Full Performance Summary

| Asset | Model | Accuracy | F1-score | ROC-AUC | Type |
|-------|-------|----------|----------|---------|------|
| **S&P 500** | HMM (best) | 0.790 | 0.827 | – | Statistical |
| **S&P 500** | Random Forest | 0.880 | 0.863 | 0.771 | Bagging |
| **S&P 500** | XGBoost ★ | **0.916** | **0.908** | 0.753 | Boosting |
| **S&P 500** | LSTM | 0.786 | 0.739 | 0.842 | Deep Learning |
| **Bitcoin** | HMM (best) | 0.775 | 0.790 | – | Statistical |
| **Bitcoin** | Random Forest | 0.914 | 0.912 | 0.942 | Bagging |
| **Bitcoin** | XGBoost | 0.921 | 0.918 | 0.939 | Boosting |
| **Bitcoin** | LSTM ★ | **0.955** | **0.951** | 0.933 | Deep Learning |
| **Gold** | HMM (best) | 0.719 | 0.774 | – | Statistical |
| **Gold** | Random Forest | 0.857 | 0.790 | 0.667 | Bagging |
| **Gold** | XGBoost | 0.857 | 0.790 | 0.701 | Boosting |
| **Gold** | LSTM | 0.857 | 0.790 | 0.850 | Deep Learning |

### Overall Model Ranking (Average F1 across all assets)

| Rank | Model | Average F1 |
|------|-------|-----------|
| 1 | XGBoost | 0.872 |
| 2 | Random Forest | 0.855 |
| 3 | LSTM | 0.827 |
| 4 | HMM (3-state) | 0.807 |
| 5 | HMM (2-state) | 0.579 |

### Cross-Asset Generalization (Trained on S&P 500)

| Target Asset | Model | Asset-Specific Acc | Generalized Acc | Change |
|---|---|---|---|---|
| Bitcoin | Random Forest | 0.914 | 0.882 | -0.032 |
| Bitcoin | XGBoost | 0.921 | 0.906 | -0.015 |
| Gold | Random Forest | 0.857 | 0.858 | +0.001 |
| **Gold** | **XGBoost** | 0.857 | **0.941** | **+0.084** |

---

## Key Findings

### RQ1 — HMM vs. 20% Rule
HMM achieves reasonable performance for S&P 500 (Acc: 0.790) without seeing any labels during training. However, the **3-state model significantly outperforms the 2-state model** for Bitcoin (+45.7% accuracy) and Gold (+24.5%), revealing that a neutral/sideways accumulation state is genuinely informative — particularly for Bitcoin's halving cycles. All supervised ML models outperform HMM consistently.

### RQ2 — Bagging vs. Boosting
**XGBoost (boosting) outperforms Random Forest (bagging)** for S&P 500 (+3.6%) and Bitcoin (+0.7%). Gold is a draw — both models fail to detect bear markets due to the lateral price movement characteristic of gold's bear periods. Across assets, XGBoost ranks first by average F1 (0.872 vs. 0.855).

### RQ3 — Cross-Asset Generalization
Results are mixed but revealing. XGBoost trained on S&P 500 **generalizes unexpectedly well to Gold** (Acc: 0.941 vs. 0.857 asset-specific) — suggesting that macroeconomic patterns driving S&P 500 regimes transfer to gold. Bitcoin, however, performs better with asset-specific training (0.921 vs. 0.906 generalized), reflecting its unique halving-driven cycles.

### The Gold Challenge
Gold's bear market in 2022–2023 was characterized by **lateral/sideways price movement** rather than a sharp decline. Technical indicators (RSI, MACD, volatility) do not signal bear conditions during sideways markets, causing all models to fail on this specific bear period. This challenges the 20% rule as a ground truth definition for gold and highlights the need for asset-specific regime definitions.

### Correlation Breakdown
A notable cross-asset finding: all pairwise correlations **increase sharply during bear markets** (e.g., S&P 500 vs. Bitcoin: 0.16 in bull → 0.53 in bear), consistent with the correlation breakdown phenomenon. This motivates the inclusion of rolling cross-asset correlations as features and supports the study's multi-asset approach.

---

## Limitations

- **Ground truth dependency:** The 20% rule is an industry convention, not a mathematical truth. Alternative thresholds may yield different regime definitions and results.
- **Look-ahead bias risk:** Feature engineering uses rolling windows anchored to past data only, but the temporal split must be strictly maintained in any future extension.
- **Gold bear detection:** The lateral nature of gold's bear markets may require domain-specific regime definitions beyond price-based thresholds.
- **LSTM overfitting:** Training history shows divergence between training and validation loss for S&P 500 and Gold, suggesting the LSTM architecture may benefit from further regularization or architecture search.
- **Single test period:** The 30% test window (2022–2025) covers one major bear market and one bull market. Results may vary across different market cycles.
- **No transaction costs:** Models are evaluated on classification accuracy only — real-world application would require accounting for trading costs and execution latency.

---

## Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.x | Data manipulation |
| `numpy` | 1.26.4 | Numerical computation |
| `yfinance` | latest | Yahoo Finance data download |
| `scikit-learn` | latest | Random Forest, preprocessing, metrics |
| `xgboost` | latest | XGBoost classifier |
| `hmmlearn` | latest | Hidden Markov Model |
| `tensorflow` | 2.16.2 | LSTM deep learning |
| `matplotlib` | latest | Visualization |
| `seaborn` | latest | Statistical visualization |

---

## License

This project is developed for academic purposes as part of a bachelor's thesis at Örebro University. All data is sourced from Yahoo Finance under their terms of service.
