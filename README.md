# FX ML Signals (Drafts Archive)

This repo collects many experimental versions of my FX direction/alert project from Dec 2024–Jan 2025. The goal across versions is the same: predict the next bar’s direction (intraday or daily), backtest in a walk-forward way that mirrors live trading, and optionally send alerts (e.g., Telegram).

## What it does (generally)
- Ingests OHLC/V data from CSV or `yfinance`.
- Builds features (returns, rolling stats, RSI/EMA/Bollinger, ATR, session flags, simple candle patterns).
- Trains a model, scores the next bar, and applies simple trade rules (filters, ATR-based exits).
- Runs day-by-day backtests that avoid look-ahead and match the live logic.
- (Optional) Sends alerts when a setup passes thresholds.

## Models I tried
Classic ML
- **XGBoost**, **LightGBM**, **CatBoost**
- **Random Forest**, **Logistic/Linear Regression**, **SVM** (a few drafts)
Neural networks
- **MLP** (scikit-learn / Keras)
- **LSTM** (Keras) on windowed sequences
Ensembles
- Simple **stacking/averaging** of multiple models (“mega model” drafts)
Other ideas
- **Multi-output regression** (predicting multiple horizons, then voting)
- **Calibrated probabilities** and confidence thresholds for fewer, higher-quality signals
- A **Telegram-bot** version for notifications

> Note: Many files explore one change at a time (features, window sizes, thresholds, filters). Not every draft is “good”—the value here is the progression.

## Data & labels
- Sources: CSV exports (broker data) or `yfinance`.
- Targets: next bar **UP/DOWN** (classification) or next-period **return** (regression).
- Timeframes: daily and intraday (e.g., 4H/2H/1H/30m/15m/5m).
- Pairs: mainly EURUSD, plus some GBPUSD/XAUUSD experiments.

## How I tried to keep it honest
- **Walk-forward** evaluation (time-based splits).
- **No look-ahead** in features or labels.
- **Session windows** (London/NY) and **news blackout** options in some drafts.
- **ATR-based** risk and simple, transparent rules.

## Repo layout (suggested)

src/ # current code you run now (pick your latest version)
archive/ # older drafts and one-off experiments
data/ # local CSVs (gitignored)
models/ # saved models (gitignored)


## Quickstart
1. Create a venv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt


2. Put input CSVs in data/ (or set up yfinance inside the script).

3. (Optional alerts) Copy .env.example → .env and fill tokens.

4. Run the latest script (example):
python src/run_pipeline.py

Notes & lessons

- Forward tests get closer to backtests when the pipeline is strictly time-aware and simple.

- Confidence thresholds + filters usually beat “always trade” rules.

- More features isn’t always better—clean data and matching live logic matter more.

Status:

This is a drafts archive. I keep the most reliable version in src/ and move older attempts to archive/. Use at your own risk—this is for learning and research.

Tech:

Python, NumPy, pandas, scikit-learn, XGBoost/LightGBM/CatBoost, Keras (for MLP/LSTM), python-dotenv, requests, (optional) Telegram Bot API.


If you want, tell me your actual entry script name and which folders you created—I’ll tailor the Quickstart and the repo layout section to match exactly.
::contentReference[oaicite:0]{index=0}
