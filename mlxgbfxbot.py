import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import datetime
import requests  # For Telegram messages
import time      # For continuous operation (sleep)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##########################################
# ATR CALCULATION FUNCTION
# (Used for Stop-Loss and Take-Profit)
##########################################
def compute_atr(data, window=14):
    """
    Calculates the Average True Range (ATR) for the given DataFrame.
    Expects columns: 'High', 'Low', 'Close'.
    Returns a pandas Series of ATR values.
    """
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = (data['High'] - data['Close'].shift(1)).abs()
    data['L-PC'] = (data['Low']  - data['Close'].shift(1)).abs()

    tr = data[['H-L','H-PC','L-PC']].max(axis=1)
    atr = tr.rolling(window).mean()

    # Clean up temporary columns
    data.drop(['H-L','H-PC','L-PC'], axis=1, inplace=True)
    return atr


##########################################
# TELEGRAM NOTIFICATION (Hard-coded)
##########################################
TELEGRAM_BOT_TOKEN = "7847987861:AAHrr1mAnYYyUK5bfrJyyc5GmggsWCEVJFw"
TELEGRAM_CHAT_ID   = "7749762504"

def send_telegram_message(message):
    """
    Sends a text message to the specified Telegram chat.
    Update TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID with valid details.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Telegram error: {response.text}")
    except Exception as e:
        print(f"Telegram send failed: {str(e)}")


##########################################
# LIST OF CURRENCY PAIRS
##########################################
pairs = [
    "GBPJPY=X",  # GBP/JPY
    "EURUSD=X",  # EUR/USD
    "USDCAD=X",  # USD/CAD
    "USDJPY=X",  # USD/JPY
    "GBPUSD=X",  # GBP/USD
    "EURGBP=X"   # EUR/GBP
]

start_date = "2018-01-01"
end_date = "2023-01-01"

feature_cols = ['Daily_Return', 'SMA_diff']

##########################################
# DICTIONARIES TO STORE MODELS & DATA
##########################################
models = {}
threshold = 0.7  # Probability threshold for a strong signal


########################################################################
# 1) TRAIN A SEPARATE MODEL FOR EACH PAIR
########################################################################
for ticker in pairs:
    print(f"\n=== TRAINING MODEL FOR {ticker} ===")
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    if len(df.columns) == 5:
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

    if df.empty:
        print(f"No data for {ticker}, skipping.")
        continue

    print(df.head())

    df.dropna(inplace=True)
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    df.dropna(inplace=True)

    df['Daily_Return'] = df['Close'].pct_change()
    df['Close_SMA_5'] = df['Close'].rolling(window=5).mean()
    df['Close_SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_diff'] = df['Close_SMA_5'] - df['Close_SMA_10']
    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df['Target']

    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for {ticker}:", accuracy)

    test_signals = X_test.copy()
    test_signals['Predicted'] = y_pred
    test_signals['Signal'] = test_signals['Predicted'].apply(lambda x: "Buy" if x == 1 else "Sell")
    print(test_signals.tail(10))

    models[ticker] = model

print("\n=== TRAINING COMPLETE FOR ALL PAIRS ===")


########################################################################
# 2) CONTINUOUS LOOP FOR LIVE CHECKS ON EACH PAIR
########################################################################
print("Starting continuous loop for live signals...")
while True:
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    for ticker in pairs:
        model = models.get(ticker, None)
        if model is None:
            continue

        df_live = yf.download(ticker, start=end_date, end=today, interval="1d")
        if df_live.empty:
            print(f"No live data for {ticker} right now.")
            continue

        if len(df_live.columns) == 5:
            df_live.columns = ["Open", "High", "Low", "Close", "Volume"]

        df_live.columns = df_live.columns.str.strip()

        df_live['Daily_Return'] = df_live['Close'].pct_change()
        df_live['Close_SMA_5'] = df_live['Close'].rolling(window=5).mean()
        df_live['Close_SMA_10'] = df_live['Close'].rolling(window=10).mean()
        df_live['SMA_diff'] = df_live['Close_SMA_5'] - df_live['Close_SMA_10']
        df_live.dropna(inplace=True)

        if df_live.empty:
            print(f"No usable rows after feature calculation for {ticker}.")
            continue

        df_live['ATR'] = compute_atr(df_live, window=14)
        df_live.dropna(inplace=True)

        if df_live.empty:
            print(f"No usable rows after ATR for {ticker}.")
            continue

        X_live = df_live[feature_cols].iloc[-1:].copy()

        live_probs = model.predict_proba(X_live)[0]
        prob_of_buy  = live_probs[1]
        prob_of_sell = live_probs[0]

        if prob_of_buy > threshold:
            last_row = df_live.iloc[-1]
            current_price = last_row['Close']
            current_atr   = last_row['ATR']

            # ATR-based 3:1 reward:risk (reduced factors)
            risk_factor   = 0.3  # <-- CHANGED
            reward_factor = 0.9  # <-- CHANGED

            stop_loss   = current_price - (risk_factor * current_atr)
            take_profit = current_price + (reward_factor * current_atr)

            msg = (
                f"{ticker}/BUY/{round(current_price,4)} "
                f"SL:{round(stop_loss,4)} "
                f"TP:{round(take_profit,4)} "
                f"[Prob_Buy={prob_of_buy:.2f}]"
            )
            print("Live Signal:", msg)
            send_telegram_message(msg)

        elif prob_of_sell > threshold:
            last_row = df_live.iloc[-1]
            current_price = last_row['Close']
            current_atr   = last_row['ATR']

            # ATR-based 3:1 reward:risk (reduced factors)
            risk_factor   = 0.3  # <-- CHANGED
            reward_factor = 0.9  # <-- CHANGED

            stop_loss   = current_price + (risk_factor * current_atr)
            take_profit = current_price - (reward_factor * current_atr)

            msg = (
                f"{ticker}/SELL/{round(current_price,4)} "
                f"SL:{round(stop_loss,4)} "
                f"TP:{round(take_profit,4)} "
                f"[Prob_Sell={prob_of_sell:.2f}]"
            )
            print("Live Signal:", msg)
            send_telegram_message(msg)
        else:
            print(
                f"{ticker}: No strong signal. Confidence too low. "
                f"(Buy={prob_of_buy:.2f}, Sell={prob_of_sell:.2f})"
            )

    print("Waiting 5 minutes before next check...\n")
    time.sleep(300)  # 300 seconds
