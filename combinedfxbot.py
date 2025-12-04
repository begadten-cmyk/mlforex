import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import datetime
import requests  # For Telegram messages
import time      # For continuous operation (sleep)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# For the candle predictor
from river import linear_model, metrics

###############################################################################
# TELEGRAM BOT CONFIG
###############################################################################
TELEGRAM_BOT_TOKEN = "7847987861:AAHrr1mAnYYyUK5bfrJyyc5GmggsWCEVJFw"
TELEGRAM_CHAT_ID   = "7749762504"

def send_telegram_message(message):
    """
    Sends a text message to the specified Telegram chat.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Telegram error: {response.text}")
    except Exception as e:
        print(f"Telegram send failed: {str(e)}")

###############################################################################
# ATR Calculation (Used if you still want ATR references, not strictly needed)
###############################################################################
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

    data.drop(['H-L','H-PC','L-PC'], axis=1, inplace=True)
    return atr

###############################################################################
# CURRENCY PAIRS + SIGNAL MODEL CONFIG
###############################################################################
pairs = [
    "GBPJPY=X",
    "EURUSD=X",
    "USDCAD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "EURGBP=X"
]

start_date_signals = "2018-01-01"
end_date_signals   = "2023-01-01"

feature_cols_signals = ['Daily_Return', 'SMA_diff']
threshold = 0.7  # Probability threshold for a strong signal

# We'll store the XGBoost models here
xgb_models = {}

###############################################################################
# 1) TRAIN XGBOOST SIGNAL MODELS
###############################################################################
def train_signal_models():
    for ticker in pairs:
        print(f"\n=== TRAINING XGBOOST SIGNAL MODEL FOR {ticker} ===")
        df = yf.download(ticker, start=start_date_signals, end=end_date_signals, interval="1d")

        if len(df.columns) == 5:
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        df.dropna(inplace=True)

        # Create a simple up/down target
        df['Tomorrow_Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
        df.dropna(inplace=True)

        # Basic features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Close_SMA_5']  = df['Close'].rolling(window=5).mean()
        df['Close_SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_diff']     = df['Close_SMA_5'] - df['Close_SMA_10']
        df.dropna(inplace=True)

        X = df[feature_cols_signals]
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
        print(f"Test Accuracy for {ticker}: {accuracy:.4f}")

        xgb_models[ticker] = model
    print("\n=== XGBOOST SIGNAL MODELS TRAINING COMPLETE ===")

###############################################################################
# 2) TRAIN / PREDICTOR FOR NEXT-DAY HIGH/LOW (LINEAR REGRESSION)
###############################################################################
start_date_candle = "2010-01-01"
end_date_candle   = "2024-12-31"

FEATURE_COLS_CANDLE = ['Daily_Return', 'MA_5', 'MA_10', 'Volume']

# We'll store 3 incremental models (High, Low, Close) per pair
candle_models = {}
candle_metrics = {}  # track error if you want

def train_candle_predictor():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    for ticker in pairs:
        print(f"\n=== Training Candle Predictor for {ticker} ===")
        df = yf.download(ticker, start=start_date_candle, end=end_date_candle, interval="1d")
        if len(df.columns) == 5:
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        # Shift for tomorrow
        df['Tomorrow_High']  = df['High'].shift(-1)
        df['Tomorrow_Low']   = df['Low'].shift(-1)
        df.dropna(inplace=True)

        # Features
        df['Daily_Return'] = df['Close'].pct_change()
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df.dropna(inplace=True)

        if df.empty:
            print(f"No usable data for {ticker} (candle predictor).")
            continue

        # Create models if not exist
        if ticker not in candle_models:
            model_high  = linear_model.LinearRegression()
            model_low   = linear_model.LinearRegression()

            candle_models[ticker] = {
                "high":  model_high,
                "low":   model_low
            }
            mae_high = metrics.MAE()
            mae_low  = metrics.MAE()
            candle_metrics[ticker] = {
                "mae_high": mae_high,
                "mae_low":  mae_low
            }
        else:
            pass

        model_high = candle_models[ticker]["high"]
        model_low  = candle_models[ticker]["low"]
        mae_high   = candle_metrics[ticker]["mae_high"]
        mae_low    = candle_metrics[ticker]["mae_low"]

        # Day-by-day incremental training
        for i in range(len(df)-1):
            row = df.iloc[i]
            x_t = row[FEATURE_COLS_CANDLE].to_dict()

            true_h = row['Tomorrow_High']
            true_l = row['Tomorrow_Low']

            # Predict
            pred_h = model_high.predict_one(x_t) or 0.0
            pred_l = model_low.predict_one(x_t)  or 0.0

            mae_high.update(true_h, pred_h)
            mae_low.update(true_l,  pred_l)

            # Learn
            model_high.learn_one(x_t, true_h)
            model_low.learn_one(x_t,  true_l)

        print(f"MAE High: {mae_high.get():.4f}, MAE Low: {mae_low.get():.4f}")

    print("\n=== Candle Predictor Training Complete ===")

def predict_nextday_range(ticker):
    """
    Predict tomorrow's High, Low using partial-fit approach for 'today' data.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    from datetime import datetime
    today_str = datetime.today().strftime('%Y-%m-%d')
    df_live = yf.download(ticker, start=today_str, end=today_str, interval="1d")

    if len(df_live.columns) == 5:
        df_live.columns = ["Open", "High", "Low", "Close", "Volume"]
    df_live.dropna(inplace=True)

    if df_live.empty:
        return None, None

    df_live['Daily_Return'] = df_live['Close'].pct_change()
    df_live['MA_5'] = df_live['Close'].rolling(5).mean()
    df_live['MA_10'] = df_live['Close'].rolling(10).mean()
    df_live.dropna(inplace=True)

    if df_live.empty:
        return None, None

    row = df_live.iloc[-1]
    x_t = row[FEATURE_COLS_CANDLE].to_dict()

    model_high = candle_models[ticker]["high"]
    model_low  = candle_models[ticker]["low"]

    # True values
    true_h = row['High']
    true_l = row['Low']

    # partial fit for "today"
    pred_h = model_high.predict_one(x_t) or 0.0
    pred_l = model_low.predict_one(x_t)  or 0.0

    candle_metrics[ticker]["mae_high"].update(true_h, pred_h)
    candle_metrics[ticker]["mae_low"].update(true_l,  pred_l)

    model_high.learn_one(x_t, true_h)
    model_low.learn_one(x_t,  true_l)

    # Now the model is updated for "today", so predict "tomorrow"
    fut_h = model_high.predict_one(x_t) or 0.0
    fut_l = model_low.predict_one(x_t)  or 0.0

    return fut_h, fut_l  # Next-day predicted High, Low

###############################################################################
# MAIN: Combine the two
###############################################################################
if __name__ == "__main__":
    # 1) Train XGBoost for signals
    train_signal_models()

    # 2) Train the candle predictor (linear regression)
    train_candle_predictor()

    print("\n=== Starting combined logic loop... ===")
    while True:
        today = datetime.datetime.today().strftime('%Y-%m-%d')

        for ticker in pairs:
            model = xgb_models.get(ticker, None)
            if model is None:
                continue

            # 2a) SIGNAL (Buy/Sell) from XGBoost
            df_live = yf.download(ticker, start=end_date_signals, end=today, interval="1d")
            if df_live.empty:
                print(f"No live data for {ticker} (signals).")
                continue
            if len(df_live.columns) == 5:
                df_live.columns = ["Open", "High", "Low", "Close", "Volume"]
            df_live.dropna(inplace=True)
            if df_live.empty:
                print(f"No rows after signals cleaning for {ticker}.")
                continue

            df_live['Daily_Return'] = df_live['Close'].pct_change()
            df_live['Close_SMA_5']  = df_live['Close'].rolling(window=5).mean()
            df_live['Close_SMA_10'] = df_live['Close'].rolling(window=10).mean()
            df_live['SMA_diff']     = df_live['Close_SMA_5'] - df_live['Close_SMA_10']
            df_live.dropna(inplace=True)
            if df_live.empty:
                continue

            X_live = df_live[feature_cols_signals].iloc[-1:].copy()
            live_probs = model.predict_proba(X_live)[0]
            prob_of_buy  = live_probs[1]
            prob_of_sell = live_probs[0]

            # 2b) Candle predictor: get tomorrow's predicted High/Low
            fut_h, fut_l = predict_nextday_range(ticker)  # None if no data
            if fut_h is None or fut_l is None:
                print(f"{ticker}: Can't get predicted High/Low.")
                continue

            # Current price
            last_row = df_live.iloc[-1]
            current_price = last_row['Close']

            # 3) Decide SL & TP with compromise logic:
            #    1:3 RR if possible, but clamp to predicted High/Low if needed.

            # Example buffer
            buffer_pips = 0.001  # minimal difference

            # If buy:
            if prob_of_buy > threshold and prob_of_buy > prob_of_sell:
                # predicted Low might be below current_price if the model expects a drop
                # set SL a bit above predicted Low to reduce risk
                # risk = (entry - SL)
                # desired TP = entry + 3*risk
                # clamp at predicted High if it goes above

                # if predicted Low is higher than current price (rare), clamp it
                sl = min(fut_l, current_price) + buffer_pips
                risk = current_price - sl
                desired_tp = current_price + 3*risk  # 1:3 ratio
                # clamp if above predicted High
                tp = min(desired_tp, fut_h - buffer_pips)

                msg = (
                    f"{ticker}/BUY/{round(current_price,4)}\n"
                    f"SL: {round(sl,4)}\n"
                    f"TP: {round(tp,4)}\n"
                    f"[Prob_Buy={prob_of_buy:.2f}, fut_H={fut_h:.4f}, fut_L={fut_l:.4f}]"
                )
                if tp <= sl:
                    msg += "\n(Warning: can't maintain 1:3; potential skip?)"
                print("Live Combined Signal:", msg)
                send_telegram_message(msg)

            # If sell:
            elif prob_of_sell > threshold and prob_of_sell > prob_of_buy:
                # predicted High might be above current_price
                # set SL a bit below predicted High to reduce risk
                sl = max(fut_h, current_price) - buffer_pips
                risk = sl - current_price
                desired_tp = current_price - 3*risk  # 1:3 ratio
                # clamp if below predicted Low
                tp = max(desired_tp, fut_l + buffer_pips)

                msg = (
                    f"{ticker}/SELL/{round(current_price,4)}\n"
                    f"SL: {round(sl,4)}\n"
                    f"TP: {round(tp,4)}\n"
                    f"[Prob_Sell={prob_of_sell:.2f}, fut_H={fut_h:.4f}, fut_L={fut_l:.4f}]"
                )
                if tp >= sl:
                    msg += "\n(Warning: can't maintain 1:3; potential skip?)"
                print("Live Combined Signal:", msg)
                send_telegram_message(msg)

            else:
                print(
                    f"{ticker}: No strong signal. "
                    f"(Buy={prob_of_buy:.2f}, Sell={prob_of_sell:.2f}, "
                    f"Pred. High={fut_h:.4f}, Pred. Low={fut_l:.4f})"
                )

        print("Waiting 5 minutes before next check...\n")
        time.sleep(300)  # Wait 5 minutes in the loop
