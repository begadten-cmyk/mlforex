import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from river import linear_model, metrics, optim

############################################################################
# CONFIG: Which currency pairs, start date, "today" (end_date)
############################################################################
pairs = [
    "GBPJPY=X",
    "EURUSD=X",
    "USDCAD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "EURGBP=X"
]

start_date = "2010-01-01"
end_date   = "2024-12-31"

FEATURE_COLS = ['Daily_Return', 'MA_5', 'MA_10', 'Volume']

############################################################################
# We'll store 3 incremental models per pair: High, Low, Close
# This dictionary will keep them in memory:
############################################################################
models_dict = {}
metrics_dict = {}

############################################################################
def train_incrementally(pairs, start_date, end_date):
    """
    1) For each pair, download daily data up to end_date.
    2) Create 'tomorrow' columns for High, Low, Close.
    3) Do day-by-day incremental training.
    4) Return the trained models + final metrics.
    """
    global models_dict, metrics_dict

    for ticker in pairs:
        print(f"\n=== Training incrementally for {ticker} ===")

        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        if len(df.columns) == 5:
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        # Create tomorrow's columns
        df['Tomorrow_High']   = df['High'].shift(-1)
        df['Tomorrow_Low']    = df['Low'].shift(-1)
        df['Tomorrow_Close']  = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # Simple features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Daily_Return'] = df['Daily_Return'].clip(-0.05, 0.05)  # clip extremes
        df['MA_5']         = df['Close'].rolling(5).mean()
        df['MA_10']        = df['Close'].rolling(10).mean()
        df.dropna(inplace=True)

        if df.empty:
            print(f"No usable data for {ticker} after feature engineering.")
            continue

        # If models for this ticker don't exist, create them
        if ticker not in models_dict:
            # Plain LinearRegression with default squared loss
            model_high  = linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01))
            model_low   = linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01))
            model_close = linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01))

            models_dict[ticker] = {
                "high":  model_high,
                "low":   model_low,
                "close": model_close
            }
            mae_high  = metrics.MAE()
            mae_low   = metrics.MAE()
            mae_close = metrics.MAE()
            metrics_dict[ticker] = {
                "mae_high": mae_high,
                "mae_low":  mae_low,
                "mae_close": mae_close
            }
        else:
            pass

        model_high  = models_dict[ticker]["high"]
        model_low   = models_dict[ticker]["low"]
        model_close = models_dict[ticker]["close"]
        mae_high    = metrics_dict[ticker]["mae_high"]
        mae_low     = metrics_dict[ticker]["mae_low"]
        mae_close   = metrics_dict[ticker]["mae_close"]

        # Day-by-day incremental training loop
        for i in range(len(df) - 1):
            row = df.iloc[i]
            x_t = row[FEATURE_COLS].to_dict()

            true_high   = row['Tomorrow_High']
            true_low    = row['Tomorrow_Low']
            true_close  = row['Tomorrow_Close']

            # Predictions
            pred_high   = model_high.predict_one(x_t)  or 0.0
            pred_low    = model_low.predict_one(x_t)   or 0.0
            pred_close  = model_close.predict_one(x_t) or 0.0

            # Update metrics
            mae_high.update(true_high, pred_high)
            mae_low.update(true_low, pred_low)
            mae_close.update(true_close, pred_close)

            # Partial fit
            model_high.learn_one(x_t, true_high)
            model_low.learn_one(x_t,   true_low)
            model_close.learn_one(x_t, true_close)

        print(f"Final MAE High:   {mae_high.get():.4f}")
        print(f"Final MAE Low:    {mae_low.get():.4f}")
        print(f"Final MAE Close:  {mae_close.get():.4f}")

    print("\n=== Incremental Training Complete ===")


def predict_tomorrow(pairs):
    """
    1) For each pair, attempt to get today's data from Yahoo (1d).
       If market is closed (weekend/holiday) or df_live is empty,
       skip partial-fitting, but still produce a forecast for tomorrow
       using the last known model state.
    2) Then predict next day's High, Low, Close.
    """
    global models_dict, metrics_dict

    # If it's weekend, shift to Friday
    # e.g., if .weekday() == 5 (Saturday) or 6 (Sunday), we go back to Friday
    today_dt = datetime.today()
    wd = today_dt.weekday()  # Monday=0, Sunday=6
    if wd == 5:   # Saturday
        # shift to Friday
        day_shift = 1
    elif wd == 6: # Sunday
        # shift to Friday
        day_shift = 2
    else:
        day_shift = 0

    adjusted_dt = today_dt - timedelta(days=day_shift)
    today_str   = adjusted_dt.strftime('%Y-%m-%d')

    print(f"\n=== Predicting tomorrow for {today_str} === (adjusted from real today if weekend)")

    for ticker in pairs:
        if ticker not in models_dict:
            print(f"{ticker} has no trained model yet, skipping.")
            continue

        print(f"\n--- {ticker} ---")

        # Attempt to download that adjusted day's bar
        df_live = yf.download(ticker, start=today_str, end=today_str, interval="1d")

        if len(df_live.columns) == 5:
            df_live.columns = ["Open", "High", "Low", "Close", "Volume"]
        df_live.dropna(inplace=True)

        # If still empty, skip partial-fitting but produce forecast from last known state
        if df_live.empty:
            print("No data for this candle (possibly weekend/holiday). Skipping partial-fit.")
            # We'll produce a forecast from last known state anyway
            # We'll pretend our x_t is None or we keep an x_last if we wanted to store it
            # For now, just do a "tomorrow" predict with no partial update
            produce_tomorrow_forecast_only(ticker)
            continue

        # If we do have data, partial-fit as usual
        df_live['Daily_Return'] = df_live['Close'].pct_change()
        df_live['Daily_Return'] = df_live['Daily_Return'].clip(-0.05, 0.05)
        df_live['MA_5']         = df_live['Close'].rolling(5).mean()
        df_live['MA_10']        = df_live['Close'].rolling(10).mean()
        df_live.dropna(inplace=True)

        if df_live.empty:
            print("No usable features for today's candle after rolling. Skipping partial-fit.")
            produce_tomorrow_forecast_only(ticker)
            continue

        row = df_live.iloc[-1]
        x_t = row[FEATURE_COLS].to_dict()

        true_high   = row['High']
        true_low    = row['Low']
        true_close  = row['Close']

        model_high  = models_dict[ticker]["high"]
        model_low   = models_dict[ticker]["low"]
        model_close = models_dict[ticker]["close"]

        mae_high  = metrics_dict[ticker]["mae_high"]
        mae_low   = metrics_dict[ticker]["mae_low"]
        mae_close = metrics_dict[ticker]["mae_close"]

        # Predictions for "today"
        pred_h = model_high.predict_one(x_t)  or 0.0
        pred_l = model_low.predict_one(x_t)   or 0.0
        pred_c = model_close.predict_one(x_t) or 0.0

        # Update metrics
        mae_high.update(true_high, pred_h)
        mae_low.update(true_low, pred_l)
        mae_close.update(true_close, pred_c)

        # Partial fit for "today"
        model_high.learn_one(x_t, true_high)
        model_low.learn_one(x_t,  true_low)
        model_close.learn_one(x_t, true_close)

        # Now produce "tomorrow" forecast
        produce_tomorrow_forecast_only(ticker)


def produce_tomorrow_forecast_only(ticker):
    """
    Called when we either skip partial-fitting (empty day)
    or after we do partial-fitting, to produce the next-day forecast
    from the model's current state.
    """
    model_high  = models_dict[ticker]["high"]
    model_low   = models_dict[ticker]["low"]

    # We can't partial-fit a new row if we don't have one,
    # but we can still produce a forecast based on the last model state
    # We'll do a dummy x_t = zeros or skip if we want
    # For demonstration, let's do a "dummy" approach:
    # In practice, you might keep track of the last known row or skip
    x_dummy = {col: 0.0 for col in FEATURE_COLS}

    fut_pred_h = model_high.predict_one(x_dummy) or 0.0
    fut_pred_l = model_low.predict_one(x_dummy)  or 0.0

    print(f"Tomorrow's Predicted High: {fut_pred_h:.4f}, Low: {fut_pred_l:.4f}")


if __name__ == "__main__":
    # 1) Train incrementally from 2010-01-01 up to 2024-12-31
    train_incrementally(pairs, start_date, end_date)
    # 2) At the end of the day, we call predict_tomorrow
    predict_tomorrow(pairs)
