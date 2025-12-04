import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from river import linear_model, metrics

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

# Suppose we want to start from 2010, or any old date:
start_date = "2010-01-01"
# "Today" is your current real-world date (you said 2024-12-31).
end_date = "2024-12-31"

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
        # NO clipping here anymore
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df.dropna(inplace=True)

        if df.empty:
            print(f"No usable data for {ticker} after feature engineering.")
            continue

        # Initialize 3 models for this pair if not already existing
        if ticker not in models_dict:
            model_high  = linear_model.LinearRegression()
            model_low   = linear_model.LinearRegression()
            model_close = linear_model.LinearRegression()
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
        
        for i in range(len(df) - 1):
            row = df.iloc[i]
            x_t = row[FEATURE_COLS].to_dict()

            true_high   = row['Tomorrow_High']
            true_low    = row['Tomorrow_Low']
            true_close  = row['Tomorrow_Close']

            pred_high   = model_high.predict_one(x_t)  or 0.0
            pred_low    = model_low.predict_one(x_t)   or 0.0
            pred_close  = model_close.predict_one(x_t) or 0.0

            mae_high.update(true_high, pred_high)
            mae_low.update(true_low, pred_low)
            mae_close.update(true_close, pred_close)

            model_high.learn_one(x_t, true_high)
            model_low.learn_one(x_t, true_low)
            model_close.learn_one(x_t, true_close)

        print(f"Final MAE High:   {mae_high.get():.4f}")
        print(f"Final MAE Low:    {mae_low.get():.4f}")
        print(f"Final MAE Close:  {mae_close.get():.4f}")

    print("\n=== Incremental Training Complete ===")


def predict_tomorrow(pairs):
    """
    1) For each pair, get today's data from Yahoo (1 day),
       partial_fit on that day, 
    2) Then predict next day's High, Low, Close.
    """
    global models_dict, metrics_dict

    today_str = datetime.today().strftime('%Y-%m-%d')
    print(f"\n=== Predicting tomorrow for {today_str} ===")
    for ticker in pairs:
        if ticker not in models_dict:
            print(f"{ticker} has no trained model yet, skipping.")
            continue

        print(f"\n--- {ticker} ---")
        df_live = yf.download(ticker, start=today_str, end=today_str, interval="1d")

        if len(df_live.columns) == 5:
            df_live.columns = ["Open", "High", "Low", "Close", "Volume"]
        df_live.dropna(inplace=True)

        if df_live.empty:
            print("No data for today's candle. Possibly market closed or data unavailable.")
            continue

        # Build same features
        df_live['Daily_Return'] = df_live['Close'].pct_change()
        # NO clipping here anymore
        df_live['MA_5'] = df_live['Close'].rolling(5).mean()
        df_live['MA_10'] = df_live['Close'].rolling(10).mean()
        df_live.dropna(inplace=True)

        if df_live.empty:
            print("No usable features for today's candle after rolling. Skipping.")
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

        pred_h = model_high.predict_one(x_t)  or 0.0
        pred_l = model_low.predict_one(x_t)   or 0.0
        pred_c = model_close.predict_one(x_t) or 0.0

        mae_high.update(true_high, pred_h)
        mae_low.update(true_low, pred_l)
        mae_close.update(true_close, pred_c)

        model_high.learn_one(x_t, true_high)
        model_low.learn_one(x_t,  true_low)
        model_close.learn_one(x_t, true_close)

        fut_pred_h = model_high.predict_one(x_t)
        fut_pred_l = model_low.predict_one(x_t)
        fut_pred_c = model_close.predict_one(x_t)

        print(f"Tomorrow's Predicted High:   {fut_pred_h:.4f}")
        print(f"Tomorrow's Predicted Low:    {fut_pred_l:.4f}")
        print(f"Tomorrow's Predicted Close:  {fut_pred_c:.4f}")

    print("\n=== Done predicting tomorrow ===\n")


if __name__ == "__main__":
    # 1) Train incrementally from 2010-01-01 up to 2024-12-31
    train_incrementally(pairs, start_date, end_date)
    # 2) At the end of the day (which is presumably 2024-12-31),
    #    we call predict_tomorrow to partial-fit today's final candle 
    #    and get the next day's forecast.
    predict_tomorrow(pairs)
