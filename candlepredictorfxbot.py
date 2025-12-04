import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import talib  # for RSI, ATR (pip install TA-Lib or talib-binary)

##############################################
# CURRENCY PAIRS
##############################################
pairs = [
    "GBPJPY=X",  # GBP/JPY
    "EURUSD=X",  # EUR/USD
    "USDCAD=X",  # USD/CAD
    "USDJPY=X",  # USD/JPY
    "GBPUSD=X",  # GBP/USD
    "EURGBP=X"   # EUR/GBP
]

start_date = "2018-01-01"
end_date   = "2023-01-01"

##############################################
# FOR EACH PAIR, PREDICT TOMORROW'S HIGH & LOW
##############################################
for ticker in pairs:
    print(f"\n============ PREDICTING NEXT-DAY CANDLE FOR {ticker} ============")

    # 1) Download daily data
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if len(df.columns) == 5:  # rename if needed
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

    if df.empty:
        print(f"No data for {ticker}. Skipping.")
        continue

    df.dropna(inplace=True)
    print(df.head())

    # 2) SHIFT to create 'Tomorrow_High' and 'Tomorrow_Low'
    df['Tomorrow_High'] = df['High'].shift(-1)
    df['Tomorrow_Low']  = df['Low'].shift(-1)
    df.dropna(inplace=True)

    # 3) Feature Engineering
    #    (Using daily return, ATR, RSI, short vs. long SMA, etc.)
    df['Daily_Return'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_diff'] = df['SMA_5'] - df['SMA_10']

    # RSI, ATR with TA-Lib
    # If talib isn't available in your environment, comment these out or handle accordingly.
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Drop any NaN from rolling or talib indicators
    df.dropna(inplace=True)

    # 4) Define Features (X) and Targets (y1=Tomorrow_High, y2=Tomorrow_Low)
    #    We'll build 2 separate models: model_high and model_low.
    features = [
        'Daily_Return',
        'SMA_diff',
        'RSI_14',
        'ATR_14'
    ]
    X = df[features]
    y_high = df['Tomorrow_High']
    y_low  = df['Tomorrow_Low']

    # 5) Train/Test Split (80/20 by time)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_high_train, y_high_test = y_high.iloc[:train_size], y_high.iloc[train_size:]
    y_low_train,  y_low_test  = y_low.iloc[:train_size],  y_low.iloc[train_size:]

    # 6) SCALE Features to avoid huge predictions
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 7) Build Two XGBoost Regressors
    model_high = xgb.XGBRegressor(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1
    )
    model_low  = xgb.XGBRegressor(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1
    )

    # Train
    model_high.fit(X_train_scaled, y_high_train)
    model_low.fit(X_train_scaled,  y_low_train)

    # Evaluate (RMSE)
    y_high_pred = model_high.predict(X_test_scaled)
    y_low_pred  = model_low.predict(X_test_scaled)

    rmse_high = np.sqrt(mean_squared_error(y_high_test, y_high_pred))
    rmse_low  = np.sqrt(mean_squared_error(y_low_test,  y_low_pred))

    print(f"RMSE (Tomorrow's High) for {ticker}: {rmse_high:.5f}")
    print(f"RMSE (Tomorrow's Low)  for {ticker}: {rmse_low:.5f}")

    # Show last 5 predictions
    compare_df = pd.DataFrame({
        'Actual_High': y_high_test.tail(5).values,
        'Pred_High': y_high_pred[-5:],
        'Actual_Low': y_low_test.tail(5).values,
        'Pred_Low': y_low_pred[-5:]
    }, index=y_low_test.tail(5).index)
    print(compare_df)

    ##############################################
    # 8) Predict NEXT DAY's High & Low (Live)
    ##############################################
    today_str = datetime.datetime.today().strftime('%Y-%m-%d')
    df_live = yf.download(ticker, start=end_date, end=today_str, interval="1d")
    if len(df_live.columns) == 5:
        df_live.columns = ["Open", "High", "Low", "Close", "Volume"]
    df_live.dropna(inplace=True)

    if df_live.empty:
        print("No new data to predict tomorrow's high/low.")
        continue

    # Recompute same indicators on df_live
    df_live['Daily_Return'] = df_live['Close'].pct_change()
    df_live['SMA_5'] = df_live['Close'].rolling(5).mean()
    df_live['SMA_10'] = df_live['Close'].rolling(10).mean()
    df_live['SMA_diff'] = df_live['SMA_5'] - df_live['SMA_10']

    df_live['RSI_14'] = talib.RSI(df_live['Close'], timeperiod=14)
    df_live['ATR_14'] = talib.ATR(df_live['High'], df_live['Low'], df_live['Close'], timeperiod=14)

    df_live.dropna(inplace=True)

    # If still not empty, predict the last row
    if not df_live.empty:
        X_live = df_live[features].iloc[-1:].copy()  # last row
        X_live_scaled = scaler.transform(X_live)     # scale with same scaler

        pred_high_live = model_high.predict(X_live_scaled)[0]
        pred_low_live  = model_low.predict(X_live_scaled)[0]

        print("\n-------------------------------------------------------")
        print(f"{ticker} - Predicted Next-Day High: {pred_high_live:.4f}")
        print(f"{ticker} - Predicted Next-Day Low : {pred_low_live:.4f}")
        print("-------------------------------------------------------")
    else:
        print("df_live has no rows after indicator calculation.")
