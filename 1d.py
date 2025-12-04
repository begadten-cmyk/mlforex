"""
Daily Candle Bullish/Bearish Prediction
Using a Random Forest + select Technical Indicators.
Adapted to NOT require <TIME> column, only date/open/high/low/close.
Any leftover columns (like tick volume, volume, spread) get dropped.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

##############################################################################
# 1) LOAD DAILY CSV (No <TIME> needed)
##############################################################################
def load_daily_csv(csv_file="eurusd1d.csv"):
    """
    Reads a daily CSV file, tab-delimited. 
    Required columns at least:
      <DATE>, <OPEN>, <HIGH>, <LOW>, <CLOSE>
    or something similar. We'll rename them to [date, open, high, low, close].
    We also drop leftover columns (tick vol, volume, spread, etc.).
    """
    df = pd.read_csv(csv_file, sep='\t', engine='python')

    # rename columns if they exist
    # adjust these mappings if your CSV has simpler names like 'o','h','l','c'
    df.rename(columns={
        '<DATE>': 'date',
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>':  'low',
        '<CLOSE>': 'close'
    }, inplace=True, errors='ignore')

    # or if your CSV literally has 'o','h','l','c':
    # df.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True, errors='ignore')

    # drop extraneous columns like tick volume, volume, spread
    df.drop(columns=['<TICKVOL>','<VOL>','<SPREAD>'], errors='ignore', inplace=True)

    # check mandatory columns: date, open, high, low, close
    for c in ['date','open','high','low','close']:
        if c not in df.columns:
            raise KeyError(f"CSV '{csv_file}' missing required column '{c}'")

    return df[['date','open','high','low','close']].reset_index(drop=True)

##############################################################################
# 2) COMPUTE TECHNICAL INDICATORS
##############################################################################
def compute_indicators(df, rsi_period=14):
    """
    Minimal set of indicators for demonstration:
      - RSI(14)
      - MACD(12,26,9)
      - SMA(10), SMA(30)
      - Bollinger(20, 2std)
    We remove references to 'time'; only 'date' used for indexing.
    """
    df = df.copy()
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta>0, 0.0)
    loss = -delta.where(delta<0, 0.0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    df['rsi'] = 100 - (100/(1+rs))

    # MACD(12,26,9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # SMA(10), SMA(30)
    df['sma10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['sma30'] = df['close'].rolling(window=30, min_periods=1).mean()

    # Bollinger(20,2)
    df['bb_ma'] = df['close'].rolling(20, min_periods=1).mean()
    df['bb_std'] = df['close'].rolling(20, min_periods=1).std()
    df['bb_up'] = df['bb_ma'] + 2*df['bb_std']
    df['bb_dn'] = df['bb_ma'] - 2*df['bb_std']

    # fill initial NaNs
    df.bfill(inplace=True)
    return df

##############################################################################
# 3) CREATE BULLISH/BEARISH LABEL
##############################################################################
def create_features_and_label(df, bars_to_skip=1):
    """
    We'll define X_i => [open, high, low, close, rsi, macd_line, macd_signal, macd_hist,
                         sma10, sma30, bb_ma, bb_up, bb_dn]
    Then label => 1 if next day's close>next day's open => Bullish, else 0 => Bearish
    skip=1 => next daily bar
    """
    feat_cols = [
        'open','high','low','close',
        'rsi','macd_line','macd_signal','macd_hist',
        'sma10','sma30','bb_ma','bb_up','bb_dn'
    ]

    arr = df[feat_cols].to_numpy()
    N = len(df)
    X, y = [], []
    max_i = N - bars_to_skip
    if max_i<1:
        return np.array([]), np.array([])

    for i in range(max_i):
        feats = arr[i]
        nxt_open  = df.loc[i+bars_to_skip,'open']
        nxt_close = df.loc[i+bars_to_skip,'close']
        label = 1.0 if nxt_close>nxt_open else 0.0
        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y).reshape(-1,1)

##############################################################################
# 4) MAIN
##############################################################################
def main():
    """
    We'll do daily data. We'll name the file 'eurusd1d.csv' 
    but you can rename as needed. 
    Code does NOT require a <TIME> column.
    """
    csv_file = "eurusd1d.csv"  # your daily data file
    df_raw = load_daily_csv(csv_file)

    # compute indicators
    df_ind = compute_indicators(df_raw)

    # build features, label => next day candle
    X, y = create_features_and_label(df_ind, bars_to_skip=1)
    if len(X)==0:
        print("Not enough data for training.")
        return

    # scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.ravel())

    train_acc = rf.score(X_train, y_train)
    test_acc  = rf.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f" Test Accuracy: {test_acc:.3f}")

    # final row => next day
    last_feat = X_scaled[-1].reshape(1,-1)
    prob_bullish = rf.predict_proba(last_feat)[0][1]
    direction = "Bullish" if prob_bullish>=0.5 else "Bearish"

    last_idx = len(df_ind)-1
    last_date = df_ind.loc[last_idx,'date']
    print("\n--- NEXT DAILY CANDLE PREDICTION ---")
    print(f"Last known bar date: {last_date}")
    print(f"Probability of Bullish: {prob_bullish:.4f}")
    print(f"Model says: {direction}")

if __name__=="__main__":
    main()
