"""
RANDOM FOREST SCRIPT FOR MULTIPLE TIMEFRAMES
Predict Next 4H Candle Direction with Candlestick + Stochastic Features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

##############################################################################
# 1) LOAD CSV
##############################################################################
def load_csv_tab_delimited(csv_file):
    """
    Reads a tab-delimited CSV with columns:
      <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, ...
    Renames them to [date, time, open, high, low, close].
    Drops <TICKVOL>, <VOL>, <SPREAD> if present.
    Returns a DataFrame with columns: [date, time, open, high, low, close].
    """
    df = pd.read_csv(csv_file, sep='\t', engine='python')
    df.rename(columns={
        '<DATE>': 'date',
        '<TIME>': 'time',
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>':  'low',
        '<CLOSE>': 'close'
    }, inplace=True, errors='ignore')
    df.drop(columns=['<TICKVOL>', '<VOL>', '<SPREAD>'], errors='ignore', inplace=True)

    for col in ['date','time','open','high','low','close']:
        if col not in df.columns:
            raise KeyError(f"CSV '{csv_file}' missing '{col}' column.")
    return df[['date','time','open','high','low','close']].reset_index(drop=True)

##############################################################################
# 2) COMPUTE STOCH + CANDLE PATTERNS
##############################################################################
def compute_stochastics_and_patterns(df, stoch_period=14):
    df = df.copy()
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    # Stoch K
    lows = df['low'].rolling(window=stoch_period, min_periods=1).min()
    highs = df['high'].rolling(window=stoch_period, min_periods=1).max()
    df['stochK'] = ((df['close'] - lows)/(highs - lows + 1e-9))*100
    # Stoch D: 3-simple
    df['stochD'] = df['stochK'].rolling(window=3, min_periods=1).mean()

    # Basic Patterns
    df['doji'] = 0
    df['hammer'] = 0
    df['bull_engulf'] = 0
    df['bear_engulf'] = 0

    def detect_doji(row, threshold=0.1):
        op, hi, lo, cl = row['open'], row['high'], row['low'], row['close']
        body = abs(cl - op)
        rng = hi - lo
        if rng < 1e-9: return 0
        return 1 if (body/rng) < threshold else 0

    def detect_hammer(row, body_threshold=0.2, wick_threshold=2.0):
        op, hi, lo, cl = row['open'], row['high'], row['low'], row['close']
        body = abs(cl - op)
        rng = hi - lo
        if rng < 1e-9: return 0
        dn_wick = min(op, cl) - lo
        if (body/rng) > body_threshold: return 0
        if dn_wick < wick_threshold*body: return 0
        return 1

    def detect_engulfing(prev_row, curr_row):
        op_p, cl_p = prev_row['open'], prev_row['close']
        op_c, cl_c = curr_row['open'], curr_row['close']

        # bullish engulf
        is_bearish_prev = (cl_p < op_p)
        is_bullish_curr = (cl_c > op_c)
        bull=0
        if is_bearish_prev and is_bullish_curr:
            if op_c < cl_p and cl_c > op_p:
                bull=1

        # bearish engulf
        is_bullish_prev = (cl_p > op_p)
        is_bearish_curr = (cl_c < op_c)
        bear=0
        if is_bullish_prev and is_bearish_curr:
            if op_c > cl_p and cl_c < op_p:
                bear=1

        return bull,bear

    for i in range(len(df)):
        df.loc[i, 'doji']   = detect_doji(df.loc[i])
        df.loc[i, 'hammer'] = detect_hammer(df.loc[i])

    for i in range(1, len(df)):
        b, br = detect_engulfing(df.loc[i-1], df.loc[i])
        df.loc[i,'bull_engulf']=b
        df.loc[i,'bear_engulf']=br

    df.fillna(method='bfill', inplace=True)
    return df

##############################################################################
# 3) CREATE FLAT FEATURE ROWS => (N, #features)
##############################################################################
def create_features_and_labels(df, bars_to_skip=1):
    """
    For each row i, we define X_i = [open,high,low,close,stochK,stochD,doji,hammer,bull_engulf,bear_engulf]
    Label y_i = 1 if close_of_future > open_of_future (the next 4H candle), else 0
    We skip 'bars_to_skip' bars to find that future bar.
    """
    needed_cols = [
        'open','high','low','close',
        'stochK','stochD','doji','hammer','bull_engulf','bear_engulf'
    ]
    arr = df[needed_cols].to_numpy()
    N = len(arr)
    X, y = [], []
    max_i = N - bars_to_skip
    if max_i <= 0:
        return np.array([]), np.array([])
    for i in range(max_i):
        # Features from bar i
        features = arr[i]  # shape(10,)
        # Label from bar i+bars_to_skip
        future_open  = df.loc[i+bars_to_skip, 'open']
        future_close = df.loc[i+bars_to_skip, 'close']
        label = 1.0 if future_close > future_open else 0.0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y).reshape(-1,1)

##############################################################################
# 4) TRAIN + PREDICT FOR SINGLE TIMEFRAME
##############################################################################
def train_and_predict(csv_file, label, bars_to_skip=1):
    """
    1) Load CSV
    2) stoch + patterns
    3) create features
    4) random forest
    5) final row inference
    6) returns "Bullish"/"Bearish"
    """
    df = load_csv_tab_delimited(csv_file)
    df = compute_stochastics_and_patterns(df, stoch_period=14)
    X, y = create_features_and_labels(df, bars_to_skip=bars_to_skip)
    if len(X)==0:
        print(f"{label}: Not enough data to train.")
        return None

    # scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    # We won't do a complicated val split; quick demonstration
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_full, y_train_full.ravel())

    # final row = df.loc[-1], but we do the same features for last row in X
    # The last row's features => X_scaled[-1]
    # However, if we want the *latest bar*, that is the last element in X, which is index = len(X)-1
    if len(X_scaled)==0:
        print(f"{label}: No final row to infer.")
        return None
    final_feat = X_scaled[-1].reshape(1,-1)
    prob = rf.predict_proba(final_feat)[0][1]  # prob of class=1 => bullish
    predicted_class = 1 if prob>=0.5 else 0
    direction = "Bullish" if predicted_class==1 else "Bearish"

    # get last candle info
    last_i = len(df)-1
    last_date = df.loc[last_i,'date']
    last_time = df.loc[last_i,'time']
    print(f"{label} timeframe prediction for next 4H candle (RandomForest):")
    print(f"Last known candle in {csv_file}: {last_date} {last_time}")
    print(f"Probability of Bullish: {prob:.4f}")
    print(f"Prediction: {direction}\n")

    return direction

##############################################################################
# 5) MASTER CODE (Multi-timeframe)
##############################################################################
def main():
    # (label, csvfile, bars_to_skip)
    # skip=1 => next bar => 4H candle
    # skip=2 => 2H => next 4H => skip2 => 4 hours
    timeframe_csvs = [
        ("4H",   "eurusd4h.csv",   1),
        ("2H",   "eurusd2h.csv",   2),
        ("1H",   "eurusd1h.csv",   4),
        ("30m",  "eurusd30m.csv",  8),
        ("15m",  "eurusd15m.csv", 16),
        ("5m",   "eurusd5m.csv",  48),
    ]
    predictions = []
    for tf_label, csv_file, skip_ in timeframe_csvs:
        direction = train_and_predict(csv_file, tf_label, bars_to_skip=skip_)
        predictions.append(direction)

    # Decide final
    filtered = [p for p in predictions if p is not None]
    if len(filtered)<6:
        final_call="NO TRADE (missing data or no final row somewhere)"
    else:
        if all(p=="Bullish" for p in filtered):
            final_call="BULLISH"
        elif all(p=="Bearish" for p in filtered):
            final_call="BEARISH"
        else:
            final_call="NO TRADE"

    print(f"Trade: {final_call} (ONLY IF ALL AGREE!!!)\n")

if __name__=="__main__":
    main()
