"""
RANDOM FOREST SCRIPT FOR MULTIPLE TIMEFRAMES
Predict Next 4H Candle Direction with Candlestick + Stochastic Features
(Now includes additional indicators + advanced candlestick patterns)
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
# 2) COMPUTE STOCH + ADVANCED PATTERNS + MULTIPLE INDICATORS
##############################################################################
def compute_stochastics_and_patterns(df, stoch_period=14):
    """
    Adds columns:
      stochK, stochD (stochastic oscillator),
      RSI(14), MACD(12,26,9) + macd_signal, macd_hist,
      Bollinger Bands(20), Donchian(20), pivot points,
      doji, hammer, bull_engulf, bear_engulf,
      morning_star, evening_star, bull_harami, bear_harami
    Then backfills any initial NaNs.
    """
    df = df.copy()
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    ##########################################################################
    # (A) STOCHASTICS
    ##########################################################################
    lows = df['low'].rolling(window=stoch_period, min_periods=1).min()
    highs= df['high'].rolling(window=stoch_period, min_periods=1).max()
    df['stochK'] = ((df['close'] - lows)/(highs - lows + 1e-9))*100
    df['stochD'] = df['stochK'].rolling(window=3, min_periods=1).mean()

    ##########################################################################
    # (B) RSI(14)
    ##########################################################################
    # Simplistic approach
    # 1) compute price diff
    delta = df['close'].diff(1)
    gain = delta.where(delta>0, 0.0)
    loss = -delta.where(delta<0, 0.0)
    roll_up = gain.rolling(14, min_periods=1).mean()
    roll_down = loss.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100.0 - (100.0 / (1.0 + rs))

    ##########################################################################
    # (C) MACD(12,26,9)
    ##########################################################################
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    ##########################################################################
    # (D) Bollinger Bands(20, 2std)
    ##########################################################################
    bb_period = 20
    df['bb_ma'] = df['close'].rolling(bb_period, min_periods=1).mean()
    df['bb_std'] = df['close'].rolling(bb_period, min_periods=1).std()
    df['bb_up'] = df['bb_ma'] + 2.0 * df['bb_std']
    df['bb_dn'] = df['bb_ma'] - 2.0 * df['bb_std']

    ##########################################################################
    # (E) Donchian Channels(20)
    ##########################################################################
    dch_period = 20
    df['donchian_hi'] = df['high'].rolling(dch_period, min_periods=1).max()
    df['donchian_lo'] = df['low'].rolling(dch_period, min_periods=1).min()

    ##########################################################################
    # (F) Pivot Points (simplified "previous bar" approach)
    ##########################################################################
    # pivot(i) = ( high(i-1)+ low(i-1)+ close(i-1) )/3
    # R1= 2*pivot - low(i-1), S1= 2*pivot - high(i-1) 
    # We'll do it in a simple for-loop
    pivots = []
    r1s = []
    s1s = []
    for i in range(len(df)):
        if i==0:
            pivots.append(np.nan)
            r1s.append(np.nan)
            s1s.append(np.nan)
        else:
            ph = df.loc[i-1,'high']
            pl = df.loc[i-1,'low']
            pc = df.loc[i-1,'close']
            pivot = (ph+pl+pc)/3.0
            R1 = 2*pivot - pl
            S1 = 2*pivot - ph
            pivots.append(pivot)
            r1s.append(R1)
            s1s.append(S1)
    df['pivot'] = pivots
    df['pivot_r1'] = r1s
    df['pivot_s1'] = s1s

    ##########################################################################
    # (G) Basic Patterns: doji, hammer, engulf
    ##########################################################################
    df['doji'] = 0
    df['hammer'] = 0
    df['bull_engulf'] = 0
    df['bear_engulf'] = 0

    def detect_doji(row, threshold=0.1):
        op, hi, lo, cl = row['open'], row['high'], row['low'], row['close']
        rng = hi - lo
        if rng < 1e-9:
            return 0
        body = abs(cl - op)
        return 1 if (body/rng)<threshold else 0

    def detect_hammer(row, body_threshold=0.2, wick_threshold=2.0):
        op, hi, lo, cl = row['open'], row['high'], row['low'], row['close']
        rng= hi-lo
        if rng<1e-9:
            return 0
        body= abs(cl-op)
        dn_wick= min(op,cl)-lo
        if (body/rng)>body_threshold:
            return 0
        if dn_wick < wick_threshold*body:
            return 0
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

    ##########################################################################
    # (H) Advanced Patterns: morning star, evening star, harami
    ##########################################################################
    df['morning_star'] = 0
    df['evening_star'] = 0
    df['bull_harami']  = 0
    df['bear_harami']  = 0

    def detect_morning_star(rA, rB, rC):
        # Simplistic check:
        # rA: big bearish (close < open), rB: small candle, rC: big bullish
        # final candle closes well into rA's body
        if (rA['close'] < rA['open']) and (rC['close'] > rC['open']):
            # check small body in rB
            bodyB = abs(rB['close']-rB['open'])
            rngB  = rB['high']-rB['low']
            if rngB>1e-9 and (bodyB/rngB<0.3):
                # final checks
                # ensure rC close > midpoint of rA body
                midA = (rA['open']+rA['close'])/2.0
                if rC['close']>midA:
                    return 1
        return 0

    def detect_evening_star(rA, rB, rC):
        # Opposite of morning star
        if (rA['close'] > rA['open']) and (rC['close'] < rC['open']):
            # check small body in rB
            bodyB = abs(rB['close']-rB['open'])
            rngB  = rB['high']-rB['low']
            if rngB>1e-9 and (bodyB/rngB<0.3):
                # final checks
                midA = (rA['open']+rA['close'])/2.0
                if rC['close']<midA:
                    return 1
        return 0

    def detect_bull_harami(prev_row, curr_row):
        # Bullish harami: 
        # rA = large bearish, rB = inside, bullish
        if (prev_row['close']<prev_row['open']) and (curr_row['close']>curr_row['open']):
            # check if the body of curr is inside prev
            # i.e. curr open/close within prev open/close
            if (curr_row['open']>= min(prev_row['open'],prev_row['close'])) \
               and (curr_row['close']<= max(prev_row['open'],prev_row['close'])):
                return 1
        return 0

    def detect_bear_harami(prev_row, curr_row):
        if (prev_row['close']>prev_row['open']) and (curr_row['close']<curr_row['open']):
            if (curr_row['open']<= max(prev_row['open'],prev_row['close'])) \
               and (curr_row['close']>= min(prev_row['open'],prev_row['close'])):
                return 1
        return 0

    for i in range(2,len(df)):
        # morning/evening star check i => rC, i-1 => rB, i-2 => rA
        df.loc[i,'morning_star'] = detect_morning_star(df.loc[i-2], df.loc[i-1], df.loc[i])
        df.loc[i,'evening_star'] = detect_evening_star(df.loc[i-2], df.loc[i-1], df.loc[i])

    for i in range(1,len(df)):
        # harami check
        df.loc[i,'bull_harami'] = detect_bull_harami(df.loc[i-1], df.loc[i])
        df.loc[i,'bear_harami'] = detect_bear_harami(df.loc[i-1], df.loc[i])

    ##########################################################################
    # Finally, fill any initial NaNs from rolling computations
    ##########################################################################
    df.bfill(inplace=True)
    return df

##############################################################################
# 3) CREATE FLAT FEATURE ROWS => (N, #features)
##############################################################################
def create_features_and_labels(df, bars_to_skip=1):
    """
    For each row i, define X_i = [open,high,low,close, stochK, stochD,
      rsi14, macd, macd_signal, macd_hist,
      bb_ma, bb_up, bb_dn, donchian_hi, donchian_lo,
      pivot, pivot_r1, pivot_s1,
      doji, hammer, bull_engulf, bear_engulf,
      morning_star, evening_star, bull_harami, bear_harami]
    Label y_i = 1 if close_of_future > open_of_future (the next 4H candle), else 0
    We skip 'bars_to_skip' bars to find that future bar.
    """
    # define the columns we want to feed the model
    needed_cols = [
        'open','high','low','close',
        'stochK','stochD','rsi14',
        'macd','macd_signal','macd_hist',
        'bb_ma','bb_up','bb_dn',
        'donchian_hi','donchian_lo',
        'pivot','pivot_r1','pivot_s1',
        'doji','hammer','bull_engulf','bear_engulf',
        'morning_star','evening_star','bull_harami','bear_harami'
    ]
    arr = df[needed_cols].to_numpy()
    N = len(arr)
    X, y = [], []
    max_i = N - bars_to_skip
    if max_i <= 0:
        return np.array([]), np.array([])

    for i in range(max_i):
        # Features from bar i
        features = arr[i]
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
    2) compute_stochastics_and_patterns (now includes many indicators & adv. patterns)
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

    # Quick demonstration with random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_full, y_train_full.ravel())

    # Evaluate
    train_acc = rf.score(X_train_full, y_train_full)
    test_acc  = rf.score(X_test, y_test)
    print(f"{label} Model Train Accuracy: {train_acc:.3f}")
    print(f"{label} Model  Test Accuracy: {test_acc:.3f}")

    # final row inference
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
    """
    We'll do the same multi-timeframe approach for the next 4H candle,
    but now with more indicators and advanced candlestick features.
    """
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
