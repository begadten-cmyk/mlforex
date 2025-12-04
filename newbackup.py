import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 1) Load CSV
##############################################################################
def load_csv_tab_delimited(csv_file):
    """
    Reads a tab-delimited CSV with columns like:
      <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, ...
    Renames them to [date, time, open, high, low, close].
    Drops <TICKVOL>, <VOL>, <SPREAD> if present.
    Returns a DataFrame with those 6 columns.
    """
    df = pd.read_csv(csv_file, sep='\t', engine='python')

    # Rename columns
    df.rename(columns={
        '<DATE>': 'date',
        '<TIME>': 'time',
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>':  'low',
        '<CLOSE>': 'close'
    }, inplace=True, errors='ignore')

    # Drop unneeded if exist
    df.drop(columns=['<TICKVOL>', '<VOL>', '<SPREAD>'], errors='ignore', inplace=True)

    # Ensure required
    for col in ['date','time','open','high','low','close']:
        if col not in df.columns:
            raise KeyError(f"CSV '{csv_file}' missing column '{col}'.")

    return df[['date','time','open','high','low','close']].reset_index(drop=True)


##############################################################################
# 2) Stochastic Oscillator + Additional Candle Features
##############################################################################
def compute_stochastics_and_patterns(df, stoch_period=14):
    """
    For each row (bar), compute:
      1) Stoch_K: (close - lowest_low) / (highest_high - lowest_low)*100 over 'stoch_period'
      2) Stoch_D: 3-period SMA of Stoch_K
      3) Basic candlestick pattern flags: doji, hammer, bullish_engulfing, bearish_engulfing
    Returns df with new columns: stochK, stochD, doji, hammer, bull_engulf, bear_engulf
    NOTE: thresholds are simplified for demonstration. Tweak them as needed.
    """
    # Ensure numeric
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    # 2.1) Compute Stoch_K
    lows = df['low'].rolling(window=stoch_period, min_periods=1).min()
    highs = df['high'].rolling(window=stoch_period, min_periods=1).max()
    df['stochK'] = ((df['close'] - lows) / (highs - lows + 1e-9))*100

    # 2.2) Compute Stoch_D (3-period SMA of stochK)
    df['stochD'] = df['stochK'].rolling(window=3, min_periods=1).mean()

    # 2.3) Candlestick pattern detection
    # We'll define some quick helpers:
    def detect_doji(row, threshold=0.1):
        # A doji is when the body is very small vs the total range
        op, hi, lo, cl = row['open'], row['high'], row['low'], row['close']
        body = abs(cl - op)
        range_ = hi - lo
        # If range_ is 0, avoid division by zero
        if range_ < 1e-9:
            return 0
        # ratio = body / range_. If ratio < threshold => doji
        return 1 if (body/range_) < threshold else 0

    def detect_hammer(row, body_threshold=0.2, wick_threshold=2.0):
        # Simplistic "hammer": small body near top, long lower wick
        op, hi, lo, cl = row['open'], row['high'], row['low'], row['close']
        up_wick = hi - max(op, cl)
        dn_wick = min(op, cl) - lo
        body = abs(cl - op)
        range_ = hi - lo
        if range_ < 1e-9:
            return 0

        # body ratio is small
        if (body / range_) > body_threshold:
            return 0
        # lower wick at least 2x the body
        if dn_wick < wick_threshold*body:
            return 0
        # upper wick is relatively small
        # but let's skip that to keep it simple
        return 1

    def detect_engulfing(prev_row, curr_row):
        """
        We define 'bullish engulfing' if:
          - prev candle is bearish
          - curr candle is bullish
          - curr body fully covers prev body
        Similarly for bearish. We return (bull, bear) booleans.
        """
        op_prev, cl_prev = prev_row['open'], prev_row['close']
        op_curr, cl_curr = curr_row['open'], curr_row['close']

        body_prev = abs(cl_prev - op_prev)
        body_curr = abs(cl_curr - op_curr)

        is_bearish_prev = (cl_prev < op_prev)
        is_bullish_curr = (cl_curr > op_curr)

        is_bull_engulf = 0
        if is_bearish_prev and is_bullish_curr:
            # check if curr body covers prev body
            # e.g. curr open < prev close and curr close > prev open
            if (op_curr < cl_prev) and (cl_curr > op_prev):
                is_bull_engulf = 1

        # Similarly for bearish engulfing
        is_bullish_prev = (cl_prev > op_prev)
        is_bearish_curr = (cl_curr < op_curr)
        is_bear_engulf = 0
        if is_bullish_prev and is_bearish_curr:
            if (op_curr > cl_prev) and (cl_curr < op_prev):
                is_bear_engulf = 1

        return is_bull_engulf, is_bear_engulf

    # We'll build columns doji, hammer, bull_engulf, bear_engulf
    df['doji'] = 0
    df['hammer'] = 0
    df['bull_engulf'] = 0
    df['bear_engulf'] = 0

    # Fill them
    for i in range(len(df)):
        df.loc[i, 'doji'] = detect_doji(df.loc[i])
        df.loc[i, 'hammer'] = detect_hammer(df.loc[i])

    for i in range(1, len(df)):
        be, bae = detect_engulfing(df.loc[i-1], df.loc[i])
        df.loc[i, 'bull_engulf'] = be
        df.loc[i, 'bear_engulf'] = bae

    df.fillna(method='bfill', inplace=True)
    return df


##############################################################################
# 4) MAKE SAMPLES (X,y) -> shape (N, seq_len, num_features)
##############################################################################
def make_training_arrays(df, seq_len, bars_to_skip):
    """
    Using columns:
      open, high, low, close, stochK, stochD, doji, hammer, bull_engulf, bear_engulf
    => total 10 features.
    Then skip 'bars_to_skip' to define next 4H candle direction label.
    """
    # We'll convert to Nx10 np.array first
    needed_cols = [
        'open','high','low','close','stochK','stochD',
        'doji','hammer','bull_engulf','bear_engulf'
    ]
    arr = df[needed_cols].to_numpy()

    X, y = [], []
    N = len(arr)
    max_i = N - seq_len - bars_to_skip
    if max_i < 0:
        return np.array([]), np.array([])

    for i in range(max_i+1):
        snippet = arr[i:i+seq_len]  # shape => (seq_len, 10)
        X.append(snippet)

        # next 4H candle open/close
        target_open  = arr[i+seq_len][0]  # open
        target_close = arr[i+seq_len+bars_to_skip-1][3]  # close
        label_val = 1.0 if target_close > target_open else 0.0
        y.append(label_val)

    return np.array(X), np.array(y).reshape(-1,1)


##############################################################################
# 5) BUILD CNN+LSTM
##############################################################################
def build_cnn_lstm_model(seq_len=3, num_features=10):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filters=16, kernel_size=2, activation='relu',
                            input_shape=(seq_len, num_features)))
    model.add(layers.MaxPooling1D(pool_size=1))

    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(16))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


##############################################################################
# 6) TRAIN + PREDICT FOR A SINGLE TIMEFRAME
##############################################################################
def train_and_predict(csv_file, timeframe_label, seq_len, bars_to_skip):
    """
    1) Load csv
    2) Compute stoch, pattern columns
    3) Build training samples
    4) Train CNN+LSTM
    5) Inference on final snippet
    6) Return label("Bullish"/"Bearish") + probability
    """
    df = load_csv_tab_delimited(csv_file)
    # Add stoch, pattern columns
    df = compute_stochastics_and_patterns(df, stoch_period=14)

    # Build (X,y)
    X, y = make_training_arrays(df, seq_len=seq_len, bars_to_skip=bars_to_skip)
    if len(X) == 0:
        print(f"{timeframe_label}: Not enough data after feature engineering.")
        return None, None

    # Flatten for scaling
    # shape => (N*seq_len, 10)
    N = X.shape[0]
    feature_count = X.shape[2]
    X_2d = X.reshape(N*seq_len, feature_count)

    # Scale
    scaler = MinMaxScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)
    X_scaled = X_2d_scaled.reshape(N, seq_len, feature_count)

    # Train/Val/Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    split_idx = int(len(X_train_full)*0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val   = X_train_full[split_idx:]
    y_val   = y_train_full[split_idx:]

    # Build model
    model = build_cnn_lstm_model(seq_len=seq_len, num_features=feature_count)
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=15,
              batch_size=64,
              verbose=0)

    # Real-time inference
    if len(X_scaled) < 1:
        print(f"{timeframe_label}: Not enough final samples to infer.")
        return None, None

    final_sample = X_scaled[-1:]  # shape(1, seq_len, feature_count)
    prob_bullish = model.predict(final_sample)[0][0]
    label = "Bullish" if prob_bullish >= 0.5 else "Bearish"

    # Print
    last_idx = len(df)-1
    last_date = df.loc[last_idx, 'date']
    last_time = df.loc[last_idx, 'time']

    print(f"{timeframe_label} timeframe prediction for next 4H candle:")
    print(f"Last known candle in {csv_file}: {last_date} {last_time}")
    print(f"Probability of Bullish: {prob_bullish:.4f}")
    print(f"Prediction: {label}\n")

    return label, prob_bullish


##############################################################################
# 7) MASTER SCRIPT (ALL 6 TIMEFRAMES)
##############################################################################
def main():
    """
    We'll define 6 CSVs/timeframes:
      4H  -> skip=1
      2H  -> skip=2
      1H  -> skip=4
      30m -> skip=8
      15m -> skip=16
      5m  -> skip=48
    We'll unify final results and see if all match => "Trade" or "NO TRADE".
    Each model also includes stoch, pattern columns => (seq_len, 10) input.
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
    for label, csv_file, skip_bars in timeframe_csvs:
        pred_label, prob = train_and_predict(csv_file, label, seq_len=3, bars_to_skip=skip_bars)
        predictions.append(pred_label)

    # Decide final trade
    filtered = [p for p in predictions if p is not None]
    if len(filtered) < 6:
        final_call = "NO TRADE (some timeframe missing data)"
    else:
        if all(p == "Bullish" for p in filtered):
            final_call = "BULLISH"
        elif all(p == "Bearish" for p in filtered):
            final_call = "BEARISH"
        else:
            final_call = "NO TRADE"

    print(f"Trade: {final_call} (ONLY IF ALL AGREE!!!)\n")


if __name__ == "__main__":
    main()
