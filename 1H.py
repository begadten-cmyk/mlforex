import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 1) LOAD 1H CSV
##############################################################################
def load_csv_tab_delimited(csv_file="eurgbp1h.csv"):
    """
    Reads a tab-delimited CSV with columns like:
      <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, ...
    Renames them to [date, time, open, high, low, close], drops extras.
    Returns DataFrame with columns: [date, time, open, high, low, close].

    NOTE: The filename is "eurusd1h.csv" but this code is for 1H data.
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

    df.drop(columns=['<TICKVOL>', '<VOL>', '<SPREAD>'], inplace=True, errors='ignore')

    for col in ['date','time','open','high','low','close']:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    return df[['date','time','open','high','low','close']].reset_index(drop=True)

##############################################################################
# 2) 1H -> Candle Features
##############################################################################
def candle_features_1h(row):
    """
    [open, high, low, close] -> [candle_type, wicks_up, wicks_down, body_size].
    candle_type=1 if bullish else 0
    """
    op, hi, lo, cl = row
    candle_type = 1.0 if cl > op else 0.0
    body_size = abs(cl - op)
    if cl > op:
        wick_up = hi - cl
        wick_down = op - lo
    else:
        wick_up = hi - op
        wick_down = cl - lo

    return np.array([candle_type, wick_up, wick_down, body_size], dtype=np.float32)

##############################################################################
# 3) CREATE TRAINING SAMPLES (NEXT 4-HOUR CANDLE)
##############################################################################
def make_train_samples(ohlc_array, seq_len=3):
    """
    - ohlc_array: Nx4 of [open, high, low, close] (1H data).
    - We skip 4 bars after the snippet to define the next 4-hour candle:
        4H_open  = ohlc_array[i+seq_len][0]
        4H_close = ohlc_array[i+seq_len+3][3]  # total of 4 bars = 4 hours
    - label = 1 if 4H_close > 4H_open else 0
    """
    X = []
    y = []
    N = len(ohlc_array)
    max_i = N - seq_len - 4
    if max_i < 0:
        return np.array([]), np.array([])

    for i in range(max_i + 1):
        snippet = ohlc_array[i : i+seq_len]
        feat_seq = [candle_features_1h(r) for r in snippet]
        feat_seq = np.array(feat_seq)  # shape (seq_len,4)

        # define next 4H candle open/close
        four_h_open  = ohlc_array[i+seq_len][0]
        four_h_close = ohlc_array[i+seq_len+3][3]
        label_val = 1.0 if four_h_close > four_h_open else 0.0

        X.append(feat_seq)
        y.append(label_val)

    X = np.array(X)
    y = np.array(y).reshape(-1,1)
    return X, y

##############################################################################
# 4) CNN+LSTM MODEL (Replaces old LSTM-only)
##############################################################################
def build_lstm_model(seq_len=3, num_features=4):
    """
    CNN+LSTM hybrid for 1H data -> next 4H label.
    """
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
# 5) MAIN
##############################################################################
def main():
    #-----------------------------------------------------------------------
    # A) LOAD & PREP (1H data)
    #-----------------------------------------------------------------------
    df = load_csv_tab_delimited("eurgbp1h.csv")  
    ohlc_array = df[['open','high','low','close']].to_numpy()

    #-----------------------------------------------------------------------
    # B) CREATE TRAIN/VAL/TEST SAMPLES
    #-----------------------------------------------------------------------
    seq_len = 3
    X, y = make_train_samples(ohlc_array, seq_len=seq_len)
    if len(X) == 0:
        print("Not enough data to form any training samples with next-4H labeling.")
        return

    scaler = MinMaxScaler()
    X_2d = X.reshape(-1, X.shape[2])  
    X_2d_scaled = scaler.fit_transform(X_2d)
    X_scaled = X_2d_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    split_idx = int(len(X_train_full)*0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val   = X_train_full[split_idx:]
    y_val   = y_train_full[split_idx:]

    model = build_lstm_model(seq_len=seq_len, num_features=4)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        verbose=0
    )

    #-----------------------------------------------------------------------
    # C) REAL-TIME INFERENCE
    #-----------------------------------------------------------------------
    if len(ohlc_array) < seq_len:
        print("Not enough data to do real-time inference.")
        return

    final_snippet = ohlc_array[-seq_len:]
    snippet_feats = [candle_features_1h(r) for r in final_snippet]
    snippet_feats = np.array(snippet_feats).reshape(1, seq_len, 4)

    scaler2 = MinMaxScaler()
    scaler2.fit(X_2d)
    snippet_2d = snippet_feats.reshape(seq_len, 4)
    snippet_scaled = scaler2.transform(snippet_2d)
    snippet_scaled = snippet_scaled.reshape(1, seq_len, 4)

    prob_bullish = model.predict(snippet_scaled)[0][0]
    label = "Bullish" if prob_bullish >= 0.5 else "Bearish"

    last_idx = len(df)-1
    last_date = df.iloc[last_idx]['date']
    last_time = df.iloc[last_idx]['time']

    print("\n--- NEXT 4H CANDLE PREDICTION (using 1H data) ---")
    print(f"Last known 1H candle in your CSV: {last_date} {last_time}")
    print(f"Probability of Bullish: {prob_bullish:.4f}")
    print(f"Prediction: {label}")

if __name__ == "__main__":
    main()
