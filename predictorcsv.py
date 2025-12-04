import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 1) LOAD CSV, KEEP <DATE> AND <TIME>, RENAME OTHERS
##############################################################################
def load_csv_tab_delimited(csv_file="eurgbp_4H_5_years.csv"):
    """
    Reads a tab-delimited CSV with columns like:
      <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>, etc.
    Keeps <DATE> and <TIME> for reference, renames <OPEN>, <HIGH>, <LOW>, <CLOSE> 
    to open, high, low, close, and drops <TICKVOL>, <VOL>, <SPREAD> if present.
    Returns a DataFrame with columns: [date, time, open, high, low, close].
    """
    df = pd.read_csv(csv_file, sep='\t', engine='python')

    # Rename columns to something consistent
    df.rename(columns={
        '<DATE>': 'date',
        '<TIME>': 'time',
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close'
    }, inplace=True, errors='ignore')

    # Drop columns we do not need
    df.drop(columns=['<TICKVOL>', '<VOL>', '<SPREAD>'], inplace=True, errors='ignore')

    # Ensure we have date, time, open, high, low, close
    for col in ['date','time','open','high','low','close']:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Return just these 6 columns in that order
    return df[['date','time','open','high','low','close']].reset_index(drop=True)

##############################################################################
# 2) FEATURE ENGINEERING
##############################################################################
def convert_ohlc_to_candle_features(ohlc_row):
    """
    [open, high, low, close] -> [candle_type, wicks_up, wicks_down, body_size].
      candle_type = 1 if bullish, else 0
    """
    op, hi, lo, cl = ohlc_row
    candle_type = 1.0 if cl > op else 0.0
    body_size = abs(cl - op)

    if cl > op:
        wick_up = hi - cl
        wick_down = op - lo
    else:
        wick_up = hi - op
        wick_down = cl - lo

    return np.array([candle_type, wick_up, wick_down, body_size], dtype=np.float32)

def create_sequences(ohlc_array, seq_len=3):
    """
    Given Nx4 array of [open, high, low, close],
    returns:
      X: shape [N-seq_len, seq_len, 4]
      y: shape [N-seq_len, 1]
    where y is the next candle's type (0 or 1).
    """
    X, y = [], []
    N = len(ohlc_array)
    for i in range(N - seq_len):
        slice_ohlc = ohlc_array[i : i+seq_len]
        # Convert each row to candle features
        features = [convert_ohlc_to_candle_features(row) for row in slice_ohlc]
        features = np.array(features)  # shape (seq_len,4)

        # Label is the next row's candle type
        next_row = ohlc_array[i + seq_len]
        next_type = 1.0 if next_row[3] > next_row[0] else 0.0
        X.append(features)
        y.append(next_type)

    X = np.array(X)
    y = np.array(y).reshape(-1,1)
    return X, y

##############################################################################
# 3) BUILD & TRAIN LSTM
##############################################################################
def build_lstm_model(seq_len=3, num_features=4):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(32, return_sequences=True, input_shape=(seq_len, num_features)))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(16))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

##############################################################################
# 4) MAIN SCRIPT - MINIMAL OUTPUT
##############################################################################
def main():
    # Load CSV, keep date/time
    df = load_csv_tab_delimited("eurgbp_4H_5_years.csv")

    # Extract just the OHLC columns for training
    # shape: (N, 4) => [open, high, low, close]
    ohlc_df = df[['open','high','low','close']].copy()
    ohlc_array = ohlc_df.to_numpy()

    # Create sequences
    seq_len = 3
    X, y = create_sequences(ohlc_array, seq_len=seq_len)

    # Scale candle features
    scaler = MinMaxScaler()
    X_2d = X.reshape(-1, X.shape[2])
    X_2d_scaled = scaler.fit_transform(X_2d)
    X_scaled = X_2d_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Split train/test (80/20) with no printing
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val   = X_train_full[split_idx:]
    y_val   = y_train_full[split_idx:]

    # Build & train model (no logs)
    model = build_lstm_model(seq_len=seq_len, num_features=4)
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=15, 
              batch_size=64, 
              verbose=0)  # <--- NO TRAINING LOGS

    # Prepare last 'seq_len' rows for prediction
    last_seq = ohlc_array[-seq_len:]
    last_feats = [convert_ohlc_to_candle_features(row) for row in last_seq]
    last_feats = np.array(last_feats).reshape(1, seq_len, 4)

    # Scale with same scaler
    last_2d = last_feats.reshape(seq_len, 4)
    last_2d_scaled = scaler.transform(last_2d)
    last_scaled = last_2d_scaled.reshape(1, seq_len, 4)

    # Predict
    prob_bullish = model.predict(last_scaled)[0][0]
    label = "Bullish" if prob_bullish >= 0.5 else "Bearish"

    # Figure out the date/time of the last known candle
    last_index = len(df) - 1
    last_date  = df.iloc[last_index]['date']
    last_time  = df.iloc[last_index]['time']

    # Print just the final line
    print(f"\n--- NEXT CANDLE PREDICTION ---")
    print(f"Last known candle: {last_date} {last_time}")
    print(f"Probability of Bullish: {prob_bullish:.4f}")
    print(f"Prediction: {label}")

if __name__ == "__main__":
    main()
