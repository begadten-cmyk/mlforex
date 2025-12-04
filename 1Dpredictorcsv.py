import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 1) LOAD CSV WITH TAB-DELIMITED 4H DATA, KEEP <DATE> & <TIME>
##############################################################################
def load_csv_tab_delimited(csv_file="eurgbp_4H_5_years.csv"):
    """
    Reads a tab-delimited CSV with columns like:
      <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>.
    Keeps <DATE> and <TIME> so we can show the final bar's date/time.
    Renames <OPEN>, <HIGH>, <LOW>, <CLOSE> => open, high, low, close.
    Drops <TICKVOL>, <VOL>, <SPREAD> if present.
    Returns a DataFrame with columns: [date, time, open, high, low, close].
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

    # Drop unneeded columns
    df.drop(columns=['<TICKVOL>', '<VOL>', '<SPREAD>'], inplace=True, errors='ignore')

    # Ensure we have date/time/open/high/low/close
    for col in ['date','time','open','high','low','close']:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    return df[['date','time','open','high','low','close']].reset_index(drop=True)

##############################################################################
# 2) CREATE SEQUENCES BUT LABEL THE *NEXT DAY* (6×4H) AS BULLISH OR BEARISH
##############################################################################
def convert_ohlc_to_candle_features(ohlc_row):
    """
    Single [open, high, low, close] => [candle_type, wicks_up, wicks_down, body_size].
      candle_type = 1 if bullish, else 0.
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


def create_sequences_for_next_day(ohlc_array, seq_len=3):
    """
    ohlc_array: Nx4 of [open, high, low, close] (4H bars)
    seq_len: how many 4H bars to use as input

    We define 'the next day' as the 6 bars *after* the input sequence.
      day_open = the 'open' of bar #1 in that day,
      day_close = the 'close' of bar #6 in that day.
    If day_close > day_open => bullish (1), else 0.

    Returns:
      X -> shape [M, seq_len, 4]  (candlestick features per each input block)
      y -> shape [M, 1]  (the next day's bullish/bearish label)
    """
    X, y = [], []
    N = len(ohlc_array)

    # For each index i, we take 'seq_len' bars as the "past",
    # then skip ahead 6 bars to define the next day's candle direction.
    for i in range(N - seq_len - 6):
        # 1) Input sequence: i .. i+seq_len-1
        slice_ohlc = ohlc_array[i : i+seq_len]

        # Convert to candlestick features
        features = [convert_ohlc_to_candle_features(row) for row in slice_ohlc]
        features = np.array(features)  # shape (seq_len,4)

        # 2) Next Day's range: i+seq_len .. i+seq_len+5
        day_start_idx = i + seq_len
        day_end_idx   = i + seq_len + 5
        day_open  = ohlc_array[day_start_idx][0]  # open of bar day_start_idx
        day_close = ohlc_array[day_end_idx][3]    # close of bar day_end_idx
        next_day_type = 1.0 if day_close > day_open else 0.0

        X.append(features)
        y.append(next_day_type)

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
# 4) MAIN - MINIMAL PRINTS, PREDICT *NEXT DAY* DIRECTION
##############################################################################
def main():
    # 1) Load CSV (4H data), keep date/time
    df = load_csv_tab_delimited("eurgbp_4H_5_years.csv")

    # 2) Convert to Nx4 array for input
    ohlc_array = df[['open','high','low','close']].to_numpy()

    # 3) Create sequences (example: last 3 bars => next day's candle direction)
    seq_len = 3
    X, y = create_sequences_for_next_day(ohlc_array, seq_len=seq_len)
    # If we have fewer bars than 3+6, X and y might be empty

    # 4) Scale features
    scaler = MinMaxScaler()
    X_flat = X.reshape(-1, X.shape[2])  # shape (num_samples*seq_len, 4)
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])

    # 5) Train/test split (80/20, then 80/20 of train => val)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    split_idx = int(len(X_train_full)*0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val   = X_train_full[split_idx:]
    y_val   = y_train_full[split_idx:]

    # 6) Build & train (no logs, verbose=0)
    model = build_lstm_model(seq_len=seq_len, num_features=4)
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=15, batch_size=64, verbose=0)

    # 7) Prepare the "last" input to predict the *next day*:
    #    We'll use the final 'seq_len' bars from the data,
    #    then define day_open/day_close for the "next day" 6 bars after that.
    #    But if we want the *very last possible day*, we have to see if there's enough data.
    #    We'll do it manually: The last index that was used in X was (N - seq_len - 6 - 1).
    #    We'll just re-create the last sample from X, y:
    if len(X) == 0:
        print("Not enough data for next-day predictions.")
        return

    # The *final sample* in X is X[-1], which used bars ending at index (N - 6 - 1).
    # That sample's 'label' is y[-1]. So let's see how we do in real life:
    final_input = X[-1:]  # shape (1, seq_len, 4)

    # Scale is already done, so final_input is scaled? Actually, final_input is from X, so it's already scaled
    # We can just predict it
    prob_bullish = model.predict(final_input)[0][0]
    label = "Bullish" if prob_bullish >= 0.5 else "Bearish"

    # 8) Identify the date/time for the "last known candle"
    #    The last sample in X used bars up to index = (N - 6 - 1) + seq_len - 1
    #    That might be confusing. Let's break it down:

    # The final sample in X is constructed from i = (N - seq_len - 6) (the last loop iteration in create_sequences_for_next_day).
    # That means it used bars from [i, i+1, i+2], i+2 = i + (seq_len-1).
    # The last bar used for the input is i+seq_len-1.
    i_final = (len(ohlc_array) - seq_len - 6)  # the last iteration
    last_bar_idx_for_input = i_final + seq_len - 1

    # We can read that bar's date/time from df
    last_date = df.iloc[last_bar_idx_for_input]['date']
    last_time = df.iloc[last_bar_idx_for_input]['time']

    # Print minimal final result
    print(f"\n--- NEXT DAY CANDLE PREDICTION ---")
    print(f"Last known 4H bar: {last_date} {last_time}")
    print(f"Will the *next day* (the next 6×4H bars) be Bullish or Bearish?")
    print(f"Probability of Bullish = {prob_bullish:.4f}")
    print(f"Prediction: {label}")

if __name__ == "__main__":
    main()
