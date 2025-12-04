import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

# ADDED: tiingo
from tiingo import TiingoClient

############################################################################
# 1) CONFIG
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

# We'll create a small helper mapping from "Yahoo style" to Tiingo style.
# In Tiingo, Forex tickers often look like "fx/eurgbp", etc.
TIINGO_MAP = {
    "GBPJPY=X": "fx/gbpjpy",
    "EURUSD=X": "fx/eurusd",
    "USDCAD=X": "fx/usdcad",
    "USDJPY=X": "fx/usdjpy",
    "GBPUSD=X": "fx/gbpusd",
    "EURGBP=X": "fx/eurgbp"
}

TIINGO_API_KEY = "b8e09c0daa1f51989692145b2b68d8c286c7dd2f"

# Multi-window / multi-horizon hyper-params
W_MAX = 5   # up to 5-day lookback
H_MAX = 3   # up to 3-day horizon

############################################################################
# HELPER: CLEAN DATA
############################################################################
def clean_data(df):
    """
    Minimal cleaning: sort, drop duplicates, remove NaNs.
    (You can add outlier filters, etc.)
    """
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.dropna(inplace=True)
    return df

############################################################################
# DATA FETCH (TIINGO) using get_dataframe(...)
############################################################################
def fetch_tiingo(symbol):
    """
    Replaces the alpha_vantage fetch with Tiingo logic, now using get_dataframe(...).
    We'll use the 'fx/eurgbp' style from TIINGO_MAP, fetch daily data from
    start_date to end_date, and return a DataFrame with columns:
    Open, High, Low, Close, Volume.
    """
    config = {}
    config['session'] = True
    config['api_key'] = TIINGO_API_KEY
    client = TiingoClient(config)

    if symbol not in TIINGO_MAP:
        print(f"No TIINGO_MAP entry for {symbol}, skipping.")
        return pd.DataFrame()

    tiingo_ticker = TIINGO_MAP[symbol]

    # Convert to ISO date
    start_iso = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_iso   = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    try:
        # Using client.get_dataframe(...) for daily historical data
        # By default, it returns columns like 'adjClose','adjHigh', etc.
        # We'll rename them to standard columns after retrieval.
        prices_df = client.get_dataframe(
            tickers=[tiingo_ticker],
            startDate=start_iso,
            endDate=end_iso,
            frequency='daily'
        )
    except Exception as e:
        print(f"Tiingo fetch error for {symbol}: {str(e)}")
        return pd.DataFrame()

    if prices_df.empty:
        print(f"No data returned for {symbol} from Tiingo.")
        return pd.DataFrame()

    # If multi-ticker, the DF can be multi-index. We'll handle single ticker case:
    if isinstance(prices_df.columns, pd.MultiIndex):
        # For a single ticker, columns look like e.g. ('fx/eurgbp','close'), etc.
        # We'll flatten them:
        prices_df.columns = [c[-1] for c in prices_df.columns]

    # Now we expect columns like 'close','high','low','open','volume'
    # Rename them to match our pipeline
    rename_map = {}
    if 'open' in prices_df.columns:   rename_map['open']   = 'Open'
    if 'high' in prices_df.columns:   rename_map['high']   = 'High'
    if 'low' in prices_df.columns:    rename_map['low']    = 'Low'
    if 'close' in prices_df.columns:  rename_map['close']  = 'Close'
    if 'volume' in prices_df.columns: rename_map['volume'] = 'Volume'

    prices_df.rename(columns=rename_map, inplace=True)

    # Some columns might be missing (like 'Volume'). We can fill or set them to 0
    for col in ['Open','High','Low','Close','Volume']:
        if col not in prices_df.columns:
            prices_df[col] = 0.0

    prices_df.sort_index(inplace=True)
    df = clean_data(prices_df)
    return df

############################################################################
# BUILD DATASET: MULTI-WINDOW, MULTI-HORIZON
############################################################################
def build_dataset(df, w_max, h_max):
    """
    Build a big (X, Y) dataset enumerating:
     - w in [1..w_max]
     - h in [1..h_max]
    We'll pad each input window to w_max*5, and each output to 3*h_max,
    so shapes are consistent.
    """

    df = df[['Open','High','Low','Close','Volume']].copy()
    df_np = df.values
    dates = df.index.to_list()
    n = len(df_np)

    all_X = []
    all_Y = []
    meta_info = []

    max_window_dim = w_max * 5   # if w=5, we have 5 days * 5 columns = 25
    max_output_dim = 3 * h_max   # if h=3, we have 3 days * 3 columns = 9

    for w in range(1, w_max+1):
        for h in range(1, h_max+1):
            for t in range(w, n - h):
                # gather last w candles for input
                window_data = df_np[t-w : t, :]  # shape (w, 5)
                window_flat_small = window_data.flatten()  # length = w*5

                # pad to length = w_max*5
                window_padded = np.zeros(max_window_dim, dtype=float)
                start_idx = max_window_dim - window_flat_small.size
                window_padded[start_idx:] = window_flat_small

                # embed w,h
                feature = np.concatenate([window_padded, [float(w), float(h)]])

                # gather next h days => each day = (Low,High,Close)
                future_data = df_np[t : t+h, :]  # shape (h, 5)
                out_list = []
                for fday in range(h):
                    day_low  = future_data[fday, 2]
                    day_high = future_data[fday, 1]
                    day_close= future_data[fday, 3]
                    out_list += [day_low, day_high, day_close]
                out_arr = np.array(out_list, dtype=float)  # shape (3*h,)

                # pad to 3*h_max
                out_full = np.full(max_output_dim, np.nan, dtype=float)
                out_full[: out_arr.size] = out_arr

                all_X.append(feature)
                all_Y.append(out_full)
                meta_info.append((w, h, t, dates[t]))

    all_X = np.array(all_X, dtype=float)
    all_Y = np.array(all_Y, dtype=float)
    return all_X, all_Y, meta_info, df

############################################################################
# MAIN
############################################################################
def main():
    symbol = "EURGBP=X"  # example
    print(f"Fetching data for {symbol} via Tiingo...")
    df = fetch_tiingo(symbol)
    if df.empty:
        print("Data is empty, aborting.")
        return

    print("Building multi-window multi-horizon dataset...")
    X, Y, meta_info, df_clean = build_dataset(df, W_MAX, H_MAX)
    print(f"Dataset size: {X.shape}, {Y.shape}")

    # remove any rows with NaN in Y
    valid_mask = ~np.isnan(Y).any(axis=1)
    X = X[valid_mask]
    Y = Y[valid_mask]
    meta_info_clean = [m for (m,v) in zip(meta_info, valid_mask) if v]
    print("After dropping NaNs, dataset size:", X.shape, Y.shape)

    # TRAIN
    base_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model = MultiOutputRegressor(base_reg)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    meta_train, meta_test = meta_info_clean[:train_size], meta_info_clean[train_size:]

    model.fit(X_train, Y_train)
    print("Model trained on multi-window multi-horizon data.")

    # Evaluate
    Y_pred = model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse_val = mean_squared_error(Y_test, Y_pred)
    rmse_val = np.sqrt(mse_val)

    # Print test info
    print(f"\n{symbol} - Overall RMSE (all windows/horizons): {rmse_val:.4f}")

    # Show last 5 rows of df_clean, plus some columns for clarity
    tail_5 = df_clean.tail(5)[['Open','High','Low','Close']]
    print(tail_5)

    # We'll define the 'latest known close' as the last row's Close
    last_idx = df_clean.index[-1]
    last_close = df_clean.iloc[-1]['Close']

    next_day_date = last_idx + pd.Timedelta(days=1)
    print(f"\nLatest known close on {last_idx.date()}: {last_close:.5f}")
    print(f"Predicting the next trading day's candle, presumably for {next_day_date.date()}.")

    # Example usage: produce next-day candle from w=W_MAX, h=1
    last_w_data = df_clean[['Open','High','Low','Close','Volume']].values[-W_MAX:]
    last_w_flat = last_w_data.flatten()

    # pad to length = W_MAX*5
    max_window_dim = W_MAX*5
    window_padded = np.zeros(max_window_dim, dtype=float)
    start_idx = max_window_dim - last_w_flat.size
    window_padded[start_idx:] = last_w_flat

    # define w=W_MAX,h=1 => add [float(W_MAX), 1.0]
    final_feature = np.concatenate([window_padded, [float(W_MAX), 1.0]])
    # predict => shape (3*h_max=3*3=9)
    pred_out = model.predict(final_feature.reshape(1, -1))[0]
    # the first 3 values = day+1 (Low,High,Close)
    next_day_low  = pred_out[0]
    next_day_high = pred_out[1]
    next_day_close= pred_out[2]

    print(f"\nPredicted Tomorrow's High: {next_day_high:.5f}, Low: {next_day_low:.5f}, Close: {next_day_close:.5f}")

if __name__ == "__main__":
    main()
