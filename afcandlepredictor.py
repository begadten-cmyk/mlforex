import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

# ADDED: alpha_vantage import
from alpha_vantage.foreignexchange import ForeignExchange

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

# We'll create a small helper mapping from your "Yahoo" style pairs
# to (from_symbol, to_symbol) for Alpha Vantage:
AV_MAP = {
    "GBPJPY=X": ("GBP", "JPY"),
    "EURUSD=X": ("EUR", "USD"),
    "USDCAD=X": ("USD", "CAD"),
    "USDJPY=X": ("USD", "JPY"),
    "GBPUSD=X": ("GBP", "USD"),
    "EURGBP=X": ("EUR", "GBP")
}

# Your Alpha Vantage API key:
ALPHA_VANTAGE_API_KEY = "INOY4FHB1PZYUPT8"

############################################################################
# 2) HELPER: CLEAN DATA
############################################################################
def clean_data(df, pair):
    """
    Remove out-of-range prices specific to each pair,
    remove extreme daily returns beyond +/- 10%,
    ensure sorted index and remove duplicates.
    Also drops rows that become NaN.
    """
    if df.empty:
        return df
    
    # Basic rename if needed
    if len(df.columns) == 5:
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
    
    # 1) Pair-based price range filters
    if pair == "EURUSD=X":
        df = df[(df['Close'] > 0.5) & (df['Close'] < 2.0)]
        df = df[(df['High']  > 0.5) & (df['High']  < 2.0)]
        df = df[(df['Low']   > 0.5) & (df['Low']   < 2.0)]
    elif pair == "GBPJPY=X":
        df = df[(df['Close'] > 50) & (df['Close'] < 300)]
        df = df[(df['High']  > 50) & (df['High']  < 300)]
        df = df[(df['Low']   > 50) & (df['Low']   < 300)]
    elif pair == "USDCAD=X":
        df = df[(df['Close'] > 0.5) & (df['Close'] < 2.0)]
        df = df[(df['High']  > 0.5) & (df['High']  < 2.0)]
        df = df[(df['Low']   > 0.5) & (df['Low']   < 2.0)]
    elif pair == "USDJPY=X":
        df = df[(df['Close'] > 50) & (df['Close'] < 300)]
        df = df[(df['High']  > 50) & (df['High']  < 300)]
        df = df[(df['Low']   > 50) & (df['Low']   < 300)]
    elif pair == "GBPUSD=X":
        df = df[(df['Close'] > 0.5) & (df['Close'] < 2.5)]
        df = df[(df['High']  > 0.5) & (df['High']  < 2.5)]
        df = df[(df['Low']   > 0.5) & (df['Low']   < 2.5)]
    elif pair == "EURGBP=X":
        df = df[(df['Close'] > 0.4) & (df['Close'] < 1.5)]
        df = df[(df['High']  > 0.4) & (df['High']  < 1.5)]
        df = df[(df['Low']   > 0.4) & (df['Low']   < 1.5)]
    
    # Sort and remove duplicates
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # Compute daily returns for outlier check
    df['Daily_Return_Tmp'] = df['Close'].pct_change()
    df = df[df['Daily_Return_Tmp'].abs() < 0.10]  # remove ±10% daily jumps
    df.drop(columns=['Daily_Return_Tmp'], inplace=True)

    df.dropna(inplace=True)
    return df

############################################################################
# 3) TRAINING + PREDICTION
############################################################################

def train_and_predict_for_pair(pair):
    print(f"\n=== PROCESSING {pair} ===")

    from_symbol, to_symbol = AV_MAP[pair]
    fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY, output_format='json')

    # "full" to get the entire historical range
    raw_data, _ = fx.get_currency_exchange_daily(
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        outputsize='full'
    )
    if not raw_data:
        print("No data downloaded from Alpha Vantage. Skipping.")
        return

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(raw_data, orient='index')
    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low':  'Low',
        '4. close': 'Close'
    }, inplace=True)
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    # Sort by date ascending
    df.sort_index(inplace=True)

    # If there's no 'Volume' column, create one (avoid KeyError)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # Filter by your chosen date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    if df.empty:
        print("No data in the specified date range. Skipping.")
        return

    # Clean
    df = clean_data(df, pair)
    if df.empty:
        print("No usable data after cleaning. Skipping.")
        return
    
    if len(df.columns) == 5:
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
    
    # SHIFT: create tomorrow's High, Low, and Close
    df['Tomorrow_High']   = df['High'].shift(-1)
    df['Tomorrow_Low']    = df['Low'].shift(-1)
    df['Tomorrow_Close']  = df['Close'].shift(-1)
    df.dropna(inplace=True)

    # Feature engineering
    df['LogClose'] = np.log(df['Close'])
    df['Daily_Return'] = df['Close'].pct_change().clip(-0.05, 0.05)
    df['MA_5']  = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df.dropna(inplace=True)

    # Log transform tomorrow's High, Low, Close
    df = df[(df['Tomorrow_High']>0) & (df['Tomorrow_Low']>0) & (df['Tomorrow_Close']>0)]
    df['LogTomorrow_High']   = np.log(df['Tomorrow_High'])
    df['LogTomorrow_Low']    = np.log(df['Tomorrow_Low'])
    df['LogTomorrow_Close']  = np.log(df['Tomorrow_Close'])
    df.dropna(inplace=True)

    if df.empty:
        print("No rows left for training after log transform. Skipping.")
        return

    # Features and targets
    feature_cols = ['LogClose','Daily_Return','MA_5','MA_10','Volume']
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    Y = df[['LogTomorrow_High','LogTomorrow_Low','LogTomorrow_Close']].values

    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Multi-output random forest
    base_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model = MultiOutputRegressor(base_reg)

    model.fit(X_train, Y_train)

    y_pred_test = model.predict(X_test)

    # Exponentiate predictions
    test_pred_high  = np.exp(y_pred_test[:,0])
    test_pred_low   = np.exp(y_pred_test[:,1])
    test_pred_close = np.exp(y_pred_test[:,2])

    test_true_high  = np.exp(Y_test[:,0])
    test_true_low   = np.exp(Y_test[:,1])
    test_true_close = np.exp(Y_test[:,2])

    from sklearn.metrics import mean_squared_error
    rmse_high   = np.sqrt(mean_squared_error(test_true_high,  test_pred_high))
    rmse_low    = np.sqrt(mean_squared_error(test_true_low,   test_pred_low))
    rmse_close  = np.sqrt(mean_squared_error(test_true_close, test_pred_close))

    print(f"{pair} - Test RMSE High: {rmse_high:.4f}, Low: {rmse_low:.4f}, Close: {rmse_close:.4f}")

    # Final "latest row" prediction
    last_row = df.iloc[-1]
    X_latest = last_row[feature_cols].values.reshape(1, -1)
    pred_log = model.predict(X_latest)[0]  # shape (3,)
    pred_high  = np.exp(pred_log[0])
    pred_low   = np.exp(pred_log[1])
    pred_close = np.exp(pred_log[2])

    print(f"Latest known close: {last_row['Close']:.5f}")
    print(f"Predicted Tomorrow's High: {pred_high:.5f}, Low: {pred_low:.5f}, Close: {pred_close:.5f}")

def main():
    for pair in pairs:
        train_and_predict_for_pair(pair)

if __name__ == "__main__":
    main()
