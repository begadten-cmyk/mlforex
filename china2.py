import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

##############################################################################
# 1) CSV LOADER
##############################################################################
def load_csv_tab_delimited(csv_file):
    """
    Reads a tab-delimited CSV with columns:
      <DATE>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>
    Returns a DataFrame with columns: [date, open, high, low, close].
    """
    df = pd.read_csv(csv_file, sep='\t', engine='python')
    
    # Rename columns
    df.rename(columns={
        '<DATE>': 'date',
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close'
    }, inplace=True, errors='ignore')
    
    # Drop unnecessary columns
    df.drop(columns=['<TICKVOL>', '<VOL>', '<SPREAD>'], errors='ignore', inplace=True)
    
    # Check required columns
    required_cols = ['date', 'open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV '{csv_file}' missing columns: {missing_cols}")
    
    # Convert to numeric and handle missing
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

##############################################################################
# 2) ENHANCED INDICATORS
##############################################################################
def compute_technical_indicators(df, stoch_period=14):
    df = df.copy()
    
    # Stochastic Oscillator
    lows = df['low'].rolling(window=stoch_period, min_periods=1).min()
    highs = df['high'].rolling(window=stoch_period, min_periods=1).max()
    df['stochK'] = ((df['close'] - lows) / (highs - lows + 1e-9)) * 100
    df['stochD'] = df['stochK'].rolling(window=3, min_periods=1).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # MAs
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # Bollinger
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + 2 * df['BB_std']
    df['BB_lower'] = df['MA20'] - 2 * df['BB_std']

    # ATR
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Lagged close features
    for lag in [1, 2, 3]:
        df[f'close_lag{lag}'] = df['close'].shift(lag)
    
    # Fill
    df.bfill(inplace=True)
    return df

##############################################################################
# 3) FEATURE ENGINEERING
##############################################################################
def create_features_labels(df, bars_to_skip=1):
    """
    We'll define X from the columns of interest, y => next bar's close>open
    skip=1 => next bar is 1 in the future (like next 4H, etc.)
    """
    feat_cols = [
        'open','high','low','close',
        'stochK','stochD','RSI','MACD','Signal','MA50','MA200','BB_upper','BB_lower','ATR'
    ] + [f'close_lag{i}' for i in [1,2,3]]
    
    arr = df[feat_cols].to_numpy()
    N = len(df)
    X, y = [], []
    max_i = N - bars_to_skip
    for i in range(N):
        if i<max_i:
            feats = arr[i]
            # label
            future_open  = df.loc[i+bars_to_skip,'open']
            future_close = df.loc[i+bars_to_skip,'close']
            lab = 1 if future_close>future_open else 0
            X.append(feats)
            y.append(lab)
    X = np.array(X)
    y = np.array(y)
    return X,y, feat_cols

##############################################################################
# 4) BUILD LSTM
##############################################################################
def build_lstm_model(input_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    # smaller architecture to speed up training but keep capacity
    model.add(LSTM(30, return_sequences=True, input_shape=(input_dim, 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(30, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

##############################################################################
# 5) TRAIN & ENSEMBLE (XGB + RF + LSTM) SINGLE TRAIN/TEST
##############################################################################
def train_predict_ensemble(csv_file, label, bars_to_skip=1):
    df = load_csv_tab_delimited(csv_file)
    if df.empty:
        print(f"{label}: empty df after load => skip.")
        return None
    
    df = compute_technical_indicators(df)
    X,y,feat_cols = create_features_labels(df, bars_to_skip=bars_to_skip)
    if len(X)<500:
        print(f"{label}: Not enough data => skip. We have only {len(X)} samples.")
        return None
    
    # Train/test => last 20% is test, first 80% is train
    cutoff = int(0.8*len(X))
    X_train, y_train = X[:cutoff], y[:cutoff]
    X_test, y_test   = X[cutoff:], y[cutoff:]
    
    # Scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl  = scaler.transform(X_test)
    
    #--- XGB
    xgb_clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_clf.fit(X_train_scl, y_train, verbose=0)
    
    #--- RF
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    rf_clf.fit(X_train_scl, y_train)
    
    #--- LSTM
    # reshape => (samples, timesteps=input_dim, 1)
    X_train_lstm = X_train_scl.reshape((X_train_scl.shape[0], X_train_scl.shape[1], 1))
    X_test_lstm  = X_test_scl.reshape((X_test_scl.shape[0], X_test_scl.shape[1], 1))
    lstm_clf = build_lstm_model(X_train_lstm.shape[1])
    from tensorflow.keras.callbacks import EarlyStopping
    # up to 20 epochs, early stop after 2 epochs no improvement
    es = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True, verbose=1)
    lstm_clf.fit(
        X_train_lstm, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[es],
        verbose=0  # turn off progress bar to reduce lag
    )
    
    # Evaluate ensemble
    xgb_probs = xgb_clf.predict_proba(X_test_scl)[:,1]
    rf_probs  = rf_clf.predict_proba(X_test_scl)[:,1]
    lstm_probs= lstm_clf.predict(X_test_lstm).flatten()
    
    ensemble_probs = (xgb_probs + rf_probs + lstm_probs) / 3.0
    ensemble_preds = (ensemble_probs>0.5).astype(int)
    
    acc = accuracy_score(y_test, ensemble_preds)
    prec= precision_score(y_test, ensemble_preds, zero_division=0)
    rec = recall_score(y_test, ensemble_preds, zero_division=0)
    f1  = f1_score(y_test, ensemble_preds, zero_division=0)
    auc = roc_auc_score(y_test, ensemble_probs)
    ll  = log_loss(y_test, ensemble_preds)
    
    print(f"\n=== {label} TIMEFRAME RESULTS ===")
    print(f"Samples => Train={cutoff}, Test={len(X_test)}")
    print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}, LogLoss={ll:.3f}")
    
    # final row => "live" inference
    final_feat = X_test_scl[-1].reshape(1,-1)
    final_feat_lstm = final_feat.reshape((1, X_test_scl.shape[1], 1))
    p_xgb = xgb_clf.predict_proba(final_feat)[0][1]
    p_rf  = rf_clf.predict_proba(final_feat)[0][1]
    p_lstm= lstm_clf.predict(final_feat_lstm)[0][0]
    final_prob = (p_xgb + p_rf + p_lstm)/3
    direction = "Bullish" if final_prob>0.5 else "Bearish"
    
    print(f"Final sample prob(bullish)={final_prob:.4f} => {direction}")
    return direction

##############################################################################
# 6) MAIN
##############################################################################
def main():
    timeframe_config = [
        # skip=1 => next candle => 4H, or 2H => skip=2 => next 4H, etc.
        # but you can define for 1D horizon or 4H horizon, etc.
        ("4H", "eurusd4h.csv",1),
        ("2H", "eurusd2h.csv",2),
        ("1H", "eurusd1h.csv",4),
    ]
    
    predictions = []
    for label, csv_file, skip_ in timeframe_config:
        pred = train_predict_ensemble(csv_file, label, skip_)
        predictions.append((label, pred))
    
    valid = [p for (l,p) in predictions if p is not None]
    if not valid:
        print("\nNo valid predictions => final no trade.")
        return
    bullish_count = sum(1 for v in valid if v=="Bullish")
    bearish_count = sum(1 for v in valid if v=="Bearish")
    print(f"\n--- FINAL CONSENSUS ---")
    print(f"Bullish: {bullish_count}, Bearish: {bearish_count}")
    if bullish_count>bearish_count:
        print("OVERALL => BULLISH")
    elif bearish_count>bullish_count:
        print("OVERALL => BEARISH")
    else:
        print("OVERALL => NEUTRAL")

if __name__=="__main__":
    main()
