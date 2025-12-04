"""
MULTILAYER PERCEPTRON SCRIPT FOR MULTIPLE TIMEFRAMES
Predict Next 4H Candle Direction with Candlestick + Stochastic Features
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

##############################################################################
# 1) LOAD CSV
##############################################################################
def load_csv_tab_delimited(csv_file):
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
# 2) STOCH + PATTERNS
##############################################################################
def compute_stochastics_and_patterns(df, stoch_period=14):
    df = df.copy()
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    lows = df['low'].rolling(window=stoch_period, min_periods=1).min()
    highs= df['high'].rolling(window=stoch_period, min_periods=1).max()
    df['stochK'] = ((df['close']-lows)/(highs-lows+1e-9))*100
    df['stochD'] = df['stochK'].rolling(window=3, min_periods=1).mean()

    df['doji']=0
    df['hammer']=0
    df['bull_engulf']=0
    df['bear_engulf']=0

    def detect_doji(row, threshold=0.1):
        op,hi,lo,cl = row['open'],row['high'],row['low'],row['close']
        rng = hi-lo
        body= abs(cl-op)
        if rng<1e-9: return 0
        return 1 if (body/rng)<threshold else 0

    def detect_hammer(row, b_thr=0.2, w_thr=2.0):
        op,hi,lo,cl = row['open'],row['high'],row['low'],row['close']
        rng= hi-lo
        if rng<1e-9: return 0
        body= abs(cl-op)
        dn_wick= min(op,cl)-lo
        if (body/rng)>b_thr: return 0
        if dn_wick < w_thr*body: return 0
        return 1

    def detect_engulfing(prev_row, curr_row):
        op_p, cl_p= prev_row['open'], prev_row['close']
        op_c, cl_c= curr_row['open'], curr_row['close']
        bull=0; bear=0
        # bull engulf
        if (cl_p<op_p) and (cl_c>op_c):
            if op_c<cl_p and cl_c>op_p:
                bull=1
        # bear engulf
        if (cl_p>op_p) and (cl_c<op_c):
            if op_c>cl_p and cl_c<op_p:
                bear=1
        return bull,bear

    for i in range(len(df)):
        df.loc[i,'doji']= detect_doji(df.loc[i])
        df.loc[i,'hammer']= detect_hammer(df.loc[i])
    for i in range(1,len(df)):
        b,bear= detect_engulfing(df.loc[i-1], df.loc[i])
        df.loc[i,'bull_engulf']=b
        df.loc[i,'bear_engulf']=bear

    df.fillna(method='bfill', inplace=True)
    return df

##############################################################################
# 3) CREATE FLAT FEATURES
##############################################################################
def create_features_and_labels(df, bars_to_skip=1):
    needed_cols = [
        'open','high','low','close',
        'stochK','stochD','doji','hammer','bull_engulf','bear_engulf'
    ]
    arr= df[needed_cols].to_numpy()
    N= len(arr)
    X,y=[],[]
    max_i= N - bars_to_skip
    if max_i<=0: return np.array([]), np.array([])

    for i in range(max_i):
        feat = arr[i]  # shape(10,)
        fut_open = df.loc[i+bars_to_skip, 'open']
        fut_close= df.loc[i+bars_to_skip, 'close']
        lbl= 1.0 if fut_close>fut_open else 0.0
        X.append(feat)
        y.append(lbl)
    return np.array(X), np.array(y).reshape(-1,1)

##############################################################################
# 4) TRAIN & PREDICT
##############################################################################
def train_and_predict(csv_file, timeframe_label, bars_to_skip=1):
    df = load_csv_tab_delimited(csv_file)
    df = compute_stochastics_and_patterns(df, stoch_period=14)
    X,y = create_features_and_labels(df,bars_to_skip=bars_to_skip)
    if len(X)==0:
        print(f"{timeframe_label}: Not enough data for training.")
        return None

    scaler= MinMaxScaler()
    X_scaled= scaler.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled,y, test_size=0.2, shuffle=False
    )

    mlp= MLPClassifier(hidden_layer_sizes=(32,16), activation='relu',
                       max_iter=200, random_state=42)
    mlp.fit(X_train_full, y_train_full.ravel())

    # final row inference
    if len(X_scaled)==0:
        print(f"{timeframe_label}: No final row to infer.")
        return None
    final_feat= X_scaled[-1].reshape(1,-1)
    prob= mlp.predict_proba(final_feat)[0][1]
    predicted_class=1 if prob>=0.5 else 0
    direction= "Bullish" if predicted_class==1 else "Bearish"

    last_i= len(df)-1
    last_date= df.loc[last_i,'date']
    last_time= df.loc[last_i,'time']
    print(f"{timeframe_label} timeframe prediction for next 4H candle (MLP):")
    print(f"Last known candle in {csv_file}: {last_date} {last_time}")
    print(f"Probability of Bullish: {prob:.4f}")
    print(f"Prediction: {direction}\n")

    return direction

##############################################################################
# 5) MASTER SCRIPT: MULTIPLE TIMEFRAMES
##############################################################################
def main():
    timeframe_csvs = [
        ("4H",   "eurusd4h.csv",   1),
        ("2H",   "eurusd2h.csv",   2),
        ("1H",   "eurusd1h.csv",   4),
        ("30m",  "eurusd30m.csv",  8),
        ("15m",  "eurusd15m.csv", 16),
        ("5m",   "eurusd5m.csv",  48),
    ]
    predictions=[]
    for tf_label, csvfile, skip_ in timeframe_csvs:
        direction= train_and_predict(csvfile, tf_label, bars_to_skip=skip_)
        predictions.append(direction)
    filtered= [p for p in predictions if p is not None]
    if len(filtered)<6:
        final_call= "NO TRADE (some timeframe missing data)"
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
