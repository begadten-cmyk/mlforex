"""
STACKED CLASSIFIER WITH MANUAL XGB PARAM SEARCH + FINAL BAR INFERENCE

We do:
1) 1H => skip=4
   2H => skip=2
   4H => skip=1
   => All try to predict next 4H candle direction (Bullish/Bearish).
2) We remove duplicates by 'date', unify them by date intersection.
3) For training:
   - Build skip-based label => out-of-fold => stacked meta-model (XGB).
   - Time-based train/test split => last 20% = test.
4) For final real-time row:
   - We keep the very last row in each timeframe (unlabeled),
     compute indicators, feed sub-models => probabilities => feed meta-model => final direction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

##############################################################################
# 1) LOAD & BUILD DATASET FOR TRAINING
##############################################################################
def load_timeframe_csv(csv_file, timeframe_label):
    """Loads a CSV with columns [date, open, high, low, close, ...], drops duplicates, sorts by date."""
    df = pd.read_csv(csv_file, sep='\t', engine='python')
    rename_map = {
        '<DATE>':'date','<OPEN>':'open','<HIGH>':'high',
        '<LOW>':'low','<CLOSE>':'close'
    }
    df.rename(columns=rename_map, inplace=True, errors='ignore')
    # drop extra columns
    for c in ['<TICKVOL>','<VOL>','<SPREAD>']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    if 'date' not in df.columns:
        raise ValueError(f"{timeframe_label}: CSV must have a 'date' column")

    df.drop_duplicates(subset=['date'], inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ensure float
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    return df

def compute_indicators(df):
    """Compute minimal real indicators: RSI(14), MACD(12,26,9), SMA(20)."""
    df = df.copy()
    # 1) RSI(14)
    delta = df['close'].diff()
    gain  = delta.where(delta>0,0.0)
    loss  = -delta.where(delta<0,0.0)
    avg_gain = gain.rolling(14,min_periods=1).mean()
    avg_loss = loss.rolling(14,min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    df['rsi14'] = 100 - (100/(1+rs))

    # 2) MACD(12,26,9)
    ema12 = df['close'].ewm(span=12,adjust=False).mean()
    ema26 = df['close'].ewm(span=26,adjust=False).mean()
    df['macd_line']   = ema12-ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9,adjust=False).mean()
    df['macd_hist']   = df['macd_line']-df['macd_signal']

    # 3) SMA(20)
    df['sma20'] = df['close'].rolling(20, min_periods=1).mean()

    # fill
    df.bfill(inplace=True)
    return df

def create_label_skipbased(df, skip_bars, timeframe_label):
    """
    For each row i, label => next 4H candle => check row i+skip_bars's open,close.
    We'll do label=1 if close>open, else 0.
    We'll produce a 'label' col, drop rows that can't be labeled.
    """
    n = len(df)
    labels = []
    max_i = n - skip_bars
    for i in range(n):
        if i<max_i:
            nxt_open  = df.loc[i+skip_bars,'open']
            nxt_close = df.loc[i+skip_bars,'close']
            lb = 1 if nxt_close>nxt_open else 0
        else:
            lb = np.nan
        labels.append(lb)
    df['label'] = labels
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"{timeframe_label}: skip={skip_bars}, final trainable={len(df)}")
    return df

def create_last_inference_row(df, timeframe_label):
    """
    For final real-time row => we do not skip forward => no label.
    Return the row index + feature values for the final row if exist.
    If none => return None.
    """
    if len(df)==0:
        return None
    idx = len(df)-1
    return idx

##############################################################################
# 2) BASE MODEL => RF, plus OOF generation
##############################################################################
def train_rf(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def generate_oof_probs(model_func, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    n = len(X)
    oof_probs = np.zeros(n)
    for train_idx, valid_idx in kf.split(X):
        XX_tr, XX_val = X[train_idx], X[valid_idx]
        yy_tr, yy_val = y[train_idx], y[valid_idx]
        base_model = model_func(XX_tr, yy_tr)
        prob = base_model.predict_proba(XX_val)[:,1]
        oof_probs[valid_idx] = prob
    final_model = model_func(X,y)
    return oof_probs, final_model

##############################################################################
# 3) XGB manual param search
##############################################################################
def xgb_manual_param_search(X, y):
    param_grid = [
        {'n_estimators':50,'max_depth':3,'learning_rate':0.1},
        {'n_estimators':50,'max_depth':3,'learning_rate':0.01},
        {'n_estimators':50,'max_depth':5,'learning_rate':0.1},
        {'n_estimators':100,'max_depth':3,'learning_rate':0.1},
    ]
    from sklearn.model_selection import KFold
    best_acc = -1
    best_params = None
    kf = KFold(n_splits=3, shuffle=False)
    import xgboost as xgb

    for params in param_grid:
        acc_list = []
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                random_state=123,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_tr, y_tr)
            p = model.predict(X_val)
            a = np.mean(p==y_val)
            acc_list.append(a)
        avg_acc = np.mean(acc_list)
        print(f"XGB param {params}, cv acc={avg_acc:.3f}")
        if avg_acc>best_acc:
            best_acc = avg_acc
            best_params = params

    print(f"Chosen best param => {best_params}, cv acc={best_acc:.3f}")
    best_model = xgb.XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        random_state=123,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    best_model.fit(X,y)
    return best_model

##############################################################################
# 4) MAIN
##############################################################################
def main():
    """
    We'll do 1H => skip=4, 2H => skip=2, 4H => skip=1 => next 4H candle direction.
    We'll unify by date intersection, do 80%/20% time-based train/test,
    produce out-of-fold predictions => meta-model (XGB param search),
    THEN do final 'real-time' inference row from each timeframe => meta-model => final direction.
    """

    # TIMEFRAMES
    timeframes = [
        ("1H", "eurusd1h.csv", 4),
        ("2H", "eurusd2h.csv", 2),
        ("4H", "eurusd4h.csv", 1)
    ]

    df_list = []
    # We'll also keep a dictionary for each timeframe's "final row index"
    # so we can do real-time inference after training
    final_inference_idx = {}
    # Also store scalers, final models, etc.
    sub_models_info = {}

    # 1) BUILD DF FOR EACH TIMEFRAME
    for lbl, csvf, skip_ in timeframes:
        # load
        df_ = load_timeframe_csv(csvf, lbl)
        # compute indicators
        df_ = compute_indicators(df_)
        # skip-based label
        df_label = create_label_skipbased(df_, skip_, lbl)

        df_label['tf_label'] = lbl
        df_list.append(df_label)

        # store final row for inference if exist
        idx_ = create_last_inference_row(df_, lbl)
        final_inference_idx[lbl] = idx_

    # 2) INTERSECT DATES
    common_dates = set(df_list[0]['date'])
    for i in range(1,len(df_list)):
        common_dates = common_dates.intersection(set(df_list[i]['date']))
    common_dates = sorted(common_dates)
    if len(common_dates)<10:
        print("Not enough common dates => stop.")
        return

    # time-based train/test => last 20% => test
    cutoff = int(0.8*len(common_dates))
    train_dates = set(common_dates[:cutoff])
    test_dates  = set(common_dates[cutoff:])

    # We'll collect => out-of-fold train + test for stacking
    all_train_merge = []
    all_test_merge  = []
    final_label_train = None
    final_label_test  = None

    # We'll also keep each timeframe's final sub-model, plus scaler
    from sklearn.model_selection import KFold

    for lbl, csvf, skip_ in timeframes:
        df_tf = [d for d in df_list if d['tf_label'].iloc[0]==lbl][0]

        df_train_tf = df_tf[df_tf['date'].isin(train_dates)].copy()
        df_train_tf.sort_values('date', inplace=True)
        df_train_tf.reset_index(drop=True, inplace=True)

        df_test_tf = df_tf[df_tf['date'].isin(test_dates)].copy()
        df_test_tf.sort_values('date', inplace=True)
        df_test_tf.reset_index(drop=True, inplace=True)

        if len(df_train_tf)<10 or len(df_test_tf)<1:
            print(f"WARNING: timeframe={lbl} => not enough train/test => skip from stacking.")
            continue

        # X,y => features
        feat_cols = ['open','close','rsi14','macd_line','macd_signal','macd_hist','sma20']
        X_train_ = df_train_tf[feat_cols].to_numpy()
        y_train_ = df_train_tf['label'].to_numpy()
        X_test_  = df_test_tf[feat_cols].to_numpy()
        y_test_  = df_test_tf['label'].to_numpy()

        # scale
        sc_ = MinMaxScaler()
        X_train_scl = sc_.fit_transform(X_train_)
        X_test_scl  = sc_.transform(X_test_)

        # out-of-fold
        def model_func(XX, yy):
            return train_rf(XX, yy)
        oof_prob, final_model = generate_oof_probs(model_func, X_train_scl, y_train_)

        # store into df
        df_train_tf['oof_prob'] = oof_prob
        test_prob = final_model.predict_proba(X_test_scl)[:,1]
        df_test_tf['test_prob'] = test_prob

        # rename
        df_tr_min = df_train_tf[['date','oof_prob','label']].copy()
        df_tr_min.rename(columns={'oof_prob':f'{lbl}_oof'}, inplace=True)
        df_te_min = df_test_tf[['date','test_prob','label']].copy()
        df_te_min.rename(columns={'test_prob':f'{lbl}_test'}, inplace=True)

        all_train_merge.append(df_tr_min)
        all_test_merge.append(df_te_min)

        if final_label_train is None:
            final_label_train = df_tr_min[['date','label']].copy()
            final_label_test  = df_te_min[['date','label']].copy()

        # Also store final sub-model, scaler => for real-time inference
        sub_models_info[lbl] = {
            'final_model': final_model,
            'scaler': sc_,
            'feat_cols': feat_cols
        }

    # Merging
    if len(all_train_merge)==0:
        print("No valid timeframe => can't stack.")
        return

    train_merged = final_label_train
    for i in range(len(all_train_merge)):
        dfm = all_train_merge[i].drop(columns='label')
        train_merged = pd.merge(train_merged, dfm, on='date', how='inner')
    test_merged = final_label_test
    for i in range(len(all_test_merge)):
        dfm = all_test_merge[i].drop(columns='label')
        test_merged = pd.merge(test_merged, dfm, on='date', how='inner')

    y_train_stacked = train_merged['label'].to_numpy()
    y_test_stacked  = test_merged['label'].to_numpy()

    # stacked columns
    train_cols = [c for c in train_merged.columns if c.endswith('_oof')]
    test_cols  = [c for c in test_merged.columns if c.endswith('_test')]
    Z_train = train_merged[train_cols].to_numpy()
    Z_test  = test_merged[test_cols].to_numpy()

    print(f"Final stacked train shape: {Z_train.shape}, test shape: {Z_test.shape}")
    print(f"Train samples: {len(Z_train)}, Test samples: {len(Z_test)}")

    if len(Z_train)<10 or len(Z_test)<1:
        print("Not enough final train/test => stop stacking.")
        return

    # do manual param search XGB
    best_model = xgb_manual_param_search(Z_train, y_train_stacked)

    # final test
    pred_test = best_model.predict(Z_test)
    test_acc = np.mean(pred_test==y_test_stacked)
    print(f"STACKED META MODEL TEST ACC: {test_acc:.3f}")

    # ========== FINAL BAR INFERENCE =============
    # We'll gather each timeframe's last bar => sub-model => prob => build stacked row => meta-model => final direction
    meta_feats = []
    all_success = True
    for lbl, csvf, skip_ in timeframes:
        if lbl not in sub_models_info:
            print(f"{lbl} not in sub_models_info => skip from final inference.")
            all_success = False
            break
        # load original DF again, because we may need the actual final row data (unlabeled)
        df_orig = load_timeframe_csv(csvf, lbl)
        df_orig = compute_indicators(df_orig)
        if len(df_orig)==0:
            print(f"{lbl} => no data => skip final inference.")
            all_success = False
            break

        last_idx = len(df_orig)-1
        # gather the same features
        feat_cols = sub_models_info[lbl]['feat_cols']
        if not all(fc in df_orig.columns for fc in feat_cols):
            print(f"{lbl} => missing columns for final inference => skip.")
            all_success = False
            break

        # build 1-row array
        row_vals = df_orig.loc[last_idx, feat_cols].values.astype(float).reshape(1,-1)
        sc_ = sub_models_info[lbl]['scaler']
        row_scl = sc_.transform(row_vals)  # scale
        sub_model = sub_models_info[lbl]['final_model']
        prob_bullish = sub_model.predict_proba(row_scl)[0][1]
        meta_feats.append(prob_bullish)

    if not all_success or len(meta_feats)<len(timeframes):
        print("Some timeframe failed final inference => no final direction.")
        return

    # now meta_feats is shape (#timeframes,)
    # feed to best_model => shape(1,#timeframes)
    meta_arr = np.array(meta_feats).reshape(1,-1)
    final_pred = best_model.predict(meta_arr)[0]
    final_dir  = "Bullish" if final_pred==1 else "Bearish"
    print(f"FINAL LIVE DIRECTION => {final_dir}")

if __name__=="__main__":
    main()
"""
STACKED CLASSIFIER WITH MANUAL XGB PARAM SEARCH + FINAL BAR INFERENCE

We do:
1) 1H => skip=4
   2H => skip=2
   4H => skip=1
   => All try to predict next 4H candle direction (Bullish/Bearish).
2) We remove duplicates by 'date', unify them by date intersection.
3) For training:
   - Build skip-based label => out-of-fold => stacked meta-model (XGB).
   - Time-based train/test split => last 20% = test.
4) For final real-time row:
   - We keep the very last row in each timeframe (unlabeled),
     compute indicators, feed sub-models => probabilities => feed meta-model => final direction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

##############################################################################
# 1) LOAD & BUILD DATASET FOR TRAINING
##############################################################################
def load_timeframe_csv(csv_file, timeframe_label):
    """Loads a CSV with columns [date, open, high, low, close, ...], drops duplicates, sorts by date."""
    df = pd.read_csv(csv_file, sep='\t', engine='python')
    rename_map = {
        '<DATE>':'date','<OPEN>':'open','<HIGH>':'high',
        '<LOW>':'low','<CLOSE>':'close'
    }
    df.rename(columns=rename_map, inplace=True, errors='ignore')
    # drop extra columns
    for c in ['<TICKVOL>','<VOL>','<SPREAD>']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    if 'date' not in df.columns:
        raise ValueError(f"{timeframe_label}: CSV must have a 'date' column")

    df.drop_duplicates(subset=['date'], inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ensure float
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    return df

def compute_indicators(df):
    """Compute minimal real indicators: RSI(14), MACD(12,26,9), SMA(20)."""
    df = df.copy()
    # 1) RSI(14)
    delta = df['close'].diff()
    gain  = delta.where(delta>0,0.0)
    loss  = -delta.where(delta<0,0.0)
    avg_gain = gain.rolling(14,min_periods=1).mean()
    avg_loss = loss.rolling(14,min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    df['rsi14'] = 100 - (100/(1+rs))

    # 2) MACD(12,26,9)
    ema12 = df['close'].ewm(span=12,adjust=False).mean()
    ema26 = df['close'].ewm(span=26,adjust=False).mean()
    df['macd_line']   = ema12-ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9,adjust=False).mean()
    df['macd_hist']   = df['macd_line']-df['macd_signal']

    # 3) SMA(20)
    df['sma20'] = df['close'].rolling(20, min_periods=1).mean()

    # fill
    df.bfill(inplace=True)
    return df

def create_label_skipbased(df, skip_bars, timeframe_label):
    """
    For each row i, label => next 4H candle => check row i+skip_bars's open,close.
    We'll do label=1 if close>open, else 0.
    We'll produce a 'label' col, drop rows that can't be labeled.
    """
    n = len(df)
    labels = []
    max_i = n - skip_bars
    for i in range(n):
        if i<max_i:
            nxt_open  = df.loc[i+skip_bars,'open']
            nxt_close = df.loc[i+skip_bars,'close']
            lb = 1 if nxt_close>nxt_open else 0
        else:
            lb = np.nan
        labels.append(lb)
    df['label'] = labels
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    print(f"{timeframe_label}: skip={skip_bars}, final trainable={len(df)}")
    return df

def create_last_inference_row(df, timeframe_label):
    """
    For final real-time row => we do not skip forward => no label.
    Return the row index + feature values for the final row if exist.
    If none => return None.
    """
    if len(df)==0:
        return None
    idx = len(df)-1
    return idx

##############################################################################
# 2) BASE MODEL => RF, plus OOF generation
##############################################################################
def train_rf(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def generate_oof_probs(model_func, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    n = len(X)
    oof_probs = np.zeros(n)
    for train_idx, valid_idx in kf.split(X):
        XX_tr, XX_val = X[train_idx], X[valid_idx]
        yy_tr, yy_val = y[train_idx], y[valid_idx]
        base_model = model_func(XX_tr, yy_tr)
        prob = base_model.predict_proba(XX_val)[:,1]
        oof_probs[valid_idx] = prob
    final_model = model_func(X,y)
    return oof_probs, final_model

##############################################################################
# 3) XGB manual param search
##############################################################################
def xgb_manual_param_search(X, y):
    param_grid = [
        {'n_estimators':50,'max_depth':3,'learning_rate':0.1},
        {'n_estimators':50,'max_depth':3,'learning_rate':0.01},
        {'n_estimators':50,'max_depth':5,'learning_rate':0.1},
        {'n_estimators':100,'max_depth':3,'learning_rate':0.1},
    ]
    from sklearn.model_selection import KFold
    best_acc = -1
    best_params = None
    kf = KFold(n_splits=3, shuffle=False)
    import xgboost as xgb

    for params in param_grid:
        acc_list = []
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                random_state=123,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_tr, y_tr)
            p = model.predict(X_val)
            a = np.mean(p==y_val)
            acc_list.append(a)
        avg_acc = np.mean(acc_list)
        print(f"XGB param {params}, cv acc={avg_acc:.3f}")
        if avg_acc>best_acc:
            best_acc = avg_acc
            best_params = params

    print(f"Chosen best param => {best_params}, cv acc={best_acc:.3f}")
    best_model = xgb.XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        random_state=123,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    best_model.fit(X,y)
    return best_model

##############################################################################
# 4) MAIN
##############################################################################
def main():
    """
    We'll do 1H => skip=4, 2H => skip=2, 4H => skip=1 => next 4H candle direction.
    We'll unify by date intersection, do 80%/20% time-based train/test,
    produce out-of-fold predictions => meta-model (XGB param search),
    THEN do final 'real-time' inference row from each timeframe => meta-model => final direction.
    """

    # TIMEFRAMES
    timeframes = [
        ("1H", "eurusd1h.csv", 4),
        ("2H", "eurusd2h.csv", 2),
        ("4H", "eurusd4h.csv", 1)
    ]

    df_list = []
    # We'll also keep a dictionary for each timeframe's "final row index"
    # so we can do real-time inference after training
    final_inference_idx = {}
    # Also store scalers, final models, etc.
    sub_models_info = {}

    # 1) BUILD DF FOR EACH TIMEFRAME
    for lbl, csvf, skip_ in timeframes:
        # load
        df_ = load_timeframe_csv(csvf, lbl)
        # compute indicators
        df_ = compute_indicators(df_)
        # skip-based label
        df_label = create_label_skipbased(df_, skip_, lbl)

        df_label['tf_label'] = lbl
        df_list.append(df_label)

        # store final row for inference if exist
        idx_ = create_last_inference_row(df_, lbl)
        final_inference_idx[lbl] = idx_

    # 2) INTERSECT DATES
    common_dates = set(df_list[0]['date'])
    for i in range(1,len(df_list)):
        common_dates = common_dates.intersection(set(df_list[i]['date']))
    common_dates = sorted(common_dates)
    if len(common_dates)<10:
        print("Not enough common dates => stop.")
        return

    # time-based train/test => last 20% => test
    cutoff = int(0.8*len(common_dates))
    train_dates = set(common_dates[:cutoff])
    test_dates  = set(common_dates[cutoff:])

    # We'll collect => out-of-fold train + test for stacking
    all_train_merge = []
    all_test_merge  = []
    final_label_train = None
    final_label_test  = None

    # We'll also keep each timeframe's final sub-model, plus scaler
    from sklearn.model_selection import KFold

    for lbl, csvf, skip_ in timeframes:
        df_tf = [d for d in df_list if d['tf_label'].iloc[0]==lbl][0]

        df_train_tf = df_tf[df_tf['date'].isin(train_dates)].copy()
        df_train_tf.sort_values('date', inplace=True)
        df_train_tf.reset_index(drop=True, inplace=True)

        df_test_tf = df_tf[df_tf['date'].isin(test_dates)].copy()
        df_test_tf.sort_values('date', inplace=True)
        df_test_tf.reset_index(drop=True, inplace=True)

        if len(df_train_tf)<10 or len(df_test_tf)<1:
            print(f"WARNING: timeframe={lbl} => not enough train/test => skip from stacking.")
            continue

        # X,y => features
        feat_cols = ['open','close','rsi14','macd_line','macd_signal','macd_hist','sma20']
        X_train_ = df_train_tf[feat_cols].to_numpy()
        y_train_ = df_train_tf['label'].to_numpy()
        X_test_  = df_test_tf[feat_cols].to_numpy()
        y_test_  = df_test_tf['label'].to_numpy()

        # scale
        sc_ = MinMaxScaler()
        X_train_scl = sc_.fit_transform(X_train_)
        X_test_scl  = sc_.transform(X_test_)

        # out-of-fold
        def model_func(XX, yy):
            return train_rf(XX, yy)
        oof_prob, final_model = generate_oof_probs(model_func, X_train_scl, y_train_)

        # store into df
        df_train_tf['oof_prob'] = oof_prob
        test_prob = final_model.predict_proba(X_test_scl)[:,1]
        df_test_tf['test_prob'] = test_prob

        # rename
        df_tr_min = df_train_tf[['date','oof_prob','label']].copy()
        df_tr_min.rename(columns={'oof_prob':f'{lbl}_oof'}, inplace=True)
        df_te_min = df_test_tf[['date','test_prob','label']].copy()
        df_te_min.rename(columns={'test_prob':f'{lbl}_test'}, inplace=True)

        all_train_merge.append(df_tr_min)
        all_test_merge.append(df_te_min)

        if final_label_train is None:
            final_label_train = df_tr_min[['date','label']].copy()
            final_label_test  = df_te_min[['date','label']].copy()

        # Also store final sub-model, scaler => for real-time inference
        sub_models_info[lbl] = {
            'final_model': final_model,
            'scaler': sc_,
            'feat_cols': feat_cols
        }

    # Merging
    if len(all_train_merge)==0:
        print("No valid timeframe => can't stack.")
        return

    train_merged = final_label_train
    for i in range(len(all_train_merge)):
        dfm = all_train_merge[i].drop(columns='label')
        train_merged = pd.merge(train_merged, dfm, on='date', how='inner')
    test_merged = final_label_test
    for i in range(len(all_test_merge)):
        dfm = all_test_merge[i].drop(columns='label')
        test_merged = pd.merge(test_merged, dfm, on='date', how='inner')

    y_train_stacked = train_merged['label'].to_numpy()
    y_test_stacked  = test_merged['label'].to_numpy()

    # stacked columns
    train_cols = [c for c in train_merged.columns if c.endswith('_oof')]
    test_cols  = [c for c in test_merged.columns if c.endswith('_test')]
    Z_train = train_merged[train_cols].to_numpy()
    Z_test  = test_merged[test_cols].to_numpy()

    print(f"Final stacked train shape: {Z_train.shape}, test shape: {Z_test.shape}")
    print(f"Train samples: {len(Z_train)}, Test samples: {len(Z_test)}")

    if len(Z_train)<10 or len(Z_test)<1:
        print("Not enough final train/test => stop stacking.")
        return

    # do manual param search XGB
    best_model = xgb_manual_param_search(Z_train, y_train_stacked)

    # final test
    pred_test = best_model.predict(Z_test)
    test_acc = np.mean(pred_test==y_test_stacked)
    print(f"STACKED META MODEL TEST ACC: {test_acc:.3f}")

    # ========== FINAL BAR INFERENCE =============
    # We'll gather each timeframe's last bar => sub-model => prob => build stacked row => meta-model => final direction
    meta_feats = []
    all_success = True
    for lbl, csvf, skip_ in timeframes:
        if lbl not in sub_models_info:
            print(f"{lbl} not in sub_models_info => skip from final inference.")
            all_success = False
            break
        # load original DF again, because we may need the actual final row data (unlabeled)
        df_orig = load_timeframe_csv(csvf, lbl)
        df_orig = compute_indicators(df_orig)
        if len(df_orig)==0:
            print(f"{lbl} => no data => skip final inference.")
            all_success = False
            break

        last_idx = len(df_orig)-1
        # gather the same features
        feat_cols = sub_models_info[lbl]['feat_cols']
        if not all(fc in df_orig.columns for fc in feat_cols):
            print(f"{lbl} => missing columns for final inference => skip.")
            all_success = False
            break

        # build 1-row array
        row_vals = df_orig.loc[last_idx, feat_cols].values.astype(float).reshape(1,-1)
        sc_ = sub_models_info[lbl]['scaler']
        row_scl = sc_.transform(row_vals)  # scale
        sub_model = sub_models_info[lbl]['final_model']
        prob_bullish = sub_model.predict_proba(row_scl)[0][1]
        meta_feats.append(prob_bullish)

    if not all_success or len(meta_feats)<len(timeframes):
        print("Some timeframe failed final inference => no final direction.")
        return

    # now meta_feats is shape (#timeframes,)
    # feed to best_model => shape(1,#timeframes)
    meta_arr = np.array(meta_feats).reshape(1,-1)
    final_pred = best_model.predict(meta_arr)[0]
    final_dir  = "Bullish" if final_pred==1 else "Bearish"
    print(f"FINAL LIVE DIRECTION => {final_dir}")

if __name__=="__main__":
    main()
