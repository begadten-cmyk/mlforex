"""
ICT-Inspired Full Logic (Bullish + Bearish) with Telegram Notification
and Continuous Scanning via `schedule`, now using BOTH 1H and 15M data,
but with LOOSER restrictions for more frequent trades.
FOR EDUCATIONAL USE ONLY
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pytz
import time
import schedule
from datetime import datetime, timedelta

##########################################
# TELEGRAM NOTIFICATION (Hard-coded)
##########################################
TELEGRAM_BOT_TOKEN = "7847987861:AAHrr1mAnYYyUK5bfrJyyc5GmggsWCEVJFw"
TELEGRAM_CHAT_ID   = "7749762504"

def send_telegram_message(message):
    """
    Sends a text message to the specified Telegram chat.
    Update TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID with valid details.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Telegram error: {response.text}")
    except Exception as e:
        print(f"Telegram send failed: {str(e)}")


############################
# STEP 1: DAILY BIAS (ICT-STYLE)
############################

def detect_daily_bias(df, lookback=7, displacement_factor=0.5):
    """
    LOOSER Settings:
      - lookback=7 (was 15)
      - displacement_factor=0.5 (was 1.0)
      1) Identify major high & low in the last `lookback` bars.
      2) Check for Market Structure Shift with a smaller displacement
         => easier to call something bullish or bearish.
      3) No discount/premium check.
    """
    if len(df) < lookback + 2:
        return None
    
    recent_df = df.iloc[-lookback:].copy()
    
    major_high = recent_df['High'].max()
    major_low  = recent_df['Low'].min()
    last_close = df['Close'].iloc[-1]
    last_open  = df['Open'].iloc[-1]
    
    # Average body in the recent window
    recent_df['Body'] = (recent_df['Close'] - recent_df['Open']).abs()
    avg_body = recent_df['Body'].mean()
    
    last_body_size = abs(last_close - last_open)
    # Now a smaller displacement factor => easier "shift"
    displacement = (last_body_size > displacement_factor * avg_body)
    
    broke_above = (last_close > major_high)
    broke_below = (last_close < major_low)
    
    bullish_shift = (broke_above and displacement)
    bearish_shift = (broke_below and displacement)
    
    if bullish_shift:
        return "Bullish"
    elif bearish_shift:
        return "Bearish"
    else:
        return None


############################
# STEP 2: DAILY LIQUIDITY POOLS
############################

def find_swing_highs_lows(df, lookback=1):
    """
    Swing High/Low logic unchanged.
    """
    highs = []
    lows = []
    for i in range(lookback, len(df) - lookback):
        if (df['High'].iloc[i] > df['High'].iloc[i-1]) and (df['High'].iloc[i] > df['High'].iloc[i+1]):
            highs.append((df.index[i], df['High'].iloc[i]))
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1]) and (df['Low'].iloc[i] < df['Low'].iloc[i+1]):
            lows.append((df.index[i], df['Low'].iloc[i]))
    return highs, lows


############################
# STEP 3: ORDER BLOCKS & FAIR VALUE GAPS (INTRADAY)
############################

def find_order_blocks(df, threshold=0.005):
    """
    Lower threshold from 0.01 to 0.005 => more OB signals.
    """
    bullish_blocks = []
    bearish_blocks = []
    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i-1]
        curr_close = df['Close'].iloc[i]
        if prev_close == 0:
            continue
        pct_change = (curr_close - prev_close) / prev_close
        
        # Bullish OB
        if pct_change > threshold and (df['Close'].iloc[i-1] < df['Open'].iloc[i-1]):
            bullish_blocks.append((df.index[i-1],
                                   df['Open'].iloc[i-1],
                                   df['Close'].iloc[i-1]))
        
        # Bearish OB
        if pct_change < -threshold and (df['Close'].iloc[i-1] > df['Open'].iloc[i-1]):
            bearish_blocks.append((df.index[i-1],
                                   df['Open'].iloc[i-1],
                                   df['Close'].iloc[i-1]))
    return bullish_blocks, bearish_blocks


def find_fair_value_gaps(df):
    """
    Unchanged logic for FVG.
    """
    fvg_list = []
    for i in range(2, len(df)):
        c1_high = df['High'].iloc[i-2]
        c1_low  = df['Low'].iloc[i-2]
        c2_high = df['High'].iloc[i-1]
        c2_low  = df['Low'].iloc[i-1]
        c3_high = df['High'].iloc[i]
        c3_low  = df['Low'].iloc[i]
        
        # Bullish FVG
        if (c2_low > c1_high) and (c3_low > c1_high):
            gap_low = c1_high
            gap_high = c2_low
            fvg_list.append((df.index[i-1], 'Bullish FVG', (gap_low, gap_high)))
        
        # Bearish FVG
        if (c2_high < c1_low) and (c3_high < c1_low):
            gap_low = c2_high
            gap_high = c1_low
            fvg_list.append((df.index[i-1], 'Bearish FVG', (gap_low, gap_high)))
    return fvg_list


############################
# STEP 4: KILL ZONES
############################

def convert_to_newyork_time(df):
    """
    Unchanged timezone conversion.
    """
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')
    return df

def filter_kill_zone(df, start_hour, end_hour):
    """
    Same kill zone function, unchanged.
    """
    mask = (df.index.hour >= start_hour) & (df.index.hour <= end_hour)
    return df[mask]


############################
# STEP 5: OTE (Optimal Trade Entry)
############################

def calculate_ote_zone_bullish(swing_low_price, swing_high_price):
    diff = swing_high_price - swing_low_price
    fib_618 = swing_high_price - 0.618 * diff
    fib_79  = swing_high_price - 0.79  * diff
    ote_lower = min(fib_618, fib_79)
    ote_upper = max(fib_618, fib_79)
    return (ote_lower, ote_upper)

def calculate_ote_zone_bearish(swing_high_price, swing_low_price):
    diff = swing_high_price - swing_low_price
    fib_618 = swing_low_price + 0.618 * diff
    fib_79  = swing_low_price + 0.79  * diff
    ote_lower = min(fib_618, fib_79)
    ote_upper = max(fib_618, fib_79)
    return (ote_lower, ote_upper)


def is_price_in_bullish_ote(df, swing_low_price, swing_high_price, tolerance=0.002):
    """
    Tolerance increased from 0.0 to 0.002 => price can be 'slightly' out of the zone but still count.
    """
    if df.empty:
        return False, (None, None)
    ote_lower, ote_upper = calculate_ote_zone_bullish(swing_low_price, swing_high_price)
    last_close = df['Close'].iloc[-1]
    in_zone = (last_close >= ote_lower - tolerance) and (last_close <= ote_upper + tolerance)
    return in_zone, (ote_lower, ote_upper)


def is_price_in_bearish_ote(df, swing_high_price, swing_low_price, tolerance=0.002):
    """
    Same increased tolerance for bearish checks.
    """
    if df.empty:
        return False, (None, None)
    ote_lower, ote_upper = calculate_ote_zone_bearish(swing_high_price, swing_low_price)
    last_close = df['Close'].iloc[-1]
    in_zone = (last_close >= ote_lower - tolerance) and (last_close <= ote_upper + tolerance)
    return in_zone, (ote_lower, ote_upper)


############################
# STEP 6: RISK MANAGEMENT (Stops, Position Sizing)
############################

def place_stop_loss_long(swing_low_price, buffer_pct=0.01):
    """
    Larger buffer_pct=0.01 (was 0.003) => SL is farther away => more trades hold.
    """
    return swing_low_price * (1.0 - buffer_pct)

def place_stop_loss_short(swing_high_price, buffer_pct=0.01):
    """
    Same for short => bigger buffer => more likely to stay in trade.
    """
    return swing_high_price * (1.0 + buffer_pct)


############################
# STEP 7: PARTIAL PROFITS
############################

def manage_take_profit_long(current_price, entry_price, partial_tp_price, full_tp_price, total_position_size):
    """
    Unchanged partial/final TP. 
    """
    realized_pnl = 0.0
    remaining_position = total_position_size
    status_message = "No action"
    
    # Partial TP
    if (current_price >= partial_tp_price) and (remaining_position == total_position_size):
        half_pos = total_position_size * 0.5
        realized_pnl += (current_price - entry_price) * half_pos
        remaining_position -= half_pos
        status_message = f"Partial TP triggered at {current_price:.5f}"
    
    # Full TP
    if (current_price >= full_tp_price) and (remaining_position > 0):
        realized_pnl += (current_price - entry_price) * remaining_position
        remaining_position = 0
        status_message = f"Full TP triggered at {current_price:.5f}"
    
    return (remaining_position, realized_pnl, status_message)

def manage_take_profit_short(current_price, entry_price, partial_tp_price, full_tp_price, total_position_size):
    """
    Same partial/final TP logic, unchanged.
    """
    realized_pnl = 0.0
    remaining_position = total_position_size
    status_message = "No action"
    
    # Partial TP
    if (current_price <= partial_tp_price) and (remaining_position == total_position_size):
        half_pos = total_position_size * 0.5
        realized_pnl += (entry_price - current_price) * half_pos
        remaining_position -= half_pos
        status_message = f"Partial TP triggered at {current_price:.5f}"
    
    # Full TP
    if (current_price <= full_tp_price) and (remaining_position > 0):
        realized_pnl += (entry_price - current_price) * remaining_position
        remaining_position = 0
        status_message = f"Full TP triggered at {current_price:.5f}"
    
    return (remaining_position, realized_pnl, status_message)


#############################################################
# NEW: INTEGRATE 15-MIN DATA ALONGSIDE 1-HOUR
#############################################################

def fetch_and_filter_15m(symbol, kill_zone, intraday_period="1mo"):
    """
    Same function, no structural change. 
    We'll get more signals from other loosened parameters 
    (like bigger tolerance in OTE).
    """
    df_15m = yf.download(symbol, period=intraday_period, interval='15m', progress=False)
    if isinstance(df_15m.columns, pd.MultiIndex):
        df_15m.columns = df_15_m.columns.droplevel(1)
    df_15m.dropna(inplace=True)
    df_15m.sort_index(inplace=True)
    
    if df_15m.empty:
        return pd.DataFrame()  # return empty if no data
    
    df_15m = convert_to_newyork_time(df_15m)
    kz_start, kz_end = kill_zone
    df_15m_kz = filter_kill_zone(df_15m, kz_start, kz_end)
    return df_15m_kz


############################
# FULL PIPELINE
############################

def ict_full_pipeline(symbol="GBPJPY=X", 
                      daily_period="6mo", 
                      intraday_period="1mo", 
                      intraday_interval="1h", 
                      kill_zone_hours=(7, 10),
                      bias_lookback=7,           # was 15
                      displacement_factor=0.5,   # was 1.0
                      buffer_pct=0.01,          # was 0.003
                      account_size=10000,
                      risk_per_trade=0.01):
    """
    Comprehensive pipeline with LOOSER restrictions:
      - lookback=7
      - displacement_factor=0.5
      - bigger buffer_pct=0.01
      - OTE tolerance=0.002
      - OB threshold=0.005
    Everything else is the same.
    """

    results = {}

    ########## STEP 1: FETCH DAILY DATA, DETECT DAILY BIAS ##########
    daily_df = yf.download(symbol, period=daily_period, interval='1d', progress=False)
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = daily_df.columns.droplevel(1)

    daily_df.dropna(inplace=True)
    daily_df.sort_index(inplace=True)

    if len(daily_df) < bias_lookback + 2:
        results["signal"] = None
        results["reason"] = "Not enough daily data for bias"
        return results
    
    daily_bias = detect_daily_bias(daily_df, lookback=bias_lookback, displacement_factor=displacement_factor)
    results["daily_bias"] = daily_bias

    ########## STEP 2: DAILY LIQUIDITY POOLS ##########
    daily_highs, daily_lows = find_swing_highs_lows(daily_df, lookback=1)
    results["num_daily_highs"] = len(daily_highs)
    results["num_daily_lows"] = len(daily_lows)
    if not daily_highs or not daily_lows:
        results["signal"] = None
        results["reason"] = "No daily swing found"
        return results

    # Most recent swing high & low
    recent_swing_high = daily_highs[-1]
    recent_swing_low  = daily_lows[-1]
    sh_price = recent_swing_high[1]
    sl_price = recent_swing_low[1]

    ########## STEP 3: FETCH 1-HOUR INTRADAY ##########
    intraday_df = yf.download(symbol, period=intraday_period, interval=intraday_interval, progress=False)
    if isinstance(intraday_df.columns, pd.MultiIndex):
        intraday_df.columns = intraday_df.columns.droplevel(1)
    intraday_df.dropna(inplace=True)
    intraday_df.sort_index(inplace=True)
    if intraday_df.empty:
        results["signal"] = None
        results["reason"] = "No intraday data (1H)"
        return results

    intraday_df = convert_to_newyork_time(intraday_df)

    # OB / FVG (on 1H)
    bull_obs, bear_obs = find_order_blocks(intraday_df, threshold=0.005)  # was 0.01
    fvg_list = find_fair_value_gaps(intraday_df)
    results["bullish_OB_count"] = len(bull_obs)
    results["bearish_OB_count"] = len(bear_obs)
    results["fvg_count"] = len(fvg_list)

    ########## STEP 4: KILL ZONE on 1H ##########
    kz_start, kz_end = kill_zone_hours
    kz_df_1h = filter_kill_zone(intraday_df, kz_start, kz_end)
    results["kill_zone_count"] = len(kz_df_1h)
    if kz_df_1h.empty:
        results["signal"] = None
        results["reason"] = "No 1H candles in kill zone"
        return results

    ########## STEP 5: FETCH & FILTER 15M DATA ##########
    kz_df_15m = fetch_and_filter_15m(symbol, kill_zone_hours, intraday_period=intraday_period)
    if kz_df_15m.empty:
        results["signal"] = None
        results["reason"] = "No 15m candles in kill zone"
        return results
    
    ########## STEP 6: OTE Check (Bullish or Bearish) ##########
    from math import isclose  # optional if you want additional comparisons

    if daily_bias == "Bullish" and (sh_price > sl_price):
        in_ote_1h, ote_zone_1h = is_price_in_bullish_ote(kz_df_1h, sl_price, sh_price, tolerance=0.002)
        in_ote_15m, ote_zone_15m = is_price_in_bullish_ote(kz_df_15m, sl_price, sh_price, tolerance=0.002)

        results["in_ote_1h"] = in_ote_1h
        results["in_ote_15m"] = in_ote_15m

        if in_ote_1h and in_ote_15m:
            entry_price = kz_df_15m['Close'].iloc[-1]
            stop_loss_price = place_stop_loss_long(sl_price, buffer_pct=buffer_pct)
            
            risk_amt = account_size * risk_per_trade
            risk_per_unit = entry_price - stop_loss_price
            if risk_per_unit <= 0:
                results["signal"] = None
                results["reason"] = "Invalid risk logic (stop above entry for bullish)"
                return results
            
            position_size = risk_amt / risk_per_unit

            partial_tp_price = sh_price
            full_tp_price = sh_price + 0.002
            current_price = entry_price

            rem_pos, realized_pnl, tp_msg = manage_take_profit_long(
                current_price=current_price,
                entry_price=entry_price,
                partial_tp_price=partial_tp_price,
                full_tp_price=full_tp_price,
                total_position_size=position_size
            )

            results["signal"] = "BUY"
            results["entry_price"] = entry_price
            results["stop_loss"] = stop_loss_price
            results["position_size"] = position_size
            results["remaining_position"] = rem_pos
            results["realized_pnl"] = realized_pnl
            results["tp_message"] = tp_msg
            results["reason"] = "LOOSER BIAS: Bullish OTE on 1H + 15M alignment."

            msg = (
                f"Signal: BUY {symbol}\n"
                f"Entry: {entry_price:.5f}\n"
                f"Stop: {stop_loss_price:.5f}\n"
                f"Position Size: {position_size:.2f}\n"
                f"Reason: {results['reason']}"
            )
            send_telegram_message(msg)
            return results
        else:
            results["signal"] = None
            results["reason"] = "LOOSER: Bullish bias but OTE not aligned on both 1H & 15M."
            return results

    elif daily_bias == "Bearish" and (sh_price > sl_price):
        in_ote_1h, ote_zone_1h = is_price_in_bearish_ote(kz_df_1h, sh_price, sl_price, tolerance=0.002)
        in_ote_15m, ote_zone_15m = is_price_in_bearish_ote(kz_df_15m, sh_price, sl_price, tolerance=0.002)

        results["in_ote_1h"] = in_ote_1h
        results["in_ote_15m"] = in_ote_15m

        if in_ote_1h and in_ote_15m:
            entry_price = kz_df_15m['Close'].iloc[-1]
            stop_loss_price = place_stop_loss_short(sh_price, buffer_pct=buffer_pct)
            
            risk_amt = account_size * risk_per_trade
            risk_per_unit = stop_loss_price - entry_price
            if risk_per_unit <= 0:
                results["signal"] = None
                results["reason"] = "Invalid risk logic (stop below entry for bearish)"
                return results

            position_size = risk_amt / risk_per_unit

            partial_tp_price = sl_price
            full_tp_price    = sl_price - 0.002
            current_price    = entry_price

            rem_pos, realized_pnl, tp_msg = manage_take_profit_short(
                current_price=current_price,
                entry_price=entry_price,
                partial_tp_price=partial_tp_price,
                full_tp_price=full_tp_price,
                total_position_size=position_size
            )

            results["signal"] = "SELL"
            results["entry_price"] = entry_price
            results["stop_loss"] = stop_loss_price
            results["position_size"] = position_size
            results["remaining_position"] = rem_pos
            results["realized_pnl"] = realized_pnl
            results["tp_message"] = tp_msg
            results["reason"] = "LOOSER BIAS: Bearish OTE on 1H + 15M alignment."

            msg = (
                f"Signal: SELL {symbol}\n"
                f"Entry: {entry_price:.5f}\n"
                f"Stop: {stop_loss_price:.5f}\n"
                f"Position Size: {position_size:.2f}\n"
                f"Reason: {results['reason']}"
            )
            send_telegram_message(msg)
            return results
        else:
            results["signal"] = None
            results["reason"] = "LOOSER: Bearish bias but OTE not aligned on both 1H & 15M."
            return results

    else:
        results["signal"] = None
        results["reason"] = "LOOSER: No valid daily bias or not enough range for OTE."
        return results


############################
# CONTINUOUS BACKGROUND SCAN
############################

def run_ict_pipeline_continuous():
    """
    Runs the LOOSER version of the ICT pipeline for multiple pairs,
    using 1-6 am for EUR/GBP pairs and 7-12 for USD-based pairs.
    """
    # Pairs that should use 1–6 am (modified from 2–5)
    london_pairs = [
        "GBPJPY=X",
        "EURUSD=X",
        "GBPUSD=X",
        "EURGBP=X"
    ]
    
    # Pairs that should use 7–12 am 
    ny_pairs = [
        "USDCAD=X",
        "USDJPY=X",
        "GBPJPY=X"
    ]
    
    # 1) Process London pairs with kill zone (1,6)
    for symbol in london_pairs:
        results = ict_full_pipeline(
            symbol=symbol,
            daily_period="6mo",
            intraday_period="1mo",
            intraday_interval="1h",
            kill_zone_hours=(1, 6),  # Looser
            bias_lookback=7,        # looser
            displacement_factor=0.5,# looser
            buffer_pct=0.01,        # bigger stop buffer
            account_size=10000,
            risk_per_trade=0.01
        )
        
        print(f"\n=== ICT Full Pipeline Results for {symbol} ===")
        for k, v in results.items():
            print(f"{k}: {v}")
    
    # 2) Process NY pairs with kill zone (7,12)
    for symbol in ny_pairs:
        results = ict_full_pipeline(
            symbol=symbol,
            daily_period="6mo",
            intraday_period="1mo",
            intraday_interval="1h",
            kill_zone_hours=(7, 12), # extended to noon
            bias_lookback=7,
            displacement_factor=0.5,
            buffer_pct=0.01,
            account_size=10000,
            risk_per_trade=0.01
        )
        
        print(f"\n=== ICT Full Pipeline Results for {symbol} ===")
        for k, v in results.items():
            print(f"{k}: {v}")

    print(f"\n=== ICT Full Pipeline Results for {symbol} ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    # Use 'schedule' to run every 15 minutes (or any interval).
    schedule.every(15).minutes.do(run_ict_pipeline_continuous)

    print("Starting continuous background process... (Press Ctrl+C to stop)")
    
    while True:
        schedule.run_pending()
        time.sleep(1)
