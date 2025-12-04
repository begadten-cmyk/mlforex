"""
ICT-Inspired Full Logic (Bullish + Bearish) with Telegram Notification
FOR EDUCATIONAL USE ONLY
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pytz
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
    url = f"https://api.telegram.org/bot7847987861:AAHrr1mAnYYyUK5bfrJyyc5GmggsWCEVJFw/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Telegram error: {response.text}")
    except Exception as e:
        print(f"Telegram send failed: {str(e)}")


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

def detect_daily_bias(df, lookback=30, displacement_factor=1.5):
    """
    A more ICT-aligned daily bias detection function:
      1) Identify major high & low in the last `lookback` bars.
      2) Check for a Market Structure Shift (MSS) with a displacement candle:
         - Bullish if we close above the major high with wide-range body
         - Bearish if we close below the major low with wide-range body
      3) Determine if we're in Premium (above midpoint) or Discount (below midpoint).
      4) If bullish shift + discount => "Bullish"
         If bearish shift + premium => "Bearish"
         Else => None
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
    displacement = (last_body_size > displacement_factor * avg_body)
    
    broke_above = (last_close > major_high)
    broke_below = (last_close < major_low)
    
    bullish_shift = (broke_above and displacement)
    bearish_shift = (broke_below and displacement)
    
    midpoint = (major_high + major_low) / 2.0
    zone = "Discount" if last_close < midpoint else "Premium"
    
    if bullish_shift and zone == "Discount":
        return "Bullish"
    elif bearish_shift and zone == "Premium":
        return "Bearish"
    else:
        return None


############################
# STEP 2: DAILY LIQUIDITY POOLS
############################

def find_swing_highs_lows(df, lookback=1):
    """
    Swing High: High[i] > High[i-1] & High[i] > High[i+1]
    Swing Low:  Low[i] < Low[i-1] & Low[i] < Low[i+1]
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

def find_order_blocks(df, threshold=0.01):
    """
    Naive approach for OB:
    - Bullish OB: last down-candle prior to a strong up candle (pct_change > threshold)
    - Bearish OB: last up-candle prior to a strong down candle (pct_change < -threshold)
    Returns lists of (timestamp, open, close).
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
    3-candle pattern for FVG:
    - Bullish FVG if c2_low > c1_high AND c3_low > c1_high
    - Bearish FVG if c2_high < c1_low AND c3_high < c1_low
    Returns list of tuples: (index_of_displacement_candle, type, (gap_low, gap_high)).
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
    Convert df.index to US/Eastern if it's naive or UTC.
    """
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')
    return df

def filter_kill_zone(df, start_hour, end_hour):
    """
    Return subset of df where df.index.hour is in [start_hour, end_hour].
    """
    mask = (df.index.hour >= start_hour) & (df.index.hour <= end_hour)
    return df[mask]


############################
# STEP 5: OTE (Optimal Trade Entry)
############################

def calculate_ote_zone_bullish(swing_low_price, swing_high_price):
    """
    For bullish scenario: 61.8% - 79% retracement from low to high.
    """
    diff = swing_high_price - swing_low_price
    fib_618 = swing_high_price - 0.618 * diff
    fib_79  = swing_high_price - 0.79  * diff
    ote_lower = min(fib_618, fib_79)
    ote_upper = max(fib_618, fib_79)
    return (ote_lower, ote_upper)

def calculate_ote_zone_bearish(swing_high_price, swing_low_price):
    """
    For bearish scenario: 61.8% - 79% retracement from high down to low.
    """
    diff = swing_high_price - swing_low_price
    fib_618 = swing_low_price + 0.618 * diff
    fib_79  = swing_low_price + 0.79  * diff
    ote_lower = min(fib_618, fib_79)
    ote_upper = max(fib_618, fib_79)
    return (ote_lower, ote_upper)


def is_price_in_bullish_ote(df, swing_low_price, swing_high_price, tolerance=0.0):
    """
    Check if the last candle's Close is within OTE zone for a bullish scenario.
    """
    if df.empty:
        return False, (None, None)
    ote_lower, ote_upper = calculate_ote_zone_bullish(swing_low_price, swing_high_price)
    last_close = df['Close'].iloc[-1]
    in_zone = (last_close >= ote_lower - tolerance) and (last_close <= ote_upper + tolerance)
    return in_zone, (ote_lower, ote_upper)


def is_price_in_bearish_ote(df, swing_high_price, swing_low_price, tolerance=0.0):
    """
    Check if the last candle's Close is within OTE zone for a bearish scenario.
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

def place_stop_loss_long(swing_low_price, buffer_pct=0.003):
    """
    For a bullish trade, place SL some % below swing_low_price.
    """
    return swing_low_price * (1.0 - buffer_pct)

def place_stop_loss_short(swing_high_price, buffer_pct=0.003):
    """
    For a bearish trade, place SL some % above swing_high_price.
    """
    return swing_high_price * (1.0 + buffer_pct)


############################
# STEP 7: PARTIAL PROFITS
############################

def manage_take_profit_long(current_price, entry_price, partial_tp_price, full_tp_price, total_position_size):
    """
    Partial profit for long:
     - If current_price >= partial_tp_price => close 50%
     - If current_price >= full_tp_price => close remainder
    """
    realized_pnl = 0.0
    remaining_position = total_position_size
    status_message = "No action"
    
    if (current_price >= partial_tp_price) and (remaining_position == total_position_size):
        half_pos = total_position_size * 0.5
        realized_pnl += (current_price - entry_price) * half_pos
        remaining_position -= half_pos
        status_message = f"Partial TP triggered at {current_price:.5f}"
    
    if (current_price >= full_tp_price) and (remaining_position > 0):
        realized_pnl += (current_price - entry_price) * remaining_position
        remaining_position = 0
        status_message = f"Full TP triggered at {current_price:.5f}"
    
    return (remaining_position, realized_pnl, status_message)

def manage_take_profit_short(current_price, entry_price, partial_tp_price, full_tp_price, total_position_size):
    """
    Partial profit for short:
     - If current_price <= partial_tp_price => close 50%
     - If current_price <= full_tp_price => close remainder
    """
    realized_pnl = 0.0
    remaining_position = total_position_size
    status_message = "No action"
    
    if (current_price <= partial_tp_price) and (remaining_position == total_position_size):
        half_pos = total_position_size * 0.5
        realized_pnl += (entry_price - current_price) * half_pos  # profit if price dropped
        remaining_position -= half_pos
        status_message = f"Partial TP triggered at {current_price:.5f}"
    
    if (current_price <= full_tp_price) and (remaining_position > 0):
        realized_pnl += (entry_price - current_price) * remaining_position
        remaining_position = 0
        status_message = f"Full TP triggered at {current_price:.5f}"
    
    return (remaining_position, realized_pnl, status_message)


############################
# FULL PIPELINE
############################

def ict_full_pipeline(symbol="EURUSD=X", 
                      daily_period="6mo", 
                      intraday_period="1mo", 
                      intraday_interval="1h", 
                      kill_zone_hours=(7, 10),
                      bias_lookback=30, 
                      displacement_factor=1.5,
                      buffer_pct=0.003,
                      account_size=10000,
                      risk_per_trade=0.01):
    """
    Comprehensive pipeline for both Bullish & Bearish:
    1) Fetch Daily + Intraday
    2) Detect daily bias => "Bullish", "Bearish", or None
    3) Mark daily liquidity (swing highs/lows)
    4) Intraday OB/FVG + Kill Zone
    5) OTE check for either bullish or bearish
    6) Stop + position sizing
    7) Partial profits
    8) Telegram notification if a signal is triggered
    """

    results = {}

    # STEP 1: FETCH DAILY DATA, DETECT DAILY BIAS
    # -----------------------------------------------------
    # Ensure we only download a single symbol at a time (string, not list)
    daily_df = yf.download(symbol, period=daily_period, interval='1d', progress=False)
    if isinstance(daily_df.columns, pd.MultiIndex):
        # Flatten columns if multi-index
        daily_df.columns = daily_df.columns.droplevel(1)

    daily_df.dropna(inplace=True)
    daily_df.sort_index(inplace=True)

    if len(daily_df) < bias_lookback + 2:
        results["signal"] = None
        results["reason"] = "Not enough daily data for bias"
        return results
    
    daily_bias = detect_daily_bias(daily_df, lookback=bias_lookback, displacement_factor=displacement_factor)
    results["daily_bias"] = daily_bias

    # STEP 2: DAILY LIQUIDITY POOLS
    # -----------------------------------------------------
    daily_highs, daily_lows = find_swing_highs_lows(daily_df, lookback=1)
    results["num_daily_highs"] = len(daily_highs)
    results["num_daily_lows"] = len(daily_lows)
    if not daily_highs or not daily_lows:
        results["signal"] = None
        results["reason"] = "No daily swing found"
        return results

    # Most recent swing high & low
    recent_swing_high = daily_highs[-1]  # (Timestamp, price)
    recent_swing_low  = daily_lows[-1]   # (Timestamp, price)
    sh_price = recent_swing_high[1]
    sl_price = recent_swing_low[1]

    # STEP 3: INTRADAY DATA
    # -----------------------------------------------------
    intraday_df = yf.download(symbol, period=intraday_period, interval=intraday_interval, progress=False)
    if isinstance(intraday_df.columns, pd.MultiIndex):
        # Flatten columns if multi-index
        intraday_df.columns = intraday_df.columns.droplevel(1)

    intraday_df.dropna(inplace=True)
    intraday_df.sort_index(inplace=True)
    if intraday_df.empty:
        results["signal"] = None
        results["reason"] = "No intraday data"
        return results

    intraday_df = convert_to_newyork_time(intraday_df)

    # OB / FVG (Not strictly required to place trade but included for context)
    bull_obs, bear_obs = find_order_blocks(intraday_df, threshold=0.01)
    fvg_list = find_fair_value_gaps(intraday_df)
    results["bullish_OB_count"] = len(bull_obs)
    results["bearish_OB_count"] = len(bear_obs)
    results["fvg_count"] = len(fvg_list)

    # STEP 4: KILL ZONE
    # -----------------------------------------------------
    kz_start, kz_end = kill_zone_hours
    kz_df = filter_kill_zone(intraday_df, kz_start, kz_end)
    results["kill_zone_count"] = len(kz_df)
    if kz_df.empty:
        results["signal"] = None
        results["reason"] = "No candles in kill zone"
        return results

    # STEP 5: OTE Check (Bullish or Bearish)
    # -----------------------------------------------------
    # BULLISH SCENARIO
    if daily_bias == "Bullish" and (sh_price > sl_price):
        in_ote, ote_zone = is_price_in_bullish_ote(kz_df, sl_price, sh_price, tolerance=0.0001)
        results["in_ote"] = in_ote
        results["ote_zone"] = ote_zone
        if in_ote:
            # Step 6: Place Stop, Position Sizing
            entry_price = kz_df['Close'].iloc[-1]
            stop_loss_price = place_stop_loss_long(sl_price, buffer_pct=buffer_pct)
            
            risk_amt = account_size * risk_per_trade
            risk_per_unit = entry_price - stop_loss_price
            if risk_per_unit <= 0:
                results["signal"] = None
                results["reason"] = "Invalid risk logic (stop above entry for bullish)"
                return results
            
            position_size = risk_amt / risk_per_unit

            # STEP 7: Partial Profits
            partial_tp_price = sh_price
            full_tp_price    = sh_price + 0.002
            current_price    = entry_price

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
            results["reason"] = "Bullish OTE with daily bias."

            # STEP 8: SEND TELEGRAM NOTIFICATION
            msg = (f"Signal: BUY {symbol}\n"
                   f"Entry: {entry_price:.5f}\n"
                   f"Stop: {stop_loss_price:.5f}\n"
                   f"Position Size: {position_size:.2f}\n"
                   f"Reason: {results['reason']}")
            send_telegram_message(msg)
            return results
        else:
            results["signal"] = None
            results["reason"] = "Bullish bias but no bullish OTE alignment."
            return results

    # BEARISH SCENARIO
    elif daily_bias == "Bearish" and (sh_price > sl_price):
        in_ote, ote_zone = is_price_in_bearish_ote(kz_df, sh_price, sl_price, tolerance=0.0001)
        results["in_ote"] = in_ote
        results["ote_zone"] = ote_zone
        if in_ote:
            # Step 6: Place Stop, Position Sizing
            entry_price = kz_df['Close'].iloc[-1]
            stop_loss_price = place_stop_loss_short(sh_price, buffer_pct=buffer_pct)
            
            risk_amt = account_size * risk_per_trade
            risk_per_unit = stop_loss_price - entry_price  # short => stop - entry
            if risk_per_unit <= 0:
                results["signal"] = None
                results["reason"] = "Invalid risk logic (stop below entry for bearish)"
                return results

            position_size = risk_amt / risk_per_unit

            # STEP 7: Partial Profits
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
            results["reason"] = "Bearish OTE with daily bias."

            msg = (f"Signal: SELL {symbol}\n"
                   f"Entry: {entry_price:.5f}\n"
                   f"Stop: {stop_loss_price:.5f}\n"
                   f"Position Size: {position_size:.2f}\n"
                   f"Reason: {results['reason']}")
            send_telegram_message(msg)
            return results
        else:
            results["signal"] = None
            results["reason"] = "Bearish bias but no bearish OTE alignment."
            return results

    else:
        # If daily_bias is None or price range doesn't make sense
        results["signal"] = None
        results["reason"] = "No valid daily bias or not enough range for OTE."
        return results


############################
# EXAMPLE USAGE
############################

if __name__ == "__main__":
    symbol = "EURUSD=X"
    results = ict_full_pipeline(
        symbol=symbol,
        daily_period="6mo",
        intraday_period="1mo",
        intraday_interval="1h",
        kill_zone_hours=(7, 10),  # e.g. 7 AM - 10 AM NY time
        bias_lookback=30,
        displacement_factor=1.5,
        buffer_pct=0.003,
        account_size=10000,
        risk_per_trade=0.01
    )

    print(f"\n=== ICT Full Pipeline Results for {symbol} ===")
    for k, v in results.items():
        print(f"{k}: {v}")
