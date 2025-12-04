import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

###################################################
# OPTIONAL: TELEGRAM NOTIFICATION SETUP
###################################################
import requests

TELEGRAM_BOT_TOKEN = "7847987861:AAHrr1mAnYYyUK5bfrJyyc5GmggsWCEVJFw"
TELEGRAM_CHAT_ID   = "7749762504"

def send_telegram_message(message):
    """
    Sends a message to your Telegram bot.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials are not set. Skipping Telegram notification.")
        return
    
    url = f"https://api.telegram.org/bot7847987861:AAHrr1mAnYYyUK5bfrJyyc5GmggsWCEVJFw/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print(f"Message sent to Telegram: {message}")
        else:
            print(f"Error sending message. Response code: {response.status_code}")
    except Exception as e:
        print(f"Exception when sending Telegram message: {e}")




###################################################
# SIMPLE MOVING AVERAGE (MA) CROSSOVER
###################################################
def moving_average_strategy(df, short_window=50, long_window=200):
    """
    Example fallback logic:
    - Calculate two moving averages (e.g., 50 and 200).
    - If short MA crosses above long MA => 'BUY'
    - If short MA crosses below long MA => 'SELL'
    - Otherwise => 'WAIT'
    """
    if len(df) < long_window:
        return None  # Not enough data for MAs
    
    df['MA_short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_long']  = df['Close'].rolling(window=long_window).mean()
    
    prev_short = df['MA_short'].iloc[-2]
    prev_long  = df['MA_long'].iloc[-2]
    curr_short = df['MA_short'].iloc[-1]
    curr_long  = df['MA_long'].iloc[-1]
    
    # Detect crossover
    if (prev_short <= prev_long) and (curr_short > curr_long):
        return "BUY"
    elif (prev_short >= prev_long) and (curr_short < curr_long):
        return "SELL"
    else:
        return "WAIT"


###################################################
# GPT INTEGRATION (PLACEHOLDER)
###################################################
def get_gpt_recommendation(df):
    """
    Placeholder function for GPT-based logic.
    Replace with your real GPT calls if desired.
    """
    # Ensure single-column Series
    last_closes = df['Close'].tail(5).round(3).tolist()
    
    # Example prompt (not actually used):
    prompt = (
        f"Recent GBP/JPY closes: {last_closes}\n"
        "Provide a trade recommendation (BUY, SELL, or WAIT) and a brief reason:"
    )
    
    # For now, we pretend GPT said "WAIT":
    gpt_output = "WAIT. The market is flat right now."
    
    # Parse
    if "BUY" in gpt_output.upper():
        return "BUY"
    elif "SELL" in gpt_output.upper():
        return "SELL"
    else:
        return "WAIT"


###################################################
# ICT FULL LOGIC
###################################################

def detect_daily_bias(df, lookback=30, displacement_factor=1.5):
    """
    ICT-style daily bias detection:
      1) Identify major range high/low in last `lookback` bars.
      2) Check Market Structure Shift with displacement candle.
      3) Check if price is in Premium or Discount.
      4) Return "Bullish", "Bearish", or None.
    
    FIX: Cast each to float to avoid Series comparisons.
    """
    if len(df) < lookback + 2:
        return None
    
    recent_df = df.iloc[-lookback:].copy()
    
    # Compute major high/low as floats
    major_high = float(recent_df['High'].max())
    major_low  = float(recent_df['Low'].min())
    
    # Last candle values as floats
    last_close = float(df['Close'].iloc[-1])
    last_open  = float(df['Open'].iloc[-1])
    
    # Average body in the recent window
    recent_df['Body'] = (recent_df['Close'] - recent_df['Open']).abs()
    avg_body = float(recent_df['Body'].mean())
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

def find_swing_highs_lows(df, lookback=1):
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        # Swing High
        if (df['High'].iloc[i] > df['High'].iloc[i-1]) and (df['High'].iloc[i] > df['High'].iloc[i+1]):
            highs.append((df.index[i], df['High'].iloc[i]))
        # Swing Low
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1]) and (df['Low'].iloc[i] < df['Low'].iloc[i+1]):
            lows.append((df.index[i], df['Low'].iloc[i]))
    return highs, lows

def find_order_blocks(df, threshold=0.01):
    bullish_blocks, bearish_blocks = [], []
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

def convert_to_newyork_time(df):
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')
    return df

def filter_kill_zone(df, start_hour, end_hour):
    mask = (df.index.hour >= start_hour) & (df.index.hour <= end_hour)
    return df[mask]

def calculate_ote_zone(swing_low_price, swing_high_price):
    diff = swing_high_price - swing_low_price
    fib_618 = swing_high_price - 0.618 * diff
    fib_79  = swing_high_price - 0.79  * diff
    return (min(fib_618, fib_79), max(fib_618, fib_79))

def is_price_in_ote(df, swing_low_price, swing_high_price, tolerance=0.0):
    if df.empty:
        return False, (None, None)
    ote_lower, ote_upper = calculate_ote_zone(swing_low_price, swing_high_price)
    last_close = df['Close'].iloc[-1]
    in_zone = (last_close >= ote_lower - tolerance) and (last_close <= ote_upper + tolerance)
    return in_zone, (ote_lower, ote_upper)

def place_stop_loss_long(swing_low_price, buffer_pct=0.003):
    return swing_low_price * (1.0 - buffer_pct)

def place_stop_loss_short(swing_high_price, buffer_pct=0.003):
    return swing_high_price * (1.0 + buffer_pct)

def manage_take_profit(current_price,
                       entry_price,
                       partial_tp_price,
                       full_tp_price,
                       total_position_size):
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
    
    return remaining_position, realized_pnl, status_message

def ict_full_pipeline(symbol="GBPJPY=X", 
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
    1) Fetch daily & intraday data
    2) Detect daily bias
    3) Daily liquidity
    4) Intraday OB/FVG + Kill Zone
    5) OTE check
    6) Stop + position sizing
    7) Partial profits
    Returns dict with "signal": "BUY"/"SELL"/None + details
    """
    results = {}

    # STEP 1: Daily Data & Bias
    daily_df = yf.download(symbol, period=daily_period, interval='1d', progress=False)
    daily_df.dropna(inplace=True)
    daily_df.sort_index(inplace=True)

    if len(daily_df) < bias_lookback + 2:
        results["signal"] = None
        results["reason"] = "Not enough daily data"
        return results
    
    daily_bias = detect_daily_bias(daily_df, lookback=bias_lookback, displacement_factor=displacement_factor)
    results["daily_bias"] = daily_bias

    # STEP 2: Daily Liquidity
    daily_highs, daily_lows = find_swing_highs_lows(daily_df, lookback=1)
    results["num_daily_highs"] = len(daily_highs)
    results["num_daily_lows"]  = len(daily_lows)
    if not daily_highs or not daily_lows:
        results["signal"] = None
        results["reason"] = "No daily swings found"
        return results
    
    # Most recent swings
    recent_swing_high = daily_highs[-1]  # (Timestamp, price)
    recent_swing_low  = daily_lows[-1]   # (Timestamp, price)
    sh_price = float(recent_swing_high[1])
    sl_price = float(recent_swing_low[1])

    # STEP 3: Intraday
    intraday_df = yf.download(symbol, period=intraday_period, interval=intraday_interval, progress=False)
    intraday_df.dropna(inplace=True)
    intraday_df.sort_index(inplace=True)
    if intraday_df.empty:
        results["signal"] = None
        results["reason"] = "No intraday data"
        return results

    intraday_df = convert_to_newyork_time(intraday_df)
    bull_obs, bear_obs = find_order_blocks(intraday_df, threshold=0.01)
    fvg_list = find_fair_value_gaps(intraday_df)
    results["bullish_OB_count"] = len(bull_obs)
    results["bearish_OB_count"] = len(bear_obs)
    results["fvg_count"] = len(fvg_list)

    # STEP 4: Kill Zone
    kz_start, kz_end = kill_zone_hours
    kz_df = filter_kill_zone(intraday_df, kz_start, kz_end)
    results["kill_zone_count"] = len(kz_df)

    # STEP 5: OTE check (Bullish scenario)
    if daily_bias == "Bullish" and (sh_price > sl_price) and (len(kz_df) > 0):
        in_ote, ote_zone = is_price_in_ote(kz_df, sl_price, sh_price, tolerance=0.0001)
        results["in_ote"] = in_ote
        results["ote_zone"] = ote_zone
    else:
        in_ote = False
    
    if not in_ote:
        results["signal"] = None
        results["reason"] = "No OTE alignment or daily bias not bullish"
        return results

    # If bullish OTE triggered
    entry_price = float(kz_df['Close'].iloc[-1])
    stop_loss_price = place_stop_loss_long(sl_price, buffer_pct=buffer_pct)
    
    # Step 6: Position sizing
    risk_amt = account_size * risk_per_trade
    risk_per_unit = entry_price - stop_loss_price
    if risk_per_unit <= 0:
        results["signal"] = None
        results["reason"] = "Invalid risk logic (stop above entry)"
        return results
    
    position_size = risk_amt / risk_per_unit

    # Step 7: Partial profits
    partial_tp_price = sh_price
    full_tp_price = sh_price + 0.002  # e.g., a tiny bit above
    current_price = entry_price
    rem_pos, realized_pnl, tp_msg = manage_take_profit(
        current_price=current_price,
        entry_price=entry_price,
        partial_tp_price=partial_tp_price,
        full_tp_price=full_tp_price,
        total_position_size=position_size
    )

    # Construct final
    results["signal"] = "BUY"
    results["entry_price"] = entry_price
    results["stop_loss"]   = stop_loss_price
    results["position_size"] = position_size
    results["remaining_position"] = rem_pos
    results["realized_pnl"] = realized_pnl
    results["tp_message"] = tp_msg
    results["reason"] = "Bullish OTE with daily bias"

    return results


###################################################
# FETCH 1-MINUTE GBP/JPY DATA FOR GRAPH + GPT + MA
###################################################
def fetch_gbpjpy_data():
    """
    Example: 1-minute interval, last 1 day for GBP/JPY.
    """
    df = yf.download(tickers='GBPJPY=X', interval='1m', period='1d')
    df.dropna(inplace=True)
    return df


###################################################
# MAIN APP FLOW
###################################################
def main():
    # A) Fetch 1-minute data (for MA + GPT + Plot)
    gbp_jpy_data = fetch_gbpjpy_data()
    if gbp_jpy_data.empty:
        print("No data fetched for GBP/JPY. Exiting.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(gbp_jpy_data['Close'], label='GBP/JPY Close')
    plt.title('GBP/JPY Close Price (1-Minute)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()
    
    # B) MA Strategy
    ma_signal = moving_average_strategy(gbp_jpy_data)
    print(f"[MA Strategy Signal]: {ma_signal}")
    
    # C) ICT Logic on Higher TF
    ict_result = ict_full_pipeline(
        symbol="GBPJPY=X",
        daily_period="6mo",
        intraday_period="1mo",
        intraday_interval="1h",
        kill_zone_hours=(7, 10),
        bias_lookback=30,
        displacement_factor=1.5,
        buffer_pct=0.003,
        account_size=10000,
        risk_per_trade=0.01
    )
    ict_signal = ict_result.get("signal", None)
    print(f"[ICT Logic Signal]: {ict_signal}, reason: {ict_result.get('reason','')}")
    
    # D) GPT Logic
    gpt_signal = get_gpt_recommendation(gbp_jpy_data)
    print(f"[GPT Recommendation]: {gpt_signal}")
    
    # E) Combine signals (ICT > GPT > MA)
    final_signal = None
    rationale = "No rationale"
    
    if ict_signal in ["BUY", "SELL"]:
        final_signal = ict_signal
        rationale = f"ICT logic triggered: {ict_result.get('reason','')}"
    else:
        if gpt_signal in ["BUY", "SELL"]:
            final_signal = gpt_signal
            rationale = "GPT-based recommendation."
        else:
            if ma_signal in ["BUY", "SELL"]:
                final_signal = ma_signal
                rationale = "MA crossover fallback."
            else:
                final_signal = "WAIT"
                rationale = "All signals = WAIT/None."
    
    print(f"[FINAL SIGNAL]: {final_signal}, rationale: {rationale}")
    
    # F) Send Telegram if BUY/SELL
    if final_signal in ["BUY", "SELL"]:
        msg = f"GBP/JPY Trade Alert: {final_signal}\nReason: {rationale}"
        send_telegram_message(msg)
    else:
        print("No trade alert sent. Final signal is WAIT or None.")


if __name__ == "__main__":
    main()
