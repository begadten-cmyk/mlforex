import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime, timedelta

###################################
# TELEGRAM STUB (not used in backtest)
###################################
def send_telegram_message(msg):
    # We won't really send Telegrams in a backtest, so it's a no-op here.
    pass

###################################
# ICT LOGIC HELPERS
###################################
def detect_daily_bias(df, lookback=30, displacement_factor=1.5):
    if len(df) < lookback + 2:
        return None
    recent_df = df.iloc[-lookback:].copy()
    major_high = recent_df['High'].max()
    major_low  = recent_df['Low'].min()
    last_close = df['Close'].iloc[-1]
    last_open  = df['Open'].iloc[-1]
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

def find_swing_highs_lows(df, lookback=1):
    highs = []
    lows = []
    for i in range(lookback, len(df) - lookback):
        if (df['High'].iloc[i] > df['High'].iloc[i-1]) and (df['High'].iloc[i] > df['High'].iloc[i+1]):
            highs.append((df.index[i], df['High'].iloc[i]))
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1]) and (df['Low'].iloc[i] < df['Low'].iloc[i+1]):
            lows.append((df.index[i], df['Low'].iloc[i]))
    return highs, lows

def convert_to_newyork_time(df):
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    return df.tz_convert('US/Eastern')

def filter_kill_zone(df, start_hour, end_hour):
    mask = (df.index.hour >= start_hour) & (df.index.hour <= end_hour)
    return df[mask]

def calculate_ote_zone_bullish(swing_low_price, swing_high_price):
    diff = swing_high_price - swing_low_price
    fib_618 = swing_high_price - 0.618 * diff
    fib_79  = swing_high_price - 0.79  * diff
    return (min(fib_618, fib_79), max(fib_618, fib_79))

def calculate_ote_zone_bearish(swing_high_price, swing_low_price):
    diff = swing_high_price - swing_low_price
    fib_618 = swing_low_price + 0.618 * diff
    fib_79  = swing_low_price + 0.79  * diff
    return (min(fib_618, fib_79), max(fib_618, fib_79))

def is_price_in_bullish_ote(price, swing_low_price, swing_high_price, tolerance=0.0):
    ote_lower, ote_upper = calculate_ote_zone_bullish(swing_low_price, swing_high_price)
    return (price >= ote_lower - tolerance) and (price <= ote_upper + tolerance)

def is_price_in_bearish_ote(price, swing_high_price, swing_low_price, tolerance=0.0):
    ote_lower, ote_upper = calculate_ote_zone_bearish(swing_high_price, swing_low_price)
    return (price >= ote_lower - tolerance) and (price <= ote_upper + tolerance)

def place_stop_loss_long(swing_low_price, buffer_pct=0.003):
    return swing_low_price * (1.0 - buffer_pct)

def place_stop_loss_short(swing_high_price, buffer_pct=0.003):
    return swing_high_price * (1.0 + buffer_pct)

##################################
# BACKTESTING ENGINE
##################################
def backtest_ict(symbol="EURUSD=X",
                 start_date="2023-01-01",
                 end_date="2023-08-01",
                 bias_lookback=30,
                 displacement_factor=1.5,
                 kill_zone_hours=(7, 10),
                 buffer_pct=0.003,
                 account_size=100000.0,
                 risk_per_trade=0.01):
    """
    1) Fetch all daily data from start_date to end_date.
    2) For each day (from bias_lookback onward), we pretend it's "today".
       - We have daily data up to 'yesterday'.
       - Determine daily bias, find swing highs/lows.
       - If there's a bias, then load that day's intraday data (for the kill zone).
       - Check if OTE is triggered in that kill zone. If yes, place a trade.
       - Manage partial & full TP within the same day. If not reached, the trade ends at day close (for simplicity).
    3) Track each trade, final PnL, stats, etc.
    """

    # 1) Download daily data
    daily_df = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)
    # Flatten columns if multi-index
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = daily_df.columns.droplevel(1)
    daily_df.dropna(inplace=True)
    daily_df.sort_index(inplace=True)

    # We will store trades in a list of dicts
    trades = []
    equity = account_size

    # We'll iterate from day # bias_lookback up to the end
    all_dates = daily_df.index.unique().tolist()
    
    for idx in range(bias_lookback, len(all_dates)):
        current_day = all_dates[idx]
        # 'yesterday' daily data is everything up to the day before current_day
        # so that we only "know" that info up to that day
        up_to_yesterday = daily_df.loc[:current_day].iloc[:-1]
        if len(up_to_yesterday) < bias_lookback:
            continue  # skip if not enough data

        # Step A: Determine daily bias from up_to_yesterday
        day_bias = detect_daily_bias(up_to_yesterday, lookback=bias_lookback, displacement_factor=displacement_factor)
        if day_bias is None:
            # No trade if no bias
            continue

        # Step B: find daily swing highs/lows from up_to_yesterday
        daily_highs, daily_lows = find_swing_highs_lows(up_to_yesterday, lookback=1)
        if not daily_highs or not daily_lows:
            continue
        recent_swing_high = daily_highs[-1]
        recent_swing_low  = daily_lows[-1]
        sh_price = recent_swing_high[1]
        sl_price = recent_swing_low[1]
        if sh_price <= sl_price:
            # If the most recent swing high is below or equal to the low, skip
            continue

        # Step C: On 'current_day', we load intraday data for that single day only
        # We'll fetch from current_day 00:00 to current_day 23:59
        day_start = pd.Timestamp(current_day).floor("D")  # midnight
        day_end   = day_start + pd.Timedelta(days=1)      # next midnight
        intraday_df = yf.download(symbol, start=day_start, end=day_end, interval='1h', progress=False)
        if isinstance(intraday_df.columns, pd.MultiIndex):
            intraday_df.columns = intraday_df.columns.droplevel(1)
        intraday_df.dropna(inplace=True)
        intraday_df.sort_index(inplace=True)
        if intraday_df.empty:
            continue
        intraday_df = convert_to_newyork_time(intraday_df)

        # Filter kill zone
        kz_df = filter_kill_zone(intraday_df, kill_zone_hours[0], kill_zone_hours[1])
        if kz_df.empty:
            continue

        # We'll check the *last close in kill zone* to see if OTE is triggered
        last_kz_close = kz_df['Close'].iloc[-1]

        # Step D: If bias is bullish, check bullish OTE
        if day_bias == "Bullish":
            in_ote = is_price_in_bullish_ote(last_kz_close, sl_price, sh_price, tolerance=0.0001)
            if in_ote:
                # If we have a bullish OTE, place trade
                entry_price = last_kz_close
                stop_loss_price = place_stop_loss_long(sl_price, buffer_pct=buffer_pct)
                risk_amt = equity * risk_per_trade
                risk_per_unit = entry_price - stop_loss_price
                if risk_per_unit <= 0:
                    continue

                position_size = risk_amt / risk_per_unit

                # We'll see if partial/ full TPs are triggered on *any candle after entry* within that day
                partial_tp_price = sh_price
                full_tp_price    = sh_price + 0.002

                # We simulate candle by candle from the entry time to the end of the day
                # to see if partial or full TP triggers
                # We'll also forcibly close at day's last candle if it hasn't hit full TP
                trade_open = True
                realized_pnl = 0
                remaining_pos = position_size
                entry_dt = kz_df.index[-1]  # time of the last kill-zone candle

                # Intraday data after the last kill-zone candle
                post_entry_df = intraday_df.loc[intraday_df.index >= entry_dt]
                for dt, row in post_entry_df.iterrows():
                    current_px = row['Close']

                    # Partial TP?
                    if (current_px >= partial_tp_price) and (remaining_pos == position_size):
                        # close 50%
                        half_pos = position_size * 0.5
                        realized_pnl += (current_px - entry_price) * half_pos
                        remaining_pos -= half_pos

                    # Full TP?
                    if (current_px >= full_tp_price) and (remaining_pos > 0):
                        realized_pnl += (current_px - entry_price) * remaining_pos
                        remaining_pos = 0
                        trade_open = False
                        break

                # If trade still open at end of day, close at last candle:
                if trade_open:
                    close_price = post_entry_df['Close'].iloc[-1]
                    # no partial triggers now, so we close everything
                    realized_pnl += (close_price - entry_price) * remaining_pos
                    remaining_pos = 0
                    trade_open = False

                # update equity
                equity += realized_pnl

                trades.append({
                    "date": current_day,
                    "bias": "Bullish",
                    "entry_time": entry_dt,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss_price,
                    "position_size": position_size,
                    "realized_pnl": realized_pnl,
                    "end_equity": equity
                })

        # Step E: If bias is bearish, check bearish OTE
        elif day_bias == "Bearish":
            in_ote = is_price_in_bearish_ote(last_kz_close, sh_price, sl_price, tolerance=0.0001)
            if in_ote:
                entry_price = last_kz_close
                stop_loss_price = place_stop_loss_short(sh_price, buffer_pct=buffer_pct)
                risk_amt = equity * risk_per_trade
                risk_per_unit = stop_loss_price - entry_price
                if risk_per_unit <= 0:
                    continue

                position_size = risk_amt / risk_per_unit

                partial_tp_price = sl_price
                full_tp_price    = sl_price - 0.002

                trade_open = True
                realized_pnl = 0
                remaining_pos = position_size
                entry_dt = kz_df.index[-1]

                post_entry_df = intraday_df.loc[intraday_df.index >= entry_dt]
                for dt, row in post_entry_df.iterrows():
                    current_px = row['Close']
                    # For short, we want price to drop
                    # Partial TP
                    if (current_px <= partial_tp_price) and (remaining_pos == position_size):
                        half_pos = position_size * 0.5
                        # short profit is entry_price - current_px
                        realized_pnl += (entry_price - current_px) * half_pos
                        remaining_pos -= half_pos

                    # Full TP
                    if (current_px <= full_tp_price) and (remaining_pos > 0):
                        realized_pnl += (entry_price - current_px) * remaining_pos
                        remaining_pos = 0
                        trade_open = False
                        break

                # close at day's end if still open
                if trade_open:
                    close_price = post_entry_df['Close'].iloc[-1]
                    realized_pnl += (entry_price - close_price) * remaining_pos
                    remaining_pos = 0
                    trade_open = False

                equity += realized_pnl

                trades.append({
                    "date": current_day,
                    "bias": "Bearish",
                    "entry_time": entry_dt,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss_price,
                    "position_size": position_size,
                    "realized_pnl": realized_pnl,
                    "end_equity": equity
                })

    ###############################################
    # Summarize Results
    ###############################################
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades were taken during the backtest.")
        return None

    # Basic stats
    total_trades = len(trades_df)
    wins = sum(trades_df['realized_pnl'] > 0)
    losses = sum(trades_df['realized_pnl'] < 0)
    win_rate = wins / total_trades * 100.0
    net_pnl = trades_df['realized_pnl'].sum()
    final_equity = account_size + net_pnl  # or simply trades_df['end_equity'].iloc[-1]

    print("\n========== BACKTEST RESULTS ==========")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%")
    print(f"Net PnL: {net_pnl:.2f}")
    print(f"Start Equity: {account_size:.2f}, End Equity: {final_equity:.2f}")
    print("======================================\n")
    print("All Trades:")
    print(trades_df)
    return trades_df


##################################
# RUN BACKTEST
##################################
if __name__ == "__main__":
    results_df = backtest_ict(
        symbol="EURUSD=X",
        start_date="2022-01-01",  # pick your backtest start
        end_date="2023-08-01",    # pick your backtest end
        bias_lookback=30,
        displacement_factor=1.5,
        kill_zone_hours=(7, 10),
        buffer_pct=0.003,
        account_size=100000.0,
        risk_per_trade=0.01
    )
