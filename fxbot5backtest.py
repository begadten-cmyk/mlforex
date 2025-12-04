import pandas as pd
import numpy as np
import yfinance as yf
import datetime

##########################################
# 1) ICT Logic Functions (Loosened)
##########################################

def detect_daily_bias(df, lookback=7, displacement_factor=0.5):
    """
    LOOSER SETTINGS:
    - lookback=7 (instead of 15)
    - displacement_factor=0.5 (instead of 1.0)
    """
    if len(df) < lookback + 2:
        return None

    recent_df = df.iloc[-lookback:].copy()

    # major_high, major_low as scalars
    major_high = recent_df['High'].max()
    if hasattr(major_high, 'item'):
        major_high = major_high.item()

    major_low = recent_df['Low'].min()
    if hasattr(major_low, 'item'):
        major_low = major_low.item()

    last_close = df['Close'].iloc[-1]
    if hasattr(last_close, 'item'):
        last_close = last_close.item()
    last_open = df['Open'].iloc[-1]
    if hasattr(last_open, 'item'):
        last_open = last_open.item()

    # Average body
    recent_df['Body'] = (recent_df['Close'] - recent_df['Open']).abs()
    avg_body = recent_df['Body'].mean()
    if hasattr(avg_body, 'item'):
        avg_body = avg_body.item()

    last_body_size = abs(last_close - last_open)
    # Smaller displacement => more likely bullish/bearish shift
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

def find_swing_highs_lows(df, lookback=1):
    """
    Same logic, just ensuring scalar comparisons.
    """
    highs = []
    lows = []
    for i in range(lookback, len(df) - lookback):
        high_i    = df['High'].iloc[i]
        high_prev = df['High'].iloc[i - 1]
        high_next = df['High'].iloc[i + 1]
        low_i     = df['Low'].iloc[i]
        low_prev  = df['Low'].iloc[i - 1]
        low_next  = df['Low'].iloc[i + 1]

        # Convert if needed
        if hasattr(high_i, 'item'):
            high_i = high_i.item()
        if hasattr(high_prev, 'item'):
            high_prev = high_prev.item()
        if hasattr(high_next, 'item'):
            high_next = high_next.item()
        if hasattr(low_i, 'item'):
            low_i = low_i.item()
        if hasattr(low_prev, 'item'):
            low_prev = low_prev.item()
        if hasattr(low_next, 'item'):
            low_next = low_next.item()

        if (high_i > high_prev) and (high_i > high_next):
            highs.append((df.index[i], high_i))
        if (low_i < low_prev) and (low_i < low_next):
            lows.append((df.index[i], low_i))
    return highs, lows

def find_order_blocks(df, threshold=0.005):
    """
    Lower threshold => 0.005 instead of 0.01
    => More OB signals
    """
    bullish_blocks = []
    bearish_blocks = []
    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i - 1]
        curr_close = df['Close'].iloc[i]
        if hasattr(prev_close, 'item'):
            prev_close = prev_close.item()
        if hasattr(curr_close, 'item'):
            curr_close = curr_close.item()

        if prev_close == 0:
            continue
        pct_change = (curr_close - prev_close) / prev_close

        prev_close_val = df['Close'].iloc[i-1]
        prev_open_val  = df['Open'].iloc[i-1]
        if hasattr(prev_close_val, 'item'):
            prev_close_val = prev_close_val.item()
        if hasattr(prev_open_val, 'item'):
            prev_open_val = prev_open_val.item()

        # Bullish OB
        if pct_change > threshold and (prev_close_val < prev_open_val):
            blocks_open  = df['Open'].iloc[i-1]
            blocks_close = df['Close'].iloc[i-1]
            if hasattr(blocks_open, 'item'):
                blocks_open = blocks_open.item()
            if hasattr(blocks_close, 'item'):
                blocks_close = blocks_close.item()
            bullish_blocks.append((df.index[i-1], blocks_open, blocks_close))

        # Bearish OB
        if pct_change < -threshold and (prev_close_val > prev_open_val):
            blocks_open  = df['Open'].iloc[i-1]
            blocks_close = df['Close'].iloc[i-1]
            if hasattr(blocks_open, 'item'):
                blocks_open = blocks_open.item()
            if hasattr(blocks_close, 'item'):
                blocks_close = blocks_close.item()
            bearish_blocks.append((df.index[i-1], blocks_open, blocks_close))

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

        for var in [c1_high, c1_low, c2_high, c2_low, c3_high, c3_low]:
            if hasattr(var, 'item'):
                var = var.item()

        c1_high = float(c1_high); c1_low = float(c1_low)
        c2_high = float(c2_high); c2_low = float(c2_low)
        c3_high = float(c3_high); c3_low = float(c3_low)

        if (c2_low > c1_high) and (c3_low > c1_high):
            gap_low = c1_high
            gap_high = c2_low
            fvg_list.append((df.index[i-1], 'Bullish FVG', (gap_low, gap_high)))
        if (c2_high < c1_low) and (c3_high < c1_low):
            gap_low = c2_high
            gap_high = c1_low
            fvg_list.append((df.index[i-1], 'Bearish FVG', (gap_low, gap_high)))
    return fvg_list

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

def manage_take_profit_long(current_price, entry_price, partial_tp_price, full_tp_price, total_position_size):
    realized_pnl = 0.0
    remaining_position = total_position_size

    if (current_price >= partial_tp_price) and (remaining_position == total_position_size):
        half_pos = total_position_size * 0.5
        realized_pnl += (current_price - entry_price) * half_pos
        remaining_position -= half_pos

    if (current_price >= full_tp_price) and (remaining_position > 0):
        realized_pnl += (current_price - entry_price) * remaining_position
        remaining_position = 0
    return (remaining_position, realized_pnl, "TP done")

def manage_take_profit_short(current_price, entry_price, partial_tp_price, full_tp_price, total_position_size):
    realized_pnl = 0.0
    remaining_position = total_position_size

    if (current_price <= partial_tp_price) and (remaining_position == total_position_size):
        half_pos = total_position_size * 0.5
        realized_pnl += (entry_price - current_price) * half_pos
        remaining_position -= half_pos

    if (current_price <= full_tp_price) and (remaining_position > 0):
        realized_pnl += (entry_price - current_price) * remaining_position
        remaining_position = 0
    return (remaining_position, realized_pnl, "TP done")

def place_stop_loss_long(swing_low_price, buffer_pct=0.01):
    """
    Larger buffer_pct=0.01 => looser stops => more frequent trades
    """
    return float(swing_low_price) * (1.0 - buffer_pct)

def place_stop_loss_short(swing_high_price, buffer_pct=0.01):
    """
    Larger buffer_pct=0.01 => looser stops => more frequent trades
    """
    return float(swing_high_price) * (1.0 + buffer_pct)

##########################################
# 2) MULTI-TRADE run_ict_pipeline_day
##########################################
def run_ict_pipeline_day(symbol,
                         daily_df,
                         intraday_df_1h,
                         intraday_df_15m,
                         kill_zone_hours=(1,12),  
                         bias_lookback=7,          # Looser
                         displacement_factor=0.5,  # Looser
                         buffer_pct=0.01,          # Looser
                         account_size=10000,
                         risk_per_trade=0.01):
    """
    Multi-trade approach with looser constraints:
    - bias_lookback=7
    - displacement_factor=0.5
    - buffer_pct=0.01
    """
    all_trades = []

    # 1) Daily bias
    if len(daily_df) < bias_lookback + 2:
        return all_trades
    
    daily_bias = detect_daily_bias(daily_df, lookback=bias_lookback, displacement_factor=displacement_factor)
    daily_highs, daily_lows = find_swing_highs_lows(daily_df, lookback=1)
    if (not daily_highs) or (not daily_lows):
        return all_trades
    
    sh_price = daily_highs[-1][1]
    sl_price = daily_lows[-1][1]

    df_15m_kz = intraday_df_15m.between_time(f"{kill_zone_hours[0]}:00", f"{kill_zone_hours[1]}:00").copy()
    if df_15m_kz.empty:
        return all_trades

    for ts in df_15m_kz.index:
        close_price = df_15m_kz['Close'].loc[ts]
        if hasattr(close_price, 'item'):
            close_price = close_price.item()

        # BULLISH
        if (daily_bias == "Bullish") and (sh_price > sl_price):
            ote_lower, ote_upper = calculate_ote_zone_bullish(sl_price, sh_price)
            # We'll allow a small tolerance in the next comparison
            tol = 0.002  
            if (close_price >= (ote_lower - tol)) and (close_price <= (ote_upper + tol)):
                entry_price = close_price
                stop_loss_price = place_stop_loss_long(sl_price, buffer_pct=buffer_pct)
                risk_amt = account_size * risk_per_trade
                risk_per_unit = entry_price - stop_loss_price
                if risk_per_unit > 0:
                    position_size = risk_amt / risk_per_unit
                    partial_tp_price = sh_price
                    full_tp_price    = sh_price + 0.002
                    current_price    = entry_price

                    rem_pos, realized_pnl, _ = manage_take_profit_long(
                        current_price=current_price,
                        entry_price=entry_price,
                        partial_tp_price=partial_tp_price,
                        full_tp_price=full_tp_price,
                        total_position_size=position_size
                    )
                    trade_info = {
                        "timestamp": ts,
                        "signal": "BUY",
                        "entry_price": entry_price,
                        "stop_loss": stop_loss_price,
                        "pnl": realized_pnl,
                        "reason": "LOOSER Bullish OTE on 15m bar",
                        "position_size": position_size
                    }
                    all_trades.append(trade_info)

        # BEARISH
        elif (daily_bias == "Bearish") and (sh_price > sl_price):
            ote_lower, ote_upper = calculate_ote_zone_bearish(sh_price, sl_price)
            tol = 0.002
            if (close_price >= (ote_lower - tol)) and (close_price <= (ote_upper + tol)):
                entry_price = close_price
                stop_loss_price = place_stop_loss_short(sh_price, buffer_pct=buffer_pct)
                risk_amt = account_size * risk_per_trade
                risk_per_unit = stop_loss_price - entry_price
                if risk_per_unit > 0:
                    position_size = risk_amt / risk_per_unit
                    partial_tp_price = sl_price
                    full_tp_price    = sl_price - 0.002
                    current_price    = entry_price

                    rem_pos, realized_pnl, _ = manage_take_profit_short(
                        current_price=current_price,
                        entry_price=entry_price,
                        partial_tp_price=partial_tp_price,
                        full_tp_price=full_tp_price,
                        total_position_size=position_size
                    )
                    trade_info = {
                        "timestamp": ts,
                        "signal": "SELL",
                        "entry_price": entry_price,
                        "stop_loss": stop_loss_price,
                        "pnl": realized_pnl,
                        "reason": "LOOSER Bearish OTE on 15m bar",
                        "position_size": position_size
                    }
                    all_trades.append(trade_info)

    return all_trades

##########################################
# 3) The Day-by-Day Backtest
##########################################
def get_trading_days(start_date="2024-01-01", end_date="2024-12-30"):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_days = pd.date_range(start, end, freq='D')
    trading_days = [d for d in all_days if d.weekday() < 5]
    return trading_days

def fetch_data_for_2024(symbol="GBPJPY=X"):
    """
    We'll keep the same data approach: daily for entire year,
    1H for entire year, 15m for last 2 months of 2024.
    """
    daily_data = yf.download(
        symbol,
        start="2024-01-01",
        end="2024-12-31",
        interval="1d",
        progress=False
    )
    daily_data.dropna(inplace=True)
    daily_data.sort_index(inplace=True)

    intraday_1h = yf.download(
        symbol,
        start="2024-01-01",
        end="2024-12-31",
        interval="1h",
        progress=False
    )
    intraday_1h.dropna(inplace=True)
    intraday_1h.sort_index(inplace=True)
    if intraday_1h.index.tz is None:
        intraday_1h.index = intraday_1h.index.tz_localize('UTC')
    else:
        intraday_1h.index = intraday_1h.index.tz_convert('UTC')

    intraday_15m = yf.download(
        symbol,
        start="2024-11-01",
        end="2024-12-31",
        interval="15m",
        progress=False
    )
    intraday_15m.dropna(inplace=True)
    intraday_15m.sort_index(inplace=True)
    if intraday_15m.index.tz is None:
        intraday_15m.index = intraday_15m.index.tz_localize('UTC')
    else:
        intraday_15m.index = intraday_15m.index.tz_convert('UTC')

    return daily_data, intraday_1h, intraday_15m

def backtest_ict_strategy_2024(symbols):
    trading_days = get_trading_days("2024-01-01", "2024-12-30")
    all_trades = []

    for sym in symbols:
        daily_data, intraday_1h, intraday_15m = fetch_data_for_2024(sym)
        daily_data.index = daily_data.index.tz_localize(None)

        for current_day in trading_days:
            current_day_utc = pd.Timestamp(current_day, tz='UTC')
            daily_slice = daily_data.loc[:current_day].copy()
            if daily_slice.empty:
                continue
            
            start_intraday_utc = current_day_utc - pd.Timedelta(days=1)
            end_intraday_utc   = current_day_utc + pd.Timedelta(days=1)

            intraday_1h_slice = intraday_1h.loc[start_intraday_utc:end_intraday_utc].copy()
            intraday_15m_slice = intraday_15m.loc[start_intraday_utc:end_intraday_utc].copy()

            if intraday_1h_slice.empty or intraday_15m_slice.empty:
                continue
            
            # Use the LOOSER param values
            day_trades = run_ict_pipeline_day(
                symbol=sym,
                daily_df=daily_slice,
                intraday_df_1h=intraday_1h_slice,
                intraday_df_15m=intraday_15m_slice,
                kill_zone_hours=(1,12),   # 1am–12pm
                bias_lookback=7,         # was 15
                displacement_factor=0.5, # was 1.0
                buffer_pct=0.01,         # was 0.003
                account_size=10000,
                risk_per_trade=0.01
            )
            if day_trades:
                for td in day_trades:
                    new_trade = td.copy()
                    new_trade["date"] = current_day
                    new_trade["symbol"] = sym
                    all_trades.append(new_trade)

    return pd.DataFrame(all_trades)

##########################################
# 4) MAIN
##########################################
if __name__ == "__main__":
    my_symbols = [
        "GBPJPY=X",
        "EURUSD=X",
        "USDCAD=X",
        "USDJPY=X",
        "GBPUSD=X",
        "EURGBP=X"
    ]
    
    df_all_trades = backtest_ict_strategy_2024(my_symbols)

    print("\n=== BACKTEST RESULTS (Summary) ===")
    print(df_all_trades)
    
    if not df_all_trades.empty:
        total_pnl = df_all_trades["pnl"].sum()
        wins = (df_all_trades["pnl"] > 0).sum()
        losses = (df_all_trades["pnl"] < 0).sum()
        total_trades = len(df_all_trades)
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins}, Losses: {losses}")
        if (wins+losses) > 0:
            win_rate = 100.0 * wins / (wins+losses)
            print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")
    else:
        print("No trades triggered in this backtest window.")
