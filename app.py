import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

# --- Configuration ---
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 1000.0
FEES_PCT = 0.0006  # 0.06% est. taker fee per trade (entry + exit)

# Strategy Parameters
SMA_PERIOD_1 = 57
SMA_PERIOD_2 = 124
BAND_WIDTH = 0.05
STATIC_STOP_PCT = 0.02
TAKE_PROFIT_PCT = 0.16
III_WINDOW = 35

# III Leverage Thresholds
III_T_LOW = 0.13
III_T_HIGH = 0.18
LEV_LOW = 0.5
LEV_MID = 4.5
LEV_HIGH = 2.45

def fetch_binance_data(symbol, start_date):
    """Fetches daily OHLC data from Binance public API."""
    print(f"Fetching data for {symbol} starting {start_date}...")
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while True:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ts,
            "limit": limit
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        start_ts = data[-1][0] + 86400000  # Move to next day
        
        # Stop if we reached current time
        if len(data) < limit:
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    float_cols = ["open", "high", "low", "close", "volume"]
    df[float_cols] = df[float_cols].astype(float)
    
    return df[["date", "open", "high", "low", "close"]].set_index("date")

def calculate_indicators(df):
    """Calculates SMAs and III."""
    df = df.copy()
    
    # SMAs
    df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    
    # Bands
    df['upper_band'] = df['sma_1'] * (1 + BAND_WIDTH)
    df['lower_band'] = df['sma_1'] * (1 - BAND_WIDTH)
    
    # III Calculation (Rolling 35 days)
    # III = |Sum(LogReturns)| / Sum(|LogReturns|)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    def calc_rolling_iii(x):
        abs_net_dir = abs(x.sum())
        path_len = x.abs().sum()
        if path_len == 0: return 0
        return abs_net_dir / path_len

    # Use rolling apply for III (Note: this can be slow on huge datasets, but fine for daily)
    df['iii'] = df['log_ret'].rolling(window=III_WINDOW).apply(calc_rolling_iii, raw=True)
    
    return df

def get_leverage(iii):
    if pd.isna(iii): return 1.0
    if iii < III_T_LOW: return LEV_LOW
    elif iii < III_T_HIGH: return LEV_MID
    else: return LEV_HIGH

def run_backtest(df):
    """Executes the state machine and trade logic."""
    print("Running backtest strategy logic...")
    
    equity_curve = [INITIAL_CAPITAL]
    capital = INITIAL_CAPITAL
    position = None  # {type: 'LONG'/'SHORT', entry_price: float, size: float, stop: float, tp: float}
    
    # State Machine Variables
    cross_flag = 0 
    
    history = []
    
    # Iterate row by row to replicate state machine accurately
    # Start loop after enough data for indicators
    start_idx = max(SMA_PERIOD_2, III_WINDOW) + 1
    
    for i in range(start_idx, len(df)):
        curr_date = df.index[i]
        curr_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- 1. UPDATE STATE MACHINE (Based on previous close vs SMA) ---
        # Note: Strategy Logic generates signal at T based on T-1 Close
        
        sma_1 = prev_row['sma_1']
        sma_2 = prev_row['sma_2']
        prev_close = df.iloc[i-2]['close'] # Close of T-2
        last_close = prev_row['close']     # Close of T-1
        
        # Detect Crosses (T-2 to T-1)
        if prev_close < sma_1 and last_close > sma_1:
            cross_flag = 1
        elif prev_close > sma_1 and last_close < sma_1:
            cross_flag = -1
            
        # Reset Flag if exited bands
        upper_band = prev_row['upper_band']
        lower_band = prev_row['lower_band']
        
        if last_close > upper_band or last_close < lower_band:
            cross_flag = 0
            
        # --- 2. GENERATE SIGNAL ---
        signal = "FLAT"
        
        # Long Logic
        if last_close > upper_band: signal = "LONG"
        elif last_close > sma_1 and cross_flag == 1: signal = "LONG"
        
        # Short Logic
        elif last_close < lower_band: signal = "SHORT"
        elif last_close < sma_1 and cross_flag == -1: signal = "SHORT"
        
        # Filter Logic (SMA 2)
        if signal == "LONG" and last_close < sma_2: signal = "FLAT"
        if signal == "SHORT" and last_close > sma_2: signal = "FLAT"
        
        # --- 3. EXECUTE TRADES ---
        # Today's price action
        today_open = curr_row['open']
        today_high = curr_row['high']
        today_low = curr_row['low']
        today_close = curr_row['close']
        
        # Determine Leverage for TODAY based on YESTERDAY's III
        lev = get_leverage(prev_row['iii'])
        
        # Check Existing Position (SL/TP or Signal Flip)
        if position:
            # Check SL/TP first (assuming they exist in the market)
            # Assumption: SL hit first if Low < SL, unless Open gap causes issues (simplified here)
            
            pnl_pct = 0
            exit_price = None
            exit_reason = None
            
            # LONG EXIT CHECKS
            if position['type'] == 'LONG':
                if today_low <= position['stop']:
                    exit_price = position['stop']
                    exit_reason = "STOP_LOSS"
                elif today_high >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = "TAKE_PROFIT"
                elif signal != "LONG": # Signal Flip or Flat
                    exit_price = today_open
                    exit_reason = "SIGNAL_CHANGE"
                
                if exit_price:
                    # Calculate PnL
                    # (Exit - Entry) / Entry * Leverage
                    raw_ret = (exit_price - position['entry_price']) / position['entry_price']
                    pnl_pct = raw_ret * position['leverage']
            
            # SHORT EXIT CHECKS
            elif position['type'] == 'SHORT':
                if today_high >= position['stop']:
                    exit_price = position['stop']
                    exit_reason = "STOP_LOSS"
                elif today_low <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = "TAKE_PROFIT"
                elif signal != "SHORT":
                    exit_price = today_open
                    exit_reason = "SIGNAL_CHANGE"
                    
                if exit_price:
                    # (Entry - Exit) / Entry * Leverage
                    raw_ret = (position['entry_price'] - exit_price) / position['entry_price']
                    pnl_pct = raw_ret * position['leverage']

            # Process Exit
            if exit_price:
                # Apply Fees
                fee_impact = FEES_PCT * position['leverage'] 
                net_pnl = pnl_pct - fee_impact
                
                capital = capital * (1 + net_pnl)
                position = None # Flat
                
                # If we exited due to signal change, we might re-enter immediately below
                # But for simplicity, we trade on Open, so Signal Change exit happens at Open
                # If signal is reversed, we can enter new position same bar? 
                # Yes, if we exited at Open, we can enter at Open.
        
        # Entry Logic (If flat)
        if position is None and signal != "FLAT":
            entry_price = today_open
            
            # Setup Stops/TP
            if signal == "LONG":
                sl_price = entry_price * (1 - STATIC_STOP_PCT)
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            else:
                sl_price = entry_price * (1 + STATIC_STOP_PCT)
                tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
            
            # Check if Candle immediately hits SL/TP (Intraday volatility)
            # Simplified: Assume entry successful, check SL/TP next loop OR same loop?
            # We must check same loop for validity.
            
            # Immediate Stop Check (bad luck entry)
            hit_sl = False
            if signal == "LONG" and today_low < sl_price: hit_sl = True
            if signal == "SHORT" and today_high > sl_price: hit_sl = True
            
            if hit_sl:
                # Immediate loss
                pnl_pct = -STATIC_STOP_PCT * lev
                fee_impact = FEES_PCT * lev
                capital = capital * (1 + pnl_pct - fee_impact)
            else:
                # Position Established
                position = {
                    'type': signal,
                    'entry_price': entry_price,
                    'stop': sl_price,
                    'tp': tp_price,
                    'leverage': lev,
                    'entry_date': curr_date
                }
                # Apply Entry Fee immediately? usually applied on PnL calc, but let's deduct from 'virtual' equity if we tracked realized.
                # We will deduct entry fee upon exit calculation for simplicity, or effectively reduce size.
                # Actually, standard is: Capital is locked.
                
        equity_curve.append(capital)
        history.append({'date': curr_date, 'equity': capital, 'signal': signal, 'leverage': lev if position else 0})

    return pd.DataFrame(history).set_index('date')

def calculate_metrics(df):
    """Calculates Sortino, Sharpe, and Quarterly Returns."""
    df['returns'] = df['equity'].pct_change()
    
    # Annualization factor (Crypto trades 365 days)
    ann_factor = 365
    
    total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
    cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0]) ** (365 / len(df)) - 1
    
    daily_std = df['returns'].std()
    downside_std = df[df['returns'] < 0]['returns'].std()
    
    sharpe = (df['returns'].mean() / daily_std) * np.sqrt(ann_factor) if daily_std > 0 else 0
    sortino = (df['returns'].mean() / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0
    
    # Quarterly Capital
    quarterly = df['equity'].resample('QE').last()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Final Capital': df['equity'].iloc[-1],
        'Quarterly': quarterly
    }

def project_growth(equity_df, days_forward=365):
    """Projects future equity using log-linear regression."""
    # Fit y = a * e^(bx)  <->  ln(y) = ln(a) + bx
    
    y = np.log(equity_df['equity'].values)
    x = np.arange(len(y))
    
    # Linear fit on log equity
    coeffs = np.polyfit(x, y, 1) # [slope, intercept]
    
    # Generate future dates
    last_date = equity_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_forward + 1)]
    future_x = np.arange(len(x), len(x) + days_forward)
    
    # Project
    predicted_log_y = coeffs[0] * future_x + coeffs[1]
    predicted_equity = np.exp(predicted_log_y)
    
    return pd.Series(predicted_equity, index=future_dates)

# --- Execution ---

# 1. Fetch
df = fetch_binance_data(SYMBOL, START_DATE)

# 2. Indicators
df = calculate_indicators(df)
df.dropna(inplace=True)

# 3. Backtest
res_df = run_backtest(df)

# 4. Metrics
metrics = calculate_metrics(res_df)

# 5. Projection
projection = project_growth(res_df)

# --- Output & Visualization ---
print("-" * 40)
print(f"BACKTEST RESULTS: {SYMBOL} ({START_DATE} to Now)")
print("-" * 40)
print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
print(f"Final Capital:   ${metrics['Final Capital']:.2f}")
print(f"Total Return:    {metrics['Total Return']*100:.2f}%")
print(f"CAGR:            {metrics['CAGR']*100:.2f}%")
print(f"Sharpe Ratio:    {metrics['Sharpe Ratio']:.2f}")
print(f"Sortino Ratio:   {metrics['Sortino Ratio']:.2f}")
print("-" * 40)
print("Capital Start of Each Quarter (Last 8):")
print(metrics['Quarterly'].tail(8))
print("-" * 40)

# Plotting
plt.figure(figsize=(12, 6))

# Historical Equity
plt.plot(res_df.index, res_df['equity'], label='Historical Equity', color='blue')

# Projection
plt.plot(projection.index, projection.values, label='1-Year Projection', color='green', linestyle='--')

plt.title(f"Tumbler Strategy Backtest & Projection\nSMA({SMA_PERIOD_1}/{SMA_PERIOD_2}) | Dynamic Leverage")
plt.xlabel("Date")
plt.ylabel("Capital (USD) - Log Scale")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()
