import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION (MATCHING MAIN (27).PY)
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 10000.0
FEE_PCT = 0.0006  # 0.06% est. slippage+fees

# TUMBLER_PARAMS from main.py
TUMBLER_PARAMS = {
    "SMA1": 32, 
    "SMA2": 114, 
    "STOP": 0.043, 
    "TAKE_PROFIT": 0.126, 
    "III_WIN": 27, 
    "FLAT_THRESH": 0.356, 
    "BAND": 0.077, 
    "LEVS": [0.079, 4.327, 3.868], 
    "III_TH": [0.058, 0.259]
}

# Normalization Constants from main.py
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0
NORMALIZATION_FACTOR = TARGET_STRAT_LEV / TUMBLER_MAX_LEV

# ==========================================
# DATA FETCHING
# ==========================================
def fetch_binance_data(symbol, start_date_str):
    filename = f"{symbol.lower()}_daily_{start_date_str}.csv"
    
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    print(f"Fetching data from Binance for {symbol} starting {start_date_str}...")
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    
    while start_ts < end_ts:
        params = {"symbol": symbol, "interval": "1d", "startTime": start_ts, "limit": 1000}
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data: break
            all_data.extend(data)
            start_ts = data[-1][0] + 86400000
            time.sleep(0.05)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'ig'])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open', 'high', 'low', 'close']: df[c] = df[c].astype(float)
    final_df = df[['timestamp', 'open', 'high', 'low', 'close']].set_index('timestamp')
    final_df.to_csv(filename)
    return final_df

# ==========================================
# INDICATOR CALCULATIONS
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    # SMAs
    df['sma1'] = df['close'].rolling(window=TUMBLER_PARAMS['SMA1']).mean()
    df['sma2'] = df['close'].rolling(window=TUMBLER_PARAMS['SMA2']).mean()
    
    # III Calculation
    # Note: main.py uses rolling sum of abs(log_ret)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    w = TUMBLER_PARAMS['III_WIN']
    
    # iii = path_efficiency basically
    # rolling_sum(abs(log_ret)) is path length? 
    # Wait, main.py code:
    # iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum())
    # Numerator: Absolute value of (Sum of log returns) -> Net Price Change
    # Denominator: Sum of (Absolute log returns) -> Total Path Length
    
    numerator = df['log_ret'].rolling(w).sum().abs()
    denominator = df['log_ret'].abs().rolling(w).sum()
    df['iii'] = (numerator / denominator).fillna(0)
    
    return df

# ==========================================
# BACKTEST LOGIC
# ==========================================
def run_backtest(df):
    print(f"Running Backtest with Normalization Factor: {NORMALIZATION_FACTOR:.4f} (Max Lev ~2.0x)")
    
    cash = INITIAL_CAPITAL
    equity_curve = []
    
    # State
    flat_regime = False
    
    # Pre-calc
    df = calculate_indicators(df)
    
    # Arrays for speed
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    sma1 = df['sma1'].values
    sma2 = df['sma2'].values
    iii_vals = df['iii'].values
    times = df.index
    
    start_idx = max(TUMBLER_PARAMS['SMA2'], TUMBLER_PARAMS['III_WIN']) + 1
    
    for i in range(start_idx, len(df)):
        # Data available at decision time (Yesterday's Close / Today's Open 00:00)
        # We use i-1 for indicators (decision made based on closed candle)
        # We execute at i (Open of current day)
        
        idx_prev = i - 1
        
        curr_price = opens[i] # Entry price
        day_high = highs[i]
        day_low = lows[i]
        day_close = closes[i] # Mark to market
        
        # Indicator Values
        val_sma1 = sma1[idx_prev]
        val_sma2 = sma2[idx_prev]
        val_iii = iii_vals[idx_prev]
        prev_close = closes[idx_prev]
        
        # --- LOGIC FROM MAIN.PY ---
        
        # 1. Determine Base Leverage
        raw_lev = TUMBLER_PARAMS["LEVS"][2] # High default
        if val_iii < TUMBLER_PARAMS["III_TH"][0]: 
            raw_lev = TUMBLER_PARAMS["LEVS"][0] # Low
        elif val_iii < TUMBLER_PARAMS["III_TH"][1]: 
            raw_lev = TUMBLER_PARAMS["LEVS"][1] # Mid
            
        # 2. Flat Regime Trigger
        if val_iii < TUMBLER_PARAMS["FLAT_THRESH"]:
            flat_regime = True
            
        # 3. Flat Regime Release Check
        if flat_regime:
            band1 = val_sma1 * TUMBLER_PARAMS["BAND"]
            band2 = val_sma2 * TUMBLER_PARAMS["BAND"]
            
            # Check deviation from SMAs
            # main.py: if abs(curr - sma1) <= band1 or ...
            # using prev_close as 'curr' for decision
            if abs(prev_close - val_sma1) <= band1 or abs(prev_close - val_sma2) <= band2:
                flat_regime = False
                
        # 4. Signal Generation
        signal = 0.0
        if not flat_regime:
            if prev_close > val_sma1 and prev_close > val_sma2:
                signal = 1.0 # Long
            elif prev_close < val_sma1 and prev_close < val_sma2:
                signal = -1.0 # Short
                
        # 5. Apply Leverage & Normalization
        # main.py: r_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        effective_leverage = (signal * raw_lev) * NORMALIZATION_FACTOR
        
        # --- SIMULATION ---
        
        # Position Size (USD)
        position_size = cash * effective_leverage
        
        # If flat, just append cash (assuming risk free 0)
        if abs(position_size) < 1.0:
            equity_curve.append({'date': times[i], 'equity': cash, 'lev': 0, 'iii': val_iii})
            continue
            
        # Fees (Entry)
        # Assuming we re-balance daily, we basically pay fees on the whole position every day
        # In reality, main.py only trades the delta.
        # But for conservative backtesting, let's assume simple daily re-entry or 
        # approximate holding. 
        # To be precise: We check if signal flipped. But here we simplify:
        # We pay fee on Entry, check SL/TP, pay fee on Exit.
        
        # Calculate PnL
        # Long: (Close - Open) / Open * Size
        # Short: (Open - Close) / Open * Size (Positive Size passed in usually, here signed)
        
        # Let's handle Long/Short separately for SL/TP logic
        
        entry_fee = abs(position_size) * FEE_PCT
        pnl = 0.0
        
        # Logic for SL/TP (Intraday)
        sl_pct = TUMBLER_PARAMS["STOP"]
        tp_pct = TUMBLER_PARAMS["TAKE_PROFIT"]
        
        exit_price = day_close
        hit_stop = False
        hit_tp = False
        
        if position_size > 0: # LONG
            sl_price = curr_price * (1 - sl_pct)
            tp_price = curr_price * (1 + tp_pct)
            
            if day_low <= sl_price:
                exit_price = sl_price
                hit_stop = True
            elif day_high >= tp_price:
                exit_price = tp_price
                hit_tp = True
                
        else: # SHORT
            sl_price = curr_price * (1 + sl_pct)
            tp_price = curr_price * (1 - tp_pct)
            
            if day_high >= sl_price:
                exit_price = sl_price
                hit_stop = True
            elif day_low <= tp_price:
                exit_price = tp_price
                hit_tp = True

        # Calculate PnL based on Exit Price
        # PnL = Size * (Exit - Entry) / Entry  (roughly, for linear instruments)
        # For Futures: PnL = Qty * (Exit - Entry)
        # Here position_size is in USD. 
        # PnL USD = Position_USD * (Exit/Entry - 1) * side
        
        pct_move = (exit_price - curr_price) / curr_price
        if position_size < 0: pct_move = -pct_move
        
        gross_pnl = abs(position_size) * pct_move
        exit_fee = abs(position_size) * FEE_PCT # Paid on exit notional (approx same as entry)
        
        net_pnl = gross_pnl - entry_fee - exit_fee
        cash += net_pnl
        
        equity_curve.append({
            'date': times[i], 
            'equity': cash, 
            'lev': effective_leverage,
            'iii': val_iii
        })

    return pd.DataFrame(equity_curve).set_index('date')

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    df = fetch_binance_data(SYMBOL, START_DATE)
    res = run_backtest(df)
    
    # Stats
    res['ret'] = res['equity'].pct_change()
    total_ret = (res['equity'].iloc[-1] / INITIAL_CAPITAL) - 1
    sharpe = np.sqrt(365) * (res['ret'].mean() / res['ret'].std())
    dd = (res['equity'] - res['equity'].cummax()) / res['equity'].cummax()
    max_dd = dd.min()
    
    print("\n" + "="*40)
    print("TUMBLER CONTRIBUTION (Main.py Logic)")
    print("="*40)
    print(f"Final Equity:   ${res['equity'].iloc[-1]:,.2f}")
    print(f"Total Return:   {total_ret*100:.2f}%")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print(f"Max Drawdown:   {max_dd*100:.2f}%")
    print(f"Target Max Lev: {TARGET_STRAT_LEV}x")
    print("="*40)
    
    plt.figure(figsize=(10,6))
    plt.plot(res.index, res['equity'], label='Tumbler Contribution (Norm. 2x)')
    plt.plot(res.index, (df.loc[res.index]['close']/df.loc[res.index]['close'].iloc[0])*INITIAL_CAPITAL, label='BTC Hold', alpha=0.3)
    plt.yscale('log')
    plt.title('Tumbler Strategy Contribution (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig("tumbler_contribution.png")
    print("Saved chart to tumbler_contribution.png")
