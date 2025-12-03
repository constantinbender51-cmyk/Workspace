import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from itertools import product

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '2018-01-01'
CACHE_FILE = 'btc_data.csv'

# Grid Search Ranges
# range(start, stop, step) in Python stops BEFORE the stop value.
# Adjusting to ensure we cover the requested ranges inclusive.
SMA1_RANGE = range(10, 501, 10)  # Starting at 10 to avoid SMA=1 (which is just price)
SMA2_RANGE = range(10, 181, 10)
STOP_LOSS_RANGE = np.arange(0.01, 0.105, 0.005) # 1% to 10% step 0.5%
LEVERAGE_RANGE = np.arange(1.0, 5.5, 0.5)       # 1x to 5x step 0.5

# ==========================================
# 1. DATA FETCHING
# ==========================================
def fetch_binance_data(symbol, interval, start_str):
    """
    Fetches historical klines from Binance API with pagination.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start date to milliseconds timestamp
    dt_obj = datetime.strptime(start_str, '%Y-%m-%d')
    start_ts = int(dt_obj.timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} data from {start_str}...")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not isinstance(data, list):
                print("Error fetching data:", data)
                break
                
            if len(data) == 0:
                break
                
            all_data.extend(data)
            
            # Update start time for next batch (last close time + 1ms)
            current_start = data[-1][6] + 1
            
            # Rate limit respect
            time.sleep(0.1)
            
            # Progress indicator
            last_date = datetime.fromtimestamp(data[-1][0]/1000).strftime('%Y-%m-%d')
            print(f"Fetched up to {last_date}...", end='\r')
            
            # Break if we reached current time (approx)
            if data[-1][6] > time.time() * 1000:
                break
                
        except Exception as e:
            print(f"Connection error: {e}")
            break
            
    print(f"\nTotal records fetched: {len(all_data)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    # Type conversion
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
    
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

def get_data():
    if os.path.exists(CACHE_FILE):
        print(f"Loading data from {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE)
        df.to_csv(CACHE_FILE, index=False)
    return df

# ==========================================
# 2. VECTORIZED BACKTEST ENGINE
# ==========================================
def run_grid_search(df):
    print("Pre-calculating indicators and returns matrices...")
    
    # Convert columns to numpy arrays for speed
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    n_days = len(df)
    
    # -----------------------------------
    # A. Pre-calculate all SMAs
    # -----------------------------------
    # We collect all unique periods needed for both SMA1 and SMA2 to avoid duplicate work
    all_periods = sorted(list(set(list(SMA1_RANGE) + list(SMA2_RANGE))))
    smas = {}
    
    # Using pandas rolling for efficiency, then converting to numpy
    # (Doing this loop is fast enough for ~100 indicators)
    for p in all_periods:
        smas[p] = df['Open'].rolling(window=p).mean().values
        
    # -----------------------------------
    # B. Pre-calculate Base Returns (Unleveraged) for Long and Short
    #    based on Stop Loss % (s)
    # -----------------------------------
    # Logic:
    # If Long: 
    #   Stop Price = Open * (1 - s)
    #   If Low <= Stop Price: Return = -s
    #   Else: Return = (Close - Open) / Open
    #
    # If Short:
    #   Stop Price = Open * (1 + s)
    #   If High >= Stop Price: Return = -s
    #   Else: Return = (Open - Close) / Open
    
    # We create a dictionary where key=s, value=array of returns for that s
    long_returns_map = {}
    short_returns_map = {}
    
    # Standard daily return if no stop is hit
    std_long_ret = (closes - opens) / opens
    std_short_ret = (opens - closes) / opens
    
    for s in STOP_LOSS_RANGE:
        # Long Logic
        stop_price_long = opens * (1 - s)
        hit_long_sl = lows <= stop_price_long
        
        # Combine: where stop hit, use -s, else use standard return
        r_long = np.where(hit_long_sl, -s, std_long_ret)
        long_returns_map[s] = r_long
        
        # Short Logic
        stop_price_short = opens * (1 + s)
        hit_short_sl = highs >= stop_price_short
        
        # Combine
        r_short = np.where(hit_short_sl, -s, std_short_ret)
        short_returns_map[s] = r_short

    # -----------------------------------
    # C. Grid Search Loop
    # -----------------------------------
    best_perf = -np.inf
    best_params = {}
    
    # Total iterations estimate
    total_iter = len(SMA1_RANGE) * len(SMA2_RANGE)
    print(f"Starting Grid Search over {total_iter} SMA combinations...")
    print(f"Inner loops: {len(STOP_LOSS_RANGE)} Stop Losses x {len(LEVERAGE_RANGE)} Leverages")
    
    count = 0
    start_time = time.time()
    
    # Outer Loop: SMAs (Determines Signal)
    for sma1_p in SMA1_RANGE:
        sma1_arr = smas[sma1_p]
        
        for sma2_p in SMA2_RANGE:
            if sma1_p == sma2_p: continue # Redundant
            
            sma2_arr = smas[sma2_p]
            count += 1
            
            # Calculate Signals
            # 1 = Long, -1 = Short, 0 = Flat
            # Use NaN handling: if SMA is NaN, condition is False
            
            # Long: Open > SMA1 AND Open > SMA2
            # Note: fill_value=False handles NaN at start of array
            is_long = (opens > sma1_arr) & (opens > sma2_arr)
            
            # Short: Open < SMA1 AND Open < SMA2
            is_short = (opens < sma1_arr) & (opens < sma2_arr)
            
            # We assume we are FLAT otherwise.
            # Convert booleans to 0/1 integers for math
            mask_long = is_long.astype(int)
            mask_short = is_short.astype(int)
            
            # Optimization: If almost no trades, skip
            if np.sum(mask_long) + np.sum(mask_short) < 10:
                continue

            # Middle Loop: Stop Loss (Determines Base Return vector)
            for s_val in STOP_LOSS_RANGE:
                # Get pre-calculated returns vectors
                # Multiply by masks (0 or 1) to get daily strategy return (unleveraged)
                # If flat (both masks 0), return is 0.
                
                # Vector math: 
                # daily_base_ret = (mask_long * long_returns_map[s_val]) + (mask_short * short_returns_map[s_val])
                # However, allocating new array every time is slow. 
                # We can perform the leverage calc directly on the components.
                
                r_l = long_returns_map[s_val]
                r_s = short_returns_map[s_val]
                
                # Combined daily base return for this SMA + SL combo
                daily_combined = (mask_long * r_l) + (mask_short * r_s)
                
                # Inner Loop: Leverage (Scales Return)
                for lev in LEVERAGE_RANGE:
                    
                    # Apply leverage
                    # Note: If stop loss was hit (return is -s), leveraged return is -s * lev
                    final_daily_rets = daily_combined * lev
                    
                    # Calculate Metric: Total Compounded Return
                    # Use log returns for stability and speed: sum(ln(1+r))
                    # Clamp returns to -0.999 to avoid log error if loss >= 100%
                    # (Though with max stop 10% and max lev 5x, max loss is 50%, so safe)
                    
                    # 1 + r
                    equity_curve = 1 + final_daily_rets
                    
                    # Check for bust (equity <= 0)
                    if np.any(equity_curve <= 0):
                        total_ret = -1.0 # Bust
                    else:
                        # Geometric linking
                        total_ret = np.prod(equity_curve) - 1
                    
                    if total_ret > best_perf:
                        best_perf = total_ret
                        best_params = {
                            'SMA1': sma1_p,
                            'SMA2': sma2_p,
                            'StopLoss': round(s_val * 100, 2), # %
                            'Leverage': lev,
                            'TotalReturn': round(total_ret * 100, 2) # %
                        }
            
            if count % 100 == 0:
                print(f"Processed {count}/{total_iter} SMA pairs... Best so far: {best_params.get('TotalReturn')}%", end='\r')

    end_time = time.time()
    print(f"\n\nGrid Search Complete in {end_time - start_time:.2f} seconds.")
    return best_params

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Get Data
    df = get_data()
    print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
    print("------------------------------------------------")
    
    # 2. Run Optimization
    result = run_grid_search(df)
    
    # 3. Output Results
    print("------------------------------------------------")
    print("OPTIMIZATION RESULTS")
    print("------------------------------------------------")
    print(f"Best Total Return: {result['TotalReturn']}%")
    print(f"Parameters:")
    print(f"  SMA 1 Period:    {result['SMA1']}")
    print(f"  SMA 2 Period:    {result['SMA2']}")
    print(f"  Stop Loss:       {result['StopLoss']}%")
    print(f"  Leverage:        {result['Leverage']}x")
    print("------------------------------------------------")
    print("Note: Returns are compounded. Slippage and fees are not included.")
