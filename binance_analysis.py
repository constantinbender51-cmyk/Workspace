import pandas as pd
import numpy as np
import requests
import time
import datetime as dt
import os

# --- Configuration (Non-Optimized Parameters) ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '1 Jan, 2018'
SMA_PERIOD_120 = 120 
ANNUALIZATION_FACTOR = 365 

# Fixed Strategy Parameters
STATIC_CENTER_DISTANCE = 0.010 # Fixed 1.0% center distance
LEVERAGE_FACTOR = 5.0        # 5x Leverage

# Optimized Risk Parameters (Fixed during this search)
STOP_LOSS_PERCENT = 0.065     # 6.5% Stop Loss
REENTRY_PROXIMITY_PERCENT = 0.070 # 7.0% Re-entry Proximity

# --- Grid Search Parameters (Exponent) ---
GRID_E_START = 0.10
GRID_E_END = 1.00
GRID_E_STEP = 0.01

# --- 1. Data Fetching Utilities ---

def date_to_milliseconds(date_str):
    """Convert date string to UTC timestamp in milliseconds"""
    return int(dt.datetime.strptime(date_str, '%d %b, %Y').timestamp() * 1000)

def fetch_klines(symbol, interval, start_str):
    """Fetches historical klines data from Binance."""
    print(f"-> Fetching {symbol} {interval} data starting from {start_str}...")
    start_ts = date_to_milliseconds(start_str)
    base_url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    
    while True:
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'limit': 1000
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status() 
            klines = response.json()
            
            if not klines:
                break

            all_data.extend(klines)
            start_ts = klines[-1][0] + 1
            time.sleep(0.5) 
            
            if len(klines) < 1000:
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
        
    print(f"-> Data fetch complete. Total candles: {len(all_data)}")
    
    df = pd.DataFrame(all_data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])

    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
    df = df.set_index('Open Time')
    df = df[['Open', 'High', 'Low', 'Close']]
    
    return df.dropna()

# --- 2. Metric Calculations ---

def calculate_sharpe_ratio(returns, annualization_factor=ANNUALIZATION_FACTOR, risk_free_rate=0):
    """Calculates the Annualized Sharpe Ratio."""
    if returns.empty or len(returns) <= 1:
        return -np.inf 
    excess_return = returns - risk_free_rate
    mean_excess_return = excess_return.mean()
    std_dev = excess_return.std()
    
    if std_dev == 0:
        return 0.0

    sharpe = (mean_excess_return * annualization_factor) / (std_dev * np.sqrt(annualization_factor))
    return sharpe

# --- 3. Optimization Logic ---

def run_strategy_for_optimization(df_data, exponent):
    """
    Applies the strategy for a given exponent and returns Sharpe Ratio.
    """
    df = df_data.copy()
    center = STATIC_CENTER_DISTANCE
    leverage = LEVERAGE_FACTOR
    sl_percent = STOP_LOSS_PERCENT
    reentry_prox = REENTRY_PROXIMITY_PERCENT
    
    # 1. Calculate SMA 120 and Raw Daily Returns
    df[f'SMA_{SMA_PERIOD_120}'] = df['Close'].rolling(window=SMA_PERIOD_120).mean()
    df['Daily_Return_Raw'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Look-ahead Prevention & Core Indicators ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA_120'] = df[f'SMA_{SMA_PERIOD_120}'].shift(1)
    df['Proximity_to_SMA'] = np.abs((df['Yesterday_Close'] - df['Yesterday_SMA_120']) / df['Yesterday_SMA_120'])
    
    df = df.dropna().copy()
    
    if df.empty:
        return -np.inf

    # Initialize Series for returns
    strategy_returns = pd.Series(index=df.index, dtype=float)
    sl_cooldown = False # State variable
    
    # Iterate through the DataFrame for day-by-day logic (required for state management)
    for i in range(len(df)):
        index = df.index[i]
        
        entry_price = df.loc[index, 'Yesterday_Close']
        yesterday_sma = df.loc[index, 'Yesterday_SMA_120']
        proximity = df.loc[index, 'Proximity_to_SMA']
        
        # --- 1. Determine Base Position and Direction ---
        direction = np.where(entry_price > yesterday_sma, 1, -1)
        distance_d = np.abs((entry_price - yesterday_sma) / yesterday_sma)
        
        # Multiplier (M) calculation
        distance_scaler = 1.0 / center
        scaled_distance = distance_d * distance_scaler
        epsilon = 1e-10 
        
        # Denominator: (1/Scaled_Dist) + Scaled_Dist - 1
        denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
        
        # Multiplier = 1 / (Denominator)^exponent
        multiplier = np.where(denominator <= 0, 0, 1.0 / (denominator ** exponent))
        
        position_size_base = direction * multiplier

        # --- 2. Apply Re-entry/Cooldown Filter (State Management) ---
        
        if sl_cooldown:
            if proximity <= reentry_prox:
                sl_cooldown = False 
                
        if sl_cooldown:
            position_size_base = 0.0
            daily_return = 0.0
            
        else:
            # --- 3. Stop Loss Logic ---
            
            current_low = df.loc[index, 'Low']
            current_high = df.loc[index, 'High']
            raw_return = df.loc[index, 'Daily_Return_Raw']
            
            stop_price = np.where(
                direction == 1,
                entry_price * (1 - sl_percent),
                entry_price * (1 + sl_percent)
            )
            
            if sl_percent > 0.0 and (
                (direction == 1 and current_low <= stop_price) or 
                (direction == -1 and current_high >= stop_price)
            ):
                sl_return = np.log(stop_price / entry_price)
                daily_return = sl_return
                sl_cooldown = True

            else:
                daily_return = raw_return

            # --- 4. Final Strategy Return ---
            strategy_return = daily_return * position_size_base * leverage
            strategy_returns[index] = strategy_return
        
        # Store the daily return
        strategy_returns[index] = daily_return * position_size_base * leverage


    # Final Sharpe Calculation
    return calculate_sharpe_ratio(strategy_returns.dropna())

def run_grid_search(df):
    """
    Performs a 1D grid search over the exponent (E).
    """
    print("\n" + "=" * 80)
    print("STARTING 1D GRID SEARCH OPTIMIZATION (IN-SAMPLE: FIRST 50% OF DATA)")
    print("Target Metric: Annualized Sharpe Ratio")
    print(f"Exponent (E) Range: {GRID_E_START:.2f} to {GRID_E_END:.2f} in {GRID_E_STEP:.2f} steps")
    print("Fixed Risk: SL={STOP_LOSS_PERCENT*100:.1f}%, Re-entry={REENTRY_PROXIMITY_PERCENT*100:.1f}%")
    print("=" * 80)
    
    # 1. Slice data to first 50% (In-Sample period)
    half_index = len(df) // 2
    df_in_sample = df.iloc[:half_index]
    
    # 2. Define grid search space
    exponents = [round(e, 2) for e in np.arange(GRID_E_START, GRID_E_END + GRID_E_STEP/2, GRID_E_STEP)]

    best_sharpe = -np.inf
    optimal_e = GRID_E_START
    
    # 3. Run 1D search
    total_runs = len(exponents)
    current_run = 0

    for exponent in exponents:
        current_run += 1
            
        if current_run % 10 == 0:
             print(f"   Running optimization... Progress: {current_run}/{total_runs}")

        try:
            sharpe = run_strategy_for_optimization(df_in_sample, exponent)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                optimal_e = exponent
        
        except Exception:
            continue

    if best_sharpe == -np.inf:
        print("\nOptimization failed: No valid Sharpe ratio could be calculated.")
        return None
        
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print(f"Optimal Exponent (Sharpe Max): {optimal_e:.2f}")
    print(f"Max Annualized Sharpe Ratio (In-Sample): {best_sharpe:.4f}")
    print("=" * 80)
    
    return optimal_e

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Fetch data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Run 1D grid search
        optimal_e = run_grid_search(df_data)
        
        if optimal_e is not None:
             print(f"\nOptimization successful. Optimal Exponent found: {optimal_e:.2f}.")
        else:
             print("\nOptimization failed to find a valid optimal exponent.")
