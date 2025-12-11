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
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data

# Fixed Strategy Parameters
STATIC_CENTER_DISTANCE = 0.010 # Fixed 1.0% center distance
LEVERAGE_FACTOR = 5.0        # 5x Leverage

# --- 2D Grid Search Parameters ---
# Stop Loss Percentage (S)
GRID_S_START = 0.00 
GRID_S_END = 0.10
GRID_S_STEP = 0.005

# Re-entry Proximity Percentage (P)
GRID_P_START = 0.01 
GRID_P_END = 0.10
GRID_P_STEP = 0.005

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

def run_strategy_for_optimization(df_data, sl_percent, reentry_prox):
    """
    Applies the strategy for a given SL and Re-entry Proximity and returns Sharpe Ratio.
    """
    df = df_data.copy()
    center = STATIC_CENTER_DISTANCE
    leverage = LEVERAGE_FACTOR
    
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
    sl_cooldown = False # State variable: True if SL was hit yesterday
    
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
        epsilon = 1e-6 
        denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
        multiplier = np.where(denominator == 0, 0, 1.0 / denominator)
        position_size_base = direction * multiplier

        # --- 2. Apply Re-entry/Cooldown Filter (State Management) ---
        
        if sl_cooldown:
            # If in cooldown, check if the price has returned to the re-entry proximity
            if proximity <= reentry_prox:
                sl_cooldown = False # Exit cooldown, resume trading
                
        if sl_cooldown:
            # If still in cooldown, force position to zero and skip SL check
            position_size_base = 0.0
            daily_return = 0.0
            
        else:
            # --- 3. Stop Loss Logic (Only runs if not in Cooldown) ---
            
            current_low = df.loc[index, 'Low']
            current_high = df.loc[index, 'High']
            raw_return = df.loc[index, 'Daily_Return_Raw']
            
            # Stop Price (S% loss relative to entry price)
            stop_price = np.where(
                direction == 1,
                entry_price * (1 - sl_percent),
                entry_price * (1 + sl_percent)
            )
            
            sl_triggered = False
            
            # Check for Stop Loss Trigger 
            if sl_percent > 0.0 and (
                (direction == 1 and current_low <= stop_price) or 
                (direction == -1 and current_high >= stop_price)
            ):
                # SL hit: calculate return based on SL price (log return)
                sl_return = np.log(stop_price / entry_price)
                daily_return = sl_return
                sl_cooldown = True # Enter cooldown state

            else:
                # SL not hit: use full close-to-close return
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
    Performs a 2D grid search over the Stop-Loss (S) and Re-entry Proximity (P) parameters.
    """
    print("\n" + "=" * 80)
    print("STARTING 2D GRID SEARCH OPTIMIZATION (IN-SAMPLE: FIRST 50% OF DATA)")
    print("Target Metric: Annualized Sharpe Ratio")
    print(f"Stop Loss (S) Range: {GRID_S_START*100:.1f}% to {GRID_S_END*100:.1f}% in {GRID_S_STEP*100:.1f}% steps")
    print(f"Re-entry Proximity (P) Range: {GRID_P_START*100:.1f}% to {GRID_P_END*100:.1f}% in {GRID_P_STEP*100:.1f}% steps")
    print("=" * 80)
    
    # 1. Slice data to first 50% (In-Sample period)
    half_index = len(df) // 2
    df_in_sample = df.iloc[:half_index]
    
    # 2. Define grid search spaces
    s_factors = [round(s, 3) for s in np.arange(GRID_S_START, GRID_S_END + GRID_S_STEP/2, GRID_S_STEP)]
    p_factors = [round(p, 3) for p in np.arange(GRID_P_START, GRID_P_END + GRID_P_STEP/2, GRID_P_STEP)]

    best_sharpe = -np.inf
    optimal_s = 0.0
    optimal_p = 0.0
    
    # 3. Run 2D search
    total_runs = len(s_factors) * len(p_factors)
    current_run = 0

    for sl_percent in s_factors:
        for reentry_prox in p_factors:
            current_run += 1
            
            # Print progress every 100 runs
            if current_run % 100 == 0:
                 print(f"   Running optimization... Progress: {current_run}/{total_runs}")

            try:
                sharpe = run_strategy_for_optimization(df_in_sample, sl_percent, reentry_prox)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    optimal_s = sl_percent
                    optimal_p = reentry_prox
            
            except Exception:
                # Silently skip errors (e.g., if insufficient data length for SMA calculation in the slice)
                continue

    if best_sharpe == -np.inf:
        print("\nOptimization failed: No valid Sharpe ratio could be calculated.")
        return None
        
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print(f"Optimal Stop Loss (S): {optimal_s*100:.1f}%")
    print(f"Optimal Re-entry Proximity (P): {optimal_p*100:.1f}%")
    print(f"Max Annualized Sharpe Ratio (In-Sample): {best_sharpe:.4f}")
    print("=" * 80)
    
    return optimal_s, optimal_p

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Fetch data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Run 2D grid search
        optimal_params = run_grid_search(df_data)
        
        if optimal_params is not None:
             optimal_s, optimal_p = optimal_params
             print(f"\nOptimization successful. Optimal parameters found: SL={optimal_s*100:.1f}%, Re-entry={optimal_p*100:.1f}%.")
        else:
             print("\nOptimization failed to find a valid optimal parameter set.")
