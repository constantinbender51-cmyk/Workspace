import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import datetime as dt
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from matplotlib.ticker import ScalarFormatter

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '1 Jan, 2018'
SMA_PERIOD_120 = 120 
PLOT_FILE = 'strategy_results.png' # Retained for consistency
SERVER_PORT = 8080 # Retained for consistency
RESULTS_DIR = 'results'
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data

# --- Grid Search Parameters ---
GRID_M_START = 7
GRID_M_END = 60
GRID_M_STEP = 1

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
    df['Close'] = pd.to_numeric(df['Close'])
    df = df.set_index('Open Time')
    df = df[['Open', 'Close']].astype(float)
    
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

def calculate_total_return(cumulative_returns):
    """Calculates the Total Return in percentage."""
    if cumulative_returns.empty:
        return 0.0
    return (cumulative_returns.iloc[-1] - 1) * 100

# --- 3. Backtesting Logic for Grid Search ---

def run_strategy_for_optimization(df, avg_dist_period):
    """
    Applies the 120 SMA trading strategy with dynamic center distance based on M-day average distance.
    Returns Sharpe Ratio and Total Return.
    """
    df_data = df.copy()
    
    # 1. Calculate 120 SMA & Daily Returns
    df_data[f'SMA_{SMA_PERIOD_120}'] = df_data['Close'].rolling(window=SMA_PERIOD_120).mean()
    df_data['Daily_Return'] = np.log(df_data['Close'] / df_data['Close'].shift(1))

    # --- Calculation for Dynamic Center (Requires current data) ---
    # Calculate the absolute daily distance from SMA 120 (decimal format)
    df_data['Raw_Distance'] = np.abs((df_data['Close'] - df_data[f'SMA_{SMA_PERIOD_120}']) / df_data[f'SMA_{SMA_PERIOD_120}'])
    
    # Calculate the M-day rolling average of the distance
    df_data['Rolling_Avg_Distance'] = df_data['Raw_Distance'].rolling(window=avg_dist_period).mean()
    
    # --- Look-ahead Prevention for all components ---
    df_data['Yesterday_Close'] = df_data['Close'].shift(1)
    df_data['Yesterday_SMA_120'] = df_data[f'SMA_{SMA_PERIOD_120}'].shift(1)
    
    # Lag the rolling average distance to be used as the center for today's trade
    df_data['Center_Distance'] = df_data['Rolling_Avg_Distance'].shift(1)

    # Drop NaNs after all lagging/rolling calculations
    df_clean = df_data.dropna().copy()
    
    if df_clean.empty:
        return -np.inf, 0.0
    
    # ----------------------------------------------------
    # Strategy: Dynamic Position Sizing
    # ----------------------------------------------------
    
    # 1. Determine Position Direction based on SMA crossover (Lagged)
    df_clean['Bullish'] = df_clean['Yesterday_Close'] > df_clean['Yesterday_SMA_120']
    df_clean['Direction'] = np.where(df_clean['Bullish'], 1, -1)
    
    # 2. Calculate Distance (D): Absolute decimal distance from the SMA (lagged)
    df_clean['Distance'] = np.abs((df_clean['Yesterday_Close'] - df_clean['Yesterday_SMA_120']) / df_clean['Yesterday_SMA_120'])

    # 3. Calculate Multiplier (M) using the dynamically calculated Center
    # Scaler = 1 / Center_Distance 
    distance_scaler = 1.0 / np.maximum(df_clean['Center_Distance'], 1e-10) # Protect against zero division
    
    scaled_distance = df_clean['Distance'] * distance_scaler
    
    epsilon = 1e-6 
    # M_magnitude = 1 / ( (1 / (D * Scaler)) + (D * Scaler) - 1 )
    denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
    
    # Calculate Multiplier (Position Size Magnitude)
    df_clean['Multiplier'] = np.where(denominator == 0, 0, 1.0 / denominator)

    # 4. Final Position Size = Direction * Multiplier
    df_clean['Position_Size'] = df_clean['Direction'] * df_clean['Multiplier']
    
    # 5. Calculate Strategy Returns
    df_clean['Strategy_Return'] = df_clean['Daily_Return'] * df_clean['Position_Size']
    df_clean['Cumulative_Strategy_Return'] = np.exp(df_clean['Strategy_Return'].cumsum())
    
    # Calculate metrics
    sharpe = calculate_sharpe_ratio(df_clean['Strategy_Return'])
    total_return = calculate_total_return(df_clean['Cumulative_Strategy_Return'])
    
    return sharpe, total_return

def run_grid_search(df):
    """
    Performs a grid search over the first 50% of the data to find the optimal averaging window.
    """
    print("\n" + "=" * 60)
    print("STARTING GRID SEARCH OPTIMIZATION (IN-SAMPLE: FIRST 50% OF DATA)")
    print("Target Metric: Annualized Sharpe Ratio")
    print(f"Window (M) Range: {GRID_M_START} to {GRID_M_END} in {GRID_M_STEP} day steps")
    print("=" * 60)
    
    # 1. Slice data to first 50% (In-Sample period)
    half_index = len(df) // 2
    df_in_sample = df.iloc[:half_index].copy()
    
    # 2. Define grid search space (7 to 60 in 1 steps)
    avg_dist_periods = list(range(GRID_M_START, GRID_M_END + 1, GRID_M_STEP))

    best_sharpe = -np.inf
    optimal_M = GRID_M_START
    
    # 3. Run search
    for avg_dist_period in avg_dist_periods:
        try:
            sharpe, total_return = run_strategy_for_optimization(df_in_sample, avg_dist_period)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                optimal_M = avg_dist_period
        
        except Exception as e:
            # Silently handle errors for periods that might be too short for the 120 SMA
            continue 

    if best_sharpe == -np.inf:
        print("\nOptimization failed: No valid Sharpe ratio could be calculated.")
        return None
        
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print(f"Optimal Averaging Window M (Sharpe Max): {optimal_M} days")
    print(f"Max Annualized Sharpe Ratio (In-Sample): {best_sharpe:.4f}")
    print("=" * 60)
    
    return optimal_M

# --- Main Execution ---

# Note: Plotting and web-serving functions are intentionally excluded from the main execution
# block in optimization mode.

if __name__ == '__main__':
    # 1. Fetch data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Run grid search
        optimal_M = run_grid_search(df_data)
        
        if optimal_M is not None:
             print(f"\nOptimization successful. Optimal Averaging Window found: {optimal_M} days.")
        else:
             print("\nOptimization failed to find a valid optimal Averaging Window.")
