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
PLOT_FILE = 'strategy_results.png' # Retained for consistency, but not used in this run mode
SERVER_PORT = 8080 # Retained for consistency, but not used in this run mode
RESULTS_DIR = 'results'
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data

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
            # print(f"   Fetched up to: {dt.datetime.fromtimestamp(start_ts / 1000).strftime('%Y-%m-%d')}")
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
    if returns.empty:
        return 0.0
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

def run_strategy_for_optimization(df_data, center_distance):
    """
    Applies the dynamic sizing strategy for a specific center_distance and returns Sharpe Ratio.
    """
    df = df_data.copy()
    
    # Calculate 120 SMA
    df[f'SMA_{SMA_PERIOD_120}'] = df['Close'].rolling(window=SMA_PERIOD_120).mean()
    
    # Calculate daily returns (log returns)
    df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Prepare Lagged Data (Look-ahead Bias Prevention) ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA_120'] = df[f'SMA_{SMA_PERIOD_120}'].shift(1)
    
    # Drop NaNs from SMAs and shifting
    df = df.dropna()
    
    # --- Dynamic Position Sizing Calculation ---
    
    # 1. Calculate Distance (D): Absolute decimal distance from the SMA (lagged)
    df['Distance'] = np.abs((df['Yesterday_Close'] - df['Yesterday_SMA_120']) / df['Yesterday_SMA_120'])

    # 2. Calculate Multiplier (M) using the provided formula:
    # M = 1 / ( (1 / (D * Scaler)) + (D * Scaler) - 1 )
    # Scaler = 1 / center_distance
    distance_scaler = 1.0 / center_distance
    scaled_distance = df['Distance'] * distance_scaler
    
    # Use epsilon to prevent division by zero when price is exactly on the SMA (D=0)
    epsilon = 1e-6 
    denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
    
    # Calculate Multiplier, handle possible division by zero in the outer term (though unlikely with epsilon)
    df['Multiplier'] = np.where(denominator == 0, 0, 1.0 / denominator)

    # 3. Determine Direction (Long/Short)
    df['Direction'] = np.where(
        df['Yesterday_Close'] > df['Yesterday_SMA_120'],
        1,
        -1
    )

    # 4. Final Position Size = Direction * Multiplier
    df['Position_Size'] = df['Direction'] * df['Multiplier']
    
    # 5. Calculate Strategy Returns
    df['Strategy_Return'] = df['Daily_Return'] * df['Position_Size']
    df['Cumulative_Strategy_Return'] = np.exp(df['Strategy_Return'].cumsum())
    
    # Calculate metrics
    sharpe = calculate_sharpe_ratio(df['Strategy_Return'])
    total_return = calculate_total_return(df['Cumulative_Strategy_Return'])
    
    return sharpe, total_return

def run_grid_search(df):
    """
    Performs a grid search over the first 50% of the data to find the optimal center distance.
    """
    print("\n" + "=" * 60)
    print("STARTING GRID SEARCH OPTIMIZATION (IN-SAMPLE: FIRST 50% OF DATA)")
    print("Target Metric: Annualized Sharpe Ratio")
    print("Parameter Range: 1.0% to 8.0% in 0.1% steps")
    print("=" * 60)
    
    # 1. Slice data to first 50% (In-Sample period)
    half_index = len(df) // 2
    df_in_sample = df.iloc[:half_index].copy()
    
    # 2. Define grid search space
    start_center = 0.010
    end_center = 0.080
    step_center = 0.001
    
    # Create the range of center distances (0.010, 0.011, ..., 0.080)
    center_distances = [round(c, 3) for c in np.arange(start_center, end_center + step_center/2, step_center)]

    best_sharpe = -np.inf
    optimal_center = None
    
    results = []

    # 3. Run search
    for center_distance in center_distances:
        try:
            sharpe, total_return = run_strategy_for_optimization(df_in_sample, center_distance)
            
            results.append({
                'Center': f"{center_distance:.3f}",
                'Sharpe': sharpe,
                'Return': total_return
            })
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                optimal_center = center_distance
        
        except Exception as e:
            # Handle potential errors during backtest for a specific parameter
            print(f"Error for center={center_distance*100:.1f}%: {e}")
            continue

    if optimal_center is None:
        print("\nOptimization failed: Could not run backtest for any parameter set.")
        return None
        
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print(f"Optimal Center Distance (Sharpe): {optimal_center*100:.1f}%")
    print(f"Max Annualized Sharpe Ratio: {best_sharpe:.4f}")
    print("=" * 60)
    
    return optimal_center

# --- 4. Main Execution ---

if __name__ == '__main__':
    # 1. Fetch data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Run grid search
        optimal_center = run_grid_search(df_data)
        
        # Note: Plotting and serving are disabled in this optimization mode.
        # The optimal parameter is now available for further out-of-sample testing.
