import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import datetime as dt
import os

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '1 Jan, 2018'
SMA_PERIOD_120 = 120 
MOMENTUM_PERIOD = 60 # Lookback period for calculating the dynamic center distance (60 days)
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data

# --- Grid Search Parameters ---
GRID_K_START = 0.00
GRID_K_END = 0.50
GRID_K_STEP = 0.001

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

def run_strategy_for_optimization(df_data, k_factor):
    """
    Applies the dynamic sizing strategy for a specific k_factor and returns Sharpe Ratio/Total Return.
    """
    df = df_data.copy()
    
    # 1. Calculate 120 SMA & Daily Returns
    df[f'SMA_{SMA_PERIOD_120}'] = df['Close'].rolling(window=SMA_PERIOD_120).mean()
    df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Look-ahead Prevention ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA_120'] = df[f'SMA_{SMA_PERIOD_120}'].shift(1)
    
    # Calculate 60-day momentum
    df['Momentum_60d'] = (df['Close'] / df['Close'].shift(MOMENTUM_PERIOD)) - 1
    
    # Dynamic Center = k * |Momentum_60d| (Lagged)
    df['Center_Distance'] = k_factor * np.abs(df['Momentum_60d'].shift(1))

    # Drop NaNs after all lagging/rolling calculations
    df = df.dropna()
    
    if df.empty:
        return -np.inf, 0.0
    
    # 2. Calculate Distance (D): Absolute decimal distance from the SMA (lagged)
    df['Distance'] = np.abs((df['Yesterday_Close'] - df['Yesterday_SMA_120']) / df['Yesterday_SMA_120'])

    # 3. Calculate Multiplier (M)
    # Scaler = 1 / Center_Distance 
    distance_scaler = 1.0 / np.maximum(df['Center_Distance'], 1e-10) # Use max to prevent div by zero
    
    scaled_distance = df['Distance'] * distance_scaler
    
    epsilon = 1e-6 
    # M = 1 / ( (1 / (D * Scaler)) + (D * Scaler) - 1 )
    denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
    
    # Calculate Multiplier (Position Size Magnitude)
    df['Multiplier'] = np.where(denominator == 0, 0, 1.0 / denominator)

    # 4. Determine Direction (Long/Short)
    df['Direction'] = np.where(
        df['Yesterday_Close'] > df['Yesterday_SMA_120'],
        1,
        -1
    )

    # 5. Final Position Size = Direction * Multiplier
    df['Position_Size'] = df['Direction'] * df['Multiplier']
    
    # 6. Calculate Strategy Returns & Cumulative Returns
    df['Strategy_Return'] = df['Daily_Return'] * df['Position_Size']
    df['Cumulative_Strategy_Return'] = np.exp(df['Strategy_Return'].cumsum())
    
    # Calculate metrics
    sharpe = calculate_sharpe_ratio(df['Strategy_Return'])
    total_return = calculate_total_return(df['Cumulative_Strategy_Return'])
    
    return sharpe, total_return

def run_grid_search(df):
    """
    Performs a grid search over the first 50% of the data to find the optimal k factor.
    """
    print("\n" + "=" * 60)
    print("STARTING GRID SEARCH OPTIMIZATION (IN-SAMPLE: FIRST 50% OF DATA)")
    print("Target Metric: Annualized Sharpe Ratio")
    print(f"Parameter Range (k): {GRID_K_START:.3f} to {GRID_K_END:.3f} in {GRID_K_STEP:.3f} steps")
    print("=" * 60)
    
    # 1. Slice data to first 50% (In-Sample period)
    half_index = len(df) // 2
    df_in_sample = df.iloc[:half_index].copy()
    
    # 2. Define grid search space (0.00 to 0.50 in 0.001 steps)
    k_factors = [round(k, 3) for k in np.arange(GRID_K_START, GRID_K_END + GRID_K_STEP/2, GRID_K_STEP)]

    best_sharpe = -np.inf
    optimal_k = 0.0
    
    # 3. Run search
    for k_factor in k_factors:
        try:
            sharpe, total_return = run_strategy_for_optimization(df_in_sample, k_factor)
            
            # Print intermediate results sparingly
            # if k_factor % 0.05 == 0:
            #     print(f"   k={k_factor:.3f} -> Sharpe: {sharpe:.4f}")
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                optimal_k = k_factor
        
        except Exception as e:
            print(f"Error for k={k_factor:.3f}: {e}")
            continue

    if best_sharpe == -np.inf:
        print("\nOptimization failed: No valid Sharpe ratio could be calculated.")
        return None
        
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print(f"Optimal Factor k (Sharpe Max): {optimal_k:.3f}")
    print(f"Max Annualized Sharpe Ratio (In-Sample): {best_sharpe:.4f}")
    print("=" * 60)
    
    return optimal_k

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Fetch data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Run grid search
        optimal_k = run_grid_search(df_data)
        
        if optimal_k is not None:
             print(f"\nOptimization successful. Optimal k factor found: {optimal_k:.3f}")
        else:
             print("\nOptimization failed to find a valid optimal k factor.")
