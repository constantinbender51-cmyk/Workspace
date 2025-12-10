#!/usr/bin/env python3
"""
Binance OHLCV Analysis Script
Fetches data from 2018, computes returns, returns of returns, SMAs, and resistance signals.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01'
SMA_WINDOWS = list(range(10, 401, 10))  # 10, 20, ..., 400
LOOKBACK_WINDOW = 400  # days
FUTURE_DAYS = 10  # for returns of returns weighting
TRAILING_STOP = 0.02  # 2%
PROXIMITY_SCALING = 1/0.05  # 1/0.05 = 20

# Initialize exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})

def fetch_ohlcv_data(symbol, timeframe, since):
    """Fetch OHLCV data from Binance"""
    print(f"Fetching {symbol} data from {since}...")
    
    all_data = []
    since_timestamp = exchange.parse8601(since + 'T00:00:00Z')
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_timestamp)
            if not ohlcv:
                break
            
            since_timestamp = ohlcv[-1][0] + 1
            all_data.extend(ohlcv)
            
            # Print progress
            latest_date = exchange.iso8601(ohlcv[-1][0])
            print(f"Fetched up to {latest_date}")
            
            # Break if we've reached current date
            if len(ohlcv) < 1000:  # Binance returns max 1000 candles per request
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Total data points: {len(df)}")
    return df

def calculate_returns(df):
    """Calculate returns and returns of returns"""
    # Returns: (close - open) / open
    df['returns'] = (df['close'] - df['open']) / df['open']
    
    # Cumulative returns (sum of returns up to day x)
    df['cumulative_returns'] = df['returns'].cumsum()
    
    # Returns of returns: rate of change of returns
    # Using: (return_t - return_{t-1}) / abs(return_{t-1}) to handle zero returns
    df['returns_of_returns'] = df['returns'].diff() / df['returns'].shift(1).abs()
    df['returns_of_returns'].replace([np.inf, -np.inf], 0, inplace=True)
    df['returns_of_returns'].fillna(0, inplace=True)
    
    return df

def calculate_smas(df):
    """Calculate Simple Moving Averages of returns for different windows"""
    for window in SMA_WINDOWS:
        # Calculate SMA (average of returns over window)
        df[f'sma_{window}'] = df['returns'].rolling(window=window, min_periods=window).mean()
        # Calculate cumulative SMA (sum of SMA values up to day x)
        df[f'cumulative_sma_{window}'] = df[f'sma_{window}'].cumsum()
    return df

def calculate_proximity(current_value, reference_value):
    """
    Calculate proximity as 1/(distance * (1/0.05))
    where distance = abs(current_value - reference_value) / abs(reference_value)
    """
    if reference_value == 0:
        return 0
    
    distance = abs(current_value - reference_value) / abs(reference_value)
    if distance == 0:
        return np.inf  # Perfect match
    
    proximity = 1 / (distance * PROXIMITY_SCALING)
    return proximity

def calculate_position_proximity(current_value, reference_value):
    """
    Calculate proximity for position management (distance * 1/0.05)
    Used when we want farness to increase signal
    """
    if reference_value == 0:
        return 0
    
    distance = abs(current_value - reference_value) / abs(reference_value)
    return distance * PROXIMITY_SCALING

def calculate_sma_significance(df, day_idx):
    """
    Calculate significance of each SMA at day_idx
    Significance = (sum of weighted future returns_of_returns) * proximity
    """
    significance_dict = {}
    
    # Get future returns_of_returns with decaying weights
    future_ror_sum = 0
    for i in range(1, FUTURE_DAYS + 1):
        if day_idx + i < len(df):
            weight = 1 / i  # decaying weight: 1, 1/2, 1/3, ..., 1/10
            future_ror_sum += df['returns_of_returns'].iloc[day_idx + i] * weight
    
    # Calculate significance for each SMA window
    for window in SMA_WINDOWS:
        cum_sma_col = f'cumulative_sma_{window}'
        if day_idx >= window - 1:  # Ensure we have enough data for SMA
            cum_sma_value = df[cum_sma_col].iloc[day_idx]
            cum_returns = df['cumulative_returns'].iloc[day_idx]
            
            # Calculate proximity of cumulative returns to cumulative SMA
            proximity = calculate_proximity(cum_returns, cum_sma_value)
            
            # Significance = weighted future ror sum * proximity
            significance = future_ror_sum * proximity
            significance_dict[window] = significance
        else:
            significance_dict[window] = 0
    
    return significance_dict

def calculate_resistance(df, n):
    """
    Calculate resistance at day n (last day of lookback window)
    Resistance = sum(proximities to past cumulative returns * returns_of_returns)
               + sum(proximities to SMAs * SMA significances)
    """
    # We now ensure n >= LOOKBACK_WINDOW before calling this function
    # No need to check since we start at LOOKBACK_WINDOW
    
    resistance = 0
    cum_returns_n = df['cumulative_returns'].iloc[n]
    
    # Part 1: Sum over lookback days
    part1_sum = 0
    for x in range(n - LOOKBACK_WINDOW + 1, n + 1):
        cum_returns_x = df['cumulative_returns'].iloc[x]
        ror_x = df['returns_of_returns'].iloc[x]
        
        proximity = calculate_proximity(cum_returns_n, cum_returns_x)
        part1_sum += proximity * ror_x
    
    # Part 2: Sum over SMAs
    part2_sum = 0
    sma_significances = {}
    
    for window in SMA_WINDOWS:
        cum_sma_col = f'cumulative_sma_{window}'
        if n >= window - 1:
            cum_sma_value = df[cum_sma_col].iloc[n]
            
            # Calculate significance at day n
            significance_dict = calculate_sma_significance(df, n)
            significance = significance_dict.get(window, 0)
            
            # Calculate proximity to cumulative SMA
            proximity = calculate_proximity(cum_returns_n, cum_sma_value)
            
            part2_sum += proximity * significance
            sma_significances[window] = significance
    
    resistance = part1_sum + part2_sum
    
    return resistance, sma_significances

def run_analysis():
    """Main analysis function"""
    # Step 1: Fetch data
    df = fetch_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE)
    
    if len(df) < LOOKBACK_WINDOW:
        print(f"Error: Need at least {LOOKBACK_WINDOW} days of data, got {len(df)}")
        return
    
    # Step 2: Calculate returns
    df = calculate_returns(df)
    
    # Step 3: Calculate SMAs
    df = calculate_smas(df)
    
    # Step 4: Initialize tracking variables
    entry_flag = False
    entry_day = None
    entry_cum_returns = None
    position = 0  # 0: no position, 1: long, -1: short
    trailing_stop = None
    results = []
    
    # Threshold placeholder - to be determined empirically
    RESISTANCE_THRESHOLD = 0  # You'll need to set this based on analysis
    
    print("\nStarting analysis...")
    
    # Step 5: Iterate through days with enough lookback
    # Start at day 401 (index 400) to have full 400-day lookback window
    for n in range(LOOKBACK_WINDOW, len(df)):
        current_date = df.index[n]
        
        # Calculate resistance
        resistance, sma_significances = calculate_resistance(df, n)
        
        # Check for entry signal
        if not entry_flag and resistance > RESISTANCE_THRESHOLD:
            entry_flag = True
            entry_day = n
            entry_cum_returns = df['cumulative_returns'].iloc[n]
            
            # Determine long/short based on distance sign
            # Using position proximity calculation (distance * scaling)
            distance = (df['cumulative_returns'].iloc[n] - df['cumulative_returns'].iloc[n-1]) \
                      / abs(df['cumulative_returns'].iloc[n-1]) if df['cumulative_returns'].iloc[n-1] != 0 else 0
            
            if distance > 0:
                position = 1  # long
            else:
                position = -1  # short
            
            trailing_stop = df['close'].iloc[n] * (1 - TRAILING_STOP) if position == 1 else \
                           df['close'].iloc[n] * (1 + TRAILING_STOP)
            
            print(f"\nEntry signal on {current_date.date()}:")
            print(f"  Resistance: {resistance:.6f}")
            print(f"  Position: {'Long' if position == 1 else 'Short'}")
            print(f"  Entry cumulative returns: {entry_cum_returns:.6f}")
            print(f"  Entry price: {df['close'].iloc[n]:.2f}")
            print(f"  Initial trailing stop: {trailing_stop:.2f}")
        
        # Position management if in a trade
        if entry_flag and position != 0:
            current_price = df['close'].iloc[n]
            current_cum_returns = df['cumulative_returns'].iloc[n]
            
            # Calculate proximity for position management
            position_proximity = calculate_position_proximity(current_cum_returns, entry_cum_returns)
            
            # Update trailing stop for long position
            if position == 1:
                new_stop = current_price * (1 - TRAILING_STOP)
                trailing_stop = max(trailing_stop, new_stop)
                
                # Check stop loss
                if current_price <= trailing_stop:
                    print(f"\nStop loss hit on {current_date.date()}:")
                    print(f"  Exit price: {current_price:.2f}")
                    print(f"  Trailing stop: {trailing_stop:.2f}")
                    print(f"  Position proximity at exit: {position_proximity:.6f}")
                    entry_flag = False
                    position = 0
                    entry_day = None
                    entry_cum_returns = None
                    trailing_stop = None
            
            # Update trailing stop for short position
            elif position == -1:
                new_stop = current_price * (1 + TRAILING_STOP)
                trailing_stop = min(trailing_stop, new_stop)
                
                # Check stop loss
                if current_price >= trailing_stop:
                    print(f"\nStop loss hit on {current_date.date()}:")
                    print(f"  Exit price: {current_price:.2f}")
                    print(f"  Trailing stop: {trailing_stop:.2f}")
                    print(f"  Position proximity at exit: {position_proximity:.6f}")
                    entry_flag = False
                    position = 0
                    entry_day = None
                    entry_cum_returns = None
                    trailing_stop = None
            
            # Record daily status
            results.append({
                'date': current_date,
                'resistance': resistance,
                'position': position,
                'position_proximity': position_proximity,
                'price': current_price,
                'cumulative_returns': current_cum_returns,
                'trailing_stop': trailing_stop
            })
        else:
            # Record resistance even when not in position
            results.append({
                'date': current_date,
                'resistance': resistance,
                'position': 0,
                'position_proximity': 0,
                'price': df['close'].iloc[n],
                'cumulative_returns': df['cumulative_returns'].iloc[n],
                'trailing_stop': None
            })
        
        # Progress indicator
        if (n - LOOKBACK_WINDOW + 1) % 100 == 0:
            print(f"Processed up to {current_date.date()}")
    
    # Step 6: Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    # Step 7: Summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total days analyzed: {len(results_df)} (starting from day 401)")
    print(f"Days with positions: {len(results_df[results_df['position'] != 0])}")
    print(f"Maximum resistance: {results_df['resistance'].max():.6f}")
    print(f"Minimum resistance: {results_df['resistance'].min():.6f}")
    print(f"Average resistance: {results_df['resistance'].mean():.6f}")
    print(f"Standard deviation of resistance: {results_df['resistance'].std():.6f}")
    
    # Save results
    results_df.to_csv('analysis_results.csv')
    df.to_csv('ohlcv_with_indicators.csv')
    
    print("\nResults saved to:")
    print("  - analysis_results.csv (trading signals and positions)")
    print("  - ohlcv_with_indicators.csv (raw data with all indicators)")
    
    return df, results_df

if __name__ == "__main__":
    print("Binance OHLCV Analysis Script")
    print("="*60)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Start Date: {START_DATE}")
    print(f"Lookback Window: {LOOKBACK_WINDOW} days")
    print(f"SMA Windows: {SMA_WINDOWS[:5]}...{SMA_WINDOWS[-5:]}")
    print(f"Future Days for ROR weights: {FUTURE_DAYS}")
    print(f"Trailing Stop: {TRAILING_STOP*100}%")
    print("="*60 + "\n")
    
    try:
        df, results_df = run_analysis()
        print("\nAnalysis complete!")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()