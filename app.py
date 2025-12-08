import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01T00:00:00Z'

# Indicators
SMA_LONG_PERIOD = 365
SMA_FILTER_PERIOD = 90

# Strategy Parameters
# "Close is 7% from SMA": We assume this is the 'proximity' threshold. 
# Price must be within this % distance of SMA365 to be considered a 'retest'.
PROXIMITY_THRESHOLD = 0.07 

# "Horizontal" definition: If absolute daily slope of SMA90 is less than this, we stay flat.
# 0.0005 means if SMA changes less than 0.05% in a day, it's horizontal.
SLOPE_FLAT_THRESHOLD = 0.0005 

# -----------------------------------------------------------------------------
# DATA FETCHING
# -----------------------------------------------------------------------------
def fetch_historical_data(symbol, timeframe, start_str):
    print(f"Fetching {symbol} data starting from {start_str}...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'} # Using futures for long/short, or spot? Defaulting to spot data for chart.
    })
    
    # Switch to spot if preferred, usually longer history on spot for backtesting logic
    exchange = ccxt.binance() 

    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            since = ohlcv[-1][0] + 1  # Move to next timestamp
            all_ohlcv += ohlcv
            
            # Progress indicator
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"Fetched up to {current_date.date()}... ({len(all_ohlcv)} candles)")
            
            if current_date > datetime.now() - timedelta(days=1):
                break
                
            time.sleep(0.5) # Be nice to API limits
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Drop duplicates just in case
    df = df[~df.index.duplicated(keep='first')]
    return df

# -----------------------------------------------------------------------------
# STRATEGY LOGIC
# -----------------------------------------------------------------------------
def apply_strategy(df):
    print("\nCalculating indicators and signals...")
    
    # 1. Indicators
    df['SMA365'] = df['close'].rolling(window=SMA_LONG_PERIOD).mean()
    df['SMA90'] = df['close'].rolling(window=SMA_FILTER_PERIOD).mean()
    
    # 2. Distance Calculations
    # Positive if price > SMA, Negative if price < SMA
    df['dist_raw'] = df['close'] - df['SMA365']
    df['dist_pct'] = df['dist_raw'] / df['SMA365']
    
    # 3. Derivative of Distance (1-day rate)
    # If derivative > 0, price is moving AWAY from SMA (or up towards it if below) relative to yesterday
    df['dist_deriv'] = df['dist_pct'].diff()
    
    # 4. Slope of SMA 90
    # Normalized slope: (SMA_today - SMA_yesterday) / SMA_yesterday
    df['sma90_slope'] = df['SMA90'].pct_change()
    
    # 5. Logic Loop
    # We iterate because "Coming from > 7%" implies a state memory
    
    positions = [] # 1 for Long, -1 for Short, 0 for Flat
    
    # State variables
    position = 0
    was_extended_bullish = False # Tracks if we were recently > 7%
    was_extended_bearish = False # Tracks if we were recently < -7%
    
    # Convert to standard python lists/numpy for speed in loop
    close = df['close'].values
    dist_pct = df['dist_pct'].values
    dist_deriv = df['dist_deriv'].values
    sma90_slope = df['sma90_slope'].values
    index = df.index
    
    out_pos = np.zeros(len(df))
    
    for i in range(1, len(df)):
        # Skip until we have enough data for SMAs
        if np.isnan(close[i]) or np.isnan(dist_pct[i]) or np.isnan(sma90_slope[i]):
            out_pos[i] = 0
            continue
            
        # A. Update Extension State
        # If we are far away, mark as extended
        if dist_pct[i] > PROXIMITY_THRESHOLD: 
            was_extended_bullish = True
            was_extended_bearish = False # Reset opposite
        elif dist_pct[i] < -PROXIMITY_THRESHOLD:
            was_extended_bearish = True
            was_extended_bullish = False
            
        # Reset extension if we cross the SMA (trend change invalidates pullback logic)
        # If sign of distance changes, we crossed the SMA
        if (dist_pct[i] > 0 and dist_pct[i-1] < 0) or (dist_pct[i] < 0 and dist_pct[i-1] > 0):
            was_extended_bullish = False
            was_extended_bearish = False
            
        # B. Check for Retest Trigger
        
        # LONG SIGNAL CONDITIONS:
        # 1. Price is above SMA (Positive Distance)
        # 2. We came from > 7% (was_extended_bullish is True)
        # 3. We are currently within the retest zone (Distance < 7%)
        # 4. Derivative switched sign from Negative (getting closer) to Positive (moving away/bouncing)
        
        long_trigger = False
        short_trigger = False
        
        # Check Long Retest
        if (dist_pct[i] > 0 and 
            was_extended_bullish and 
            dist_pct[i] <= PROXIMITY_THRESHOLD):
            
            # Derivative switch: Yesterday < 0, Today > 0
            if dist_deriv[i-1] < 0 and dist_deriv[i] > 0:
                long_trigger = True
                
        # Check Short Retest
        if (dist_pct[i] < 0 and 
            was_extended_bearish and 
            dist_pct[i] >= -PROXIMITY_THRESHOLD):
            
            # Derivative switch: Yesterday > 0 (getting less negative/closer), Today < 0 (getting more negative/away)
            if dist_deriv[i-1] > 0 and dist_deriv[i] < 0:
                short_trigger = True

        # C. Apply SMA 90 Slope Filter
        is_flat_slope = abs(sma90_slope[i]) < SLOPE_FLAT_THRESHOLD
        
        # D. Determine Position for TOMORROW (Simulating entry on next day open/close hold)
        # The prompt says: "Hold the positions one day then re-enter the next."
        # This effectively means we re-evaluate daily.
        
        if long_trigger:
            if not is_flat_slope:
                position = 1
            else:
                position = 0 # Stay flat if slope is horizontal
        elif short_trigger:
            if not is_flat_slope:
                position = -1
            else:
                position = 0
        else:
            # If no NEW signal, do we hold previous?
            # Prompt: "Hold the positions one day then re-enter the next."
            # This implies the signal dictates the day's trade. 
            # If there is NO signal today, what do we do?
            # Standard "retest" strategies usually hold the trend until invalidated.
            # HOWEVER, the prompt implies a daily check. 
            # "If after retest slope... is horizontal stays flat until next signal."
            # Interpretation: Once a signal hits, we enter. We stay in that state.
            # BUT, we must check the slope daily.
            
            if position != 0 and is_flat_slope:
                position = 0 # Exit if slope goes flat
            
            # If slope is fine, we keep the previous position
            pass 

        out_pos[i] = position

    df['position'] = out_pos
    return df

# -----------------------------------------------------------------------------
# BACKTEST EXECUTION
# -----------------------------------------------------------------------------
def calculate_performance(df):
    # Strategy Returns: Position (yesterday) * Market Return (today)
    df['market_return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['market_return']
    
    # Cumulative Returns
    df['equity_curve'] = (1 + df['strategy_return']).cumprod()
    df['buy_hold_curve'] = (1 + df['market_return']).cumprod()
    
    # Stats
    total_return = df['equity_curve'].iloc[-1] - 1
    bh_return = df['buy_hold_curve'].iloc[-1] - 1
    
    print("\n---------------------------------------------------")
    print(f"Backtest Results ({START_DATE} to Now)")
    print("---------------------------------------------------")
    print(f"Total Strategy Return: {total_return * 100:.2f}%")
    print(f"Buy & Hold Return:     {bh_return * 100:.2f}%")
    print(f"Final Equity (Start 1.0): {df['equity_curve'].iloc[-1]:.4f}")
    
    # Trade Count (Approximate based on position changes)
    trades = df['position'].diff().abs().sum() / 2
    print(f"Approximate Trades:    {int(trades)}")
    
    return df

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Get Data
    df = fetch_historical_data(SYMBOL, TIMEFRAME, START_DATE)
    
    if len(df) < SMA_LONG_PERIOD:
        print("Not enough data fetched to calculate 365 SMA.")
    else:
        # 2. Apply Logic
        df = apply_strategy(df)
        
        # 3. Calculate Performance
        df = calculate_performance(df)
        
        # 4. Export for manual inspection
        filename = "strategy_results.csv"
        df.to_csv(filename)
        print(f"\nDetailed logs saved to {filename}")
