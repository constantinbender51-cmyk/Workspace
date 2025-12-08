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
PROXIMITY_THRESHOLD = 0.07  # 7% distance from SMA365

# -----------------------------------------------------------------------------
# DATA FETCHING
# -----------------------------------------------------------------------------
def fetch_historical_data(symbol, timeframe, start_str):
    print(f"Fetching {symbol} data starting from {start_str}...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            since = ohlcv[-1][0] + 1
            all_ohlcv += ohlcv
            
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            if current_date > datetime.now() - timedelta(days=1):
                break
            
            time.sleep(exchange.rateLimit / 1000) 
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

# -----------------------------------------------------------------------------
# STRATEGY LOGIC (STRICT LOOKBEACK)
# -----------------------------------------------------------------------------
def apply_strategy(df):
    print("Calculating strict 1-day lag signals...")
    
    # 1. Indicators
    df['SMA365'] = df['close'].rolling(window=SMA_LONG_PERIOD).mean()
    df['SMA90'] = df['close'].rolling(window=SMA_FILTER_PERIOD).mean()
    
    # Distance: (Price - SMA) / SMA
    df['dist_pct'] = (df['close'] - df['SMA365']) / df['SMA365']
    
    # Derivative: Change in distance from previous day
    df['dist_deriv'] = df['dist_pct'].diff()
    
    # SMA 90 Slope: Daily change in SMA 90 value
    df['sma90_change'] = df['SMA90'].diff()
    
    # 2. Logic Loop
    # We use numpy arrays for faster iteration
    # Indices: i = Today (Execution), i-1 = Yesterday (Signal)
    
    close = df['close'].values
    dist_pct = df['dist_pct'].values
    dist_deriv = df['dist_deriv'].values
    sma90_change = df['sma90_change'].values
    
    # Output array for ACTUAL positions held Today
    position = np.zeros(len(df))
    
    # State Memory (Persists across days)
    current_holding = 0 # 0: Flat, 1: Long, -1: Short
    was_extended_bullish = False 
    was_extended_bearish = False 
    
    # Start loop after we have enough data (Long SMA + 2 days for diffs)
    start_idx = SMA_LONG_PERIOD + 2
    
    for i in range(start_idx, len(df)):
        # -------------------------------------------------------
        # STEP 1: DEFINE TIME INDICES
        # -------------------------------------------------------
        # We are at day 'i' (Today). We must decide what to do TODAY.
        # We can ONLY look at 'i-1' (Yesterday) and 'i-2' (Day Before).
        
        idx_yest = i - 1
        idx_prev = i - 2
        
        # -------------------------------------------------------
        # STEP 2: UPDATE INTERNAL STATE (Based on Yesterday's Close)
        # -------------------------------------------------------
        
        # Update Extension Memory
        # Did yesterday's close put us in "Extended" territory?
        if dist_pct[idx_yest] > PROXIMITY_THRESHOLD:
            was_extended_bullish = True
            was_extended_bearish = False
        elif dist_pct[idx_yest] < -PROXIMITY_THRESHOLD:
            was_extended_bearish = True
            was_extended_bullish = False
            
        # Reset Extension Memory if Yesterday crossed the SMA
        # (Sign change between DayBefore and Yesterday)
        if (dist_pct[idx_yest] > 0 and dist_pct[idx_prev] < 0) or \
           (dist_pct[idx_yest] < 0 and dist_pct[idx_prev] > 0):
            was_extended_bullish = False
            was_extended_bearish = False

        # -------------------------------------------------------
        # STEP 3: CHECK "KILL SWITCH" (SMA 90 Slope Flip)
        # -------------------------------------------------------
        # Definition: Slope sign switched between DayBefore and Yesterday.
        slope_yest = sma90_change[idx_yest]
        slope_prev = sma90_change[idx_prev]
        
        # Check strict sign flip
        slope_flipped = False
        if (slope_yest > 0 and slope_prev < 0) or \
           (slope_yest < 0 and slope_prev > 0) or \
           (slope_yest == 0):
            slope_flipped = True
            
        # -------------------------------------------------------
        # STEP 4: CHECK ENTRY SIGNALS (Retests)
        # -------------------------------------------------------
        entry_signal = 0 # 0: None, 1: Long, -1: Short
        
        # Long Entry Logic (Analyzed on Yesterday's data)
        # 1. Price > SMA
        # 2. Was Extended > 7%
        # 3. Yesterday's Distance <= 7%
        # 4. Derivative switched: DayBefore < 0 (closing in), Yesterday > 0 (bouncing away)
        if (dist_pct[idx_yest] > 0 and 
            was_extended_bullish and 
            dist_pct[idx_yest] <= PROXIMITY_THRESHOLD):
            
            if dist_deriv[idx_prev] < 0 and dist_deriv[idx_yest] > 0:
                entry_signal = 1
                
        # Short Entry Logic
        if (dist_pct[idx_yest] < 0 and 
            was_extended_bearish and 
            dist_pct[idx_yest] >= -PROXIMITY_THRESHOLD):
            
            if dist_deriv[idx_prev] > 0 and dist_deriv[idx_yest] < 0:
                entry_signal = -1

        # -------------------------------------------------------
        # STEP 5: DETERMINE POSITION FOR TODAY (i)
        # -------------------------------------------------------
        
        # Priority 1: If Slope Flipped Yesterday -> FLATTEN Today
        if slope_flipped:
            current_holding = 0
            
        # Priority 2: If Valid Entry Signal Yesterday -> ENTER Today
        # (Note: An entry signal overrides a slope flip if they happen same day, 
        # or implies a new trend starting. Usually, entries happen far from horizontal 90SMA,
        # but if conflict, Entry takes precedence as per "next signal" instruction).
        if entry_signal != 0:
            current_holding = entry_signal
            
        # Priority 3: Otherwise, HOLD previous position (Daily compounding logic)
        # The prompt says: "Hold the positions one day then re-enter the next."
        # This means we maintain the state 'current_holding' into Today.
        
        position[i] = current_holding

    df['position'] = position
    return df

# -----------------------------------------------------------------------------
# BACKTEST EXECUTION
# -----------------------------------------------------------------------------
def calculate_performance(df):
    # Calculate Strategy Returns
    # Position[i] determines exposure to Market[i] return
    # No shift needed here because position[i] was already calculated using i-1 data
    
    df['market_return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'] * df['market_return']
    
    df.dropna(subset=['strategy_return'], inplace=True)
    
    df['equity_curve'] = (1 + df['strategy_return']).cumprod()
    df['buy_hold_curve'] = (1 + df['market_return']).cumprod()
    
    total_ret = df['equity_curve'].iloc[-1] - 1
    bh_ret = df['buy_hold_curve'].iloc[-1] - 1
    
    print("\n---------------------------------------------------")
    print(f"Backtest Results ({START_DATE} - Now)")
    print("---------------------------------------------------")
    print(f"Total Strategy Return: {total_ret * 100:.2f}%")
    print(f"Buy & Hold Return:     {bh_ret * 100:.2f}%")
    print(f"Final Equity:          {df['equity_curve'].iloc[-1]:.4f}")
    
    trades = df['position'].diff().abs().sum() / 2
    print(f"Approximate Trades:    {int(trades)}")
    
    return df

if __name__ == "__main__":
    df = fetch_historical_data(SYMBOL, TIMEFRAME, START_DATE)
    if len(df) > SMA_LONG_PERIOD + 10:
        df = apply_strategy(df)
        df = calculate_performance(df)
        df.to_csv("btc_strict_lag_results.csv")
        print("\nResults saved to btc_strict_lag_results.csv")
    else:
        print("Insufficient data.")
