import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time

def fetch_binance_data(symbol='ETH/USDT', timeframe='4h', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    """
    Fetches historical OHLC data from Binance using CCXT with pagination.
    Fixes TypeError by ensuring all dates are UTC-aware.
    """
    print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    
    all_ohlcv = []
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Move to next timestamp
            
            # Simple progress indicator
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"Fetched up to {current_date}...", end='\r')
            
            if since >= end_ts:
                break
                
            time.sleep(exchange.rateLimit / 1000) 
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nTotal candles fetched: {len(all_ohlcv)}")
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # FIX: Add utc=True to make the column timezone-aware
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Filter using pd.Timestamp to ensure consistent timezone handling
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
    df = df.loc[mask].reset_index(drop=True)
    
    return df

def prepare_arrays(df):
    # 1. Close array
    close_array = df['close'].to_numpy()
    
    # 2. Percentage change array (price - last) / last
    # Note: numpy diff does (out[i] = a[i+1] - a[i]), we need change relative to previous
    pct_change = np.diff(close_array) / close_array[:-1]
    
    # 3. Insert 0 in beginning
    pct_change = np.insert(pct_change, 0, 0.0)
    
    # 4. Verify equal length
    assert len(close_array) == len(pct_change), "Arrays are not of equal length"
    
    # 5. Absolute price array (Custom Definition)
    # "Beginning with 1 all consecutive elements are the sum of all previous percentage changes"
    # This is effectively: 1 + Cumulative Sum of pct_changes
    abs_price_array = 1.0 + np.cumsum(pct_change)
    
    # 6. Verify absolute price length
    assert len(abs_price_array) == len(close_array), "Abs Price array length mismatch"
    
    return close_array, pct_change, abs_price_array

def run_analysis():
    # --- Step 1: Fetch Data ---
    # Note: 2026 is in the future relative to now (2025/2026 transition). 
    # Fetching up to 'now' if 2026 is not available.
    df = fetch_binance_data(end_date='2026-01-01T00:00:00Z')
    
    if df.empty:
        print("No data found.")
        return

    # --- Step 2: Prepare Arrays ---
    close_arr, pct_arr, abs_arr = prepare_arrays(df)
    
    total_len = len(abs_arr)
    
    # --- Step 3: Split 80/10/10 ---
    # We split based on indices
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    # Training Set (80 set)
    train_abs = abs_arr[:idx_80]
    
    # Validation Set (10 set 1)
    val_abs = abs_arr[idx_80:idx_90]
    
    # Test Set (10 set 2) - Ignored as per instructions
    # test_abs = abs_arr[idx_90:]

    print(f"\nData Split:")
    print(f"Training Set (80%): {len(train_abs)} items")
    print(f"Validation Set (10%): {len(val_abs)} items")

    # --- Step 4: Pattern Matching ---
    # "Take 10 set 1. Find all sequences of length 5."
    
    seq_length = 5
    correct_predictions = 0
    total_predictions = 0
    
    # We iterate through the validation set
    # We need at least seq_length elements to form a sequence, and +1 for the target
    print("\nRunning pattern matching (Nearest Neighbor)...")
    
    # Optimization: Pre-process training sequences into a sliding window view for speed
    # This creates a matrix where each row is a sequence of length 5 from the training set
    train_windows = np.lib.stride_tricks.sliding_window_view(train_abs[:-1], window_shape=seq_length)
    train_targets = train_abs[seq_length:] # The value immediately following each window
    
    # Iterate through Validation set
    for i in range(len(val_abs) - seq_length):
        
        # Current sequence (input) and actual continuation (target)
        current_seq = val_abs[i : i + seq_length]
        actual_continuation = val_abs[i + seq_length]
        
        # --- Find "Most Probable" inside 80 set ---
        # Since float matches are rare, we find the sequence with the MINIMUM EUCLIDEAN DISTANCE.
        # This represents the "closest historical pattern".
        
        # Calculate distances between current_seq and all training windows
        # Axis 1 means we sum the squares across the window columns
        distances = np.sum((train_windows - current_seq) ** 2, axis=1)
        
        # Find index of the best match (minimum distance)
        best_match_idx = np.argmin(distances)
        
        # The predicted continuation is the value that followed the best match in history
        # We need to adjust the prediction relative to the current level
        # (i.e. if the pattern matches shape but is at a different absolute level)
        
        # However, the prompt implies "Absolute Price", so we will take the raw value 
        # from the training set if we assume stationarity, BUT "Absolute Price" here is cumulative sum.
        # A better prediction is the *change* from the matched sequence applied to current.
        
        # Let's strictly follow: "return the most probable" (value).
        historical_seq = train_windows[best_match_idx]
        historical_next = train_targets[best_match_idx]
        
        # Logic: Did the history go UP or DOWN from the end of the sequence?
        hist_diff = historical_next - historical_seq[-1]
        
        # Apply that logic to current sequence
        predicted_continuation = current_seq[-1] + hist_diff
        
        # --- Check Accuracy ---
        # "How many times ... equals actual continuation"
        # Strict float equality is impossible. We check Directional Accuracy.
        # (Did we correctly predict if price goes up or down?)
        
        actual_diff = actual_continuation - current_seq[-1]
        
        # Check if directions match (both > 0 or both < 0)
        # Or if the error is extremely small (e.g. < 0.0001)
        
        is_same_direction = (hist_diff > 0 and actual_diff > 0) or \
                            (hist_diff < 0 and actual_diff < 0) or \
                            (hist_diff == 0 and actual_diff == 0)
                            
        if is_same_direction:
            correct_predictions += 1
            
        total_predictions += 1
        
        if total_predictions % 100 == 0:
            print(f"Processed {total_predictions} sequences...", end='\r')

    print(f"\n\n--- Results ---")
    print(f"Total Sequences Analyzed: {total_predictions}")
    print(f"Directional Accuracy (Correct Trend Prediction): {correct_predictions}")
    print(f"Accuracy Percentage: {(correct_predictions / total_predictions) * 100:.2f}%")
    
    # Note on "Equality":
    print("\n*Note: Strict numerical equality (matches to 8 decimal places) is statistically impossible")
    print("in floating point financial data. The accuracy above represents Directional Accuracy")
    print("(predicted Up/Down matches actual Up/Down).")

if __name__ == "__main__":
    run_analysis()
