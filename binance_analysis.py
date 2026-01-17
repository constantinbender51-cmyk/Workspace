import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import Counter

def fetch_binance_data(symbol='ETH/USDT', timeframe='4h', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    """
    Fetches historical OHLC data from Binance using CCXT.
    """
    print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")
    exchange = ccxt.binance()
    # Parse dates to timestamp (ms)
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    
    all_ohlcv = []
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            # Print progress
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"Fetched up to {current_date}...", end='\r')
            
            if since >= end_ts:
                break
            
            # Rate limit
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nTotal candles fetched: {len(all_ohlcv)}")
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # FIX: Ensure UTC awareness to avoid "tz-naive vs tz-aware" TypeError
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Filter strictly
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
    df = df.loc[mask].reset_index(drop=True)
    
    return df

def prepare_arrays(df):
    # 1. Close array
    close_array = df['close'].to_numpy()
    
    # 2. Percentage change array
    # Insert 0 in beginning implicitly by setting first element to 0
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    
    # 3. Absolute price array
    # "Beginning with 1 all consecutive elements are the sum of all previous percentage changes"
    # This is 1.0 + Cumulative Sum
    abs_price_array = 1.0 + np.cumsum(pct_change)
    
    # 4. Round absolute price array to lowest 0.5 step
    # "Lowest" implies floor. E.g., 1.9 -> 1.5, 1.4 -> 1.0
    abs_price_rounded = np.floor(abs_price_array * 2) / 2.0
    
    return close_array, pct_change, abs_price_rounded

def run_analysis():
    # --- Step 1: Fetch ---
    df = fetch_binance_data()
    if df.empty:
        return

    # --- Step 2: Prepare & Round ---
    close_arr, pct_arr, abs_arr = prepare_arrays(df)
    
    print("\nData Sample (First 10 Absolute Prices):")
    print(abs_arr[:10])

    # --- Step 3: Split 80/10/10 ---
    total_len = len(abs_arr)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_set = abs_arr[:idx_80]      # 80 set
    val_set = abs_arr[idx_80:idx_90]  # 10 set 1
    # test_set = abs_arr[idx_90:]     # 10 set 2 (Ignored)
    
    print(f"\nSplit Sizes -> Train: {len(train_set)}, Val: {len(val_set)}")

    # --- Step 4: Pattern Matching (Exact Match) ---
    seq_length = 5
    
    # A. Build Probability Map from Training Set (80 set)
    # Dictionary: { sequence_tuple : [list_of_next_values] }
    print("Building pattern model from Training Set...")
    patterns = {}
    
    # Sliding window over Training Set
    for i in range(len(train_set) - seq_length):
        # The sequence (key)
        seq = tuple(train_set[i : i + seq_length])
        # The value immediately following (target)
        target = train_set[i + seq_length]
        
        if seq not in patterns:
            patterns[seq] = []
        patterns[seq].append(target)
        
    print(f"Unique patterns found in training: {len(patterns)}")

    # B. Test Accuracy on Validation Set (10 set 1)
    print("Testing on Validation Set...")
    
    correct_predictions = 0
    matches_found = 0
    total_sequences_scanned = 0
    
    for i in range(len(val_set) - seq_length):
        total_sequences_scanned += 1
        
        # Current sequence in validation
        current_seq = tuple(val_set[i : i + seq_length])
        actual_continuation = val_set[i + seq_length]
        
        # Check if we saw this pattern in training
        if current_seq in patterns:
            matches_found += 1
            
            # "Find all sequences... and return the most probable"
            # Get all historical continuations for this sequence
            history = patterns[current_seq]
            
            # Find the most common value (Mode)
            # most_common(1) returns [(value, count)]
            prediction = Counter(history).most_common(1)[0][0]
            
            # Compare with actual
            if prediction == actual_continuation:
                correct_predictions += 1
    
    # --- Results ---
    print(f"\n--- Results ---")
    print(f"Total Sequences in Set 10_1: {total_sequences_scanned}")
    print(f"Sequences with Historical Match: {matches_found} (Coverage: {matches_found/total_sequences_scanned*100:.2f}%)")
    
    if matches_found > 0:
        accuracy = (correct_predictions / matches_found) * 100
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy (on matched patterns): {accuracy:.2f}%")
    else:
        print("No patterns from Validation Set were found in Training Set.")

if __name__ == "__main__":
    run_analysis()
