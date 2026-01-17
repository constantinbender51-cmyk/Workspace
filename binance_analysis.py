import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import Counter

# --- CONFIGURATION ---
STEP_SIZE = 0.5   # Step size for rounding (e.g., 0.5 means 1.0, 1.5, 2.0...)
SEQ_LENGTH = 5    # Length of the sequence to analyze

def fetch_binance_data(symbol='ETH/USDT', timeframe='4h', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    """
    Fetches historical OHLC data from Binance using CCXT.
    Includes rate limiting and timezone handling.
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
            since = ohlcv[-1][0] + 1
            
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
    
    # Ensure UTC awareness to prevent TypeError
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Filter range
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
    df = df.loc[mask].reset_index(drop=True)
    
    return df

def prepare_arrays(df):
    # 1. Close array
    close_array = df['close'].to_numpy()
    
    # 2. Percentage change array
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    
    # 3. Absolute price array (1 + Cumulative Sum)
    abs_price_array = 1.0 + np.cumsum(pct_change)
    
    # 4. Round absolute price array to lowest STEP_SIZE
    # Formula: floor(value / step) * step
    abs_price_rounded = np.floor(abs_price_array / STEP_SIZE) * STEP_SIZE
    
    return close_array, pct_change, abs_price_rounded

def run_analysis():
    # --- Step 1: Fetch ---
    df = fetch_binance_data()
    if df.empty:
        return

    # --- Step 2: Prepare & Round ---
    close_arr, pct_arr, abs_arr = prepare_arrays(df)
    
    print(f"\nConfiguration -> Step Size: {STEP_SIZE}, Seq Length: {SEQ_LENGTH}")
    print("Data Sample (First 10 Absolute Prices):")
    print(abs_arr[:10])

    # --- Step 3: Split 80/10/10 ---
    total_len = len(abs_arr)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_set = abs_arr[:idx_80]      # 80 set
    val_set = abs_arr[idx_80:idx_90]  # 10 set 1
    
    print(f"Split Sizes -> Train: {len(train_set)}, Val: {len(val_set)}")

    # --- Step 4: Pattern Matching ---
    
    # A. Build Probability Map from Training Set
    print("Building pattern model from Training Set...")
    patterns = {}
    
    for i in range(len(train_set) - SEQ_LENGTH):
        # Sequence + Target
        seq = tuple(train_set[i : i + SEQ_LENGTH])
        target = train_set[i + SEQ_LENGTH]
        
        if seq not in patterns:
            patterns[seq] = []
        patterns[seq].append(target)
        
    print(f"Unique patterns found in training: {len(patterns)}")

    # B. Test Accuracy on Validation Set
    print("Testing on Validation Set...")
    
    correct_predictions = 0
    matches_found = 0
    total_sequences_scanned = 0
    
    for i in range(len(val_set) - SEQ_LENGTH):
        total_sequences_scanned += 1
        
        current_seq = tuple(val_set[i : i + SEQ_LENGTH])
        actual_continuation = val_set[i + SEQ_LENGTH]
        
        if current_seq in patterns:
            matches_found += 1
            
            # Find the most probable continuation from history
            history = patterns[current_seq]
            prediction = Counter(history).most_common(1)[0][0]
            
            if prediction == actual_continuation:
                correct_predictions += 1
    
    # --- Results ---
    print(f"\n--- Results ---")
    print(f"Total Sequences in Set 10_1: {total_sequences_scanned}")
    print(f"Sequences with Match: {matches_found} (Coverage: {matches_found/total_sequences_scanned*100:.2f}%)")
    
    if matches_found > 0:
        accuracy = (correct_predictions / matches_found) * 100
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy (on matched patterns): {accuracy:.2f}%")
    else:
        print("No patterns from Validation Set were found in Training Set.")

if __name__ == "__main__":
    run_analysis()
