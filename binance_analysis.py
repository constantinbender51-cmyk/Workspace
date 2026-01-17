import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import Counter

# --- CONFIGURATION ---
STEP_SIZE = 0.005  # 0.5% steps (filters small noise)
SEQ_LENGTH = 5     # Pattern sequence length

def fetch_binance_data(symbol='ETH/USDT', timeframe='4h', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    
    all_ohlcv = []
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000)}...", end='\r')
            if since >= end_ts: break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"\nError: {e}")
            break
            
    print(f"\nTotal candles: {len(all_ohlcv)}")
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)
    return df.loc[mask].reset_index(drop=True)

def prepare_arrays(df):
    close_array = df['close'].to_numpy()
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    
    # Absolute Price = 1 + Cumulative Sum of changes
    abs_price_array = 1.0 + np.cumsum(pct_change)
    
    # Rounding to Step Size
    abs_price_rounded = np.floor(abs_price_array / STEP_SIZE) * STEP_SIZE
    
    return close_array, pct_change, abs_price_rounded

def run_analysis():
    # 1. Fetch & Prepare
    df = fetch_binance_data()
    if df.empty: return
    
    _, _, abs_arr = prepare_arrays(df)
    
    # 2. Split
    total_len = len(abs_arr)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_set = abs_arr[:idx_80]
    val_set = abs_arr[idx_80:idx_90]
    
    print(f"\nTraining Set: {len(train_set)} | Validation Set: {len(val_set)}")
    
    # 3. Build Model (Training)
    patterns = {}
    for i in range(len(train_set) - SEQ_LENGTH):
        seq = tuple(train_set[i : i + SEQ_LENGTH])
        target = train_set[i + SEQ_LENGTH]
        if seq not in patterns: patterns[seq] = []
        patterns[seq].append(target)
        
    print(f"Unique Patterns in Train: {len(patterns)}")
    
    # 4. Test (Validation)
    matches_found = 0
    
    # Exact Accuracy Metrics
    exact_correct = 0
    
    # Move Accuracy Metrics
    move_correct = 0
    move_total_valid = 0  # Counts only non-flat moves
    
    for i in range(len(val_set) - SEQ_LENGTH):
        current_seq = tuple(val_set[i : i + SEQ_LENGTH])
        actual_val = val_set[i + SEQ_LENGTH]
        current_val = current_seq[-1] # The last known price before prediction
        
        if current_seq in patterns:
            matches_found += 1
            
            # Predict most probable next value
            history = patterns[current_seq]
            prediction = Counter(history).most_common(1)[0][0]
            
            # A. Exact Match Accuracy
            if np.isclose(prediction, actual_val, atol=1e-9):
                exact_correct += 1
                
            # B. Move Accuracy (Directional)
            pred_diff = prediction - current_val
            actual_diff = actual_val - current_val
            
            # Ignore flat predictions OR flat outcomes
            # Since we are using rounded data, "flat" means it didn't jump a step
            if not np.isclose(pred_diff, 0, atol=1e-9) and not np.isclose(actual_diff, 0, atol=1e-9):
                move_total_valid += 1
                
                # Check Direction
                is_same_direction = (pred_diff > 0 and actual_diff > 0) or \
                                    (pred_diff < 0 and actual_diff < 0)
                                    
                if is_same_direction:
                    move_correct += 1

    # --- Results ---
    print(f"\n--- Results ---")
    print(f"Total Sequences Scanned: {len(val_set) - SEQ_LENGTH}")
    print(f"Patterns Matched: {matches_found}")
    
    if matches_found > 0:
        print(f"\n1. Exact Value Accuracy (Exact Step Match):")
        print(f"   {exact_correct}/{matches_found} -> {(exact_correct/matches_found)*100:.2f}%")
        
        print(f"\n2. Move Accuracy (Directional):")
        if move_total_valid > 0:
            print(f"   (Ignored {matches_found - move_total_valid} flat predictions/outcomes)")
            print(f"   Correct Moves: {move_correct}/{move_total_valid}")
            print(f"   Accuracy: {(move_correct/move_total_valid)*100:.2f}%")
        else:
            print("   No valid moves found (all predictions or outcomes were flat).")
            
    else:
        print("No patterns matched.")

if __name__ == "__main__":
    run_analysis()
