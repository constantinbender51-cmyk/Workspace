import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import http.server
import socketserver
import time
import os
import glob
from datetime import datetime
from collections import Counter

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTH = 5

# Step Size (0.10 = 10% log-steps).
LOG_STEP_SIZE = 0.005

def fetch_binance_data(symbol='ETH/USDT', timeframe='4h', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    print(f"Fetching {symbol}...")
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
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

def prepare_arrays(df, step_size):
    close_array = df['close'].to_numpy()
    
    # 1. Percentage Change
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    
    # 2. Absolute Price Index (High Precision)
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # 3. Logarithmic Rounding (The Grid)
    abs_price_log = np.log(abs_price_raw)
    
    # ABSOLUTE GRID INDICES (Integers representing the "Floor" level)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    
    # Convert back to float price for plotting
    abs_price_rounded = np.exp(grid_indices * step_size)
    
    return df['timestamp'], close_array, pct_change, abs_price_raw, abs_price_rounded, grid_indices

def run_prediction_logic(grid_indices):
    print("\n\n--- RUNNING PREDICTION ALGORITHM ---")
    
    # Split Data (80/10/10)
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_seq = grid_indices[:idx_80]
    val_seq = grid_indices[idx_80:idx_90]
    
    # 1. Build Model from 80 Set
    patterns = {}
    for i in range(len(train_seq) - SEQ_LENGTH):
        seq = tuple(train_seq[i : i + SEQ_LENGTH])
        target = train_seq[i + SEQ_LENGTH]
        
        if seq not in patterns:
            patterns[seq] = []
        patterns[seq].append(target)
        
    print(f"Unique Patterns Learned: {len(patterns)}")
    
    # 2. Test on 10 Set 1
    matches_found = 0
    total_scanned = 0
    
    # Accuracy Metrics
    exact_correct = 0
    move_correct = 0
    move_total_valid = 0 # Count of sequences where neither pred nor actual was flat
    
    for i in range(len(val_seq) - SEQ_LENGTH):
        total_scanned += 1
        
        # Current Sequence
        current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
        current_level = current_seq[-1] # The last known level
        actual_next_level = val_seq[i + SEQ_LENGTH]
        
        if current_seq in patterns:
            matches_found += 1
            
            # Find most probable next LEVEL (Mode)
            history = patterns[current_seq]
            predicted_level = Counter(history).most_common(1)[0][0]
            
            # --- A. Exact Accuracy ---
            if predicted_level == actual_next_level:
                exact_correct += 1
                
            # --- B. Move Accuracy (Directional) ---
            pred_diff = predicted_level - current_level
            actual_diff = actual_next_level - current_level
            
            # Ignore if Prediction is Flat OR Actual Outcome is Flat
            if pred_diff != 0 and actual_diff != 0:
                move_total_valid += 1
                
                # Check if Directions Match
                same_direction = (pred_diff > 0 and actual_diff > 0) or \
                                 (pred_diff < 0 and actual_diff < 0)
                
                if same_direction:
                    move_correct += 1
                
    # 3. Print Results
    print("\n--- PREDICTION RESULTS ---")
    print(f"Total Sequences in Validation: {total_scanned}")
    print(f"Sequences Matched in History: {matches_found}")
    
    if matches_found > 0:
        # Exact
        exact_acc = (exact_correct / matches_found) * 100
        print(f"\n1. Exact Match Accuracy: {exact_acc:.2f}% ({exact_correct}/{matches_found})")
        
        # Move
        print(f"\n2. Move Accuracy (Directional):")
        if move_total_valid > 0:
            move_acc = (move_correct / move_total_valid) * 100
            print(f"   Valid Moves (Non-flat): {move_total_valid}")
            print(f"   Correct Direction: {move_correct}")
            print(f"   Accuracy: {move_acc:.2f}%")
        else:
            print("   No valid moves found (all matched predictions or outcomes were flat).")
            
    else:
        print("No matches found.")
    print("--------------------------------\n")

def generate_plots(timestamps, close, pct, abs_raw, abs_rounded, step_size, filename):
    print(f"Generating plot: {filename}...")
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    axs[0].plot(timestamps, close, color='blue', linewidth=1)
    axs[0].set_title('1. Raw Close Price (USDT)')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(timestamps, pct, color='gray', linewidth=0.5, alpha=0.8)
    axs[1].set_title('2. Percentage Changes (Unrounded)')
    axs[1].set_ylim(-0.10, 0.10)
    axs[1].grid(True, alpha=0.3)
    
    axs[2].plot(timestamps, abs_raw, color='green', linewidth=1)
    axs[2].set_title('3. Absolute Price Index (High Precision)')
    axs[2].grid(True, alpha=0.3)
    
    axs[3].step(timestamps, abs_rounded, color='red', linewidth=1.5, where='mid')
    axs[3].set_title(f'4. Absolute Price Index (Log-Rounded: {step_size*100}% steps)')
    axs[3].grid(True, alpha=0.3)
    
    # Horizontal Lines Logic
    min_log = np.log(np.min(abs_rounded[abs_rounded > 0])) 
    max_log = np.log(np.max(abs_rounded))
    grid_interval = 10 * step_size
    start_level = np.floor(min_log / grid_interval) * grid_interval
    current_level = start_level
    
    while current_level < max_log + grid_interval:
        y_val = np.exp(current_level)
        axs[3].axhline(y=y_val, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        axs[3].text(timestamps.iloc[0], y_val, ' 10-Step', fontsize=8, va='bottom', color='black', alpha=0.7)
        current_level += grid_interval

    plt.xlabel('Date')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to '{filename}'")

def start_server(image_filename):
    for f in glob.glob("analysis_plot_*.png"):
        if f != image_filename:
            try:
                os.remove(f)
            except:
                pass

    html_content = f"""
    <html>
    <head>
        <title>Crypto Analysis</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
        <meta http-equiv="Pragma" content="no-cache" />
        <meta http-equiv="Expires" content="0" />
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f4; }}
            .container {{ background: white; padding: 20px; display: inline-block; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Absolute Sequence Analysis</h1>
            <p><strong>Step Size:</strong> {LOG_STEP_SIZE*100}%</p>
            <p><small>Image ID: {image_filename}</small></p>
            <img src="{image_filename}" alt="Plots">
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
        
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\n--- SERVER STARTED ---")
            print(f"View at: http://localhost:{PORT}")
            print(f"Press Ctrl+C to stop")
            httpd.serve_forever()
    except OSError:
        print(f"\nERROR: Port {PORT} is busy.")

def run_analysis():
    df = fetch_binance_data()
    if df.empty: return
    
    timestamps, close, pct, abs_raw, abs_rounded, grid_indices = prepare_arrays(df, step_size=LOG_STEP_SIZE)
    
    # Run Prediction Logic (Absolute Sequence + Move Accuracy)
    run_prediction_logic(grid_indices)
    
    unique_id = int(time.time())
    filename = f"analysis_plot_{unique_id}.png"
    
    generate_plots(timestamps, close, pct, abs_raw, abs_rounded, LOG_STEP_SIZE, filename)
    start_server(filename)

if __name__ == "__main__":
    run_analysis()
