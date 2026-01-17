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
import sys  # Added for safe exit
from datetime import datetime
from collections import Counter

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTH = 5

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.1
GRID_STEPS = 100

# Ensemble Threshold
ENSEMBLE_ACC_THRESHOLD = 70.0

def fetch_binance_data(symbol='ETH/USDT', timeframe='30m', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    print(f"Fetching {symbol}...")
    exchange = ccxt.binance({
        'enableRateLimit': True,  # CCXT built-in rate limiter
    })
    
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    all_ohlcv = []
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            if not ohlcv:
                print("No more data received.")
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=None)}...", end='\r')
            
            if since >= end_ts:
                break
                
            # Sleep slightly longer than required to be safe
            time.sleep(exchange.rateLimit / 1000 * 1.1)

        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
            # STOP IMMEDIATELY ON 429 OR DDOS
            print(f"\n\nCRITICAL: Rate Limit Exceeded or DDoS Protection Triggered (429).")
            print(f"Reason: {e}")
            print("Stopping immediately to prevent IP ban.")
            sys.exit(1) # Exit the entire script with error code
            
        except ccxt.NetworkError as e:
            print(f"\nNetwork Error: {e}. Retrying in 10s...")
            time.sleep(10)
            
        except Exception as e:
            print(f"\nUnexpected Error: {e}")
            break
            
    print(f"\nData fetch complete. Total rows: {len(all_ohlcv)}")
    
    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

def get_grid_indices(df, step_size):
    """Converts price series to log-integer grid indices."""
    close_array = df['close'].to_numpy()
    
    # Absolute Price Index calculation
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # Log Rounding
    abs_price_log = np.log(abs_price_raw)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    return grid_indices

def train_and_evaluate(df, step_size):
    """
    Returns dictionary containing metrics and model data needed for the ensemble.
    """
    grid_indices = get_grid_indices(df, step_size)
    
    # Split
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_seq = grid_indices[:idx_80]
    val_seq = grid_indices[idx_80:idx_90]
    
    # Train (Build Patterns)
    patterns = {}
    for i in range(len(train_seq) - SEQ_LENGTH):
        seq = tuple(train_seq[i : i + SEQ_LENGTH])
        target = train_seq[i + SEQ_LENGTH]
        if seq not in patterns: patterns[seq] = []
        patterns[seq].append(target)
        
    # Evaluate on Validation Set
    move_correct = 0
    move_total_valid = 0
    
    for i in range(len(val_seq) - SEQ_LENGTH):
        current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
        current_level = current_seq[-1]
        actual_next_level = val_seq[i + SEQ_LENGTH]
        
        if current_seq in patterns:
            history = patterns[current_seq]
            # Simple majority vote
            predicted_level = Counter(history).most_common(1)[0][0]
            
            pred_diff = predicted_level - current_level
            actual_diff = actual_next_level - current_level
            
            # We only evaluate if the model predicts a move AND a move actually happened
            if pred_diff != 0 and actual_diff != 0:
                move_total_valid += 1
                same_direction = (pred_diff > 0 and actual_diff > 0) or \
                                 (pred_diff < 0 and actual_diff < 0)
                if same_direction:
                    move_correct += 1
                    
    accuracy = (move_correct / move_total_valid * 100) if move_total_valid > 0 else 0.0
    
    return {
        'step_size': step_size,
        'accuracy': accuracy,
        'trade_count': move_total_valid,
        'patterns': patterns,
        'val_seq': val_seq  # Needed for ensemble alignment
    }

def run_combined_metric(high_acc_configs):
    """
    Calculates the accuracy of the Ensemble Logic.
    Logic: If Any(>70% configs) says UP and None says DOWN -> Trade UP.
    Verification: Uses the Maximum Step Size among the voters to verify the move.
    """
    if not high_acc_configs:
        return 0.0, 0

    # All configs align on the same time axis (derived from same DF), 
    # so we can iterate by index using the validation sequence length of the first config.
    ref_seq_len = len(high_acc_configs[0]['val_seq'])
    
    combined_correct = 0
    combined_total = 0
    
    for i in range(ref_seq_len - SEQ_LENGTH):
        up_votes = []
        down_votes = []
        
        # Gather predictions from all qualified configs
        for cfg in high_acc_configs:
            val_seq = cfg['val_seq']
            current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
            
            if current_seq in cfg['patterns']:
                history = cfg['patterns'][current_seq]
                predicted_level = Counter(history).most_common(1)[0][0]
                current_level = current_seq[-1]
                diff = predicted_level - current_level
                
                if diff > 0:
                    up_votes.append(cfg)
                elif diff < 0:
                    down_votes.append(cfg)
        
        # Apply Ensemble Logic: 
        # 1. At least one UP
        # 2. NO conflicting votes (DOWN)
        # (This logic targets Long-Only or Short-Only depending on how you flip it, 
        # here checking for UP moves purely)
        if len(up_votes) > 0 and len(down_votes) == 0:
            # Tie-breaker / Verification: Pick Maximum Step Size among voters
            # Why? Because if the coarsest grid sees a move, the move is likely significant.
            best_cfg = max(up_votes, key=lambda x: x['step_size'])
            
            # Verify using the Chosen Config's Grid
            chosen_val_seq = best_cfg['val_seq']
            
            curr_lvl = chosen_val_seq[i + SEQ_LENGTH - 1] # The end of the input sequence
            next_lvl = chosen_val_seq[i + SEQ_LENGTH]     # The actual target
            
            actual_diff = next_lvl - curr_lvl
            
            # We only count it if the price actually moved on this specific grid
            if actual_diff != 0:
                combined_total += 1
                if actual_diff > 0: # We predicted UP
                    combined_correct += 1
                    
    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    return acc, combined_total

def run_grid_search():
    df = fetch_binance_data()
    if df.empty: 
        print("DataFrame is empty. Exiting.")
        return

    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    
    results = []
    print(f"\n\n--- STARTING GRID SEARCH ({GRID_STEPS} Steps) ---")
    
    for step in step_sizes:
        print(f"Testing Step Size: {step:.5f}...", end=" ")
        res = train_and_evaluate(df, step)
        results.append(res)
        print(f"Acc: {res['accuracy']:.2f}% | Trades: {res['trade_count']}")
        
    # --- COMBINED PREDICTOR ---
    print(f"\n--- CALCULATING COMBINED PREDICTOR ---")
    high_acc_configs = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
    print(f"Qualifying Configs (> {ENSEMBLE_ACC_THRESHOLD}%): {len(high_acc_configs)}")
    
    cmb_acc, cmb_count = run_combined_metric(high_acc_configs)
    print(f"Combined Predictor Accuracy: {cmb_acc:.2f}%")
    print(f"Combined Predictor Trades:   {cmb_count}")

    # --- PLOTTING ---
    unique_id = int(time.time())
    filename = f"grid_search_{unique_id}.png"
    
    accuracies = [r['accuracy'] for r in results]
    trade_counts = [r['trade_count'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Log Step Size')
    ax1.set_ylabel('Directional Accuracy (%)', color='tab:blue')
    ax1.plot(step_sizes, accuracies, marker='o', markersize=3, color='tab:blue', label='Accuracy')
    ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':', label=f'Threshold ({ENSEMBLE_ACC_THRESHOLD}%)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Number of Trades', color='tab:orange')
    ax2.plot(step_sizes, trade_counts, marker='x', markersize=3, linestyle='--', color='tab:orange', label='Trades')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title(f'Accuracy vs. Trade Volume\nCombined Predictor: {cmb_acc:.2f}% ({cmb_count} trades)')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"\nGrid search complete. Plot saved to {filename}")
    
    # Pass combined stats to server to display in HTML
    start_server(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count)

def start_server(image_filename, steps, accs, counts, cmb_acc, cmb_count):
    # Clean up old images
    for f in glob.glob("grid_search_*.png"):
        if f != image_filename:
            try: os.remove(f)
            except: pass

    table_rows = ""
    for s, a, c in zip(steps, accs, counts):
        # Highlight rows that met the threshold
        bg_style = "background-color: #e6fffa;" if a > ENSEMBLE_ACC_THRESHOLD else ""
        # Bold text for very high accuracy
        weight = "font-weight: bold;" if a > ENSEMBLE_ACC_THRESHOLD else ""
        table_rows += f"<tr style='{bg_style} {weight}'><td>{s:.5f}</td><td>{a:.2f}%</td><td>{c}</td></tr>"

    html_content = f"""
    <html>
    <head>
        <title>Grid Search & Ensemble Results</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
        <meta http-equiv="Pragma" content="no-cache" />
        <meta http-equiv="Expires" content="0" />
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 20px; background: #f4f4f4; color: #333; }}
            .container {{ background: white; padding: 30px; display: inline-block; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 900px; width: 100%; }}
            .metric-box {{ background: #2d3748; color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            table {{ margin: 20px auto; border-collapse: collapse; width: 100%; font-size: 14px; }}
            th {{ background: #4a5568; color: white; padding: 12px; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f1f1f1; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Grid Search Strategy Results</h1>
            
            <div class="metric-box">
                <h2>Ensemble Predictor</h2>
                <div style="font-size: 36px; font-weight: bold; margin: 10px 0;">{cmb_acc:.2f}%</div>
                <div>Total Trades: {cmb_count}</div>
                <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">Logic: Consensus of configs > {ENSEMBLE_ACC_THRESHOLD}% acc</div>
            </div>
            
            <img src="{image_filename}" alt="Grid Search Plot">
            
            <h3>Individual Configuration Performance</h3>
            <table>
                <tr><th>Grid Step Size (Log)</th><th>Accuracy</th><th>Trade Count</th></tr>
                {table_rows}
            </table>
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
            print(f"View report at: http://localhost:{PORT}")
            print(f"Press Ctrl+C to stop")
            httpd.serve_forever()
    except OSError:
        print(f"\nERROR: Port {PORT} is busy. Try changing the PORT variable.")
    except KeyboardInterrupt:
        print("\nServer stopped by user.")

if __name__ == "__main__":
    try:
        run_grid_search()
    except KeyboardInterrupt:
        print("\nScript stopped by user.")
