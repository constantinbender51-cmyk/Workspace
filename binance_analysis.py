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

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.1
GRID_STEPS = 100

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

def calculate_metrics(df, step_size):
    """
    Returns (Accuracy %, Trade Count)
    """
    close_array = df['close'].to_numpy()
    
    # 1. Absolute Price Index
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # 2. Log Rounding
    abs_price_log = np.log(abs_price_raw)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    
    # 3. Split
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_seq = grid_indices[:idx_80]
    val_seq = grid_indices[idx_80:idx_90]
    
    # 4. Train
    patterns = {}
    for i in range(len(train_seq) - SEQ_LENGTH):
        seq = tuple(train_seq[i : i + SEQ_LENGTH])
        target = train_seq[i + SEQ_LENGTH]
        if seq not in patterns: patterns[seq] = []
        patterns[seq].append(target)
        
    # 5. Test
    move_correct = 0
    move_total_valid = 0
    
    for i in range(len(val_seq) - SEQ_LENGTH):
        current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
        current_level = current_seq[-1]
        actual_next_level = val_seq[i + SEQ_LENGTH]
        
        if current_seq in patterns:
            history = patterns[current_seq]
            predicted_level = Counter(history).most_common(1)[0][0]
            
            # Check Move
            pred_diff = predicted_level - current_level
            actual_diff = actual_next_level - current_level
            
            # Ignore Flats
            if pred_diff != 0 and actual_diff != 0:
                move_total_valid += 1
                same_direction = (pred_diff > 0 and actual_diff > 0) or \
                                 (pred_diff < 0 and actual_diff < 0)
                if same_direction:
                    move_correct += 1
                    
    accuracy = (move_correct / move_total_valid * 100) if move_total_valid > 0 else 0.0
    return accuracy, move_total_valid

def run_grid_search():
    df = fetch_binance_data()
    if df.empty: return

    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    accuracies = []
    trade_counts = []
    
    print(f"\n\n--- STARTING GRID SEARCH ({GRID_STEPS} Steps) ---")
    
    for step in step_sizes:
        print(f"Testing Step Size: {step:.5f}...", end=" ")
        acc, count = calculate_metrics(df, step)
        accuracies.append(acc)
        trade_counts.append(count)
        print(f"Acc: {acc:.2f}% | Trades: {count}")
        
    # --- PLOTTING (Dual Axis) ---
    unique_id = int(time.time())
    filename = f"grid_search_{unique_id}.png"
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Axis 1: Accuracy (Blue)
    ax1.set_xlabel('Log Step Size')
    ax1.set_ylabel('Directional Accuracy (%)', color='tab:blue')
    ax1.plot(step_sizes, accuracies, marker='o', color='tab:blue', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Axis 2: Trade Count (Orange)
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Number of Trades', color='tab:orange')
    ax2.plot(step_sizes, trade_counts, marker='x', linestyle='--', color='tab:orange', label='Trades')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title(f'Accuracy vs. Trade Volume\n(Range: {GRID_MIN} - {GRID_MAX})')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"\nGrid search complete. Plot saved to {filename}")
    start_server(filename, step_sizes, accuracies, trade_counts)

def start_server(image_filename, steps, accs, counts):
    for f in glob.glob("grid_search_*.png"):
        if f != image_filename:
            try: os.remove(f)
            except: pass

    # Build Table with Extra Column
    table_rows = ""
    for s, a, c in zip(steps, accs, counts):
        table_rows += f"<tr><td>{s:.5f}</td><td>{a:.2f}%</td><td>{c}</td></tr>"

    html_content = f"""
    <html>
    <head>
        <title>Grid Search Results</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
        <meta http-equiv="Pragma" content="no-cache" />
        <meta http-equiv="Expires" content="0" />
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f4; }}
            .container {{ background: white; padding: 20px; display: inline-block; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            table {{ margin: 0 auto; border-collapse: collapse; width: 80%; }}
            th, td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Grid Search Results</h1>
            <p><strong>Config:</strong> 10 Steps between {GRID_MIN} and {GRID_MAX}</p>
            
            <img src="{image_filename}" alt="Grid Search Plot">
            
            <h3>Data Points</h3>
            <table>
                <tr><th>Step Size</th><th>Accuracy</th><th>Trades</th></tr>
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
            print(f"View at: http://localhost:{PORT}")
            print(f"Press Ctrl+C to stop")
            httpd.serve_forever()
    except OSError:
        print(f"\nERROR: Port {PORT} is busy.")

if __name__ == "__main__":
    run_grid_search()
