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
GRID_MAX = 0.05
GRID_STEPS = 20

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

def calculate_accuracy(df, step_size):
    """
    Runs the full prediction logic for a single step size and returns the Directional Accuracy.
    """
    close_array = df['close'].to_numpy()
    
    # 1. Absolute Price Index (High Precision)
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # 2. Logarithmic Rounding (The Grid)
    abs_price_log = np.log(abs_price_raw)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    
    # 3. Split Data (80/10/10)
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_seq = grid_indices[:idx_80]
    val_seq = grid_indices[idx_80:idx_90]
    
    # 4. Build Model (Training)
    patterns = {}
    for i in range(len(train_seq) - SEQ_LENGTH):
        seq = tuple(train_seq[i : i + SEQ_LENGTH])
        target = train_seq[i + SEQ_LENGTH]
        if seq not in patterns:
            patterns[seq] = []
        patterns[seq].append(target)
        
    # 5. Test (Validation)
    move_correct = 0
    move_total_valid = 0
    
    for i in range(len(val_seq) - SEQ_LENGTH):
        current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
        current_level = current_seq[-1]
        actual_next_level = val_seq[i + SEQ_LENGTH]
        
        if current_seq in patterns:
            # Predict
            history = patterns[current_seq]
            predicted_level = Counter(history).most_common(1)[0][0]
            
            # Move Accuracy Check
            pred_diff = predicted_level - current_level
            actual_diff = actual_next_level - current_level
            
            # Ignore Flat Predictions or Flat Outcomes
            if pred_diff != 0 and actual_diff != 0:
                move_total_valid += 1
                same_direction = (pred_diff > 0 and actual_diff > 0) or \
                                 (pred_diff < 0 and actual_diff < 0)
                if same_direction:
                    move_correct += 1
                    
    # Return Accuracy
    if move_total_valid > 0:
        return (move_correct / move_total_valid) * 100
    else:
        return 0.0

def run_grid_search():
    # 1. Fetch Data Once
    df = fetch_binance_data()
    if df.empty: return

    # 2. Define Grid
    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    accuracies = []
    
    print(f"\n\n--- STARTING GRID SEARCH ({GRID_STEPS} Steps) ---")
    print(f"Range: {GRID_MIN} to {GRID_MAX}\n")
    
    # 3. Loop
    for step in step_sizes:
        print(f"Testing Step Size: {step:.5f}...", end=" ")
        acc = calculate_accuracy(df, step)
        accuracies.append(acc)
        print(f"Accuracy: {acc:.2f}%")
        
    # 4. Plot
    unique_id = int(time.time())
    filename = f"grid_search_{unique_id}.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(step_sizes, accuracies, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)
    
    plt.title(f'Directional Accuracy vs. Step Size\n(Range: {GRID_MIN} - {GRID_MAX})', fontsize=14)
    plt.xlabel('Log Step Size', fontsize=12)
    plt.ylabel('Directional Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate max point
    max_acc = max(accuracies)
    max_step = step_sizes[accuracies.index(max_acc)]
    plt.annotate(f'Max: {max_acc:.2f}%', xy=(max_step, max_acc), xytext=(max_step, max_acc + 1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig(filename)
    plt.close()
    print(f"\nGrid search complete. Plot saved to {filename}")
    
    # 5. Serve
    start_server(filename, step_sizes, accuracies)

def start_server(image_filename, steps, accs):
    # Cleanup old files
    for f in glob.glob("grid_search_*.png"):
        if f != image_filename:
            try: os.remove(f)
            except: pass

    # Build Table Rows
    table_rows = ""
    for s, a in zip(steps, accs):
        table_rows += f"<tr><td>{s:.5f}</td><td>{a:.2f}%</td></tr>"

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
                <tr><th>Step Size</th><th>Accuracy</th></tr>
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
