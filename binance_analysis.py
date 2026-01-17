import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import http.server
import socketserver
import threading
import time
from datetime import datetime
from collections import Counter

# --- CONFIGURATION ---
PORT = 8080
STEP_SIZE = 0.05  # Step size for the Absolute Price rounding
SEQ_LENGTH = 5     # Length of sequence to match

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

def prepare_arrays(df):
    close_array = df['close'].to_numpy()
    
    # 1. Raw Percentage Change (NO ROUNDING)
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    
    # 2. Absolute Price Index (High Precision)
    #    Start at 1.0, multiply by (1 + change)
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # 3. Rounded Absolute Price (The only thing we round)
    #    Floors to the nearest STEP_SIZE
    abs_price_rounded = np.floor(abs_price_raw / STEP_SIZE) * STEP_SIZE
    
    return df['timestamp'], close_array, pct_change, abs_price_raw, abs_price_rounded

def generate_plots(timestamps, close, pct, abs_raw, abs_rounded):
    print("\nGenerating plots...")
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 1. Close Price
    axs[0].plot(timestamps, close, color='blue', linewidth=1)
    axs[0].set_title('1. Raw Close Price (USDT)')
    axs[0].grid(True, alpha=0.3)
    
    # 2. Percentage Changes (Raw)
    axs[1].plot(timestamps, pct, color='gray', linewidth=0.5, alpha=0.8)
    axs[1].set_title('2. Percentage Changes (Unrounded / Full Precision)')
    axs[1].set_ylim(-0.10, 0.10)
    axs[1].grid(True, alpha=0.3)
    
    # 3. Calculated Absolute Price (Raw)
    axs[2].plot(timestamps, abs_raw, color='green', linewidth=1)
    axs[2].set_title('3. Absolute Price Index (High Precision Cumprod)')
    axs[2].grid(True, alpha=0.3)
    
    # 4. Rounded Absolute Price
    axs[3].step(timestamps, abs_rounded, color='red', linewidth=1, where='mid')
    axs[3].set_title(f'4. Absolute Price Index (Rounded to {STEP_SIZE})')
    axs[3].grid(True, alpha=0.3)
    
    plt.xlabel('Date')
    plt.savefig('analysis_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to 'analysis_plot.png'")

def start_server():
    html_content = """
    <html>
    <head>
        <title>Crypto Analysis</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f4; }
            .container { background: white; padding: 20px; display: inline-block; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Absolute Price Analysis</h1>
            <p><strong>Step Size:</strong> """ + str(STEP_SIZE) + """ | <strong>Mode:</strong> CumProd (Compounding)</p>
            <img src="analysis_plot.png" alt="Plots">
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
        
    Handler = http.server.SimpleHTTPRequestHandler
    # Allow address reuse to prevent "Address already in use" errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n--- SERVER STARTED ---")
        print(f"View at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop")
        httpd.serve_forever()

def run_analysis():
    # 1. Fetch
    df = fetch_binance_data()
    if df.empty: return
    
    # 2. Process
    timestamps, close, pct, abs_raw, abs_rounded = prepare_arrays(df)
    
    # 3. Analyze Patterns (Using ROUNDED Absolute Price as requested)
    # Note: Matching exact absolute levels across years (1.0 vs 30.0) is difficult.
    total_len = len(abs_rounded)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_set = abs_rounded[:idx_80]
    val_set = abs_rounded[idx_80:idx_90]
    
    print(f"\nTraining Set Size: {len(train_set)}")
    print(f"Validation Set Size: {len(val_set)}")
    
    # Build Map
    patterns = {}
    for i in range(len(train_set) - SEQ_LENGTH):
        seq = tuple(train_set[i : i + SEQ_LENGTH])
        target = train_set[i + SEQ_LENGTH]
        if seq not in patterns: patterns[seq] = []
        patterns[seq].append(target)
        
    print(f"Unique Patterns in Train: {len(patterns)}")
    
    # Test
    matches = 0
    for i in range(len(val_set) - SEQ_LENGTH):
        seq = tuple(val_set[i : i + SEQ_LENGTH])
        if seq in patterns:
            matches += 1
            
    print(f"Patterns Matched in Validation: {matches}")
    if matches == 0:
        print("(Note: 0 matches is expected if Price Index in 2025 is significantly higher than 2020-2024)")

    # 4. Plot & Serve
    generate_plots(timestamps, close, pct, abs_raw, abs_rounded)
    start_server()

if __name__ == "__main__":
    run_analysis()
