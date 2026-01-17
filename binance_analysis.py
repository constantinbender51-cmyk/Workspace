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

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTH = 5

# Step Size (0.10 = 10% steps)
# Try changing this to 0.05 or 0.20 to see the grid lines shift
LOG_STEP_SIZE = 0.10  

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
    
    # 3. Logarithmic Rounding
    abs_price_log = np.log(abs_price_raw)
    abs_price_log_rounded = np.floor(abs_price_log / step_size) * step_size
    abs_price_rounded = np.exp(abs_price_log_rounded)
    
    # Debug info
    unique_levels = len(np.unique(abs_price_rounded))
    print(f"\n[DEBUG] Step Size: {step_size}")
    print(f"[DEBUG] Unique Steps Found: {unique_levels}")
        
    return df['timestamp'], close_array, pct_change, abs_price_raw, abs_price_rounded

def generate_plots(timestamps, close, pct, abs_raw, abs_rounded, step_size, filename):
    print(f"Generating plot: {filename}...")
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 1. Close Price
    axs[0].plot(timestamps, close, color='blue', linewidth=1)
    axs[0].set_title('1. Raw Close Price (USDT)')
    axs[0].grid(True, alpha=0.3)
    
    # 2. Percentage Changes
    axs[1].plot(timestamps, pct, color='gray', linewidth=0.5, alpha=0.8)
    axs[1].set_title('2. Percentage Changes (Unrounded)')
    axs[1].set_ylim(-0.10, 0.10)
    axs[1].grid(True, alpha=0.3)
    
    # 3. Absolute Price (Raw)
    axs[2].plot(timestamps, abs_raw, color='green', linewidth=1)
    axs[2].set_title('3. Absolute Price Index (High Precision)')
    axs[2].grid(True, alpha=0.3)
    
    # 4. Absolute Price (Log Rounded) + Horizontal Lines
    axs[3].step(timestamps, abs_rounded, color='red', linewidth=1.5, where='mid')
    axs[3].set_title(f'4. Absolute Price Index (Log-Rounded: {step_size*100}% steps)')
    axs[3].grid(True, alpha=0.3)
    
    # --- DRAW HORIZONTAL LINES EVERY 10 STEPS ---
    # We calculate the levels in Log space, then convert to linear for plotting
    min_log = np.log(np.min(abs_rounded[abs_rounded > 0])) # avoid log(0)
    max_log = np.log(np.max(abs_rounded))
    
    # Start from the nearest "10th step" below the minimum
    # "10 steps" means a jump of 10 * step_size
    grid_interval = 10 * step_size
    
    start_level = np.floor(min_log / grid_interval) * grid_interval
    current_level = start_level
    
    while current_level < max_log + grid_interval:
        # Convert back to linear price for the plot
        y_val = np.exp(current_level)
        
        # Draw the line
        axs[3].axhline(y=y_val, color='black', linestyle='--', linewidth=1, alpha=0.6)
        
        # Add a text label to identify the line
        axs[3].text(timestamps.iloc[0], y_val, f' 10-Step Level', 
                    color='black', fontsize=8, verticalalignment='bottom')
        
        current_level += grid_interval
    # --------------------------------------------

    plt.xlabel('Date')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to '{filename}'")

def start_server(image_filename):
    # Cleanup old images
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
            <h1>Logarithmic Price Analysis</h1>
            <p><strong>Step Mode:</strong> Logarithmic | <strong>Step Size:</strong> {LOG_STEP_SIZE*100}%</p>
            <p><strong>Grid Lines:</strong> Every 10 steps ({LOG_STEP_SIZE*1000}% change)</p>
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
        print(f"\nERROR: Port {PORT} is busy. Stop previous script.")

def run_analysis():
    df = fetch_binance_data()
    if df.empty: return
    
    timestamps, close, pct, abs_raw, abs_rounded = prepare_arrays(df, step_size=LOG_STEP_SIZE)
    
    unique_id = int(time.time())
    filename = f"analysis_plot_{unique_id}.png"
    
    generate_plots(timestamps, close, pct, abs_raw, abs_rounded, LOG_STEP_SIZE, filename)
    start_server(filename)

if __name__ == "__main__":
    run_analysis()
