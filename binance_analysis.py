import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import http.server
import socketserver
import time
from datetime import datetime

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTH = 5

# LOG_STEP_SIZE = 0.05  (5% steps - Normal)
# LOG_STEP_SIZE = 10.0  (1000% steps - Should be a flat line)
LOG_STEP_SIZE = 0.05 

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
    
    # 1. Percentage Change
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    
    # 2. Absolute Price Index (High Precision)
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # 3. Logarithmic Rounding
    # We take the log, floor it to the nearest STEP, then exponentiate back.
    abs_price_log = np.log(abs_price_raw)
    abs_price_log_rounded = np.floor(abs_price_log / LOG_STEP_SIZE) * LOG_STEP_SIZE
    abs_price_rounded = np.exp(abs_price_log_rounded)
    
    return df['timestamp'], close_array, pct_change, abs_price_raw, abs_price_rounded

def generate_plots(timestamps, close, pct, abs_raw, abs_rounded):
    print("\nGenerating plots...")
    
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
    
    # 4. Absolute Price (Log Rounded)
    axs[3].step(timestamps, abs_rounded, color='red', linewidth=1.5, where='mid')
    axs[3].set_title(f'4. Absolute Price Index (Log-Rounded: {LOG_STEP_SIZE*100}% steps)')
    axs[3].grid(True, alpha=0.3)
    
    plt.xlabel('Date')
    plt.savefig('analysis_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to 'analysis_plot.png'")

def start_server():
    # CACHE BUSTER: We generate a unique timestamp
    timestamp = int(time.time())
    
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
            <img src="analysis_plot.png?t={timestamp}" alt="Plots">
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
        
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n--- SERVER STARTED ---")
        print(f"View at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop")
        httpd.serve_forever()

def run_analysis():
    df = fetch_binance_data()
    if df.empty: return
    
    timestamps, close, pct, abs_raw, abs_rounded = prepare_arrays(df)
    
    # Plot & Serve
    generate_plots(timestamps, close, pct, abs_raw, abs_rounded)
    start_server()

if __name__ == "__main__":
    run_analysis()
