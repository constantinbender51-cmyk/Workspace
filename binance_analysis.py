import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for server environments
import matplotlib.pyplot as plt
import http.server
import socketserver
import threading
import webbrowser
import os
from collections import Counter
from datetime import datetime
import time

# --- CONFIGURATION ---
PORT = 8080
PCT_STEP = 0.001  # Group changes into 0.1% buckets
SEQ_LENGTH = 5

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
    
    # 2. Quantized (for matching)
    pct_change_quantized = np.round(pct_change / PCT_STEP) * PCT_STEP
    
    # 3. Absolute Price Index (Cumprod)
    # Start at 1.0, multiply by (1 + change)
    multipliers = 1.0 + pct_change
    abs_price_array = np.cumprod(multipliers)
    
    return df['timestamp'], close_array, pct_change, pct_change_quantized, abs_price_array

def generate_plots(timestamps, close, pct, pct_quant, abs_price):
    print("\nGenerating plots...")
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 1. Close Price
    axs[0].plot(timestamps, close, color='blue', linewidth=1)
    axs[0].set_title('1. Raw Close Price (USDT)')
    axs[0].grid(True, alpha=0.3)
    
    # 2. Absolute Price Index (Calculated)
    axs[1].plot(timestamps, abs_price, color='green', linewidth=1)
    axs[1].set_title('2. Calculated Absolute Price Index (Base 1.0)')
    axs[1].grid(True, alpha=0.3)
    
    # 3. Percentage Changes
    axs[2].plot(timestamps, pct, color='gray', linewidth=0.5, alpha=0.7)
    axs[2].set_title('3. Percentage Changes (Raw)')
    axs[2].set_ylim(-0.10, 0.10) # Limit to +/- 10% for readability
    axs[2].grid(True, alpha=0.3)
    
    # 4. Quantized Changes (What the bot sees)
    axs[3].step(timestamps, pct_quant, color='red', linewidth=0.5, where='mid')
    axs[3].set_title(f'4. Quantized Changes (Step Size: {PCT_STEP})')
    axs[3].set_ylim(-0.10, 0.10)
    axs[3].grid(True, alpha=0.3)
    
    plt.xlabel('Date')
    plt.savefig('analysis_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to 'analysis_plot.png'")

def start_server():
    # Create a simple HTML file
    html_content = """
    <html>
    <head>
        <title>Crypto Pattern Analysis</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 20px; background: #f0f0f0; }
            img { max-width: 95%; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 5px; }
            .card { background: white; padding: 20px; display: inline-block; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Analysis Result</h1>
            <p>Visualizing: Close Price, Calculated Index, and Volatility Inputs</p>
            <img src="analysis_plot.png" alt="Analysis Plots">
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
        
    # Start Server
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n--- SERVER STARTED ---")
        print(f"Open your browser at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop")
        httpd.serve_forever()

def run_analysis():
    # 1. Fetch
    df = fetch_binance_data()
    if df.empty: return
    
    # 2. Process
    timestamps, close, pct, pct_quant, abs_price = prepare_arrays(df)
    
    # 3. Plot
    generate_plots(timestamps, close, pct, pct_quant, abs_price)
    
    # 4. Serve
    start_server()

if __name__ == "__main__":
    run_analysis()
