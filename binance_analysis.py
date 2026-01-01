import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
PAIR = 'XBTUSD' 
INTERVAL = 10080 # Weekly data for maximum depth
PORT = 8080

def fetch_kraken_data(pair, interval):
    """Fetches historical weekly OHLC from Kraken and resamples to Monthly."""
    url = "https://api.kraken.com/0/public/OHLC"
    params = {'pair': pair, 'interval': interval}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get('error'): return pd.DataFrame()

        result_data = data['result']
        key = [k for k in result_data.keys() if k != 'last'][0]
        df = pd.DataFrame(result_data[key], columns=['time','open','high','low','close','vwap','vol','count'])
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        numeric = ['open','high','low','close']
        df[numeric] = df[numeric].apply(pd.to_numeric)
        
        # Resample to Monthly for cleaner structural analysis
        df.set_index('time', inplace=True)
        df_m = df.resample('MS').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        
        return df_m
    except:
        return pd.DataFrame()

def create_plot(df):
    if df.empty: return None

    # Setup Figure with a GREY background
    # We use a linear scale now, which makes the parabolic nature of BTC very evident
    fig, ax = plt.subplots(figsize=(15, 8), facecolor='#f1f2f6')
    
    # Explicitly set the plotting area to grey so WHITE spans are visible
    ax.set_facecolor('#e0e0e0') 
    
    # --- Pattern 2: Consolidation (30% Range for 1 Year) ---
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    df['roll_max'] = df['high'].rolling(window=indexer).max()
    df['roll_min'] = df['low'].rolling(window=indexer).min()
    df['range_pct'] = (df['roll_max'] - df['roll_min']) / df['roll_min']
    
    consolidation_mask = pd.Series(False, index=df.index)
    consolidation_indices = df[df['range_pct'] <= 0.30].index

    for idx in consolidation_indices:
        start_date = df.loc[idx, 'time']
        end_idx = min(idx + 11, len(df) - 1)
        end_date = df.loc[end_idx, 'time']
        consolidation_mask.loc[idx:end_idx] = True
        
        # Mark consolidation PURE WHITE
        ax.axvspan(start_date, end_date, color='white', alpha=1.0, zorder=1)

    # --- Pattern 1: Local High (2yr Radius) ---
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    
    local_highs = df[df['close'] == df['h_max']]
    ax.scatter(local_highs['time'], local_highs['close'], color='#ff4757', s=160, marker='v', 
               edgecolors='black', zorder=5, label='Local High (2yr)')

    # --- Pattern 3: Local Low (1yr Radius, Excluding Consolidation) ---
    df_lows_clean = df.copy()
    df_lows_clean.loc[consolidation_mask, 'low'] = np.inf
    df_lows_clean['l_min'] = df_lows_clean['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df_lows_clean.loc[df_lows_clean.index[-12:], 'l_min'] = -1.0
    
    local_lows = df_lows_clean[df_lows_clean['low'] == df_lows_clean['l_min']]
    ax.scatter(local_lows['time'], local_lows['low'], color='#2ed573', s=160, marker='^', 
               edgecolors='black', zorder=5, label='Local Low (1yr, Clean)')

    # Main Price Line
    ax.plot(df['time'], df['close'], color='#2f3542', linewidth=2.5, zorder=4, label='BTC Price')

    # Styling (Linear Scale)
    ax.set_yscale('linear') # Changed from 'log' to 'linear'
    ax.set_title(f"Scientific Structural Analysis: {PAIR} (Linear Scale)", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, which='major', color='white', linestyle='-', alpha=0.4, zorder=2)
    ax.legend(facecolor='white', framealpha=1, loc='upper left')

    # Adjust layout to fit everything
    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    plot = create_plot(df)
    
    current = f"${df['close'].iloc[-1]:,.2f}" if not df.empty else "N/A"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kraken Linear Chart</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f2f6; text-align: center; margin: 0; padding: 40px; }}
            .container {{ background: white; display: inline-block; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
            h1 {{ color: #2f3542; margin-bottom: 5px; }}
            .price-tag {{ font-size: 24px; font-weight: bold; color: #57606f; margin-bottom: 25px; }}
            .legend-box {{ text-align: left; margin: 30px auto 0; max-width: 900px; background: #fafafa; padding: 20px; border-radius: 12px; border: 1px solid #eee; }}
            .item {{ margin: 10px 0; display: flex; align-items: center; }}
            .box {{ width: 16px; height: 16px; margin-right: 12px; border-radius: 3px; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bitcoin Market Structure</h1>
            <div class="price-tag">Current: {current}</div>
            <img src="data:image/png;base64,{plot}">
            
            <div class="legend-box">
                <strong>Analysis Guide:</strong>
                <div class="item"><span class="box" style="background:#ff4757"></span> <b>Local High:</b> Highest Monthly Close within ±2 years.</div>
                <div class="item"><span class="box" style="background:#2ed573"></span> <b>Local Low:</b> Lowest Monthly Low within ±1 year (skipping consolidation).</div>
                <div class="item"><span class="box" style="background:white"></span> <b>White Span:</b> Price stayed within a 30% range for 1 year+.</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    print(f"Starting Linear Pattern Server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)