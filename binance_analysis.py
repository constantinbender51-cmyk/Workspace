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
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get('error'): return pd.DataFrame()

        result_data = data['result']
        key = [k for k in result_data.keys() if k != 'last'][0]
        df = pd.DataFrame(result_data[key], columns=['time','open','high','low','close','vwap','vol','count'])
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        numeric = ['open','high','low','close']
        df[numeric] = df[numeric].apply(pd.to_numeric)
        
        # Resample to Monthly for structural analysis
        df.set_index('time', inplace=True)
        df_m = df.resample('MS').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        
        return df_m
    except Exception as e:
        print(f"Fetch error: {e}")
        return pd.DataFrame()

def create_plot(df):
    if df.empty: return None

    # Setup Figure with Gray Background to ensure markers pop
    fig, ax = plt.subplots(figsize=(15, 8), facecolor='#f1f2f6')
    ax.set_facecolor('#e0e0e0') 
    
    # --- Pattern 1: Local High (2yr Radius) ---
    # Window = 49 (24 pre + 1 curr + 24 post)
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    
    local_highs_mask = df['close'] == df['h_max']
    local_highs = df[local_highs_mask]
    ax.scatter(local_highs['time'], local_highs['close'], color='#ff4757', s=160, marker='v', 
               edgecolors='black', zorder=5, label='Local High (2yr)')

    # --- Pattern 2: Pre-Peak Stability Point ---
    # Logic: First point preceding a peak that has no 50% move in a radius of 1 month.
    # Radius of 1 month = 3 months centered window (1 pre + 1 curr + 1 post).
    
    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()

    pre_peak_points = []
    for peak_idx in local_highs.index:
        # Search backward from the peak index
        # Radius of 1 month means we need at least 1 index before and after the point to calculate.
        for i in range(peak_idx - 1, 1, -1):
            if df.loc[i, 'range_3m_centered'] <= 0.50:
                pre_peak_points.append(df.iloc[i])
                break # Only find the first (closest) stability point preceding the peak
                
    if pre_peak_points:
        stab_df = pd.DataFrame(pre_peak_points)
        ax.scatter(stab_df['time'], stab_df['close'], color='#8e44ad', s=150, marker='d', 
                   edgecolors='white', zorder=7, label='Pre-Peak Stability (50% Range / 1mo Radius)')

    # --- Pattern 3: Local Low (1yr Radius) ---
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    
    local_lows = df[df['low'] == df['l_min']]
    ax.scatter(local_lows['time'], local_lows['low'], color='#2ed573', s=160, marker='^', 
               edgecolors='black', zorder=5, label='Local Low (1yr)')

    # Main Price Line
    ax.plot(df['time'], df['close'], color='#2f3542', linewidth=2.5, zorder=4, label='BTC Price')

    # Scientific Layout (Linear Scale)
    ax.set_yscale('linear')
    ax.set_title(f"Market Structure Analysis: {PAIR} (Kraken)", fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, which='major', color='white', linestyle='-', alpha=0.5, zorder=2)
    ax.legend(facecolor='white', framealpha=1, loc='upper left', fontsize=10)

    plt.tight_layout()

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
        <title>Kraken Market Analysis</title>
        <style>
            body {{ font-family: -apple-system, system-ui, sans-serif; background: #f1f2f6; text-align: center; padding: 40px; margin: 0; }}
            .card {{ background: white; display: inline-block; padding: 40px; border-radius: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.06); }}
            h1 {{ color: #2d3436; margin-bottom: 8px; }}
            .price {{ font-size: 24px; color: #636e72; margin-bottom: 30px; font-weight: 300; }}
            .legend {{ text-align: left; margin: 30px auto 0; max-width: 900px; background: #fafafa; padding: 20px; border-radius: 12px; border: 1px solid #eee; }}
            .item {{ margin: 10px 0; display: flex; align-items: center; font-size: 14px; color: #2d3436; }}
            .marker {{ width: 14px; height: 14px; margin-right: 12px; border: 1px solid #999; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Bitcoin Market Structure</h1>
            <div class="price">Current Price: <strong>{current}</strong></div>
            <img src="data:image/png;base64,{plot}" alt="BTC Market Chart">
            
            <div class="legend">
                <strong>Analysis Legend:</strong>
                <div class="item"><span class="marker" style="background:#ff4757; clip-path: polygon(50% 100%, 0 0, 100% 0);"></span> <b>Local High:</b> Highest Close in a 2-year radius.</div>
                <div class="item"><span class="marker" style="background:#8e44ad; transform: rotate(45deg);"></span> <b>Pre-Peak Stability:</b> First point before a peak with &lt; 50% range in a 1-month radius.</div>
                <div class="item"><span class="marker" style="background:#2ed573; clip-path: polygon(50% 0%, 0% 100%, 100% 100%);"></span> <b>Local Low:</b> Lowest Low in a 1-year radius.</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    print("Server starting on http://localhost:8080")
    app.run(host='0.0.0.0', port=PORT)