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

    # Setup Figure
    fig, ax = plt.subplots(figsize=(15, 8), facecolor='#f1f2f6')
    ax.set_facecolor('#e0e0e0') 
    
    # --- Pattern 1: Local High (2yr Radius) ---
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    
    local_highs_mask = df['close'] == df['h_max']
    local_highs = df[local_highs_mask]
    ax.scatter(local_highs['time'], local_highs['close'], color='#ff4757', s=160, marker='v', 
               edgecolors='black', zorder=5, label='Local High (2yr)')

    # --- New Pattern: Stability Point before Peak ---
    # Logic: For each peak, look backwards. Find the first point where the preceding 6 months 
    # had a range (High-Low)/Low <= 50%.
    
    stability_points = []
    
    # Calculate 6-month rolling range looking backwards
    # window=6, min_periods=6 means we need 6 full months of history
    # Threshold increased to 50%
    df['range_6m'] = (df['high'].rolling(window=6).max() - df['low'].rolling(window=6).min()) / df['low'].rolling(window=6).min()

    for peak_idx in local_highs.index:
        # Search backwards from the peak index
        search_range = range(peak_idx - 1, 5, -1) # Stop at index 5 because we need 6 months of data
        for i in search_range:
            if df.loc[i, 'range_6m'] <= 0.50:
                stability_points.append(df.iloc[i])
                break # Found the most recent stability point before this peak

    if stability_points:
        stab_df = pd.DataFrame(stability_points)
        ax.scatter(stab_df['time'], stab_df['close'], color='#3498db', s=120, marker='o', 
                   edgecolors='black', zorder=6, label='Pre-Peak Stability (6mo < 50%)')

    # --- Pattern 3: Local Low (1yr Radius) ---
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    
    local_lows = df[df['low'] == df['l_min']]
    ax.scatter(local_lows['time'], local_lows['low'], color='#2ed573', s=160, marker='^', 
               edgecolors='black', zorder=5, label='Local Low (1yr)')

    # Main Price Line
    ax.plot(df['time'], df['close'], color='#2f3542', linewidth=2.5, zorder=4, label='BTC Price')

    # Styling (Linear Scale)
    ax.set_yscale('linear')
    ax.set_title(f"Scientific Structural Analysis: {PAIR}", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, which='major', color='white', linestyle='-', alpha=0.4, zorder=2)
    ax.legend(facecolor='white', framealpha=1, loc='upper left')

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
        <title>Kraken Analysis</title>
        <style>
            body {{ font-family: sans-serif; background: #f1f2f6; text-align: center; margin: 0; padding: 40px; }}
            .container {{ background: white; display: inline-block; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
            .legend-box {{ text-align: left; margin: 30px auto 0; max-width: 900px; background: #fafafa; padding: 20px; border-radius: 12px; border: 1px solid #eee; }}
            .item {{ margin: 10px 0; display: flex; align-items: center; }}
            .box {{ width: 16px; height: 16px; margin-right: 12px; border-radius: 3px; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bitcoin Market Structure</h1>
            <div style="font-size: 20px; margin-bottom: 20px;">Price: {current}</div>
            <img src="data:image/png;base64,{plot}">
            
            <div class="legend-box">
                <strong>Analysis Guide:</strong>
                <div class="item"><span class="box" style="background:#ff4757"></span> <b>Local High:</b> Highest Close within ±2 years.</div>
                <div class="item"><span class="box" style="background:#3498db"></span> <b>Stability Point:</b> Point before peak where preceding 6 months range was &lt; 50%.</div>
                <div class="item"><span class="box" style="background:#2ed573"></span> <b>Local Low:</b> Lowest Low within ±1 year.</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)