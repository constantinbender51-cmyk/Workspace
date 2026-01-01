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

def create_plot_and_vector(df):
    if df.empty: return None, []

    # --- Step 1: Identify Key Points ---
    
    # Local Highs (2yr Radius)
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    local_highs = df[df['close'] == df['h_max']].index.tolist()

    # Pre-Peak Stability Points
    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stability_points = []
    for peak_idx in local_highs:
        for i in range(peak_idx - 1, 1, -1):
            if df.loc[i, 'range_3m_centered'] <= 0.50:
                stability_points.append(i)
                break
    
    # Local Lows (1yr Radius)
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    local_lows = df[df['low'] == df['l_min']].index.tolist()

    # --- Step 2: Build the Vector ---
    # Create an ordered list of events: (index, type)
    events = []
    for idx in local_highs: events.append((idx, 'HIGH'))
    for idx in stability_points: events.append((idx, 'STABILITY'))
    for idx in local_lows: events.append((idx, 'LOW'))
    
    # Sort events by time index
    events.sort()

    # Vector initialization
    vector = np.full(len(df), np.nan)
    
    # Fill logic:
    # HIGH -> LOW: -1
    # LOW -> STABILITY: 0
    # STABILITY -> HIGH: 1
    for i in range(len(events) - 1):
        curr_idx, curr_type = events[i]
        next_idx, next_type = events[i+1]
        
        val = np.nan
        if curr_type == 'HIGH' and next_type == 'LOW': val = -1
        elif curr_type == 'LOW' and next_type == 'STABILITY': val = 0
        elif curr_type == 'STABILITY' and next_type == 'HIGH': val = 1
        
        if not np.isnan(val):
            vector[curr_idx:next_idx] = val

    df['vector'] = vector

    # --- Step 3: Plotting ---
    fig, ax = plt.subplots(figsize=(15, 8), facecolor='#f1f2f6')
    ax.set_facecolor('#e0e0e0') 

    # Plot markers
    ax.scatter(df.loc[local_highs, 'time'], df.loc[local_highs, 'close'], 
               color='#ff4757', s=160, marker='v', edgecolors='black', zorder=5, label='Local High')
    ax.scatter(df.loc[stability_points, 'time'], df.loc[stability_points, 'close'], 
               color='#8e44ad', s=150, marker='d', edgecolors='white', zorder=7, label='Stability Point')
    ax.scatter(df.loc[local_lows, 'time'], df.loc[local_lows, 'low'], 
               color='#2ed573', s=160, marker='^', edgecolors='black', zorder=5, label='Local Low')

    # Main Price Line
    ax.plot(df['time'], df['close'], color='#2f3542', linewidth=2.5, zorder=4, label='BTC Price')

    ax.set_yscale('linear')
    ax.set_title(f"Market Structure & State Vector: {PAIR}", fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, which='major', color='white', linestyle='-', alpha=0.5, zorder=2)
    ax.legend(facecolor='white', framealpha=1, loc='upper left')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Extract the vector as a clean list for the UI
    clean_vector = []
    for i, val in enumerate(df['vector']):
        date_str = df.loc[i, 'time'].strftime('%Y-%m')
        v_val = "N/A" if np.isnan(val) else int(val)
        clean_vector.append({"date": date_str, "val": v_val})

    return base64.b64encode(buf.getvalue()).decode(), clean_vector

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    plot_data, vector_data = create_plot_and_vector(df)
    current = f"${df['close'].iloc[-1]:,.2f}" if not df.empty else "N/A"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kraken Vector Analysis</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f1f2f6; margin: 0; padding: 40px; color: #2d3436; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.06); }}
            .price {{ font-size: 24px; text-align: center; margin-bottom: 30px; }}
            .chart-box {{ text-align: center; }}
            img {{ max-width: 100%; height: auto; border-radius: 12px; }}
            
            .vector-display {{ 
                margin-top: 40px; 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); 
                gap: 10px; 
                max-height: 300px; 
                overflow-y: auto; 
                padding: 20px;
                background: #fafafa;
                border-radius: 12px;
                border: 1px solid #eee;
            }}
            .vec-item {{ padding: 8px; border-radius: 6px; text-align: center; font-size: 11px; font-weight: bold; border: 1px solid #ddd; }}
            .val-1 {{ background: #ff7675; color: white; }}
            .val-0 {{ background: #dfe6e9; color: #636e72; }}
            .val-plus1 {{ background: #55efc4; color: #00b894; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align:center">Market State Vector [1, 0, -1]</h1>
            <div class="price">Current BTC: <strong>{current}</strong></div>
            
            <div class="chart-box">
                <img src="data:image/png;base64,{plot_data}">
            </div>

            <h3>Monthly Signal Vector</h3>
            <div class="vector-display">
                {{% for item in vector_data %}}
                <div class="vec-item {{ 'val-1' if item.val == -1 else ('val-0' if item.val == 0 else ('val-plus1' if item.val == 1 else '')) }}">
                    {{ item.date }}<br>
                    <span style="font-size: 16px;">{{ item.val }}</span>
                </div>
                {{% endfor %}}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, vector_data=vector_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)