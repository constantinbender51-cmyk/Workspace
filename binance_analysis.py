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
    local_high_indices = df[df['close'] == df['h_max']].index.tolist()

    # Pre-Peak Stability Points (50% range, 1mo radius)
    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stability_indices = []
    for peak_idx in local_high_indices:
        for i in range(peak_idx - 1, 1, -1):
            if df.loc[i, 'range_3m_centered'] <= 0.50:
                stability_indices.append(i)
                break
    
    # Local Lows (1yr Radius)
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    local_low_indices = df[df['low'] == df['l_min']].index.tolist()

    # --- Step 2: Build the Vector ---
    events = []
    for idx in local_high_indices: events.append((idx, 'HIGH'))
    for idx in stability_indices: events.append((idx, 'STABILITY'))
    for idx in local_low_indices: events.append((idx, 'LOW'))
    events.sort()

    vector = np.full(len(df), np.nan)
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
    fig, ax = plt.subplots(figsize=(15, 7), facecolor='#f1f2f6')
    ax.set_facecolor('#e0e0e0') 

    ax.plot(df['time'], df['close'], color='#2f3542', linewidth=2, zorder=3, label='BTC Price')
    
    ax.scatter(df.loc[local_high_indices, 'time'], df.loc[local_high_indices, 'close'], 
               color='#ff4757', s=140, marker='v', edgecolors='black', zorder=5, label='Local High')
    ax.scatter(df.loc[stability_indices, 'time'], df.loc[stability_indices, 'close'], 
               color='#8e44ad', s=130, marker='d', edgecolors='white', zorder=5, label='Stability Point')
    ax.scatter(df.loc[local_low_indices, 'time'], df.loc[local_low_indices, 'low'], 
               color='#2ed573', s=140, marker='^', edgecolors='black', zorder=5, label='Local Low')

    ax.set_yscale('linear')
    ax.set_title(f"Kraken Market Analysis: {PAIR}", fontsize=16, fontweight='bold')
    ax.grid(True, which='major', color='white', alpha=0.5, zorder=1)
    ax.legend(loc='upper left', framealpha=1)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    clean_vector = []
    for i, row in df.iterrows():
        v = row['vector']
        clean_vector.append({
            "date": row['time'].strftime('%Y-%m'),
            "val": "N/A" if np.isnan(v) else int(v)
        })

    return base64.b64encode(buf.getvalue()).decode(), clean_vector

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty:
        return "<h1>Error fetching data from Kraken API.</h1>"
        
    plot_base64, vector_list = create_plot_and_vector(df)
    current_price = f"${df['close'].iloc[-1]:,.2f}"
    
    # Pass current_price and the vector to the template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Vector Analysis</title>
        <style>
            body { font-family: -apple-system, sans-serif; background: #f1f2f6; margin: 0; padding: 20px; color: #2d3436; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
            .header { text-align: center; margin-bottom: 30px; }
            .chart-box { text-align: center; margin-bottom: 30px; }
            img { max-width: 100%; border-radius: 8px; border: 1px solid #eee; }
            
            .vector-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(85px, 1fr)); 
                gap: 8px; 
                max-height: 400px; 
                overflow-y: auto; 
                padding: 15px;
                background: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #eee;
            }
            .cell { padding: 8px; border-radius: 6px; text-align: center; font-size: 11px; border: 1px solid #ddd; }
            .val-neg1 { background: #ff7675; color: white; border-color: #d63031; }
            .val-0 { background: #dfe6e9; color: #636e72; border-color: #b2bec3; }
            .val-pos1 { background: #55efc4; color: #006266; border-color: #00b894; font-weight: bold; }
            .val-na { background: white; color: #ccc; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Bitcoin Market State Analysis</h1>
                <p>Kraken Pair: <strong>XBTUSD</strong> | Current Price: <strong>{{ current_price }}</strong></p>
            </div>
            
            <div class="chart-box">
                <img src="data:image/png;base64,{{ plot_base64 }}">
            </div>

            <h3>Monthly Signal Vector [-1, 0, 1]</h3>
            <p style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                <b>-1:</b> High to Low | <b>0:</b> Low to Stability | <b>1:</b> Stability to High
            </p>
            <div class="vector-grid">
                {% for item in vector_list %}
                <div class="cell {% if item.val == -1 %}val-neg1{% elif item.val == 0 %}val-0{% elif item.val == 1 %}val-pos1{% else %}val-na{% endif %}">
                    {{ item.date }}<br>
                    <span style="font-size: 16px;">{{ item.val }}</span>
                </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, 
                                  plot_base64=plot_base64, 
                                  vector_list=vector_list, 
                                  current_price=current_price)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)