import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import random
from flask import Flask, render_template_string
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
PAIR = 'XBTUSD' 
INTERVAL = 10080 # Weekly data
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
        
        # Resample to Monthly
        df.set_index('time', inplace=True)
        df_m = df.resample('MS').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        
        return df_m
    except Exception as e:
        print(f"Fetch error: {e}")
        return pd.DataFrame()

def generate_warped_reality(df):
    """
    Creates an alternate reality by expanding/contracting each month
    by a random factor between -1 and 1.
    """
    if df.empty: return pd.DataFrame()
    
    daily_stream = []
    
    # 1. Expand and Warp
    # We iterate through the monthly dataframe.
    # Each row represents "1 month" of price action.
    for _, row in df.iterrows():
        # Random warp factor between -1 and 1
        warp = random.uniform(-1, 1)
        
        # Base days = 30. Calculate delta.
        # e.g., 0.1 -> +3 days. -0.2 -> -6 days.
        days_in_month = int(30 + (30 * warp))
        
        # Safety: Ensure at least 1 day of existence for that price period
        days_in_month = max(1, days_in_month)
        
        # Create 'days_in_month' identical records
        # This simulates the price staying in that month's range for longer/shorter
        for _ in range(days_in_month):
            daily_stream.append({
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })
            
    # 2. Reconstruct DataFrame
    warped_df = pd.DataFrame(daily_stream)
    
    # 3. Assign new Synthetic Timeline
    # We start from the same start date as the original, but the end date will drift
    start_date = df['time'].iloc[0]
    warped_df['time'] = pd.date_range(start=start_date, periods=len(warped_df), freq='D')
    
    # 4. Resample back to "Monthly" (30-day buckets)
    # This causes the "bleed" effect where Month 2 might now contain data from Month 1
    warped_df.set_index('time', inplace=True)
    
    warped_monthly = warped_df.resample('30D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()
    
    return warped_monthly

def create_plot_and_vector(df):
    if df.empty: return None, []

    # --- Generate 3 Alternate Realities ---
    warped_df_1 = generate_warped_reality(df)
    warped_df_2 = generate_warped_reality(df)
    warped_df_3 = generate_warped_reality(df)

    # --- Setup Figure (2 Subplots) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), facecolor='#f1f2f6')
    
    # === PLOT 1: Original Reality (Vector Analysis) ===
    ax1.set_facecolor('#e0e0e0') 

    # 1. Identify Key Points
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    local_high_indices = df[df['close'] == df['h_max']].index.tolist()

    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stability_indices = []
    for peak_idx in local_high_indices:
        for i in range(peak_idx - 1, 1, -1):
            if df.loc[i, 'range_3m_centered'] <= 0.50:
                stability_indices.append(i)
                break
    
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    local_low_indices = df[df['low'] == df['l_min']].index.tolist()

    # 2. Build Vector
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
        if not np.isnan(val): vector[curr_idx:next_idx] = val

    df['vector'] = vector

    # 3. Draw Plot 1
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2, zorder=3, label='BTC Price (Actual)')
    ax1.scatter(df.loc[local_high_indices, 'time'], df.loc[local_high_indices, 'close'], 
               color='#ff4757', s=140, marker='v', edgecolors='black', zorder=5, label='Local High')
    ax1.scatter(df.loc[stability_indices, 'time'], df.loc[stability_indices, 'close'], 
               color='#8e44ad', s=130, marker='d', edgecolors='white', zorder=5, label='Stability Point')
    ax1.scatter(df.loc[local_low_indices, 'time'], df.loc[local_low_indices, 'low'], 
               color='#2ed573', s=140, marker='^', edgecolors='black', zorder=5, label='Local Low')

    ax1.set_yscale('linear')
    ax1.set_title(f"Reality A: Actual Market Structure ({PAIR})", fontsize=16, fontweight='bold')
    ax1.grid(True, which='major', color='white', alpha=0.5, zorder=1)
    ax1.legend(loc='upper left', framealpha=1)

    # === PLOT 2: Time Warped Realities (3 Simulations) ===
    ax2.set_facecolor('#dcdde1') 
    
    # Plot 3 distinct warped simulations
    ax2.plot(warped_df_1['time'], warped_df_1['close'], color='#2980b9', linewidth=1.2, alpha=0.9, zorder=3, label='Simulation A (Blue)')
    ax2.plot(warped_df_2['time'], warped_df_2['close'], color='#27ae60', linewidth=1.2, alpha=0.9, zorder=3, label='Simulation B (Green)')
    ax2.plot(warped_df_3['time'], warped_df_3['close'], color='#c0392b', linewidth=1.2, alpha=0.9, zorder=3, label='Simulation C (Red)')
    
    # Style
    ax2.set_yscale('linear')
    ax2.set_title("Reality B: 3 Randomized Time Warp Simulations", fontsize=16, fontweight='bold', color='#444')
    ax2.grid(True, which='major', color='white', alpha=0.5, zorder=1)
    ax2.legend(loc='upper left', framealpha=1)
    
    # Add annotation explaining the warp
    ax2.text(0.02, 0.95, "Method: 3 independent runs where monthly duration is warped by [-1.0, 1.0].\nLines diverge as random time dilations accumulate.", 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    if df.empty: return "<h1>Error fetching data from Kraken API.</h1>"
        
    plot_base64, vector_list = create_plot_and_vector(df)
    current_price = f"${df['close'].iloc[-1]:,.2f}"
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Time Warp Analysis</title>
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
                max-height: 300px; 
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
            
            .refresh-btn {
                display: inline-block;
                margin-top: 10px;
                padding: 10px 20px;
                background-color: #6c5ce7;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
            }
            .refresh-btn:hover { background-color: #5b4cc4; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Bitcoin Market: Actual vs Multi-Reality Warp</h1>
                <p>Kraken Pair: <strong>XBTUSD</strong> | Current Price: <strong>{{ current_price }}</strong></p>
                <a href="/" class="refresh-btn">Regenerate Time Warp</a>
            </div>
            
            <div class="chart-box">
                <img src="data:image/png;base64,{{ plot_base64 }}">
            </div>

            <h3>Reality A: Monthly Signal Vector [-1, 0, 1]</h3>
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