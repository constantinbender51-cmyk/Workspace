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
    Creates an alternate reality by:
    1. Time Warp: Randomly expanding/contracting month duration.
    2. Return Warp: Randomly drifting the return multiplier, affecting all future price moves.
    """
    if df.empty: return pd.DataFrame()
    
    daily_stream = []
    
    # --- Step 1: Time Warp Expansion ---
    for _, row in df.iterrows():
        # Random time warp factor between -1 and 1
        time_warp = random.uniform(-1, 1)
        # Base days = 30. Calculate delta.
        days_in_month = int(30 + (30 * time_warp))
        days_in_month = max(1, days_in_month)
        
        # Create identical records (prices stay original for now)
        for _ in range(days_in_month):
            daily_stream.append({
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })
            
    # Reconstruct DataFrame & Resample to Monthly
    warped_df = pd.DataFrame(daily_stream)
    
    if warped_df.empty: return pd.DataFrame()

    start_date = df['time'].iloc[0]
    warped_df['time'] = pd.date_range(start=start_date, periods=len(warped_df), freq='D')
    warped_df.set_index('time', inplace=True)
    
    # Base Warped Monthly Data (Time shifted, but original price levels)
    warped_monthly = warped_df.resample('30D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()

    # --- Step 2: Persistent Return Randomization ---
    # We will modify the *returns* of this new timeline.
    
    # Calculate original percentage returns of the Close price
    # We use 'adj_close' concept effectively
    original_returns = warped_monthly['close'].pct_change().fillna(0)
    
    # Initialize the Persistent Multiplier
    cumulative_multiplier = 1.0
    
    # New Price Series starting from the first open
    new_prices = [warped_monthly['open'].iloc[0]]
    
    # Iterate through returns to build the new price path
    # We skip index 0 (start) and process returns from index 1 onwards
    for i in range(1, len(warped_monthly)):
        # 1. Random Shock to the Multiplier
        # Small shock range to prevent runaway exponential explosion, 
        # but enough to drift significantly over time.
        shock = random.uniform(-0.05, 0.05) 
        
        # Update the persistent state
        cumulative_multiplier *= (1 + shock)
        
        # 2. Modify the Return for this month
        original_r = original_returns.iloc[i]
        new_r = original_r * cumulative_multiplier
        
        # 3. Calculate New Close based on previous New Close
        prev_price = new_prices[-1]
        new_price = prev_price * (1 + new_r)
        new_prices.append(new_price)

    # Update the dataframe with the new Close prices
    # Note: We need to handle index 0 separately or just overwrite
    # Since new_prices matches length of warped_monthly (we started with Open[0] which roughly equals Close[0] for plotting flow, 
    # but strictly Close[0] is Close[0]. Let's just overwrite 'close' with new_prices)
    
    # Fix: new_prices[0] was set to Open[0]. It should probably be Close[0] unmodified?
    # Actually, let's just accept the small drift from Open[0] or set new_prices[0] = warped_monthly['close'].iloc[0]
    new_prices[0] = warped_monthly['close'].iloc[0]
    
    # Store old close to calculate ratios for O/H/L adjustment
    old_closes = warped_monthly['close'].values
    warped_monthly['close'] = new_prices
    
    # --- Step 3: Adjust Open/High/Low proportionally ---
    # We scale O/H/L by the ratio of (New Close / Old Close) to preserve candle structure relative to level
    # This assumes the intra-month volatility scales with the price level (Log-normal assumption)
    
    ratios = warped_monthly['close'] / old_closes
    warped_monthly['open'] *= ratios
    warped_monthly['high'] *= ratios
    warped_monthly['low'] *= ratios
    
    return warped_monthly

def analyze_structure(df):
    """
    Applies the Market Structure Logic to any given DataFrame (Actual or Warped).
    Returns: df (with vector col), high_indices, stability_indices, low_indices
    """
    if df.empty: return df, [], [], []

    # 1. Local Highs (2yr Radius)
    # Window 49 = ±24 months
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    local_high_indices = df[df['close'] == df['h_max']].index.tolist()

    # 2. Pre-Peak Stability Points (50% range, 1mo radius)
    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stability_indices = []
    for peak_idx in local_high_indices:
        # Look backwards from peak
        for i in range(peak_idx - 1, 1, -1):
            if df.loc[i, 'range_3m_centered'] <= 0.50:
                stability_indices.append(i)
                break
    
    # 3. Local Lows (1yr Radius)
    # Window 25 = ±12 months
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    local_low_indices = df[df['low'] == df['l_min']].index.tolist()

    # 4. Build Vector [-1, 0, 1]
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
        
        # Vector State Logic
        if curr_type == 'HIGH' and next_type == 'LOW': val = -1
        elif curr_type == 'LOW' and next_type == 'STABILITY': val = 0
        elif curr_type == 'STABILITY' and next_type == 'HIGH': val = 1
        
        if not np.isnan(val): vector[curr_idx:next_idx] = val

    df['vector'] = vector
    return df, local_high_indices, stability_indices, local_low_indices

def create_plot_and_vector(df):
    if df.empty: return None, []

    # --- Setup Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # ==========================================
    # PLOT 1: Actual Reality
    # ==========================================
    ax1.set_facecolor('#e0e0e0') 
    
    # Apply Analysis to Actual Data
    df, highs, stabs, lows = analyze_structure(df)

    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2, zorder=3, label='BTC Price (Actual)')
    ax1.scatter(df.loc[highs, 'time'], df.loc[highs, 'close'], color='#ff4757', s=140, marker='v', edgecolors='black', zorder=5, label='Local High')
    ax1.scatter(df.loc[stabs, 'time'], df.loc[stabs, 'close'], color='#8e44ad', s=130, marker='d', edgecolors='white', zorder=5, label='Stability Point')
    ax1.scatter(df.loc[lows, 'time'], df.loc[lows, 'low'], color='#2ed573', s=140, marker='^', edgecolors='black', zorder=5, label='Local Low')

    ax1.set_yscale('linear')
    ax1.set_title(f"Reality A: Actual Market Structure ({PAIR})", fontsize=16, fontweight='bold')
    ax1.grid(True, which='major', color='white', alpha=0.5, zorder=1)
    ax1.legend(loc='upper left', framealpha=1)

    # ==========================================
    # PLOT 2: Time Warped Realities (With Persistent Return Randomization)
    # ==========================================
    ax2.set_facecolor('#dcdde1') 
    
    # Simulation 1 (Blue)
    w1 = generate_warped_reality(df)
    w1, h1, s1, l1 = analyze_structure(w1)
    ax2.plot(w1['time'], w1['close'], color='#2980b9', linewidth=1, alpha=0.8, zorder=3, label='Sim A (Blue)')
    ax2.scatter(w1.loc[h1, 'time'], w1.loc[h1, 'close'], color='#2980b9', s=40, marker='v', edgecolors='black', zorder=4)
    ax2.scatter(w1.loc[s1, 'time'], w1.loc[s1, 'close'], color='#8e44ad', s=30, marker='d', edgecolors='white', zorder=4)
    ax2.scatter(w1.loc[l1, 'time'], w1.loc[l1, 'low'], color='#2980b9', s=40, marker='^', edgecolors='black', zorder=4)

    # Simulation 2 (Green)
    w2 = generate_warped_reality(df)
    w2, h2, s2, l2 = analyze_structure(w2)
    ax2.plot(w2['time'], w2['close'], color='#27ae60', linewidth=1, alpha=0.8, zorder=3, label='Sim B (Green)')
    ax2.scatter(w2.loc[h2, 'time'], w2.loc[h2, 'close'], color='#27ae60', s=40, marker='v', edgecolors='black', zorder=4)
    ax2.scatter(w2.loc[s2, 'time'], w2.loc[s2, 'close'], color='#8e44ad', s=30, marker='d', edgecolors='white', zorder=4)
    ax2.scatter(w2.loc[l2, 'time'], w2.loc[l2, 'low'], color='#27ae60', s=40, marker='^', edgecolors='black', zorder=4)

    # Simulation 3 (Red)
    w3 = generate_warped_reality(df)
    w3, h3, s3, l3 = analyze_structure(w3)
    ax2.plot(w3['time'], w3['close'], color='#c0392b', linewidth=1, alpha=0.8, zorder=3, label='Sim C (Red)')
    ax2.scatter(w3.loc[h3, 'time'], w3.loc[h3, 'close'], color='#c0392b', s=40, marker='v', edgecolors='black', zorder=4)
    ax2.scatter(w3.loc[s3, 'time'], w3.loc[s3, 'close'], color='#8e44ad', s=30, marker='d', edgecolors='white', zorder=4)
    ax2.scatter(w3.loc[l3, 'time'], w3.loc[l3, 'low'], color='#c0392b', s=40, marker='^', edgecolors='black', zorder=4)

    ax2.set_yscale('linear')
    ax2.set_title("Reality B: 3 Simulations (Time Warp + Persistent Return Randomization)", fontsize=16, fontweight='bold', color='#444')
    ax2.grid(True, which='major', color='white', alpha=0.5, zorder=1)
    ax2.legend(loc='upper left', framealpha=1)
    
    # Legend Text
    ax2.text(0.02, 0.95, "Method: Monthly Return Multiplier drifts randomly.\nImpact: A single shock affects all future price levels.", 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Extract Vector for Reality A (Actual) to display in grid
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
                <a href="/" class="refresh-btn">Regenerate Time & Return Warp</a>
            </div>
            
            <div class="chart-box">
                <img src="data:image/png;base64,{{ plot_base64 }}">
            </div>

            <h3>Reality A: Monthly Signal Vector [-1, 0, 1]</h3>
            <div class="vector-grid">
                {% for item in vector_list %}
                <div class="cell {% if item.val == -1 %}val-neg1{% elif item.val == 0 %}val-0{% elif item.val == 1 %}val-pos1{% elif item.val == 'N/A' %}val-na{% endif %}">
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