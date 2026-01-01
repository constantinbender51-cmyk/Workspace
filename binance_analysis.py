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
from datetime import datetime, timedelta

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

def analyze_structure(df):
    """
    Applies Market Structure Logic using a STATE MACHINE to ensure no gaps.
    """
    if df.empty: return df, [], [], []
    df = df.copy()
    
    # 1. High detection (2yr radius)
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    highs = df[df['close'] == df['h_max']].index.tolist()

    # 2. Stability detection (3m centered window)
    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stabs = []
    for p in highs:
        # Search backwards from peak for closest stability
        for i in range(p - 1, 1, -1):
            if i in df.index and df.loc[i, 'range_3m_centered'] <= 0.50:
                stabs.append(i); break
    
    # 3. Low detection (1yr radius)
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    lows = df[df['low'] == df['l_min']].index.tolist()

    # 4. State Machine Vector Generation
    # Combine all events into a time-sorted line
    events = sorted([(i, 'H') for i in highs] + [(i, 'S') for i in stabs] + [(i, 'L') for i in lows])
    
    vector = np.full(len(df), np.nan)
    
    # Current State Tracker: Defaults to NaN until first event
    current_state = np.nan
    
    # Iterate through every month in the dataframe
    event_map = dict(events) # Map index -> Event Type
    
    for i in range(len(df)):
        # Check if an event happens this month
        if i in event_map:
            etype = event_map[i]
            if etype == 'H': current_state = -1 # Peak hit -> Start Correction
            elif etype == 'L': current_state = 0 # Low hit -> Start Accumulation
            elif etype == 'S': current_state = 1 # Stability hit -> Start Expansion
            
        # Assign state
        vector[i] = current_state
            
    df['vector'] = vector
    return df, highs, stabs, lows

def generate_warped_reality(df):
    """
    Creates an alternate reality by:
    1. Time Warp: Expanding/contracting month duration.
    2. RE-ANALYZING STRUCTURE on the new timeline.
    3. Persistent Return Warp: Modifying prices.
    """
    if df.empty: return pd.DataFrame()
    
    daily_stream = []
    
    for _, row in df.iterrows():
        # Time Warp
        time_warp = random.uniform(-1, 1)
        days_in_month = int(30 + (30 * time_warp))
        days_in_month = max(1, days_in_month)
        
        # We NO LONGER carry the old vector. We will recalculate it.
        for _ in range(days_in_month):
            daily_stream.append({
                'open': row['open'], 
                'high': row['high'], 
                'low': row['low'], 
                'close': row['close']
            })
            
    warped_df = pd.DataFrame(daily_stream)
    if warped_df.empty: return pd.DataFrame()

    start_date = df['time'].iloc[0]
    warped_df['time'] = pd.date_range(start=start_date, periods=len(warped_df), freq='D')
    warped_df.set_index('time', inplace=True)
    
    # Resample (Re-bucket to 30 days)
    warped_monthly = warped_df.resample('30D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()

    # --- NEW: Assign Signals AFTER Time Warp but BEFORE Price Warp ---
    warped_monthly, _, _, _ = analyze_structure(warped_monthly)

    # Persistent Return Randomization
    original_returns = warped_monthly['close'].pct_change().fillna(0)
    log_multiplier = 0.0 
    new_prices = [warped_monthly['close'].iloc[0]]
    
    for i in range(1, len(warped_monthly)):
        shock = random.uniform(-0.06, 0.06) 
        log_multiplier += shock
        
        current_multiplier = np.exp(log_multiplier)
        original_r = original_returns.iloc[i]
        new_r = original_r * current_multiplier
        new_r = max(-0.98, new_r)
        
        new_price = new_prices[-1] * (1 + new_r)
        new_prices.append(new_price)

    old_closes = warped_monthly['close'].values
    warped_monthly['close'] = new_prices
    old_closes[old_closes == 0] = 1e-9
    ratios = warped_monthly['close'] / old_closes
    
    warped_monthly['open'] *= ratios
    warped_monthly['high'] *= ratios
    warped_monthly['low'] *= ratios
    
    return warped_monthly

def create_plot_and_vector(df):
    if df.empty: return None, []
    
    # Analyze Structure on ORIGINAL data
    df, highs, stabs, lows = analyze_structure(df)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # --- PLOT 1: Actual Reality ---
    ax1.set_facecolor('#e0e0e0') 
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2.5, label='Actual BTC Price', zorder=3)
    ax1.scatter(df.loc[highs, 'time'], df.loc[highs, 'close'], color='#ff4757', s=120, marker='v', edgecolors='black', zorder=5)
    ax1.scatter(df.loc[stabs, 'time'], df.loc[stabs, 'close'], color='#8e44ad', s=120, marker='d', edgecolors='white', zorder=5)
    ax1.scatter(df.loc[lows, 'time'], df.loc[lows, 'low'], color='#2ed573', s=120, marker='^', edgecolors='black', zorder=5)
    ax1.set_yscale('linear')
    ax1.set_title("Reality A: Actual Historical Data", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2: Alternate Reality ---
    ax2.set_facecolor('#f0f0f0') 
    w = generate_warped_reality(df)
    
    # Plot Price
    ax2.plot(w['time'], w['close'], color='#2980b9', linewidth=2, alpha=1.0, label='Simulation Alpha', zorder=5)

    # Background Coloring (State Machine Logic)
    w['group'] = (w['vector'] != w['vector'].shift()).cumsum()
    w['next_time'] = w['time'].shift(-1)
    w.loc[w.index[-1], 'next_time'] = w.loc[w.index[-1], 'time'] + timedelta(days=30)

    groups = w.groupby('group').agg({
        'time': 'first',
        'next_time': 'last',
        'vector': 'first'
    })

    for _, row in groups.iterrows():
        state = row['vector']
        if not np.isnan(state): 
            face_color = None
            if state == 1:   face_color = '#d4edda' # Green
            elif state == -1: face_color = '#f8d7da' # Red
            elif state == 0:  face_color = '#e2e3e5' # Grey
            
            if face_color:
                ax2.axvspan(row['time'], row['next_time'], color=face_color, alpha=0.6, zorder=1)

    ax2.set_yscale('linear')
    ax2.set_title("Reality B: Signals Re-calculated on Warped Timeline", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Price (USD)")
    
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='#2980b9', lw=2, label='Sim Price'),
        Patch(facecolor='#d4edda', edgecolor='#c3e6cb', label='Expansion (+1)'),
        Patch(facecolor='#f8d7da', edgecolor='#f5c6cb', label='Correction (-1)'),
        Patch(facecolor='#e2e3e5', edgecolor='#d6d8db', label='Accumulation (0)')
    ]
    ax2.legend(handles=legend_elements, loc='upper left')
    ax2.grid(True, alpha=0.3, zorder=2)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110)
    plt.close()
    
    clean_v = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if np.isnan(r['vector']) else int(r['vector'])} for i, r in df.iterrows()]
    return base64.b64encode(buf.getvalue()).decode(), clean_v

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "<h1>API Data Fetch Error</h1>"
    p, v = create_plot_and_vector(df)
    
    html = """
    <!DOCTYPE html><html><head><title>BTC Reality Warp</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f1f2f6; padding: 20px; }
        .container { max-width: 1300px; margin: auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2d3436; }
        img { width: 100%; border-radius: 10px; margin: 20px 0; border: 1px solid #ddd; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 8px; height: 250px; overflow-y: scroll; background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
        .c { padding: 10px; text-align: center; font-size: 11px; border: 1px solid #ddd; border-radius: 4px; }
        .v-1 { background: #ff7675; color: white; } .v0 { background: #dfe6e9; color: #636e72; } .v1 { background: #55efc4; color: #006266; font-weight: bold; }
        .btn { display: block; width: fit-content; margin: 0 auto 20px; padding: 12px 24px; background: #6c5ce7; color: #fff; text-decoration: none; border-radius: 8px; font-weight: bold; transition: background 0.2s; }
        .btn:hover { background: #5b4cc4; }
        .desc { background: #fffbe6; padding: 15px; border-radius: 8px; border: 1px solid #ffe58f; margin-bottom: 20px; font-size: 0.9em; }
    </style></head><body>
    <div class="container">
        <h1>Bitcoin: Seamless Signal Warp (Re-calculated)</h1>
        <div class="desc">
            <strong>Signal Legend:</strong><br>
            <span style="background:#d4edda; padding:2px 5px">Green (+1)</span> Expansion | 
            <span style="background:#f8d7da; padding:2px 5px">Red (-1)</span> Correction | 
            <span style="background:#e2e3e5; padding:2px 5px">Grey (0)</span> Accumulation
        </div>
        <a href="/" class="btn">Generate New Reality</a>
        <img src="data:image/png;base64,{{p}}">
        <h3>Reality A Monthly Vector</h3>
        <div class="grid">
            {% for i in v %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><span style="font-size:1.2em;">{{i.val}}</span></div>{% endfor %}
        </div>
    </div></body></html>
    """
    return render_template_string(html, p=p, v=v)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)