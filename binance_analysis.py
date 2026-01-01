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

def analyze_structure_original(df):
    """
    ORIGINAL LOGIC: 2yr Peak, 1yr Low, Volatility Stability.
    """
    if df.empty: return df, [], [], []
    df = df.copy()
    
    # 1. High detection (2yr radius)
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    highs = df[df['close'] == df['h_max']].index.tolist()

    # 2. Stability detection (Old Method: Volatility < 50% in 3mo window)
    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stabs = []
    for p in highs:
        for i in range(p - 1, 1, -1):
            if i in df.index and df.loc[i, 'range_3m_centered'] <= 0.50:
                stabs.append(i); break
    
    # 3. Low detection (Old Method: 1yr radius)
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    lows = df[df['low'] == df['l_min']].index.tolist()

    # 4. Vector State Machine (Gapless)
    events = sorted([(i, 'H') for i in highs] + [(i, 'S') for i in stabs] + [(i, 'L') for i in lows])
    vector = np.full(len(df), np.nan)
    
    for i in range(len(events) - 1):
        idx, t = events[i]
        n_idx, n_t = events[i+1]
        val = np.nan
        if t == 'H' and n_t == 'L': val = -1
        elif t == 'L' and n_t == 'S': val = 0
        elif t == 'S' and n_t == 'H': val = 1
        # Fallbacks for sequence breaks
        elif t == 'H': val = -1
        elif t == 'L': val = 0
        elif t == 'S': val = 1
        if not np.isnan(val): vector[idx:n_idx] = val

    if events:
        last_idx, last_type = events[-1]
        last_val = np.nan
        if last_type == 'H': last_val = -1
        elif last_type == 'L': last_val = 0
        elif last_type == 'S': last_val = 1
        if not np.isnan(last_val): vector[last_idx:] = last_val
            
    df['vector'] = vector
    return df, highs, stabs, lows

def analyze_structure_new(df):
    """
    NEW LOGIC (For Random Reality Only):
    1. Peak: Asymmetric Window (2 years Past, 1 year Future).
       - Rolling Window: 37 months (24 past + 1 curr + 12 future).
       - Shifted by -12 to align the window correctly.
    """
    if df.empty: return df, [], [], []
    df = df.copy()
    
    # 1. High detection (2yr Past, 1yr Future)
    df['h_max'] = df['close'].rolling(window=37, min_periods=13).max().shift(-12)
    
    # Invalidate tail (last 1 year / 12 months) as we can't see the required future
    if len(df) > 12: df.loc[df.index[-12:], 'h_max'] = np.inf
    
    # Find Peaks
    peak_candidates = df[df['close'] == df['h_max']].index.tolist()
    
    # --- Tie Breaking Logic ---
    final_peaks = []
    if peak_candidates:
        final_peaks.append(peak_candidates[0])
        for p in peak_candidates[1:]:
            if (p - final_peaks[-1]) > 12:
                final_peaks.append(p)
    
    highs = final_peaks

    # Disable Lows and Stabs for now
    lows = []
    stabs = []

    # Vector generation (Peaks marked as -1)
    vector = np.full(len(df), np.nan)
    for h in highs:
        vector[h] = -1
            
    df['vector'] = vector
    return df, highs, stabs, lows

def generate_warped_reality(df):
    if df.empty: return pd.DataFrame()
    
    daily_stream = []
    
    # --- Step 1: Time Warp Expansion ---
    for _, row in df.iterrows():
        time_warp = random.uniform(-1, 1)
        days_in_month = int(30 + (30 * time_warp))
        days_in_month = max(1, days_in_month)
        
        for _ in range(days_in_month):
            daily_stream.append({
                'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']
            })
            
    warped_df = pd.DataFrame(daily_stream)
    if warped_df.empty: return pd.DataFrame()

    start_date = df['time'].iloc[0]
    warped_df['time'] = pd.date_range(start=start_date, periods=len(warped_df), freq='D')
    warped_df.set_index('time', inplace=True)
    
    # Resample
    warped_monthly = warped_df.resample('30D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()

    # --- Step 2: Persistent Return Randomization ---
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

    # --- Step 3: Assign Signals (Using NEW Logic - Peaks Only) ---
    # Moved to AFTER price randomization so peaks match the new reality's structure
    warped_monthly, _, _, _ = analyze_structure_new(warped_monthly)
    
    return warped_monthly

def create_plot_and_vector(df):
    if df.empty: return None, []
    
    # Analyze Structure on ORIGINAL data using ORIGINAL logic
    df, highs, stabs, lows = analyze_structure_original(df)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # --- PLOT 1: Actual Reality (Original Logic) ---
    ax1.set_facecolor('#e0e0e0') 
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2.5, label='Actual BTC Price', zorder=3)
    ax1.scatter(df.loc[highs, 'time'], df.loc[highs, 'close'], color='#ff4757', s=120, marker='v', edgecolors='black', zorder=5)
    ax1.scatter(df.loc[stabs, 'time'], df.loc[stabs, 'close'], color='#8e44ad', s=120, marker='d', edgecolors='white', zorder=5)
    ax1.scatter(df.loc[lows, 'time'], df.loc[lows, 'low'], color='#2ed573', s=120, marker='^', edgecolors='black', zorder=5)
    ax1.set_yscale('linear')
    ax1.set_title("Reality A: Actual Historical Data (Original Logic)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2: Alternate Reality (Peaks Only) ---
    ax2.set_facecolor('#f0f0f0') 
    w = generate_warped_reality(df)
    
    # Plot Price
    ax2.plot(w['time'], w['close'], color='#2980b9', linewidth=2, alpha=1.0, label='Simulation Alpha', zorder=5)

    # Mark Peaks (Where vector == -1)
    peak_rows = w[w['vector'] == -1]
    
    for _, row in peak_rows.iterrows():
        ax2.axvline(x=row['time'], color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    ax2.set_yscale('linear')
    ax2.set_title("Reality B: Peaks Only (2yr Past, 1yr Future) - Marked Black", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Price (USD)")
    
    ax2.legend(loc='upper left')
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
        <h1>Bitcoin: Two Realities, Two Logic Sets</h1>
        <div class="desc">
            <strong>Original Reality:</strong> Standard Signal Logic.<br>
            <strong>Random Reality:</strong> Peaks Only (Asymmetric 2yr Past / 1yr Future).
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