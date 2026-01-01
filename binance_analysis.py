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
    2. Persistent Return Warp: Log-normal random walk for the return multiplier.
    """
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
    
    warped_monthly = warped_df.resample('30D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()

    # --- Step 2: Persistent Return Randomization ---
    # To prevent the "only smaller" bias, we use a random walk in LOG space.
    original_returns = warped_monthly['close'].pct_change().fillna(0)
    
    # We track the 'log_multiplier' to ensure a balanced geometric walk
    log_multiplier = 0.0 
    new_prices = [warped_monthly['close'].iloc[0]]
    
    for i in range(1, len(warped_monthly)):
        # Random shock to the log-multiplier
        shock = random.uniform(-0.1, 0.1) 
        log_multiplier += shock
        
        # Convert back to linear space
        # e.g. log_mult of 0.69 -> multiplier of ~2.0
        current_multiplier = np.exp(log_multiplier)
        
        original_r = original_returns.iloc[i]
        new_r = original_r * current_multiplier
        
        # Clip returns at -99% to prevent negative prices
        new_r = max(-0.99, new_r)
        
        new_price = new_prices[-1] * (1 + new_r)
        new_prices.append(new_price)

    # Rebuild candle OHLC proportionally
    old_closes = warped_monthly['close'].values
    warped_monthly['close'] = new_prices
    ratios = warped_monthly['close'] / old_closes
    
    warped_monthly['open'] *= ratios
    warped_monthly['high'] *= ratios
    warped_monthly['low'] *= ratios
    
    return warped_monthly

def analyze_structure(df):
    if df.empty: return df, [], [], []
    df = df.copy()
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    highs = df[df['close'] == df['h_max']].index.tolist()

    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stabs = []
    for p in highs:
        for i in range(p - 1, 1, -1):
            if df.loc[i, 'range_3m_centered'] <= 0.50:
                stabs.append(i); break
    
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    lows = df[df['low'] == df['l_min']].index.tolist()

    events = sorted([(i, 'H') for i in highs] + [(i, 'S') for i in stabs] + [(i, 'L') for i in lows])
    vector = np.full(len(df), np.nan)
    for i in range(len(events) - 1):
        idx, t = events[i]; n_idx, n_t = events[i+1]
        v = -1 if t=='H' and n_t=='L' else (0 if t=='L' and n_t=='S' else (1 if t=='S' and n_t=='H' else np.nan))
        if not np.isnan(v): vector[idx:n_idx] = v
    df['vector'] = vector
    return df, highs, stabs, lows

def create_plot_and_vector(df):
    if df.empty: return None, []
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # Actual Reality
    ax1.set_facecolor('#e0e0e0') 
    df, highs, stabs, lows = analyze_structure(df)
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2, label='Actual Price', zorder=3)
    ax1.scatter(df.loc[highs, 'time'], df.loc[highs, 'close'], color='#ff4757', s=100, marker='v', edgecolors='black', zorder=5)
    ax1.scatter(df.loc[stabs, 'time'], df.loc[stabs, 'close'], color='#8e44ad', s=100, marker='d', edgecolors='white', zorder=5)
    ax1.scatter(df.loc[lows, 'time'], df.loc[lows, 'low'], color='#2ed573', s=100, marker='^', edgecolors='black', zorder=5)
    ax1.set_yscale('linear')
    ax1.set_title("Reality A: Actual (Linear Scale)", fontweight='bold')
    ax1.legend()

    # Warped Realities
    ax2.set_facecolor('#dcdde1') 
    colors = ['#2980b9', '#27ae60', '#c0392b']
    for i, color in enumerate(colors):
        w = generate_warped_reality(df)
        w, wh, ws, wl = analyze_structure(w)
        ax2.plot(w['time'], w['close'], color=color, linewidth=1, alpha=0.7, label=f'Sim {chr(65+i)}')
        # Smaller markers for simulations
        ax2.scatter(w.loc[wh, 'time'], w.loc[wh, 'close'], color=color, s=20, marker='v')
        ax2.scatter(w.loc[ws, 'time'], w.loc[ws, 'close'], color='#8e44ad', s=15, marker='d')
        ax2.scatter(w.loc[wl, 'time'], w.loc[wl, 'low'], color=color, s=20, marker='^')

    # Using LOG SCALE for the simulations so we can see all of them regardless of outcome
    ax2.set_yscale('log')
    ax2.set_title("Reality B: 3 Sims (Persistent Return Drift - LOG SCALE)", fontweight='bold')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.2)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    clean_v = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if np.isnan(r['vector']) else int(r['vector'])} for i, r in df.iterrows()]
    return base64.b64encode(buf.getvalue()).decode(), clean_v

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "<h1>API Error</h1>"
    p, v = create_plot_and_vector(df)
    
    html = """
    <!DOCTYPE html><html><head><title>BTC Warp</title>
    <style>
        body { font-family: sans-serif; background: #f1f2f6; padding: 20px; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 15px; }
        img { width: 100%; border-radius: 8px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 5px; height: 200px; overflow: auto; background: #eee; padding: 10px; }
        .c { padding: 5px; text-align: center; font-size: 10px; border: 1px solid #ccc; }
        .v-1 { background: #ff7675; } .v0 { background: #dfe6e9; } .v1 { background: #55efc4; }
        .btn { display: inline-block; padding: 10px 20px; background: #6c5ce7; color: #fff; text-decoration: none; border-radius: 5px; margin-bottom: 20px;}
    </style></head><body>
    <div class="container">
        <h1>Bitcoin: Multi-Reality Persistent Drift</h1>
        <a href="/" class="btn">New Simulations</a>
        <img src="data:image/png;base64,{{p}}">
        <h3>Actual Signal Vector</h3>
        <div class="grid">
            {% for i in v %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}
        </div>
    </div></body></html>
    """
    return render_template_string(html, p=p, v=v)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)