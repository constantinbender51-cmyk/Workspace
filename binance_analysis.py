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
    
    df['h_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    if len(df) > 24: df.loc[df.index[-24:], 'h_max'] = np.inf
    highs = df[df['close'] == df['h_max']].index.tolist()

    df['range_3m_centered'] = (df['high'].rolling(window=3, center=True).max() - 
                               df['low'].rolling(window=3, center=True).min()) / \
                              df['low'].rolling(window=3, center=True).min()
    
    stabs = []
    for p in highs:
        for i in range(p - 1, 1, -1):
            if i in df.index and df.loc[i, 'range_3m_centered'] <= 0.50:
                stabs.append(i); break
    
    df['l_min'] = df['low'].rolling(window=25, center=True, min_periods=13).min()
    if len(df) > 12: df.loc[df.index[-12:], 'l_min'] = -1.0
    lows = df[df['low'] == df['l_min']].index.tolist()

    events = sorted([(i, 'H') for i in highs] + [(i, 'S') for i in stabs] + [(i, 'L') for i in lows])
    vector = np.full(len(df), np.nan)
    
    for i in range(len(events) - 1):
        idx, t = events[i]; n_idx, n_t = events[i+1]
        val = np.nan
        if t == 'H' and n_t == 'L': val = -1
        elif t == 'L' and n_t == 'S': val = 0
        elif t == 'S' and n_t == 'H': val = 1
        elif t == 'H': val = -1
        elif t == 'L': val = 0
        elif t == 'S': val = 1
        if not np.isnan(val): vector[idx:n_idx] = val

    if events:
        last_idx, last_type = events[-1]
        last_val = -1 if last_type == 'H' else (0 if last_type == 'L' else 1)
        vector[last_idx:] = last_val
            
    df['vector'] = vector
    return df, highs, stabs, lows

def analyze_structure_new(df):
    """
    NEW LOGIC (For Random Reality):
    1. Peak: 2yr Past / 1yr Future radius, AND must be a NEW ATH.
    2. Low: Lowest price between confirmed Peaks OR between last Peak and End.
    3. Stable: 3/4 point (Time) between a Low and the month with highest 3-month return.
    """
    if df.empty: return df, [], [], []
    df = df.copy()
    
    # 1. Peak Detection (ATH Rule)
    df['h_max_window'] = df['close'].rolling(window=37, min_periods=13).max().shift(-12)
    if len(df) > 12: df.loc[df.index[-12:], 'h_max_window'] = np.inf
    df['expanding_ath'] = df['close'].expanding().max()
    
    peak_candidates = df[
        (df['close'] == df['h_max_window']) & 
        (df['close'] >= df['expanding_ath'])
    ].index.tolist()
    
    highs = []
    if peak_candidates:
        highs.append(peak_candidates[0])
        for p in peak_candidates[1:]:
            if (p - highs[-1]) > 12 and df.loc[p, 'close'] > df.loc[highs[-1], 'close']:
                highs.append(p)
    
    # 2. Low Detection (Including trailing low)
    lows = []
    if highs:
        for i in range(len(highs)):
            current_peak = highs[i]
            start_search = current_peak + 1
            # Search until next peak OR end of data
            end_search = highs[i+1] if i < len(highs) - 1 else len(df)
            
            if start_search < end_search:
                segment = df.iloc[start_search:end_search]
                if not segment.empty:
                    # idxmin gives the label of the minimum price
                    local_low = segment['close'].idxmin()
                    lows.append(local_low)

    # 3. Stability Detection
    df['ret_3m'] = df['close'].pct_change(periods=3).fillna(0)
    stabs = []
    for low_idx in lows:
        # We only define stability if this low is followed by a confirmed Peak
        next_peaks = [h for h in highs if h > low_idx]
        if not next_peaks: continue
        target_peak = next_peaks[0]
        
        segment = df.iloc[low_idx:target_peak + 1]
        if segment.empty: continue
        
        max_3m_ret_idx = segment['ret_3m'].idxmax()
        delta = max_3m_ret_idx - low_idx
        three_quarter_idx = int(low_idx + (delta * 0.75))
        stabs.append(three_quarter_idx)
        
    stabs = sorted(list(set(stabs)))

    # --- Generate Vector (State Machine) ---
    events = sorted([(i, 'H') for i in highs] + [(i, 'S') for i in stabs] + [(i, 'L') for i in lows])
    vector = np.full(len(df), np.nan)
    
    current_state = np.nan
    event_map = dict(events)
    
    for i in range(len(df)):
        if i in event_map:
            etype = event_map[i]
            if etype == 'H': current_state = -1 # Peak -> Correction
            elif etype == 'L': current_state = 0 # Low -> Accumulation
            elif etype == 'S': current_state = 1 # Stable -> Expansion
        vector[i] = current_state
            
    df['vector'] = vector
    return df, highs, stabs, lows

def generate_warped_reality(df):
    if df.empty: return pd.DataFrame()
    daily_stream = []
    for _, row in df.iterrows():
        time_warp = random.uniform(-1, 1)
        days_in_month = max(1, int(30 + (30 * time_warp)))
        for _ in range(days_in_month):
            daily_stream.append({'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']})
            
    warped_df = pd.DataFrame(daily_stream)
    start_date = df['time'].iloc[0]
    warped_df['time'] = pd.date_range(start=start_date, periods=len(warped_df), freq='D')
    warped_df.set_index('time', inplace=True)
    w_m = warped_df.resample('30D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()

    # Price Randomization
    original_returns = w_m['close'].pct_change().fillna(0)
    log_multiplier, new_prices = 0.0, [w_m['close'].iloc[0]]
    for i in range(1, len(w_m)):
        log_multiplier += random.uniform(-0.06, 0.06)
        new_r = max(-0.98, original_returns.iloc[i] * np.exp(log_multiplier))
        new_prices.append(new_prices[-1] * (1 + new_r))

    old_closes = w_m['close'].values
    w_m['close'] = new_prices
    ratios = w_m['close'] / np.where(old_closes == 0, 1e-9, old_closes)
    w_m['open'] *= ratios; w_m['high'] *= ratios; w_m['low'] *= ratios
    
    # Analyze structure
    w_m, _, _, _ = analyze_structure_new(w_m)
    return w_m

def create_plot_and_vector(df):
    if df.empty: return None, [], []
    df, highs, stabs, lows = analyze_structure_original(df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # Plot 1: Actual
    ax1.set_facecolor('#e0e0e0') 
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2, label='Actual Price')
    ax1.scatter(df.loc[highs, 'time'], df.loc[highs, 'close'], color='#ff4757', s=100, marker='v', edgecolors='black', zorder=5)
    ax1.scatter(df.loc[stabs, 'time'], df.loc[stabs, 'close'], color='#8e44ad', s=100, marker='d', edgecolors='white', zorder=5)
    ax1.scatter(df.loc[lows, 'time'], df.loc[lows, 'low'], color='#2ed573', s=100, marker='^', edgecolors='black', zorder=5)
    ax1.set_title("Reality A: Actual Historical Data", fontweight='bold')
    ax1.legend()

    # Plot 2: Random Reality
    ax2.set_facecolor('#f0f0f0') 
    w = generate_warped_reality(df)
    ax2.plot(w['time'], w['close'], color='#2980b9', linewidth=2, label='Sim Price', zorder=5)

    # Background Coloring
    w['group'] = (w['vector'] != w['vector'].shift()).cumsum()
    w['next_t'] = w['time'].shift(-1).fillna(w['time'].iloc[-1] + timedelta(days=30))
    groups = w.groupby('group').agg({'time': 'first', 'next_t': 'last', 'vector': 'first'})

    for _, row in groups.iterrows():
        if not np.isnan(row['vector']):
            color = '#f8d7da' if row['vector'] == -1 else ('#e2e3e5' if row['vector'] == 0 else '#d4edda')
            ax2.axvspan(row['time'], row['next_t'], color=color, alpha=0.6, zorder=1)

    ax2.set_title("Reality B: Randomized Signal Vector (Trailing Low Support)", fontweight='bold')
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2980b9', lw=2, label='Sim Price'),
        Patch(facecolor='#d4edda', label='Expansion (S -> P)'),
        Patch(facecolor='#f8d7da', label='Correction (P -> L)'),
        Patch(facecolor='#e2e3e5', label='Accumulation (L -> S)')
    ]
    ax2.legend(handles=legend_elements, loc='upper left')
    ax2.grid(True, alpha=0.3, zorder=2)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110); plt.close()
    
    v_a = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if np.isnan(r['vector']) else int(r['vector'])} for i, r in df.iterrows()]
    v_b = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if np.isnan(r['vector']) else int(r['vector'])} for i, r in w.iterrows()]
    
    return base64.b64encode(buf.getvalue()).decode(), v_a, v_b

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "<h1>API Data Fetch Error</h1>"
    p, v_a, v_b = create_plot_and_vector(df)
    html = """
    <!DOCTYPE html><html><head><title>BTC Reality Warp</title>
    <style>
        body { font-family: sans-serif; background: #f1f2f6; padding: 20px; }
        .container { max-width: 1400px; margin: auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
        img { width: 100%; border-radius: 10px; margin: 20px 0; }
        .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 5px; height: 200px; overflow-y: scroll; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #eee; }
        .c { padding: 8px; text-align: center; font-size: 10px; border: 1px solid #ddd; border-radius: 4px; }
        .v-1 { background: #ff7675; color: white; } .v0 { background: #dfe6e9; color: #636e72; } .v1 { background: #55efc4; color: #006266; font-weight: bold; }
        .btn { display: block; width: fit-content; margin: 0 auto 20px; padding: 12px 24px; background: #6c5ce7; color: #fff; text-decoration: none; border-radius: 8px; font-weight: bold; }
    </style></head><body>
    <div class="container">
        <h1>Bitcoin: Trailing Low Signal Warp</h1>
        <div style="background:#f8f9fa; padding:15px; border-radius:8px; border:1px solid #dee2e6; margin-bottom:20px;">
            <strong>State Progression:</strong> Following a Peak, the signal becomes <strong>Correction (-1)</strong>. 
            If a <strong>Low</strong> is identified (even if it's the last point in the data), the signal transitions to <strong>Accumulation (0)</strong>.
        </div>
        <a href="/" class="btn">Generate New Reality</a>
        <img src="data:image/png;base64,{{p}}">
        <div class="grid-container">
            <div>
                <h3>Reality A Vector</h3>
                <div class="grid">{% for i in v_a %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}</div>
            </div>
            <div>
                <h3>Reality B Vector</h3>
                <div class="grid">{% for i in v_b %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}</div>
            </div>
        </div>
    </div></body></html>"""
    return render_template_string(html, p=p, v_a=v_a, v_b=v_b)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)