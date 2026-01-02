import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import random
import time
from flask import Flask, render_template_string, Response
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

# --- ORIGINAL LOGIC (Reality A) ---
def analyze_structure_original(df):
    if df.empty: return df, [], [], []
    
    # --- Bearish Extension Logic ---
    # To avoid 'future blindness', we extend the dataframe with a price crash.
    # Radius is 24 months (window 49 = 24 past + 24 future).
    # We append 24 months of 0 price.
    orig_len = len(df)
    last_date = df['time'].iloc[-1]
    
    extension_dates = [last_date + timedelta(days=30*(i+1)) for i in range(24)]
    extension = pd.DataFrame({
        'time': extension_dates,
        'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0
    })
    
    df_ext = pd.concat([df, extension], ignore_index=True)
    
    # 1. High (2yr radius)
    df_ext['h_max'] = df_ext['close'].rolling(window=49, center=True, min_periods=25).max()
    # Note: No tail invalidation needed because we have the dummy future
    
    highs = df_ext[df_ext['close'] == df_ext['h_max']].index.tolist()

    # 2. Stability (Volatility < 50% in 3mo window)
    df_ext['range_3m_centered'] = (df_ext['high'].rolling(window=3, center=True).max() - 
                                   df_ext['low'].rolling(window=3, center=True).min()) / \
                                  df_ext['low'].rolling(window=3, center=True).min()
    
    stabs = []
    for p in highs:
        for i in range(p - 1, 1, -1):
            if i in df_ext.index and df_ext.loc[i, 'range_3m_centered'] <= 0.50:
                stabs.append(i); break
    
    # 3. Low (1yr radius)
    df_ext['l_min'] = df_ext['low'].rolling(window=25, center=True, min_periods=13).min()
    lows = df_ext[df_ext['low'] == df_ext['l_min']].index.tolist()

    # 4. Vector State Machine
    events = sorted([(i, 'H') for i in highs] + [(i, 'S') for i in stabs] + [(i, 'L') for i in lows])
    vector = np.full(len(df_ext), np.nan)
    
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
            
    # Slice back to original length
    df = df.copy()
    df['signal'] = vector[:orig_len]
    
    # Filter indices to be within range
    highs = [x for x in highs if x < orig_len]
    stabs = [x for x in stabs if x < orig_len]
    lows = [x for x in lows if x < orig_len]
    
    return df, highs, stabs, lows

# --- NEW LOGIC (Reality B) ---
def analyze_structure_new(df):
    if df.empty: return df, [], [], []
    
    # --- Bearish Extension Logic ---
    # Radius is 1 year future (window 37 = 24 past + 12 future).
    # We append 12 months of 0 price.
    orig_len = len(df)
    
    # Handle cases where df might not have 'time' column (e.g. from vectorized gen)
    # If no time, we just make up indices
    if 'time' in df.columns:
        last_date = df['time'].iloc[-1]
        extension_dates = [last_date + timedelta(days=30*(i+1)) for i in range(12)]
    else:
        extension_dates = [i for i in range(12)] # Dummy
        
    extension = pd.DataFrame({
        'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0
    }, index=range(orig_len, orig_len+12))
    
    if 'time' in df.columns:
        extension['time'] = extension_dates
        
    df_ext = pd.concat([df, extension], ignore_index=True)
    
    # 1. Peak Detection (ATH Rule)
    # Window 37 shifted -12 looks 12 months ahead.
    df_ext['h_max_window'] = df_ext['close'].rolling(window=37, min_periods=13).max().shift(-12)
    # No tail invalidation needed
    
    df_ext['expanding_ath'] = df_ext['close'].expanding().max()
    
    peak_candidates = df_ext[
        (df_ext['close'] == df_ext['h_max_window']) & 
        (df_ext['close'] >= df_ext['expanding_ath'])
    ].index.tolist()
    
    highs = []
    if peak_candidates:
        highs.append(peak_candidates[0])
        for p in peak_candidates[1:]:
            if (p - highs[-1]) > 12 and df_ext.loc[p, 'close'] > df_ext.loc[highs[-1], 'close']:
                highs.append(p)
    
    # 2. Low Detection
    lows = []
    if highs:
        for i in range(len(highs)):
            current_peak = highs[i]
            start_search = current_peak + 1
            # Search until next peak OR end of EXTENDED data
            end_search = highs[i+1] if i < len(highs) - 1 else len(df_ext)
            
            if start_search < end_search:
                segment = df_ext.iloc[start_search:end_search]
                if not segment.empty:
                    lows.append(segment['close'].idxmin())

    # 3. Stability Detection (Abs 3-Month Return 3/4 Point)
    df_ext['abs_ret_3m'] = df_ext['close'].diff(periods=3).abs().fillna(0)
    stabs = []
    
    for low_idx in lows:
        next_peaks = [h for h in highs if h > low_idx]
        if not next_peaks: continue
        target_peak = next_peaks[0]
        
        segment = df_ext.iloc[low_idx:target_peak + 1]
        if segment.empty: continue
        
        max_3m_abs_idx = segment['abs_ret_3m'].idxmax()
        delta = max_3m_abs_idx - low_idx
        three_quarter_idx = int(low_idx + (delta * 0.75))
        stabs.append(three_quarter_idx)
        
    stabs = sorted(list(set(stabs)))

    # --- Vector State Machine ---
    events = sorted([(i, 'H') for i in highs] + [(i, 'S') for i in stabs] + [(i, 'L') for i in lows])
    vector = np.full(len(df_ext), np.nan)
    
    current_state = np.nan
    event_map = dict(events)
    
    for i in range(len(df_ext)):
        if i in event_map:
            etype = event_map[i]
            if etype == 'H': current_state = -1
            elif etype == 'L': current_state = 0
            elif etype == 'S': current_state = 1
        vector[i] = current_state
            
    # Slice back
    df = df.copy()
    df['signal'] = vector[:orig_len]
    
    highs = [x for x in highs if x < orig_len]
    stabs = [x for x in stabs if x < orig_len]
    lows = [x for x in lows if x < orig_len]
    
    return df, highs, stabs, lows

def generate_warped_reality_optimized(df):
    """
    Vectorized version of warped reality generation for speed.
    """
    if df.empty: return pd.DataFrame()
    
    # 1. Vectorized Time Warp
    n_rows = len(df)
    time_warps = np.random.uniform(-1, 1, n_rows)
    days_counts = np.maximum(1, (30 + 30 * time_warps).astype(int))
    
    indices = np.repeat(np.arange(n_rows), days_counts)
    
    daily_open = df['open'].values[indices]
    daily_high = df['high'].values[indices]
    daily_low = df['low'].values[indices]
    daily_close = df['close'].values[indices]
    
    total_days = len(daily_open)
    start_date = df['time'].iloc[0]
    dates = pd.date_range(start=start_date, periods=total_days, freq='D')
    
    warped_df = pd.DataFrame({
        'time': dates,
        'open': daily_open,
        'high': daily_high,
        'low': daily_low,
        'close': daily_close
    })
    warped_df.set_index('time', inplace=True)
    
    w_m = warped_df.resample('30D').agg({
        'open': 'first', 
        'high': 'max', 
        'low': 'min', 
        'close': 'last'
    }).dropna().reset_index()

    # 2. Vectorized Price Randomization
    n_w = len(w_m)
    pct_changes = w_m['close'].pct_change().fillna(0).values
    
    shocks = np.random.uniform(-0.06, 0.06, n_w)
    shocks[0] = 0 
    log_multipliers = np.cumsum(shocks)
    multipliers = np.exp(log_multipliers)
    
    adjusted_returns = np.maximum(-0.98, pct_changes * multipliers)
    
    price_0 = w_m['close'].iloc[0]
    adjusted_returns[0] = 0 
    growth_factors = 1 + adjusted_returns
    cumulative_growth = np.cumprod(growth_factors)
    new_closes = price_0 * cumulative_growth
    
    old_closes = w_m['close'].values
    old_closes[old_closes == 0] = 1e-9
    ratios = new_closes / old_closes
    
    w_m['open'] *= ratios
    w_m['high'] *= ratios
    w_m['low'] *= ratios
    w_m['close'] = new_closes
    
    # New Logic: Analyze structure with bearish future assumption
    w_m, _, _, _ = analyze_structure_new(w_m)
    
    return w_m[['close', 'signal']]

def create_plot_and_vector(df):
    if df.empty: return None, [], []
    # Reality A Analysis
    df, highs, stabs, lows = analyze_structure_original(df)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # Plot 1: Actual
    ax1.set_facecolor('#e0e0e0') 
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2, label='Actual Price')
    ax1.scatter(df.loc[highs, 'time'], df.loc[highs, 'close'], color='#ff4757', s=100, marker='v', zorder=5)
    ax1.scatter(df.loc[stabs, 'time'], df.loc[stabs, 'close'], color='#8e44ad', s=100, marker='d', zorder=5)
    ax1.scatter(df.loc[lows, 'time'], df.loc[lows, 'low'], color='#2ed573', s=100, marker='^', zorder=5)
    ax1.set_title("Reality A: Actual Historical Data", fontweight='bold')
    ax1.legend()

    # Plot 2: Random Reality
    ax2.set_facecolor('#f0f0f0') 
    w_display = generate_warped_reality_optimized(df) 
    w_display['time'] = pd.date_range(start=df['time'].iloc[0], periods=len(w_display), freq='30D')
    
    ax2.plot(w_display['time'], w_display['close'], color='#2980b9', linewidth=2, label='Sim Price', zorder=5)

    # Background Coloring
    w_display['group'] = (w_display['signal'] != w_display['signal'].shift()).cumsum()
    w_display['next_t'] = w_display['time'].shift(-1).fillna(w_display['time'].iloc[-1] + timedelta(days=30))
    groups = w_display.groupby('group').agg({'time': 'first', 'next_t': 'last', 'signal': 'first'})

    for _, row in groups.iterrows():
        if not pd.isna(row['signal']):
            color = '#f8d7da' if row['signal'] == -1 else ('#e2e3e5' if row['signal'] == 0 else '#d4edda')
            ax2.axvspan(row['time'], row['next_t'], color=color, alpha=0.6, zorder=1)

    for t in w_display.loc[w_display.index[w_display['signal'].diff() != 0], 'time']:
        ax2.axvline(x=t, color='black', alpha=0.1, linewidth=0.5)

    ax2.set_title("Reality B: Randomized Reality (Optimized Gen)", fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, zorder=2)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110); plt.close()
    
    v_a = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if pd.isna(r['signal']) else int(r['signal'])} for i, r in df.iterrows()]
    v_b = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if pd.isna(r['signal']) else int(r['signal'])} for i, r in w_display.iterrows()]
    
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
        .btn { display: inline-block; margin: 10px 10px 20px 0; padding: 12px 24px; background: #6c5ce7; color: #fff; text-decoration: none; border-radius: 8px; font-weight: bold; }
        .btn-dl { background: #00b894; }
    </style></head><body>
    <div class="container">
        <h1>Bitcoin: Mass Reality Generation</h1>
        <div style="background:#f8f9fa; padding:15px; border-radius:8px; border:1px solid #dee2e6; margin-bottom:20px;">
            <strong>Assumption:</strong> The price crashes to zero immediately after the data ends. This forces the algorithm to confirm any recent highs as Peaks.
        </div>
        <a href="/" class="btn">Visualize New Reality</a>
        <a href="/download_csv" class="btn btn-dl">Download Training Data (1000 Realities)</a>
        
        <img src="data:image/png;base64,{{p}}">
        <div class="grid-container">
            <div>
                <h3>Reality A Vector</h3>
                <div class="grid">{% for i in v_a %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}</div>
            </div>
            <div>
                <h3>Reality B Vector (Preview)</h3>
                <div class="grid">{% for i in v_b %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}</div>
            </div>
        </div>
    </div></body></html>"""
    return render_template_string(html, p=p, v_a=v_a, v_b=v_b)

@app.route('/download_csv')
def download_csv():
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "Error"
    
    # 1. Original Data
    df_orig, _, _, _ = analyze_structure_original(df)
    output_df = df_orig[['close', 'signal']].copy()
    
    # 2. Generate 1000 Realities
    realities = []
    # We append to list then concat once for performance
    for _ in range(1000):
        w_df = generate_warped_reality_optimized(df)
        realities.append(w_df)
    
    # Concat all
    all_randoms = pd.concat(realities)
    final_df = pd.concat([output_df, all_randoms])
    
    # Convert to CSV
    csv_data = final_df.to_csv(index=False)
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=bitcoin_1000_realities.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)