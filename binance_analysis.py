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

def analyze_market_structure(df):
    """
    Unified Logic for BOTH Original and Random Realities.
    1. Peak: 2yr Past / 1yr Future radius, AND must be a NEW ATH.
    2. Low: Lowest price between confirmed Peaks OR between last Peak and End.
    3. Stable: 3/4 point (Time) between a Low and the month with highest 3-month return.
    4. Bearish Future Assumption: Appends 0s to force signal confirmation at the edge.
    """
    if df.empty: return df, [], [], []
    
    # --- Bearish Extension Logic ---
    # Append 12 months of 0 price to force peak confirmation.
    orig_len = len(df)
    
    if 'time' in df.columns:
        last_date = df['time'].iloc[-1]
        extension_dates = [last_date + timedelta(days=30*(i+1)) for i in range(12)]
    else:
        extension_dates = [i for i in range(12)]
        
    extension = pd.DataFrame({
        'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0
    }, index=range(orig_len, orig_len+12))
    
    if 'time' in df.columns:
        extension['time'] = extension_dates
        
    df_ext = pd.concat([df, extension], ignore_index=True)
    
    # 1. Peak Detection (ATH Rule)
    df_ext['h_max_window'] = df_ext['close'].rolling(window=37, min_periods=13).max().shift(-12)
    df_ext['expanding_ath'] = df_ext['close'].expanding().max()
    
    # Candidates must be local max AND >= current ATH
    peak_candidates = df_ext[
        (df_ext['close'] == df_ext['h_max_window']) & 
        (df_ext['close'] >= df_ext['expanding_ath'])
    ].index.tolist()
    
    highs = []
    if peak_candidates:
        highs.append(peak_candidates[0])
        for p in peak_candidates[1:]:
            # Ensure peaks are separated and strictly increasing (ATH logic implies increasing, but filtering for noise)
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

    # 3. Stability Detection (3/4 Point Logic)
    df_ext['abs_ret_3m'] = df_ext['close'].diff(periods=3).abs().fillna(0)
    stabs = []
    
    for low_idx in lows:
        next_peaks = [h for h in highs if h > low_idx]
        if not next_peaks: continue
        target_peak = next_peaks[0]
        
        segment = df_ext.iloc[low_idx:target_peak + 1]
        if segment.empty: continue
        
        # Month with highest absolute 3-month return
        max_3m_abs_idx = segment['abs_ret_3m'].idxmax()
        
        # 3/4 Point
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
            
    # Slice back to original length
    df = df.copy()
    df['signal'] = vector[:orig_len]
    
    # Filter indices to ensure they exist in the original data
    highs = [x for x in highs if x < orig_len]
    stabs = [x for x in stabs if x < orig_len]
    lows = [x for x in lows if x < orig_len]
    
    return df, highs, stabs, lows

def generate_warped_reality_optimized(df, randomize_returns=True):
    """
    Vectorized version of warped reality generation for speed.
    Args:
        df: Input dataframe
        randomize_returns: If True, adds random shocks to price returns. 
                           If False, only warps time (speed of market).
    """
    if df.empty: return pd.DataFrame()
    
    # 1. Vectorized Time Warp
    n_rows = len(df)
    time_warps = np.random.uniform(-1, 1, n_rows)
    # Days to repeat each monthly data point (roughly converting months to variable days)
    # Original logic used 30 days base + random variance
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
    
    # Resample back to 30D blocks (simulated months)
    w_m = warped_df.resample('30D').agg({
        'open': 'first', 
        'high': 'max', 
        'low': 'min', 
        'close': 'last'
    }).dropna().reset_index()

    # 2. Vectorized Price Randomization (Optional)
    if randomize_returns:
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
    
    # Unified Analysis
    w_m, _, _, _ = analyze_market_structure(w_m)
    
    return w_m[['close', 'signal']]

def create_plot_and_vector(df):
    if df.empty: return None, [], []
    
    # Reality A: Now using the UNIFIED logic (same as Random)
    df, highs, stabs, lows = analyze_market_structure(df)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # Plot 1: Actual
    ax1.set_facecolor('#e0e0e0') 
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2, label='Actual Price')
    ax1.scatter(df.loc[highs, 'time'], df.loc[highs, 'close'], color='#ff4757', s=100, marker='v', zorder=5)
    ax1.scatter(df.loc[stabs, 'time'], df.loc[stabs, 'close'], color='#8e44ad', s=100, marker='d', zorder=5)
    ax1.scatter(df.loc[lows, 'time'], df.loc[lows, 'low'], color='#2ed573', s=100, marker='^', zorder=5)
    ax1.set_title("Reality A: Actual Data (Unified Logic)", fontweight='bold')
    ax1.legend()

    # Plot 2: Random Reality (Always Randomized Returns for Visual Impact)
    ax2.set_facecolor('#f0f0f0') 
    w_display = generate_warped_reality_optimized(df, randomize_returns=True) 
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

    # Lines at transitions
    for t in w_display.loc[w_display.index[w_display['signal'].diff() != 0], 'time']:
        ax2.axvline(x=t, color='black', alpha=0.1, linewidth=0.5)

    ax2.set_title("Reality B: Randomized Reality (Time Warp + Price Shocks)", fontweight='bold')
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
        .btn-dl-alt { background: #0984e3; }
        .btn-group { margin-bottom: 20px; }
    </style></head><body>
    <div class="container">
        <h1>Bitcoin: Mass Reality Generation</h1>
        <div style="background:#f8f9fa; padding:15px; border-radius:8px; border:1px solid #dee2e6; margin-bottom:20px;">
            <strong>Assumption:</strong> The price crashes to zero immediately after the data ends (Bearish Future).<br>
            <strong>Logic:</strong> New ATH Peak (2yr past/1yr future) &rarr; Correction &rarr; Low &rarr; Accumulation &rarr; 3/4 Point Stable &rarr; Expansion.
        </div>
        <div class="btn-group">
            <a href="/" class="btn">Visualize New Reality</a>
            <a href="/download_csv_random" class="btn btn-dl">Download 100 (Time Warp + Price Random)</a>
            <a href="/download_csv_warped" class="btn btn-dl-alt">Download 100 (Time Warp Only)</a>
        </div>
        
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

def generate_csv_response(randomize):
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "Error"
    
    # 1. Generate 100 Realities
    realities = []
    for _ in range(100):
        w_df = generate_warped_reality_optimized(df, randomize_returns=randomize)
        realities.append(w_df)
    
    all_randoms = pd.concat(realities)
    
    # 2. Original Data LAST
    df_orig, _, _, _ = analyze_market_structure(df)
    output_df_orig = df_orig[['close', 'signal']].copy()
    
    # 3. Concatenate
    final_df = pd.concat([all_randoms, output_df_orig])
    
    csv_data = final_df.to_csv(index=False)
    
    fname = "bitcoin_100_randomized.csv" if randomize else "bitcoin_100_warped_only.csv"
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={fname}"}
    )

@app.route('/download_csv_random')
def download_csv_random():
    return generate_csv_response(randomize=True)

@app.route('/download_csv_warped')
def download_csv_warped():
    return generate_csv_response(randomize=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)