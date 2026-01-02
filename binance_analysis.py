import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import itertools
from flask import Flask, render_template_string, Response
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Configuration ---
PAIR = 'XBTUSD' 
INTERVAL = 10080 # Weekly data
PORT = 8080

# Brute Force Params
WINDOW_SIZE = 12 
SWITCHING_PENALTY_WEIGHT = 35
ACTIONS = ['Long', 'Hold', 'Short']

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

def optimize_segment(segment_prices, last_action=None):
    """Brute-force optimization for a price segment using itertools."""
    n_intervals = len(segment_prices) - 1
    if n_intervals <= 0: return []
    
    price_diffs = np.diff(segment_prices)
    best_score, best_seq = -float('inf'), None
    
    # Generate all possible move combinations (3^N)
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
        # Map moves to multipliers: Long=1, Short=-1, Hold=0
        multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in sequence])
        strategy_returns = price_diffs * multipliers
        
        total_return = np.sum(strategy_returns)
        std_dev = np.std(strategy_returns)
        risk_adj_return = total_return / (std_dev + 1e-9)
        
        # Penalize frequent switching
        switches = 0
        if last_action and sequence[0] != last_action: switches += 1
        for i in range(1, n_intervals):
            if sequence[i] != sequence[i-1]: switches += 1
        
        penalty_score = (switches * SWITCHING_PENALTY_WEIGHT) / n_intervals
        current_score = risk_adj_return - penalty_score
        
        if current_score > best_score:
            best_score, best_seq = current_score, sequence
    return best_seq

def assign_optimized_signals(df):
    """
    Unified Logic for BOTH Original and Random Realities.
    Replaces Market Structure with Brute-Force Itertools Optimization.
    """
    if df.empty: return df
    
    # --- Bearish Extension Logic (Maintains your original structure) ---
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
    
    # --- Brute Force Optimization Loop ---
    prices = df_ext['close'].values
    full_sequence = []
    last_action = None
    step_size = WINDOW_SIZE - 1
    
    for i in range(0, len(prices) - 1, step_size):
        end_idx = min(i + WINDOW_SIZE, len(prices))
        segment = prices[i:end_idx]
        if len(segment) < 2: break
        
        window_best_seq = optimize_segment(segment, last_action)
        full_sequence.extend(window_best_seq)
        if window_best_seq:
            last_action = window_best_seq[-1]

    # Map signals: Long=1, Hold=0, Short=-1
    mapping = {'Long': 1, 'Hold': 0, 'Short': -1}
    signals = [mapping[s] for s in full_sequence]
    
    # Pad or slice to match extended length
    while len(signals) < len(df_ext):
        signals.append(signals[-1] if signals else 0)
    
    # Slice back to original length
    df = df.copy()
    df['signal'] = signals[:orig_len]
    
    return df

def generate_warped_reality_optimized(df, randomize_returns=True):
    """Vectorized version of warped reality generation."""
    if df.empty: return pd.DataFrame()
    
    n_rows = len(df)
    time_warps = np.random.uniform(-1, 1, n_rows)
    days_counts = np.maximum(1, (30 + 30 * time_warps).astype(int))
    
    indices = np.repeat(np.arange(n_rows), days_counts)
    
    daily_close = df['close'].values[indices]
    total_days = len(daily_close)
    dates = pd.date_range(start=df['time'].iloc[0], periods=total_days, freq='D')
    
    warped_df = pd.DataFrame({'close': daily_close}, index=dates)
    
    # Resample to simulated months
    w_m = warped_df.resample('30D').agg({'close': 'last'}).dropna().reset_index()

    if randomize_returns:
        n_w = len(w_m)
        pct_changes = w_m['close'].pct_change().fillna(0).values
        shocks = np.random.uniform(-0.06, 0.06, n_w)
        shocks[0] = 0 
        log_multipliers = np.cumsum(shocks)
        multipliers = np.exp(log_multipliers)
        adjusted_returns = np.maximum(-0.98, pct_changes * multipliers)
        
        price_0 = w_m['close'].iloc[0]
        cumulative_growth = np.cumprod(1 + adjusted_returns)
        w_m['close'] = price_0 * cumulative_growth
    
    # Unified Analysis (Optimization instead of Market Structure)
    w_m = assign_optimized_signals(w_m)
    
    # Create the month index (1..N)
    w_m['month'] = np.arange(1, len(w_m) + 1)
    
    return w_m[['close', 'signal', 'month']]

def create_plot_and_vector(df):
    if df.empty: return None, [], []
    
    # Reality A: Apply Optimizer
    df = assign_optimized_signals(df)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor='#f1f2f6')

    # Reality A Plotting
    ax1.set_facecolor('#e0e0e0') 
    ax1.plot(df['time'], df['close'], color='#2f3542', linewidth=2, alpha=0.7)
    
    # Signal Backgrounds for Reality A
    df['next_t'] = df['time'].shift(-1).fillna(df['time'].iloc[-1] + timedelta(days=30))
    for i, row in df.iterrows():
        color = '#f8d7da' if row['signal'] == -1 else ('#e2e3e5' if row['signal'] == 0 else '#d4edda')
        ax1.axvspan(row['time'], row['next_t'], color=color, alpha=0.4)
    ax1.set_title("Reality A: Brute-Force Path Optimization")

    # Reality B: Generate Randomized Reality
    ax2.set_facecolor('#f0f0f0') 
    w_display = generate_warped_reality_optimized(df, randomize_returns=True) 
    w_display['time'] = pd.date_range(start=df['time'].iloc[0], periods=len(w_display), freq='30D')
    ax2.plot(w_display['time'], w_display['close'], color='#2980b9', linewidth=2, alpha=0.7)

    # Signal Backgrounds for Reality B
    w_display['next_t'] = w_display['time'].shift(-1).fillna(w_display['time'].iloc[-1] + timedelta(days=30))
    for i, row in w_display.iterrows():
        color = '#f8d7da' if row['signal'] == -1 else ('#e2e3e5' if row['signal'] == 0 else '#d4edda')
        ax2.axvspan(row['time'], row['next_t'], color=color, alpha=0.4)

    ax2.set_title("Reality B: Randomized & Re-Optimized Path")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110); plt.close()
    
    v_a = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if pd.isna(r['signal']) else int(r['signal'])} for i, r in df.iterrows()]
    v_b = [{"date": r['time'].strftime('%Y-%m'), "val": "N/A" if pd.isna(r['signal']) else int(r['signal'])} for i, r in w_display.iterrows()]
    return base64.b64encode(buf.getvalue()).decode(), v_a, v_b

def generate_csv_response(randomize):
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "Error"
    
    realities = []
    for _ in range(100):
        realities.append(generate_warped_reality_optimized(df, randomize_returns=randomize))
    
    df_orig = assign_optimized_signals(df)
    df_orig['month'] = np.arange(1, len(df_orig) + 1)
    output_df_orig = df_orig[['close', 'signal', 'month']].copy()
    
    final_df = pd.concat(realities + [output_df_orig])
    csv_data = final_df.to_csv(index=False)
    
    fname = "bitcoin_100_randomized_opt.csv" if randomize else "bitcoin_100_warped_opt.csv"
    return Response(csv_data, mimetype="text/csv", headers={"Content-disposition": f"attachment; filename={fname}"})

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "<h1>API Data Fetch Error</h1>"
    p, v_a, v_b = create_plot_and_vector(df)
    html = """
    <!DOCTYPE html><html><head><title>BTC Reality Warp Optimizer</title>
    <style>
        body { font-family: sans-serif; background: #f1f2f6; padding: 20px; }
        .container { max-width: 1400px; margin: auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
        img { width: 100%; border-radius: 10px; margin: 20px 0; }
        .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 5px; height: 200px; overflow-y: scroll; background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #eee; }
        .c { padding: 8px; text-align: center; font-size: 10px; border: 1px solid #ddd; border-radius: 4px; }
        .v-1 { background: #ff7675; color: white; } .v0 { background: #dfe6e9; color: #636e72; } .v1 { background: #55efc4; color: #006266; font-weight: bold; }
        .btn { display: inline-block; margin: 10px 10px 20px 0; padding: 12px 24px; background: #6c5ce7; color: #fff; text-decoration: none; border-radius: 8px; font-weight: bold; }
        .btn-dl { background: #00b894; } .btn-dl-alt { background: #0984e3; }
    </style></head><body>
    <div class="container">
        <h1>Bitcoin: Itertools Brute-Force Warp</h1>
        <div style="background:#f8f9fa; padding:15px; border-radius:8px; border:1px solid #dee2e6; margin-bottom:20px;">
            <strong>Optimization Logic:</strong> Finding the global maxima of risk-adjusted returns within a sliding {{ws}}-month window.
        </div>
        <div class="btn-group">
            <a href="/" class="btn">Visualize New Reality</a>
            <a href="/download_csv_random" class="btn btn-dl">Download 100 (Warp + Random + Opt)</a>
            <a href="/download_csv_warped" class="btn btn-dl-alt">Download 100 (Warp Only + Opt)</a>
        </div>
        <img src="data:image/png;base64,{{p}}">
        <div class="grid-container">
            <div><h3>Reality A Vector</h3><div class="grid">{% for i in v_a %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}</div></div>
            <div><h3>Reality B Vector (Preview)</h3><div class="grid">{% for i in v_b %}<div class="c {% if i.val == -1 %}v-1{% elif i.val == 0 %}v0{% elif i.val == 1 %}v1{% endif %}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}</div></div>
        </div>
    </div></body></html>"""
    return render_template_string(html, p=p, v_a=v_a, v_b=v_b, ws=WINDOW_SIZE)

@app.route('/download_csv_random')
def download_csv_random(): return generate_csv_response(randomize=True)

@app.route('/download_csv_warped')
def download_csv_warped(): return generate_csv_response(randomize=False)

if __name__ == '__main__': app.run(host='0.0.0.0', port=PORT)