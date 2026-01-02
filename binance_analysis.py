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
INTERVAL = 10080 
PORT = 8080
WINDOW_SIZE = 6
SWITCHING_PENALTY_WEIGHT = 35
ACTIONS = [-1, 0, 1] # Short, Hold, Long

# --- Global Storage ---
# We compute these once on startup to save CPU and ensure consistency
CACHED_DATA = {
    "reality_a": None,
    "realities_b": [], # List of 100 dataframes
    "plot_b64": None
}

def fetch_kraken_data(pair, interval):
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
        df.set_index('time', inplace=True)
        df_m = df.resample('MS').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        return df_m
    except Exception as e:
        print(f"Fetch error: {e}")
        return pd.DataFrame()

def optimize_segment_vectorized(segment_prices, last_action=None):
    n_intervals = len(segment_prices) - 1
    if n_intervals <= 0: return []
    price_diffs = np.diff(segment_prices)
    best_score, best_seq = -float('inf'), None
    
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
        multipliers = np.array(sequence)
        strategy_returns = price_diffs * multipliers
        total_return = np.sum(strategy_returns)
        std_dev = np.std(strategy_returns)
        risk_adj_return = total_return / (std_dev + 1e-9)
        
        switches = 0
        if last_action is not None and sequence[0] != last_action: switches += 1
        for i in range(1, n_intervals):
            if sequence[i] != sequence[i-1]: switches += 1
        
        penalty_score = (switches * SWITCHING_PENALTY_WEIGHT) / n_intervals
        current_score = risk_adj_return - penalty_score
        
        if current_score > best_score:
            best_score, best_seq = current_score, sequence
    return list(best_seq)

def apply_optimized_signals(df):
    prices = df['close'].values
    if len(prices) < 2:
        df['signal'] = 0
        return df
    full_sequence = []
    last_action = None
    step_size = WINDOW_SIZE - 1
    for i in range(0, len(prices) - 1, step_size):
        end_idx = min(i + WINDOW_SIZE, len(prices))
        segment = prices[i:end_idx]
        if len(segment) < 2: break
        window_best_seq = optimize_segment_vectorized(segment, last_action)
        full_sequence.extend(window_best_seq)
        if window_best_seq: last_action = window_best_seq[-1]
    signals = full_sequence[:len(df)]
    while len(signals) < len(df):
        signals.append(signals[-1] if signals else 0)
    df = df.copy()
    df['signal'] = signals
    return df

def generate_warped_reality(df):
    n_rows = len(df)
    time_warps = np.random.uniform(-0.8, 0.8, n_rows)
    days_counts = np.maximum(1, (30 + 30 * time_warps).astype(int))
    indices = np.repeat(np.arange(n_rows), days_counts)
    daily_close = df['close'].values[indices]
    dates = pd.date_range(start=df['time'].iloc[0], periods=len(daily_close), freq='D')
    w_m = pd.DataFrame({'close': daily_close, 'time': dates}).set_index('time').resample('30D').agg({'close': 'last'}).dropna().reset_index()
    
    # Randomize price path
    n_w = len(w_m)
    pct_changes = w_m['close'].pct_change().fillna(0).values
    shocks = np.random.uniform(-0.06, 0.06, n_w)
    shocks[0] = 0 
    multipliers = np.exp(np.cumsum(shocks))
    w_m['close'] = w_m['close'].iloc[0] * np.cumprod(1 + (pct_changes * multipliers))
    
    w_m = apply_optimized_signals(w_m)
    w_m['month'] = np.arange(1, len(w_m) + 1)
    return w_m

def precompute_all():
    """Heavy lift performed once on start."""
    print("Initializing Reality Engine... fetching and optimizing 100 realities.")
    df_raw = fetch_kraken_data(PAIR, INTERVAL)
    if df_raw.empty:
        print("Error: Could not fetch initial data.")
        return

    # Reality A
    df_a = apply_optimized_signals(df_raw)
    CACHED_DATA["reality_a"] = df_a

    # Reality B (100 variants)
    for i in range(100):
        CACHED_DATA["realities_b"].append(generate_warped_reality(df_raw))
    
    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), facecolor='#f1f2f6')
    
    # Plot Reality A
    ax1.plot(df_a['time'], df_a['close'], color='#2d3436', linewidth=2, alpha=0.7, label='Actual Price')
    for i, row in df_a.iterrows():
        color = '#00b894' if row['signal'] == 1 else ('#d63031' if row['signal'] == -1 else '#636e72')
        marker = '^' if row['signal'] == 1 else ('v' if row['signal'] == -1 else 'o')
        ax1.scatter(row['time'], row['close'], color=color, marker=marker, s=50, zorder=5)
    ax1.set_title(f"Reality A: Kraken {PAIR} Optimized Signals", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Plot Reality B (Spaghetti plot of 100)
    for i, w_df in enumerate(CACHED_DATA["realities_b"]):
        alpha = 0.25 if i == 0 else 0.08
        color = '#0984e3' if i % 2 == 0 else '#6c5ce7'
        ax2.plot(w_df['month'], w_df['close'], color=color, alpha=alpha, linewidth=1)
    
    ax2.set_yscale('log')
    ax2.set_title("Reality B: 100 Optimized Random Realities (Log Scale Overlay)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Elapsed Simulated Months")
    ax2.set_ylabel("Price (USD)")
    ax2.grid(True, which="both", linestyle=':', alpha=0.4)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110)
    CACHED_DATA["plot_b64"] = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    print("Precomputation complete.")

@app.route('/')
def index():
    if CACHED_DATA["reality_a"] is None:
        return "Engine still warming up... please refresh in a moment."
    
    df_a = CACHED_DATA["reality_a"]
    v_a = [{"date": r['time'].strftime('%Y-%m'), "val": int(r['signal'])} for i, r in df_a.iterrows()]
    
    html = """
    <!DOCTYPE html><html><head><title>BTC Reality Engine</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica; background: #f0f2f5; padding: 20px; color: #2d3436; }
        .container { max-width: 1300px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #eee; margin-bottom: 20px; padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        .stats { font-size: 0.9em; color: #636e72; background: #f8f9fa; padding: 10px; border-radius: 6px; }
        .btn { display: inline-block; padding: 10px 18px; background: #00b894; color: white; text-decoration: none; border-radius: 6px; font-weight: bold; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 5px; height: 180px; overflow-y: auto; background: #fafafa; border: 1px solid #eee; padding: 10px; margin-top: 10px; }
        .c { padding: 6px; text-align: center; font-size: 10px; border-radius: 4px; border: 1px solid #eee; }
        .s1 { background: #55efc4; color: #006266; font-weight: bold; } .s-1 { background: #ff7675; color: white; } .s0 { background: #dfe6e9; color: #636e72; }
    </style></head><body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Bitcoin Reality Engine</h1>
                <p>Optimization Model: Window=12, Penalty=35. Precomputed on Startup.</p>
            </div>
            <a href="/download" class="btn">Download All 100 Realities (CSV)</a>
        </div>
        
        <img src="data:image/png;base64,{{p}}" style="width:100%; border-radius:8px; border: 1px solid #ddd;">
        
        <h3>Reality A: Signal Vector (Latest Kraken Data)</h3>
        <div class="grid">
            {% for i in v_a %}<div class="c s{{i.val}}">{{i.date}}<br>{{i.val}}</div>{% endfor %}
        </div>
        
        <div class="stats" style="margin-top:20px;">
            Reality B contains 100 unique simulated paths, each optimized independently using the provided segment strategy.
        </div>
    </div></body></html>
    """
    return render_template_string(html, p=CACHED_DATA["plot_b64"], v_a=v_a)

@app.route('/download')
def download():
    if not CACHED_DATA["realities_b"]: return "No data cached."
    
    # Add reality_id column for the export
    export_list = []
    for i, df in enumerate(CACHED_DATA["realities_b"]):
        temp = df.copy()
        temp['reality_id'] = i
        export_list.append(temp)
    
    final_df = pd.concat(export_list)
    return Response(final_df.to_csv(index=False), mimetype="text/csv", headers={"Content-disposition": "attachment; filename=100_optimized_realities.csv"})

if __name__ == '__main__':
    precompute_all()
    app.run(host='0.0.0.0', port=PORT)