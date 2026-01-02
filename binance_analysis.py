import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import itertools
import time
import json
import random
from flask import Flask, render_template_string, Response, stream_with_context
from threading import Thread

app = Flask(__name__)

# --- Configuration ---
MAIN_PAIR = 'XBTUSD' 
INTERVAL = 10080  # Weekly data
PORT = 8080
WINDOW_SIZE = 10
SWITCHING_PENALTY_WEIGHT = 35
ACTIONS = [-1, 0, 1] 

# --- Global Storage & State ---
CACHED_DATA = {
    "reality_a": None,
    "real_assets": [],
    "plot_b64": None,
    "progress": 0,
    "status": "Idle",
    "ready": False
}

# --- Vectorization Cache ---
# We pre-compute the search space to avoid iterating 3^9 loops in Python
SEARCH_SPACE = {
    "matrix": None,           # shape (19683, 9)
    "internal_switches": None, # shape (19683,)
    "first_actions": None      # shape (19683,)
}

def precompute_search_space():
    """Generates the strategy matrix once at startup."""
    n_intervals = WINDOW_SIZE - 1
    # Generate all permutations (approx 19,683 for window 10)
    all_seqs = list(itertools.product(ACTIONS, repeat=n_intervals))
    matrix = np.array(all_seqs, dtype=np.int8)
    
    # Pre-calculate internal switch costs (switches within the sequence)
    # diff checks neighbor changes, != 0 means a switch occurred, sum counts them
    internal_switches = np.sum(np.diff(matrix, axis=1) != 0, axis=1)
    
    SEARCH_SPACE["matrix"] = matrix
    SEARCH_SPACE["internal_switches"] = internal_switches
    SEARCH_SPACE["first_actions"] = matrix[:, 0]
    print(f"Search space optimized: {len(matrix)} permutations loaded into memory.")

def get_tradable_pairs():
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('error'): return []
        pairs = []
        for name, details in data['result'].items():
            wsname = details.get('wsname', '')
            if wsname.endswith('/USD'):
                pairs.append(name)
        return pairs
    except Exception:
        return []

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
        # Resample to Monthly
        df_m = df.resample('MS').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
        df_m['pair'] = pair
        return df_m
    except Exception:
        return pd.DataFrame()

def optimize_segment_matrix(segment_prices, last_action=None):
    """
    Highly optimized vectorized solver.
    Replaces 20,000 python iterations with single numpy matrix operations.
    """
    n_intervals = len(segment_prices) - 1
    if n_intervals != (WINDOW_SIZE - 1):
        # Fallback for end-of-array segments that don't fit the matrix shape
        return [0] * n_intervals

    price_diffs = np.diff(segment_prices) # Shape (9,)
    
    # 1. Calculate Returns for all 19,683 strategies at once
    # (19683, 9) * (9,) broadcasting -> (19683, 9)
    strategy_returns = SEARCH_SPACE["matrix"] * price_diffs
    
    # 2. Vectorized Statistics
    total_returns = np.sum(strategy_returns, axis=1)
    std_devs = np.std(strategy_returns, axis=1) + 1e-9
    risk_adj = total_returns / std_devs
    
    # 3. Vectorized Penalty Calculation
    # internal switches + (1 if start != last else 0)
    if last_action is not None:
        start_switch_cost = (SEARCH_SPACE["first_actions"] != last_action).astype(int)
    else:
        start_switch_cost = 0
        
    total_switches = SEARCH_SPACE["internal_switches"] + start_switch_cost
    penalty = (total_switches * SWITCHING_PENALTY_WEIGHT) / n_intervals
    
    # 4. Score and Argmax
    final_scores = risk_adj - penalty
    best_idx = np.argmax(final_scores)
    
    return SEARCH_SPACE["matrix"][best_idx].tolist()

def apply_optimized_signals(df):
    prices = df['close'].values
    if len(prices) < WINDOW_SIZE:
        df['signal'] = 0
        df['month_idx'] = np.arange(1, len(df) + 1)
        return df
    
    full_sequence = []
    last_action = None
    step_size = WINDOW_SIZE - 1
    
    # Sliding window
    for i in range(0, len(prices) - 1, step_size):
        end_idx = min(i + WINDOW_SIZE, len(prices))
        segment = prices[i:end_idx]
        
        # If segment matches window size, use the fast matrix engine
        # Otherwise (tail end), just fill with 0s or handle gracefully
        if len(segment) == WINDOW_SIZE:
            window_best_seq = optimize_segment_matrix(segment, last_action)
        else:
            # Handle remainder
            window_best_seq = [0] * (len(segment)-1)
            
        full_sequence.extend(window_best_seq)
        if window_best_seq: last_action = window_best_seq[-1]
    
    signals = full_sequence[:len(df)]
    while len(signals) < len(df):
        signals.append(signals[-1] if signals else 0)
    
    df = df.copy()
    df['signal'] = signals
    df['month_idx'] = np.arange(1, len(df) + 1)
    return df

def precompute_worker():
    # 0. Init Math
    precompute_search_space()

    # 1. Fetch Main Reality
    CACHED_DATA["status"] = "Fetching Reality A (BTC)..."
    CACHED_DATA["progress"] = 5
    df_raw = fetch_kraken_data(MAIN_PAIR, INTERVAL)
    
    if df_raw.empty:
        CACHED_DATA["status"] = "API Error (Main Pair)"
        return

    CACHED_DATA["status"] = "Optimizing Reality A..."
    CACHED_DATA["progress"] = 10
    CACHED_DATA["reality_a"] = apply_optimized_signals(df_raw)

    # 2. Fetch List of Real Assets
    CACHED_DATA["status"] = "Scanning Market..."
    all_pairs = get_tradable_pairs()
    if MAIN_PAIR in all_pairs: all_pairs.remove(MAIN_PAIR)
    target_pairs = all_pairs[:100] # Limit to 100 for time considerations
    
    total_assets = len(target_pairs)
    processed_assets = []
    
    for i, pair in enumerate(target_pairs):
        pct = 10 + int((i / total_assets) * 85)
        CACHED_DATA["progress"] = pct
        CACHED_DATA["status"] = f"Mining {pair} ({i+1}/{total_assets})..."
        
        df = fetch_kraken_data(pair, INTERVAL)
        if not df.empty and len(df) > 15: # Need enough history
            # The apply_optimized_signals is now extremely fast
            df_opt = apply_optimized_signals(df)
            processed_assets.append(df_opt)
        
        # We still sleep for API rate limits, but the CPU work is instant
        time.sleep(0.4) 

    CACHED_DATA["real_assets"] = processed_assets
    CACHED_DATA["status"] = "Generating Visualization..."
    CACHED_DATA["progress"] = 98
    
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), facecolor='#f8f9fa')
    
    # Reality A (Main)
    df_a = CACHED_DATA["reality_a"]
    ax1.plot(df_a['time'], df_a['close'], color='#2d3436', linewidth=2, alpha=0.6)
    
    # Scatter points for signals
    # Filter to reduce SVG size if needed, but matplotlib png is raster so it's fine
    sigs = df_a['signal'].values
    colors = np.where(sigs == 1, '#00b894', np.where(sigs == -1, '#d63031', '#b2bec3'))
    ax1.scatter(df_a['time'], df_a['close'], c=colors, s=50, zorder=5)
    
    ax1.set_title(f"Target Reality: {MAIN_PAIR} (Matrix Optimized)", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Reality B (Universe)
    for i, asset_df in enumerate(processed_assets):
        if not asset_df.empty:
            # Normalize to start at 100
            start_price = asset_df['close'].iloc[0]
            if start_price > 0:
                normalized_price = (asset_df['close'] / start_price) * 100
                # Color code based on if it ended positive or negative relative to start
                final_val = normalized_price.iloc[-1]
                line_col = '#6c5ce7' if final_val > 100 else '#a29bfe'
                alpha = 0.4 if i < 10 else 0.15
                ax2.plot(asset_df['month_idx'], normalized_price, color=line_col, alpha=alpha, linewidth=1)
    
    ax2.set_yscale('log')
    ax2.set_title(f"Universe Reality: {len(processed_assets)} Global Assets (Normalized)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Months Active")
    ax2.set_ylabel("Performance (Log Scale)")
    ax2.grid(True, which="both", linestyle=':', alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    CACHED_DATA["plot_b64"] = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    CACHED_DATA["progress"] = 100
    CACHED_DATA["status"] = "Engine Ready"
    CACHED_DATA["ready"] = True

@app.route('/progress')
def progress():
    def generate():
        while True:
            yield f"data: {json.dumps({'progress': CACHED_DATA['progress'], 'status': CACHED_DATA['status'], 'ready': CACHED_DATA['ready']})}\n\n"
            if CACHED_DATA["ready"]: break
            time.sleep(0.5)
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/')
def index():
    if not CACHED_DATA["ready"]:
        return render_template_string("""
        <!DOCTYPE html><html><head><title>Matrix Engine</title>
        <style>
            body { font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; background: #2d3436; margin: 0; color: white;}
            .card { background: #353b48; padding: 40px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); width: 450px; text-align: center; border: 1px solid #4b4b4b;}
            .p-bar { background: #2f3640; height: 12px; border-radius: 6px; margin: 25px 0; overflow: hidden; border: 1px solid #7f8fa6;}
            .p-fill { background: linear-gradient(90deg, #00b894, #0984e3); height: 100%; width: 0%; transition: width 0.3s ease; }
            .status { color: #dfe6e9; font-weight: bold; margin-bottom: 5px; }
            .mono { font-family: monospace; color: #00b894; font-size: 0.9em; margin-top: 10px;}
        </style></head><body>
        <div class="card">
            <h2>Financial Reality Engine</h2>
            <div class="status" id="status">Loading Matrix...</div>
            <div class="p-bar"><div class="p-fill" id="fill"></div></div>
            <div id="pct" style="font-weight:bold; color:#0984e3">0%</div>
            <div class="mono">Vectorized Computation Active</div>
        </div>
        <script>
            const source = new EventSource('/progress');
            source.onmessage = (e) => {
                const d = JSON.parse(e.data);
                document.getElementById('fill').style.width = d.progress + '%';
                document.getElementById('pct').innerText = d.progress + '%';
                document.getElementById('status').innerText = d.status;
                if (d.ready) setTimeout(() => window.location.reload(), 600);
            };
        </script></body></html>
        """)

    v_a = [{"date": r['time'].strftime('%Y-%m'), "val": int(r['signal'])} for i, r in CACHED_DATA["reality_a"].iterrows()]
    
    html = """
    <!DOCTYPE html><html><head><title>Market Reality</title>
    <style>
        body { font-family: -apple-system, sans-serif; background: #f1f2f6; padding: 25px; color: #2d3436; }
        .container { max-width: 1400px; margin: auto; background: white; padding: 35px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.05); }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; margin-bottom: 25px; padding-bottom: 15px; }
        .btn { padding: 12px 20px; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; margin-left: 10px; font-size: 14px; }
        .btn-green { background: #00b894; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(85px, 1fr)); gap: 6px; height: 180px; overflow-y: auto; background: #fdfdfd; border: 1px solid #eee; padding: 12px; margin-top: 15px; }
        .cell { padding: 8px; text-align: center; font-size: 11px; border-radius: 5px; border: 1px solid #eee; }
        .s1 { background: #55efc4; color: #006266; font-weight: bold; } .s-1 { background: #ff7675; color: white; } .s0 { background: #dfe6e9; color: #636e72; }
    </style></head><body>
    <div class="container">
        <div class="header">
            <div><h1>Vectorized Reality Engine</h1><p>Processed {{ n_assets }} Global Assets via Matrix Broadcasting</p></div>
            <div>
                <a href="/download/all" class="btn btn-green">Export Dataset (CSV)</a>
            </div>
        </div>
        <img src="data:image/png;base64,{{p}}" style="width:100%; border-radius:10px; border: 1px solid #eee;">
        <h3 style="margin-top:30px">Main Reality Signals</h3>
        <div class="grid">
            {% for i in v_a %}<div class="cell s{{i.val}}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}
        </div>
    </div></body></html>
    """
    return render_template_string(html, p=CACHED_DATA["plot_b64"], v_a=v_a, n_assets=len(CACHED_DATA["real_assets"]))

@app.route('/download/all')
def download_all():
    if not CACHED_DATA["real_assets"]: return "No data."
    dfs = []
    df_a = CACHED_DATA["reality_a"].copy()
    df_a['asset_id'] = "BTC_MAIN"
    dfs.append(df_a)
    for i, df in enumerate(CACHED_DATA["real_assets"]):
        d = df.copy()
        d['asset_id'] = d['pair'] if 'pair' in d.columns else f"ASSET_{i}"
        dfs.append(d)
    filename = f"vectorized_market_data_{int(time.time())}.csv"
    return Response(pd.concat(dfs).to_csv(index=False), mimetype="text/csv", headers={"Content-disposition": f"attachment; filename={filename}"})

if __name__ == '__main__':
    Thread(target=precompute_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT)