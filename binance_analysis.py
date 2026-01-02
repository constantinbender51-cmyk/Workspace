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
from flask import Flask, render_template_string, Response, stream_with_context
from datetime import datetime, timedelta
from threading import Thread

app = Flask(__name__)

# --- Configuration ---
PAIR = 'XBTUSD' 
INTERVAL = 10080 
PORT = 8080
WINDOW_SIZE = 10
SWITCHING_PENALTY_WEIGHT = 35
ACTIONS = [-1, 0, 1] 

# --- Global Storage & State ---
CACHED_DATA = {
    "reality_a": None,
    "realities_randomized": [], # 100 with price shocks
    "realities_warped_only": [], # 100 with time warp only
    "plot_b64": None,
    "progress": 0,
    "status": "Idle",
    "ready": False
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
        df_m = df.resample('MS').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
        return df_m
    except Exception:
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
        std_dev = np.std(strategy_returns) + 1e-9
        risk_adj_return = total_return / std_dev
        
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

def generate_warped_reality(df, randomize_returns=True):
    n_rows = len(df)
    time_warps = np.random.uniform(-0.8, 0.8, n_rows)
    days_counts = np.maximum(1, (30 + 30 * time_warps).astype(int))
    indices = np.repeat(np.arange(n_rows), days_counts)
    
    daily_close = df['close'].values[indices]
    dates = pd.date_range(start=df['time'].iloc[0], periods=len(daily_close), freq='D')
    w_m = pd.DataFrame({'close': daily_close, 'time': dates}).set_index('time').resample('30D').agg({'close': 'last'}).dropna().reset_index()
    
    if randomize_returns:
        n_w = len(w_m)
        pct_changes = w_m['close'].pct_change().fillna(0).values
        shocks = np.random.uniform(-0.06, 0.06, n_w)
        shocks[0] = 0 
        multipliers = np.exp(np.cumsum(shocks))
        w_m['close'] = w_m['close'].iloc[0] * np.cumprod(1 + (pct_changes * multipliers))
    
    w_m = apply_optimized_signals(w_m)
    w_m['month'] = np.arange(1, len(w_m) + 1)
    return w_m

def precompute_worker():
    CACHED_DATA["status"] = "Fetching Kraken Data..."
    CACHED_DATA["progress"] = 5
    df_raw = fetch_kraken_data(PAIR, INTERVAL)
    if df_raw.empty:
        CACHED_DATA["status"] = "API Error"
        return

    CACHED_DATA["status"] = "Optimizing Reality A..."
    CACHED_DATA["progress"] = 10
    CACHED_DATA["reality_a"] = apply_optimized_signals(df_raw)

    # Batch 1: Warp Only (40% of remaining bar)
    CACHED_DATA["status"] = "Generating 100 Time-Warped Realities..."
    for i in range(100):
        CACHED_DATA["realities_warped_only"].append(generate_warped_reality(df_raw, randomize_returns=False))
        CACHED_DATA["progress"] = int(10 + (i / 100) * 40)

    # Batch 2: Randomized (40% of remaining bar)
    CACHED_DATA["status"] = "Generating 100 Warp + Random Realities..."
    for i in range(100):
        CACHED_DATA["realities_randomized"].append(generate_warped_reality(df_raw, randomize_returns=True))
        CACHED_DATA["progress"] = int(50 + (i / 100) * 40)
    
    CACHED_DATA["status"] = "Generating Comparative Plots..."
    CACHED_DATA["progress"] = 95
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), facecolor='#f8f9fa')
    
    df_a = CACHED_DATA["reality_a"]
    ax1.plot(df_a['time'], df_a['close'], color='#2d3436', linewidth=2, alpha=0.6)
    for i, row in df_a.iterrows():
        color = '#00b894' if row['signal'] == 1 else ('#d63031' if row['signal'] == -1 else '#b2bec3')
        marker = '^' if row['signal'] == 1 else ('v' if row['signal'] == -1 else 'o')
        ax1.scatter(row['time'], row['close'], color=color, marker=marker, s=50, zorder=5)
    ax1.set_title(f"Reality A: Kraken {PAIR} (Strategy Markers)", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Spaghetti plot using Randomized set
    for i, w_df in enumerate(CACHED_DATA["realities_randomized"]):
        alpha = 0.3 if i < 3 else 0.05
        ax2.plot(w_df['month'], w_df['close'], color='#6c5ce7', alpha=alpha, linewidth=1)
    
    ax2.set_yscale('log')
    ax2.set_title("Reality B: 100 Randomized Path Simulations (Log Scale)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Elapsed Months")
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
            time.sleep(0.4)
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/')
def index():
    if not CACHED_DATA["ready"]:
        return render_template_string("""
        <!DOCTYPE html><html><head><title>Reality Startup</title>
        <style>
            body { font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; background: #f1f2f6; margin: 0; }
            .card { background: white; padding: 40px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); width: 450px; text-align: center; }
            .p-bar { background: #dfe6e9; height: 12px; border-radius: 6px; margin: 25px 0; overflow: hidden; }
            .p-fill { background: linear-gradient(90deg, #00b894, #6c5ce7); height: 100%; width: 0%; transition: width 0.4s ease; }
            .status { color: #636e72; font-weight: bold; margin-bottom: 5px; }
        </style></head><body>
        <div class="card">
            <h2>Reality Engine Startup</h2>
            <div class="status" id="status">Warming Up...</div>
            <div class="p-bar"><div class="p-fill" id="fill"></div></div>
            <div id="pct" style="font-weight:bold; color:#6c5ce7">0%</div>
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
    <!DOCTYPE html><html><head><title>BTC Reality Dashboard</title>
    <style>
        body { font-family: -apple-system, sans-serif; background: #f1f2f6; padding: 25px; color: #2d3436; }
        .container { max-width: 1400px; margin: auto; background: white; padding: 35px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.05); }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; margin-bottom: 25px; padding-bottom: 15px; }
        .btn { padding: 12px 20px; color: white; text-decoration: none; border-radius: 8px; font-weight: bold; margin-left: 10px; font-size: 14px; }
        .btn-green { background: #00b894; }
        .btn-blue { background: #0984e3; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(85px, 1fr)); gap: 6px; height: 180px; overflow-y: auto; background: #fdfdfd; border: 1px solid #eee; padding: 12px; margin-top: 15px; }
        .cell { padding: 8px; text-align: center; font-size: 11px; border-radius: 5px; border: 1px solid #eee; }
        .s1 { background: #55efc4; color: #006266; font-weight: bold; } .s-1 { background: #ff7675; color: white; } .s0 { background: #dfe6e9; color: #636e72; }
    </style></head><body>
    <div class="container">
        <div class="header">
            <div><h1>Bitcoin Parallel Reality Engine</h1><p>Precomputed Signal Dataset (n=200)</p></div>
            <div>
                <a href="/download/warped" class="btn btn-blue">Export Warp-Only (CSV)</a>
                <a href="/download/random" class="btn btn-green">Export Randomized (CSV)</a>
            </div>
        </div>
        <img src="data:image/png;base64,{{p}}" style="width:100%; border-radius:10px; border: 1px solid #eee;">
        <h3 style="margin-top:30px">Reality A (Original) Strategy Signals</h3>
        <div class="grid">
            {% for i in v_a %}<div class="cell s{{i.val}}">{{i.date}}<br><b>{{i.val}}</b></div>{% endfor %}
        </div>
    </div></body></html>
    """
    return render_template_string(html, p=CACHED_DATA["plot_b64"], v_a=v_a)

@app.route('/download/<mode>')
def download(mode):
    key = "realities_randomized" if mode == "random" else "realities_warped_only"
    if not CACHED_DATA[key]: return "No data."
    dfs = []
    for i, df in enumerate(CACHED_DATA[key]):
        d = df.copy()
        d['reality_id'] = i
        if 'time' in d.columns: d = d.drop(columns=['time'])
        dfs.append(d)
    filename = f"btc_realities_{mode}.csv"
    return Response(pd.concat(dfs).to_csv(index=False), mimetype="text/csv", headers={"Content-disposition": f"attachment; filename={filename}"})

if __name__ == '__main__':
    Thread(target=precompute_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT)