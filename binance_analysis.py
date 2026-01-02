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
from datetime import timedelta

app = Flask(__name__)

# --- Configuration ---
PAIR = 'XBTUSD' 
INTERVAL = 10080 # Weekly data
PORT = 8080
WINDOW_SIZE = 10 # Keep small for performance (3^10 combinations)
SWITCHING_PENALTY_WEIGHT = 35
ACTIONS = ['Long', 'Hold', 'Short']

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
        df['close'] = pd.to_numeric(df['close'])
        df_m = df.resample('MS', on='time').agg({'close': 'last'}).dropna().reset_index()
        return df_m
    except Exception as e:
        print(f"Fetch error: {e}")
        return pd.DataFrame()

def optimize_segment(segment_prices, last_action=None):
    """Brute-force optimization for a price segment."""
    n_intervals = len(segment_prices) - 1
    if n_intervals <= 0: return []
    
    price_diffs = np.diff(segment_prices)
    best_score, best_seq = -float('inf'), None
    
    # Generate all possible move combinations (3^N)
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
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

def get_optimized_signals(df):
    """Applies the segment optimizer across the entire dataframe."""
    prices = df['close'].values
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

    # Map signals to numeric values for the dataframe
    mapping = {'Long': 1, 'Hold': 0, 'Short': -1}
    # Pad the sequence to match df length (last signal persists)
    signals = [mapping[s] for s in full_sequence]
    while len(signals) < len(df):
        signals.append(signals[-1] if signals else 0)
    
    df = df.copy()
    df['signal'] = signals[:len(df)]
    return df



def create_plot_and_vector(df):
    if df.empty: return None, [], []
    
    # Process Reality A
    df = get_optimized_signals(df)
    
    # Prepare Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), facecolor='#f1f2f6')
    
    # Plot Reality A
    ax1.plot(df['time'], df['close'], color='#2f3542', alpha=0.5)
    for i in range(len(df)-1):
        color = '#d4edda' if df['signal'].iloc[i] == 1 else ('#f8d7da' if df['signal'].iloc[i] == -1 else '#e2e3e5')
        ax1.axvspan(df['time'].iloc[i], df['time'].iloc[i+1], color=color, alpha=0.5)
    ax1.set_title("Reality A: Optimized Brute-Force Strategy")

    # Generate Reality B (Warped)
    # Using a simple warp for this example
    w_df = df.sample(frac=1).reset_index(drop=True)
    w_df['time'] = pd.date_range(start=df['time'].iloc[0], periods=len(w_df), freq='30D')
    w_df = get_optimized_signals(w_df)
    
    ax2.plot(w_df['time'], w_df['close'], color='#2980b9', alpha=0.5)
    for i in range(len(w_df)-1):
        color = '#d4edda' if w_df['signal'].iloc[i] == 1 else ('#f8d7da' if w_df['signal'].iloc[i] == -1 else '#e2e3e5')
        ax2.axvspan(w_df['time'].iloc[i], w_df['time'].iloc[i+1], color=color, alpha=0.5)
    ax2.set_title("Reality B: Warped & Re-Optimized")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100); plt.close()
    
    v_a = [{"date": r['time'].strftime('%Y-%m'), "val": int(r['signal'])} for _, r in df.iterrows()]
    v_b = [{"date": r['time'].strftime('%Y-%m'), "val": int(r['signal'])} for _, r in w_df.iterrows()]
    
    return base64.b64encode(buf.getvalue()).decode(), v_a, v_b

@app.route('/')
def index():
    df = fetch_kraken_data(PAIR, INTERVAL)
    if df.empty: return "<h1>Data Error</h1>"
    p, v_a, v_b = create_plot_and_vector(df)
    
    html = """
    <!DOCTYPE html><html><head><title>Itertools Optimizer</title>
    <style>
        body { font-family: sans-serif; background: #f1f2f6; padding: 20px; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 15px; }
        img { width: 100%; border-radius: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 5px; height: 150px; overflow-y: auto; background: #eee; padding: 10px; }
        .v1 { background: #c3e6cb; } .v-1 { background: #f5c6cb; } .v0 { background: #e2e3e5; }
        .c { font-size: 10px; padding: 5px; text-align: center; border-radius: 3px; }
    </style></head><body>
    <div class="container">
        <h1>Itertools Brute-Force Backtest</h1>
        <p>Window Size: {{ws}} | Switching Penalty: {{sp}}</p>
        <img src="data:image/png;base64,{{p}}">
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
            <div><h4>Reality A Signals</h4><div class="grid">{% for i in v_a %}<div class="c v{{i.val}}">{{i.date}}<br>{{i.val}}</div>{% endfor %}</div></div>
            <div><h4>Reality B Signals</h4><div class="grid">{% for i in v_b %}<div class="c v{{i.val}}">{{i.date}}<br>{{i.val}}</div>{% endfor %}</div></div>
        </div>
    </div></body></html>
    """
    return render_template_string(html, p=p, v_a=v_a, v_b=v_b, ws=WINDOW_SIZE, sp=SWITCHING_PENALTY_WEIGHT)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
