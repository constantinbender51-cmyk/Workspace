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
from threading import Thread
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- Configuration ---
PAIR = 'XBTUSD' 
INTERVAL = 10080  # Weekly data
PORT = 8080
WINDOW_SIZE = 10
SWITCHING_PENALTY_WEIGHT = 35
ACTIONS = [-1, 0, 1] 
N_LAGS = 5  # Number of past periods to use as features

# --- Global Storage & State ---
CACHED_DATA = {
    "df": None,
    "plot_b64": None,
    "model_stats": {},
    "progress": 0,
    "status": "Idle",
    "ready": False
}

# --- Vectorization Cache ---
SEARCH_SPACE = {
    "matrix": None,           # shape (19683, 9)
    "internal_switches": None, # shape (19683,)
    "first_actions": None      # shape (19683,)
}

def precompute_search_space():
    """Generates the strategy matrix once at startup."""
    n_intervals = WINDOW_SIZE - 1
    all_seqs = list(itertools.product(ACTIONS, repeat=n_intervals))
    matrix = np.array(all_seqs, dtype=np.int8)
    
    internal_switches = np.sum(np.diff(matrix, axis=1) != 0, axis=1)
    
    SEARCH_SPACE["matrix"] = matrix
    SEARCH_SPACE["internal_switches"] = internal_switches
    SEARCH_SPACE["first_actions"] = matrix[:, 0]
    print(f"Search space optimized: {len(matrix)} permutations loaded into memory.")

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
        # Resample to Weekly (using Friday as anchor to align with crypto weeks roughly or just generic W)
        df_w = df.resample('W').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
        df_w['pair'] = pair
        return df_w
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def optimize_segment_matrix(segment_prices, last_action=None):
    """
    Finds the sequence of actions that maximizes returns for a specific window of future prices.
    This acts as the 'Teacher' generating the ideal labels.
    """
    n_intervals = len(segment_prices) - 1
    if n_intervals != (WINDOW_SIZE - 1):
        return [0] * n_intervals

    price_diffs = np.diff(segment_prices) # Shape (9,)
    
    strategy_returns = SEARCH_SPACE["matrix"] * price_diffs
    
    total_returns = np.sum(strategy_returns, axis=1)
    std_devs = np.std(strategy_returns, axis=1) + 1e-9
    risk_adj = total_returns / std_devs
    
    if last_action is not None:
        start_switch_cost = (SEARCH_SPACE["first_actions"] != last_action).astype(int)
    else:
        start_switch_cost = 0
        
    total_switches = SEARCH_SPACE["internal_switches"] + start_switch_cost
    penalty = (total_switches * SWITCHING_PENALTY_WEIGHT) / n_intervals
    
    final_scores = risk_adj - penalty
    best_idx = np.argmax(final_scores)
    
    return SEARCH_SPACE["matrix"][best_idx].tolist()

def generate_ideal_signals(df):
    """Generates the target variable (y) using the optimization engine."""
    prices = df['close'].values
    if len(prices) < WINDOW_SIZE:
        df['ideal_signal'] = 0
        return df
    
    full_sequence = []
    last_action = None
    step_size = WINDOW_SIZE - 1
    
    for i in range(0, len(prices) - 1, step_size):
        end_idx = min(i + WINDOW_SIZE, len(prices))
        segment = prices[i:end_idx]
        
        if len(segment) == WINDOW_SIZE:
            window_best_seq = optimize_segment_matrix(segment, last_action)
        else:
            window_best_seq = [0] * (len(segment)-1)
            
        full_sequence.extend(window_best_seq)
        if window_best_seq: last_action = window_best_seq[-1]
    
    signals = full_sequence[:len(df)]
    while len(signals) < len(df):
        signals.append(signals[-1] if signals else 0)
    
    df['ideal_signal'] = signals
    df['month_idx'] = df['time'].dt.month
    return df

def train_logistic_regression(df):
    """
    Trains a Logistic Regression model to predict the 'ideal_signal'
    based on lagged close prices and month index.
    """
    # 1. Feature Engineering
    # Create lag features for close prices
    feature_cols = ['month_idx']
    
    # We create normalized lag features (price relative to current price) 
    # to help the model generalize better than raw absolute numbers
    for i in range(1, N_LAGS + 1):
        col_name = f'close_lag_{i}'
        # Using pct_change or log returns is usually better, but request was "n periods close"
        # We will use raw lag close, but StandardScaling later helps.
        df[col_name] = df['close'].shift(i)
        feature_cols.append(col_name)

    # Drop NaNs created by shifting
    df_clean = df.dropna().copy()
    
    if len(df_clean) < 50:
        return df, {"error": "Not enough data to train"}

    X = df_clean[feature_cols]
    y = df_clean['ideal_signal']

    # 2. Split Data
    # We train on the first 80%, test on the last 20% to see if it learned the pattern
    split_idx = int(len(df_clean) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 5. Predict (on whole dataset for visualization)
    X_all_scaled = scaler.transform(X)
    predictions = model.predict(X_all_scaled)
    df_clean['predicted_signal'] = predictions

    # 6. Evaluation
    y_pred_test = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_test)
    
    stats = {
        "accuracy": round(acc * 100, 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "coefficients": model.coef_.tolist(),
        "classes": model.classes_.tolist()
    }
    
    return df_clean, stats

def processing_pipeline():
    precompute_search_space()

    # 1. Fetch Data
    CACHED_DATA["status"] = "Fetching Weekly BTC/USD..."
    CACHED_DATA["progress"] = 10
    df_raw = fetch_kraken_data(PAIR, INTERVAL)
    
    if df_raw.empty:
        CACHED_DATA["status"] = "API Error"
        return

    # 2. Generate Labels (The "Teacher")
    CACHED_DATA["status"] = "Calculating Ideal Trajectory..."
    CACHED_DATA["progress"] = 30
    df_labeled = generate_ideal_signals(df_raw)

    # 3. Train Model (The "Student")
    CACHED_DATA["status"] = "Training Logistic Regression..."
    CACHED_DATA["progress"] = 60
    df_final, stats = train_logistic_regression(df_labeled)
    CACHED_DATA["df"] = df_final
    CACHED_DATA["model_stats"] = stats

    # 4. Visualization
    CACHED_DATA["status"] = "Generating Plot..."
    CACHED_DATA["progress"] = 90
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor='#f8f9fa', sharex=True)
    
    # Plot 1: Price and Ideal Signals
    ax1.plot(df_final['time'], df_final['close'], color='#2d3436', linewidth=1.5, alpha=0.7, label='Price')
    
    # Color points based on Ideal Signal
    ideal = df_final['ideal_signal'].values
    colors_ideal = np.where(ideal == 1, '#00b894', np.where(ideal == -1, '#d63031', '#b2bec3'))
    ax1.scatter(df_final['time'], df_final['close'], c=colors_ideal, s=30, zorder=5, label='Ideal Entry')
    ax1.set_title(f"Ground Truth: Optimized Ideal Trajectory ({PAIR})", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Plot 2: Price and Predicted Signals
    ax2.plot(df_final['time'], df_final['close'], color='#2d3436', linewidth=1.5, alpha=0.3)
    
    pred = df_final['predicted_signal'].values
    colors_pred = np.where(pred == 1, '#0984e3', np.where(pred == -1, '#e17055', '#b2bec3'))
    
    # Offset markers slightly to distinguish from line
    ax2.scatter(df_final['time'], df_final['close'], c=colors_pred, s=30, zorder=5, marker='s')
    
    # Highlight Test Set Area
    split_idx = int(len(df_final) * 0.8)
    if split_idx < len(df_final):
        split_date = df_final['time'].iloc[split_idx]
        ax2.axvline(x=split_date, color='#636e72', linestyle='--')
        ax2.text(split_date, df_final['close'].min(), ' Test Data Start \u2192', color='#636e72', fontsize=9, verticalalignment='bottom')

    ax2.set_title(f"Model Prediction: Logistic Regression (Accuracy: {stats['accuracy']}%)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    CACHED_DATA["plot_b64"] = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    CACHED_DATA["progress"] = 100
    CACHED_DATA["status"] = "Training Complete"
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
        <!DOCTYPE html><html><head><title>AI Training</title>
        <style>
            body { font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; background: #2d3436; margin: 0; color: white;}
            .card { background: #353b48; padding: 40px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); width: 450px; text-align: center; border: 1px solid #4b4b4b;}
            .p-bar { background: #2f3640; height: 12px; border-radius: 6px; margin: 25px 0; overflow: hidden; border: 1px solid #7f8fa6;}
            .p-fill { background: linear-gradient(90deg, #00b894, #0984e3); height: 100%; width: 0%; transition: width 0.3s ease; }
            .status { color: #dfe6e9; font-weight: bold; margin-bottom: 5px; }
            .mono { font-family: monospace; color: #00b894; font-size: 0.9em; margin-top: 10px;}
        </style></head><body>
        <div class="card">
            <h2>Logistic Regression Trainer</h2>
            <div class="status" id="status">Initializing...</div>
            <div class="p-bar"><div class="p-fill" id="fill"></div></div>
            <div id="pct" style="font-weight:bold; color:#0984e3">0%</div>
            <div class="mono">Target: Weekly BTC/USD</div>
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

    df = CACHED_DATA["df"]
    # Get last 20 rows for table
    table_data = []
    if df is not None:
        tail = df.tail(20)
        for _, r in tail.iterrows():
            table_data.append({
                "date": r['time'].strftime('%Y-%m-%d'),
                "close": f"{r['close']:.2f}",
                "ideal": int(r['ideal_signal']),
                "pred": int(r['predicted_signal'])
            })
    
    html = """
    <!DOCTYPE html><html><head><title>Model Results</title>
    <style>
        body { font-family: -apple-system, sans-serif; background: #f1f2f6; padding: 25px; color: #2d3436; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 35px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.05); }
        .header { border-bottom: 1px solid #eee; margin-bottom: 25px; padding-bottom: 15px; }
        .stats-box { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat { background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #e1e1e1; }
        .stat h3 { margin: 0 0 5px 0; color: #0984e3; font-size: 24px; }
        .stat span { font-size: 12px; color: #636e72; text-transform: uppercase; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }
        th, td { padding: 10px; border-bottom: 1px solid #eee; text-align: center; }
        th { background: #f1f2f6; }
        .s1 { color: #00b894; font-weight: bold; } .s-1 { color: #d63031; font-weight: bold; } .s0 { color: #b2bec3; }
        .match-true { background: rgba(0, 184, 148, 0.1); }
        .match-false { background: rgba(214, 48, 49, 0.1); }
    </style></head><body>
    <div class="container">
        <div class="header">
            <h1>Logistic Regression vs Market Reality</h1>
            <p>Training Features: Last {{ lags }} weekly closes + Month Index | Target: {{ pair }}</p>
        </div>
        
        <div class="stats-box">
            <div class="stat"><h3>{{ stats.accuracy }}%</h3><span>Test Accuracy</span></div>
            <div class="stat"><h3>{{ stats.train_size }}</h3><span>Training Weeks</span></div>
            <div class="stat"><h3>{{ stats.test_size }}</h3><span>Test Weeks</span></div>
        </div>

        <img src="data:image/png;base64,{{p}}" style="width:100%; border-radius:10px; border: 1px solid #eee;">
        
        <h3>Recent Predictions</h3>
        <table>
            <thead><tr><th>Date</th><th>Close</th><th>Ideal (Target)</th><th>Predicted (Model)</th><th>Result</th></tr></thead>
            <tbody>
            {% for row in table %}
                <tr>
                    <td>{{ row.date }}</td>
                    <td>${{ row.close }}</td>
                    <td class="s{{ row.ideal }}">{{ row.ideal }}</td>
                    <td class="s{{ row.pred }}">{{ row.pred }}</td>
                    <td>
                        {% if row.ideal == row.pred %}<span style="color:#00b894">MATCH</span>
                        {% else %}<span style="color:#d63031">MISS</span>{% endif %}
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div></body></html>
    """
    return render_template_string(html, p=CACHED_DATA["plot_b64"], table=table_data, stats=CACHED_DATA["model_stats"], pair=PAIR, lags=N_LAGS)

if __name__ == '__main__':
    Thread(target=processing_pipeline, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT)