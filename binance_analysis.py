import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, jsonify
from datetime import datetime
import time
import random
import threading

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTCUSDT'
START_YEAR = 2018

# OPTIMIZATION CONFIG (No Stop Loss)
WEIGHT_STEPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
POPULATION_SIZE = 100
GENERATIONS = 30

# --- GLOBAL STATE ---
GLOBAL_STATE = {
    'status': 'Idle',
    'progress': 0,
    'best_weights': [1.0] * 6,
    'metrics': {},
    'plot_url': None,
    'corr_html': None,
    'is_optimizing': False
}

# --- STRATEGY DEFINITIONS (The "Found Specs") ---
STRAT_MACD_1H = {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]}
STRAT_MACD_4H = {'params': [(6, 8, 4), (84, 324, 96), (22, 86, 14)], 'weights': [0.29, 0.58, 0.64]}
STRAT_MACD_1D = {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]}

STRAT_SMA_1H = {'params': [10, 80, 380], 'weights': [0.0, 1.0, 0.8]}
STRAT_SMA_4H = {'params': [20, 120, 260], 'weights': [0.4, 0.4, 1.0]}
STRAT_SMA_1D = {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}

# --- DATA FETCHING & PROCESSING ---
def fetch_binance_data(symbol, interval, start_year):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    limit = 1000
    all_data = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_start, 'limit': limit}
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data: break
            all_data.extend(data)
            current_start = data[-1][0] + 1
            if len(data) < limit: break 
            time.sleep(0.05)
        except: break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['close']]

def calculate_macd_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite = np.zeros(len(prices))
    for (f, s, sig_p), w in zip(params, weights):
        fast = prices.ewm(span=f, adjust=False).mean()
        slow = prices.ewm(span=s, adjust=False).mean()
        macd = fast - slow
        sig_line = macd.ewm(span=sig_p, adjust=False).mean()
        composite += np.where(macd > sig_line, 1.0, -1.0) * w
    total_w = sum(weights)
    return composite / total_w if total_w > 0 else composite

def calculate_sma_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite = np.zeros(len(prices))
    for p, w in zip(params, weights):
        sma = prices.rolling(window=p).mean()
        composite += np.where(prices > sma, 1.0, -1.0) * w
    total_w = sum(weights)
    composite = np.nan_to_num(composite)
    return composite / total_w if total_w > 0 else composite

def get_aligned_data():
    """Returns signals (N x 6) and returns aligned to 1H"""
    if hasattr(app, 'cached_data'): return app.cached_data

    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    df_4h = df_1h.resample('4h').last().dropna()
    df_1d = df_1h.resample('1D').last().dropna()
    
    # Calculate Signals
    p1 = calculate_macd_pos(df_1h['close'], STRAT_MACD_1H)
    p2 = calculate_macd_pos(df_4h['close'], STRAT_MACD_4H)
    p3 = calculate_macd_pos(df_1d['close'], STRAT_MACD_1D)
    p4 = calculate_sma_pos(df_1h['close'], STRAT_SMA_1H)
    p5 = calculate_sma_pos(df_4h['close'], STRAT_SMA_4H)
    p6 = calculate_sma_pos(df_1d['close'], STRAT_SMA_1D)
    
    # Align HTF to 1H (Fix Lookahead)
    s2 = pd.Series(p2, index=df_4h.index + pd.Timedelta(hours=3)).reindex(df_1h.index, method='ffill').fillna(0)
    s3 = pd.Series(p3, index=df_1d.index + pd.Timedelta(hours=23)).reindex(df_1h.index, method='ffill').fillna(0)
    s5 = pd.Series(p5, index=df_4h.index + pd.Timedelta(hours=3)).reindex(df_1h.index, method='ffill').fillna(0)
    s6 = pd.Series(p6, index=df_1d.index + pd.Timedelta(hours=23)).reindex(df_1h.index, method='ffill').fillna(0)
    
    signals = np.column_stack([p1, s2.values, s3.values, p4, s5.values, s6.values])
    returns = df_1h['close'].pct_change().fillna(0).values
    
    app.cached_data = {'signals': signals, 'returns': returns, 'index': df_1h.index}
    return app.cached_data

# --- OPTIMIZATION ENGINE (WEIGHTS ONLY) ---

def evaluate_ensemble(weights, signals, returns):
    w_arr = np.array(weights)
    total_w = np.sum(w_arr)
    if total_w == 0: return -10.0
    
    # Position = Weighted Average (Normalized -1 to 1)
    ensemble_pos = np.dot(signals, w_arr) / total_w
    
    # Shift for Execution (Close -> Open)
    ensemble_pos = np.concatenate(([0], ensemble_pos[:-1]))
    
    strat_rets = ensemble_pos * returns
    
    mean = np.mean(strat_rets)
    std = np.std(strat_rets)
    if std < 1e-9: return -10.0
    # Annualized Sharpe (1H data)
    return (mean / std) * np.sqrt(8760)

def run_optimization():
    GLOBAL_STATE['is_optimizing'] = True
    GLOBAL_STATE['progress'] = 0
    
    data = get_aligned_data()
    signals = data['signals']
    returns = data['returns']
    
    population = [[random.choice(WEIGHT_STEPS) for _ in range(6)] for _ in range(POPULATION_SIZE)]
    
    best_sharpe = -999
    best_weights = [1.0] * 6
    
    for gen in range(GENERATIONS):
        GLOBAL_STATE['progress'] = int((gen / GENERATIONS) * 100)
        GLOBAL_STATE['status'] = f"Optimizing Weights (Gen {gen+1}/{GENERATIONS})"
        
        scores = []
        for ind in population:
            s = evaluate_ensemble(ind, signals, returns)
            scores.append((ind, s))
            if s > best_sharpe:
                best_sharpe = s
                best_weights = ind[:]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        next_gen = [x[0] for x in scores[:4]] # Elitism
        
        while len(next_gen) < POPULATION_SIZE:
            p1 = max(random.sample(scores, 4), key=lambda x:x[1])[0]
            p2 = max(random.sample(scores, 4), key=lambda x:x[1])[0]
            
            # Crossover
            pt = random.randint(1, 5)
            c1 = p1[:pt] + p2[pt:]
            
            # Mutate
            if random.random() < 0.2:
                idx = random.randint(0, 5)
                c1[idx] = random.choice(WEIGHT_STEPS)
            next_gen.append(c1)
            
        population = next_gen

    GLOBAL_STATE['best_weights'] = best_weights
    generate_final_report(best_weights, data)
    GLOBAL_STATE['is_optimizing'] = False
    GLOBAL_STATE['progress'] = 100
    GLOBAL_STATE['status'] = "Done"

def generate_final_report(weights, data):
    signals = data['signals']
    returns = data['returns']
    
    w_arr = np.array(weights)
    total_w = np.sum(w_arr)
    ensemble_pos = np.dot(signals, w_arr) / total_w
    
    # Execution Shift
    final_pos = np.concatenate(([0], ensemble_pos[:-1]))
    strat_rets = final_pos * returns
    
    cum_ret = np.cumprod(1 + strat_rets)
    total_ret = (cum_ret[-1] - 1) * 100
    
    std = np.std(strat_rets)
    sharpe = (np.mean(strat_rets) / std * np.sqrt(8760)) if std > 0 else 0
    
    dd = (cum_ret - np.maximum.accumulate(cum_ret)) / np.maximum.accumulate(cum_ret)
    max_dd = np.min(dd) * 100
    
    GLOBAL_STATE['metrics'] = {
        'Total Return': f"{total_ret:,.2f}%",
        'Sharpe Ratio': f"{sharpe:.4f}",
        'Max Drawdown': f"{max_dd:.2f}%"
    }
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    idx = data['index']
    base_curve = np.cumprod(1 + returns)
    base_curve = base_curve / base_curve[0]
    
    ax1.plot(idx, cum_ret, color='#00ff88', label='Ensemble Strategy')
    ax1.plot(idx, base_curve, color='white', alpha=0.3, label='BTC Buy & Hold')
    ax1.set_yscale('log')
    ax1.set_title("Equity Curve (Log Scale)")
    ax1.legend()
    ax1.grid(True, alpha=0.1)
    
    ax2.plot(idx, final_pos, color='#00e5ff', linewidth=0.5)
    ax2.set_title("Net Leverage (-1.0 to 1.0)")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.1)
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e1e1e')
        for s in ax.spines.values(): s.set_color('#444')
        ax.tick_params(colors='white')
        
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    GLOBAL_STATE['plot_url'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Correlation
    labels = ['MACD 1H', 'MACD 4H', 'MACD 1D', 'SMA 1H', 'SMA 4H', 'SMA 1D']
    sig_df = pd.DataFrame(signals, index=idx, columns=labels)
    sig_df['Ensemble'] = ensemble_pos
    GLOBAL_STATE['corr_html'] = sig_df.corr().round(2).to_html(classes='table table-dark table-sm', border=0)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    if not GLOBAL_STATE['plot_url'] and not GLOBAL_STATE['is_optimizing']:
        threading.Thread(target=run_optimization).start()
        
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Ensemble Strategy Final Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <meta http-equiv="refresh" content="3">
        <style>
            body { background-color: #0f0f0f; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            .header-section { padding: 40px 0; border-bottom: 1px solid #333; margin-bottom: 30px; }
            .metric-card { background: #1a1a1a; border: 1px solid #333; padding: 20px; border-radius: 8px; text-align: center; }
            .metric-val { font-size: 2rem; font-weight: 700; margin-bottom: 5px; }
            .card { background: #1a1a1a; border: 1px solid #333; margin-bottom: 20px; }
            .weight-box { background: #252525; border-radius: 6px; padding: 10px; text-align: center; height: 100%; }
            .guide-step { border-left: 3px solid #00ff88; padding-left: 15px; margin-bottom: 25px; }
            h2, h4, h5 { color: #f0f0f0; }
            .text-accent { color: #00ff88; }
            .text-warn { color: #ffab00; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-section text-center">
                <h1 class="display-5 fw-bold text-accent">Ensemble Strategy Report</h1>
                <p class="lead text-muted">Multi-Timeframe MACD & SMA • Dynamic Weighting • Risk Normalized</p>
            </div>

            {% if state.is_optimizing %}
            <div class="alert alert-dark text-center border-secondary">
                <h4>Running Final Optimization... {{ state.progress }}%</h4>
                <div class="progress mt-2" style="height: 6px;">
                    <div class="progress-bar bg-success" style="width: {{ state.progress }}%"></div>
                </div>
            </div>
            {% endif %}

            {% if state.plot_url %}
            <!-- Executive Summary -->
            <div class="row mb-5">
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-val text-accent">{{ state.metrics['Total Return'] }}</div>
                        <div class="text-muted text-uppercase small">Total Return (2018-Now)</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-val text-info">{{ state.metrics['Sharpe Ratio'] }}</div>
                        <div class="text-muted text-uppercase small">Sharpe Ratio</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <div class="metric-val text-danger">{{ state.metrics['Max Drawdown'] }}</div>
                        <div class="text-muted text-uppercase small">Max Drawdown</div>
                    </div>
                </div>
            </div>

            <!-- Optimized Composition -->
            <div class="card p-4 mb-4">
                <h4 class="mb-4">Strategy Composition (Optimized Contribution)</h4>
                <div class="row g-3">
                    {% set labels = ['MACD 1H', 'MACD 4H', 'MACD 1D', 'SMA 1H', 'SMA 4H', 'SMA 1D'] %}
                    {% for i in range(6) %}
                    <div class="col-6 col-md-2">
                        <div class="weight-box">
                            <div class="small text-muted mb-1">{{ labels[i] }}</div>
                            <div class="h3 fw-bold text-white">{{ state.best_weights[i] }}</div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                <div class="mt-3 text-muted small text-center">
                    Values represent the relative weight (0.0 - 1.0) of each sub-strategy in the final vote.
                </div>
            </div>

            <!-- Charts -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card p-2">
                        <img src="data:image/png;base64,{{ state.plot_url }}" class="img-fluid rounded">
                    </div>
                </div>
            </div>

            <!-- Implementation Guide -->
            <div class="row">
                <div class="col-lg-7">
                    <div class="card p-4 h-100">
                        <h4 class="mb-4 text-accent">🚀 Live Execution Protocol</h4>
                        
                        <div class="guide-step">
                            <h5>1. Trading Schedule</h5>
                            <p class="mb-0 text-muted">Execute logic <strong>hourly</strong>, immediately after the candle close (e.g., HH:00:05). This ensures signals align with the backtest.</p>
                        </div>

                        <div class="guide-step">
                            <h5>2. Data Gathering</h5>
                            <p class="mb-0 text-muted">
                                - <strong>1H:</strong> Update every hour.<br>
                                - <strong>4H:</strong> Update only on 4H closes (00, 04, 08...).<br>
                                - <strong>1D:</strong> Update only on Daily close (00:00 UTC).
                            </p>
                        </div>

                        <div class="guide-step">
                            <h5>3. Position Calculation</h5>
                            <code class="d-block bg-black p-3 rounded mb-2 text-warning">
                                Net_Signal = (W1*S1 + W2*S2 + ... + W6*S6) / Sum(Weights)
                            </code>
                            <p class="mb-0 text-muted">Result is a value between -1.0 (Short) and 1.0 (Long).</p>
                        </div>

                        <div class="guide-step">
                            <h5>4. Execution</h5>
                            <p class="mb-0 text-muted">Adjust exchange position to match <code>Net_Signal</code>. <br>Example: If Net_Signal is <strong>0.65</strong>, ensure your Net Long exposure equals 65% of your trading capital.</p>
                        </div>
                    </div>
                </div>

                <!-- Correlations -->
                <div class="col-lg-5">
                    <div class="card p-4 h-100">
                        <h5 class="mb-3">Signal Correlation</h5>
                        <p class="small text-muted">Low correlation between sub-strategies improves diversification and reduces drawdown.</p>
                        <div class="table-responsive">
                            {{ state.corr_html | safe }}
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}
            
            <footer class="text-center py-4 text-muted small">
                Generated by Ensemble Strategy Optimizer • No Financial Advice
            </footer>
        </div>
    </body>
    </html>
    """, state=GLOBAL_STATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
