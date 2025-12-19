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

# OPTIMIZATION CONFIG
WEIGHT_STEPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
POPULATION_SIZE = 50
GENERATIONS = 15

# --- GLOBAL STATE ---
GLOBAL_STATE = {
    'status': 'Idle',
    'progress': 0,
    'best_weights': [1.0] * 6, # Default to equal weights
    'metrics': {},
    'plot_url': None,
    'corr_html': None,
    'is_optimizing': False
}

# --- STRATEGY PARAMETERS ---
STRAT_MACD_1H = {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]}
STRAT_MACD_4H = {'params': [(6, 8, 4), (84, 324, 96), (22, 86, 14)], 'weights': [0.29, 0.58, 0.64]}
STRAT_MACD_1D = {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]}

STRAT_SMA_1H = {'params': [10, 80, 380], 'weights': [0.0, 1.0, 0.8]}
STRAT_SMA_4H = {'params': [20, 120, 260], 'weights': [0.4, 0.4, 1.0]}
STRAT_SMA_1D = {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}

# --- DATA & SIGNALS ---

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
    composite_signal = np.zeros(len(prices))
    
    for (fast, slow, sig_p), w in zip(params, weights):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig_p, adjust=False).mean()
        
        sig = np.where(macd_line > signal_line, 1.0, -1.0)
        composite_signal += (sig * w)
        
    total_w = sum(weights)
    if total_w == 0: return composite_signal
    return composite_signal / total_w

def calculate_sma_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite_signal = np.zeros(len(prices))
    
    for p, w in zip(params, weights):
        sma = prices.rolling(window=p).mean()
        sig = np.where(prices > sma, 1.0, -1.0)
        # Convert nan to 0
        sig = np.nan_to_num(sig)
        composite_signal += (sig * w)
        
    total_w = sum(weights)
    if total_w == 0: return composite_signal
    return composite_signal / total_w

def get_aligned_signals():
    """Fetches data and returns the 6 raw signal arrays aligned to 1H"""
    if hasattr(app, 'cached_signals'): return app.cached_signals, app.cached_returns, app.cached_index

    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    df_4h = df_1h.resample('4h').last().dropna()
    df_1d = df_1h.resample('1D').last().dropna()
    
    # 1. Calc Raw Positions
    p1 = calculate_macd_pos(df_1h['close'], STRAT_MACD_1H) # MACD 1H
    p2 = calculate_macd_pos(df_4h['close'], STRAT_MACD_4H) # MACD 4H
    p3 = calculate_macd_pos(df_1d['close'], STRAT_MACD_1D) # MACD 1D
    p4 = calculate_sma_pos(df_1h['close'], STRAT_SMA_1H)   # SMA 1H
    p5 = calculate_sma_pos(df_4h['close'], STRAT_SMA_4H)   # SMA 4H
    p6 = calculate_sma_pos(df_1d['close'], STRAT_SMA_1D)   # SMA 1D
    
    # 2. Convert to Series for Index Manip
    s1 = pd.Series(p1, index=df_1h.index)
    s2 = pd.Series(p2, index=df_4h.index)
    s3 = pd.Series(p3, index=df_1d.index)
    s4 = pd.Series(p4, index=df_1h.index)
    s5 = pd.Series(p5, index=df_4h.index)
    s6 = pd.Series(p6, index=df_1d.index)
    
    # 3. Shift HTF Indices (Lookahead Fix)
    s2.index = s2.index + pd.Timedelta(hours=3)
    s5.index = s5.index + pd.Timedelta(hours=3)
    s3.index = s3.index + pd.Timedelta(hours=23)
    s6.index = s6.index + pd.Timedelta(hours=23)
    
    # 4. Reindex to 1H
    target_idx = df_1h.index
    s2 = s2.reindex(target_idx, method='ffill').fillna(0)
    s3 = s3.reindex(target_idx, method='ffill').fillna(0)
    s5 = s5.reindex(target_idx, method='ffill').fillna(0)
    s6 = s6.reindex(target_idx, method='ffill').fillna(0)
    
    # 5. Pack into matrix (N x 6)
    signals = np.column_stack([
        s1.values, s2.values, s3.values, 
        s4.values, s5.values, s6.values
    ])
    
    returns = df_1h['close'].pct_change().fillna(0).values
    
    # Cache
    app.cached_signals = signals
    app.cached_returns = returns
    app.cached_index = df_1h.index
    
    return signals, returns, df_1h.index

# --- OPTIMIZATION ENGINE ---

def evaluate_ensemble(weights, signals, returns):
    """
    weights: list of 6 floats
    signals: (N x 6) matrix
    returns: (N,) array
    """
    w_arr = np.array(weights)
    total_w = np.sum(w_arr)
    if total_w == 0: return -10.0
    
    # Weighted Average Position (Normalized to 1.0)
    # Shape: (N,)
    ensemble_pos = np.dot(signals, w_arr) / total_w
    
    # Shift by 1 (Trade Execution Delay)
    ensemble_pos = np.concatenate(([0], ensemble_pos[:-1]))
    
    strat_rets = ensemble_pos * returns
    
    # Sharpe
    mean = np.mean(strat_rets)
    std = np.std(strat_rets)
    if std == 0: return -10.0
    sharpe = (mean / std) * np.sqrt(24 * 365)
    return sharpe

def run_optimization_task():
    GLOBAL_STATE['is_optimizing'] = True
    GLOBAL_STATE['status'] = "Fetching Data..."
    GLOBAL_STATE['progress'] = 0
    
    signals, returns, _ = get_aligned_signals()
    
    # Genetic Algorithm
    population = []
    for _ in range(POPULATION_SIZE):
        ind = [random.choice(WEIGHT_STEPS) for _ in range(6)]
        population.append(ind)
        
    best_sharpe = -999
    best_ind = [1.0] * 6
    
    for gen in range(GENERATIONS):
        GLOBAL_STATE['status'] = f"Optimizing Weights (Gen {gen+1}/{GENERATIONS})"
        GLOBAL_STATE['progress'] = int((gen / GENERATIONS) * 100)
        
        fitness_scores = []
        for ind in population:
            s = evaluate_ensemble(ind, signals, returns)
            fitness_scores.append((ind, s))
            if s > best_sharpe:
                best_sharpe = s
                best_ind = ind[:]
                
        # Selection
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        next_gen = [x[0] for x in fitness_scores[:5]] # Elitism
        
        while len(next_gen) < POPULATION_SIZE:
            # Tourney
            parent1 = max(random.sample(fitness_scores, 3), key=lambda x:x[1])[0]
            parent2 = max(random.sample(fitness_scores, 3), key=lambda x:x[1])[0]
            
            # Crossover
            pt = random.randint(1, 5)
            c1 = parent1[:pt] + parent2[pt:]
            
            # Mutate
            if random.random() < 0.2:
                idx = random.randint(0, 5)
                c1[idx] = random.choice(WEIGHT_STEPS)
                
            next_gen.append(c1)
            
        population = next_gen
        
    GLOBAL_STATE['best_weights'] = best_ind
    GLOBAL_STATE['status'] = "Optimization Complete"
    GLOBAL_STATE['progress'] = 100
    
    # Generate final results with best weights
    generate_final_report(best_ind)
    GLOBAL_STATE['is_optimizing'] = False

def generate_final_report(weights):
    signals, returns, idx = get_aligned_signals()
    
    w_arr = np.array(weights)
    total_w = np.sum(w_arr)
    ensemble_pos = np.dot(signals, w_arr) / total_w if total_w > 0 else np.zeros(len(returns))
    
    # Trade execution shift
    ensemble_pos_shifted = np.concatenate(([0], ensemble_pos[:-1]))
    strategy_returns = ensemble_pos_shifted * returns
    
    # Metrics
    cum_ret = np.cumprod(1 + strategy_returns)
    total_ret = (cum_ret[-1] - 1) * 100
    std = np.std(strategy_returns)
    sharpe = (np.mean(strategy_returns) / std * np.sqrt(8760)) if std != 0 else 0
    
    roll_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - roll_max) / roll_max
    max_dd = np.min(drawdown) * 100
    
    GLOBAL_STATE['metrics'] = {
        'Total Return': f"{total_ret:,.2f}%",
        'Sharpe Ratio': f"{sharpe:.4f}",
        'Max Drawdown': f"{max_dd:.2f}%"
    }
    
    # Correlation
    labels = ['MACD_1H', 'MACD_4H', 'MACD_1D', 'SMA_1H', 'SMA_4H', 'SMA_1D']
    sig_df = pd.DataFrame(signals, index=idx, columns=labels)
    sig_df['Ensemble'] = ensemble_pos
    GLOBAL_STATE['corr_html'] = sig_df.corr().round(2).to_html(classes='table table-dark table-sm', border=0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    base_ret = np.cumprod(1 + returns)
    base_ret = base_ret / base_ret[0]
    
    ax1.plot(idx, cum_ret, color='#00ff88', label='Optimized Ensemble')
    ax1.plot(idx, base_ret, color='white', alpha=0.3, label='BTC Buy & Hold')
    ax1.set_yscale('log')
    ax1.set_title("Strategy Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.1)
    
    ax2.plot(idx, ensemble_pos, color='#00e5ff', linewidth=0.5)
    ax2.set_title("Target Leverage (-1.0 to 1.0)")
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

# --- FLASK ---

@app.route('/')
def index():
    # Trigger optimization if not done/running
    if not GLOBAL_STATE['plot_url'] and not GLOBAL_STATE['is_optimizing']:
        threading.Thread(target=run_optimization_task).start()
        
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ensemble Optimizer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <meta http-equiv="refresh" content="2"> <!-- Auto refresh for updates -->
        <style>
            body { background: #121212; color: #e0e0e0; font-family: 'Segoe UI'; padding: 20px; }
            .card { background: #1e1e1e; border: 1px solid #333; margin-bottom: 20px; }
            .badge-strat { font-size: 0.9em; padding: 8px; width: 100%; display: block; }
            .live-step { border-left: 2px solid #00ff88; padding-left: 15px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="text-center mb-4 text-success">🧪 Optimized Ensemble Strategy</h2>
            
            {% if state.is_optimizing %}
            <div class="alert alert-info text-center">
                <h4>Optimizing Strategy Weights... {{ state.progress }}%</h4>
                <div class="progress" style="height: 5px;">
                    <div class="progress-bar" style="width: {{ state.progress }}%"></div>
                </div>
            </div>
            {% endif %}

            {% if state.plot_url %}
            <!-- Weights Display -->
            <div class="row mb-4">
                <div class="col-12"><h5 class="text-muted mb-3">Optimal Strategy Contribution (Weights)</h5></div>
                {% set labels = ['MACD 1H', 'MACD 4H', 'MACD 1D', 'SMA 1H', 'SMA 4H', 'SMA 1D'] %}
                {% for i in range(6) %}
                <div class="col-md-2 col-4 mb-2">
                    <div class="card p-2 text-center">
                        <small class="text-muted">{{ labels[i] }}</small>
                        <div class="fw-bold" style="color: #00ff88; font-size: 1.2em;">{{ state.best_weights[i] }}</div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- Metrics -->
            <div class="row mb-4 text-center">
                <div class="col-4">
                    <h3>{{ state.metrics['Total Return'] }}</h3><small>Total Return</small>
                </div>
                <div class="col-4">
                    <h3 class="text-info">{{ state.metrics['Sharpe Ratio'] }}</h3><small>Sharpe Ratio</small>
                </div>
                <div class="col-4">
                    <h3 class="text-danger">{{ state.metrics['Max Drawdown'] }}</h3><small>Max Drawdown</small>
                </div>
            </div>

            <div class="card p-2">
                <img src="data:image/png;base64,{{ state.plot_url }}" class="img-fluid">
            </div>
            
            <!-- LIVE TRADING INSTRUCTIONS -->
            <div class="card p-4 mt-4">
                <h3 class="text-white mb-4">🚀 Live Trading Implementation Guide</h3>
                
                <div class="live-step">
                    <h5 class="text-success">Step 1: Hourly Execution Cycle</h5>
                    <p>This strategy <strong>must</strong> be executed exactly once per hour, ideally 5-10 seconds after the hour closes (e.g., 09:00:05 UTC). This ensures the 1H candle is fully closed and data is finalized.</p>
                </div>

                <div class="live-step">
                    <h5 class="text-success">Step 2: Signal Updates</h5>
                    <ul>
                        <li><strong>Every Hour (e.g., 09:00, 10:00):</strong> Update the three 1H strategies (MACD 1H, SMA 1H) using the new closed candle.</li>
                        <li><strong>Every 4 Hours (00:00, 04:00...):</strong> If the current hour marks a 4H close, update the three 4H strategies. Otherwise, use the signal from the last closed 4H bar.</li>
                        <li><strong>Every Day (00:00):</strong> If the day just closed, update the three 1D strategies. Otherwise, use the signal from the last daily close.</li>
                    </ul>
                </div>

                <div class="live-step">
                    <h5 class="text-success">Step 3: Calculation & Rebalancing</h5>
                    <p>Calculate the <code>Target Position</code> using the optimized weights shown above:</p>
                    <code class="d-block bg-dark p-2 rounded mb-2">
                        Target = ( (W1 * Sig1) + (W2 * Sig2) + ... + (W6 * Sig6) ) / SUM(Weights)
                    </code>
                    <p>Example: If the result is <code>0.65</code>, your portfolio should be <strong>65% Long</strong>. If it is <code>-0.2</code>, you should be <strong>20% Short</strong>.</p>
                    <p><strong>Action:</strong> Compare <code>Target</code> with your <code>Current Position</code>. If they differ by more than a threshold (e.g., 5%), execute a trade to align them.</p>
                </div>
            </div>
            
            <div class="mt-4">
                <h5>Correlation Matrix</h5>
                {{ state.corr_html | safe }}
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """, state=GLOBAL_STATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
