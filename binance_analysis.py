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
# Stop Loss: 0.1% to 20% (Represented as 0.001 to 0.2)
SL_MIN, SL_MAX = 0.001, 0.20

# Updated GA Parameters (Doubled)
POPULATION_SIZE = 100
GENERATIONS = 30

# --- GLOBAL STATE ---
GLOBAL_STATE = {
    'status': 'Idle',
    'progress': 0,
    'best_params': {'weights': [1.0]*6, 'sl': 0.05},
    'metrics': {},
    'plot_url': None,
    'corr_html': None,
    'is_optimizing': False
}

# --- STRATEGY DEFINITIONS ---
STRAT_MACD_1H = {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]}
STRAT_MACD_4H = {'params': [(6, 8, 4), (84, 324, 96), (22, 86, 14)], 'weights': [0.29, 0.58, 0.64]}
STRAT_MACD_1D = {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]}

STRAT_SMA_1H = {'params': [10, 80, 380], 'weights': [0.0, 1.0, 0.8]}
STRAT_SMA_4H = {'params': [20, 120, 260], 'weights': [0.4, 0.4, 1.0]}
STRAT_SMA_1D = {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}

# --- DATA FETCHING ---
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
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

# --- RAW SIGNAL GENERATION ---
def calculate_macd_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite = np.zeros(len(prices))
    for (f, s, sig_p), w in zip(params, weights):
        fast = prices.ewm(span=f, adjust=False).mean()
        slow = prices.ewm(span=s, adjust=False).mean()
        macd = fast - slow
        sig_line = macd.ewm(span=sig_p, adjust=False).mean()
        # Sign: 1 if MACD>Sig, -1 if MACD<Sig
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
    # Handle NaNs from rolling
    composite = np.nan_to_num(composite)
    return composite / total_w if total_w > 0 else composite

def get_aligned_data():
    """Returns aligned signals (N x 6) and Price Data"""
    if hasattr(app, 'cached_data'): 
        return app.cached_data

    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    df_4h = df_1h.resample('4h').last().dropna()
    df_1d = df_1h.resample('1D').last().dropna()
    
    c1, c4, cd = df_1h['close'], df_4h['close'], df_1d['close']
    
    # Calc raw positions
    p1 = calculate_macd_pos(c1, STRAT_MACD_1H); p4 = calculate_sma_pos(c1, STRAT_SMA_1H)
    p2 = calculate_macd_pos(c4, STRAT_MACD_4H); p5 = calculate_sma_pos(c4, STRAT_SMA_4H)
    p3 = calculate_macd_pos(cd, STRAT_MACD_1D); p6 = calculate_sma_pos(cd, STRAT_SMA_1D)
    
    # Align HTF to 1H (Fix Lookahead: Shift HTF index to bar close time)
    # 4H closes +3h after open, 1D closes +23h after open
    s2 = pd.Series(p2, index=df_4h.index + pd.Timedelta(hours=3)).reindex(df_1h.index, method='ffill').fillna(0)
    s5 = pd.Series(p5, index=df_4h.index + pd.Timedelta(hours=3)).reindex(df_1h.index, method='ffill').fillna(0)
    s3 = pd.Series(p3, index=df_1d.index + pd.Timedelta(hours=23)).reindex(df_1h.index, method='ffill').fillna(0)
    s6 = pd.Series(p6, index=df_1d.index + pd.Timedelta(hours=23)).reindex(df_1h.index, method='ffill').fillna(0)
    
    signals = np.column_stack([p1, s2.values, s3.values, p4, s5.values, s6.values])
    
    # Cache everything needed for vectorized backtest
    # We need OHLC for accurate SL checks (Low for Longs, High for Shorts)
    app.cached_data = {
        'signals': signals,
        'open': df_1h['open'].values,
        'high': df_1h['high'].values,
        'low': df_1h['low'].values,
        'close': df_1h['close'].values,
        'index': df_1h.index
    }
    return app.cached_data

# --- VECTORIZED BACKTEST ENGINE ---

def apply_stop_loss_vectorized(raw_pos, sl_pct, open_p, low_p, high_p, close_p):
    """
    Applies Stop Loss logic vectorially based on Trade Regimes.
    A 'Trade Regime' is defined by the sign of the raw position.
    If Low < Entry * (1-SL) during a Long regime, kill pos.
    """
    # 1. Identify Regimes (Long=1, Short=-1, Neutral=0)
    # We use sign to group "continuous" trades. 
    # Even if leverage changes 0.5 -> 0.8, it's one "Long Trade".
    regime = np.sign(raw_pos)
    
    # 2. Create unique IDs for each contiguous regime
    # diff!=0 marks change points. cumsum gives unique ID.
    change_pts = (regime != np.roll(regime, 1))
    change_pts[0] = True
    trade_ids = np.cumsum(change_pts)
    
    # 3. Calculate 'Entry Price' for each candle (Vectorized)
    # Entry price is the OPEN price of the FIRST candle in the regime
    # We broadcast this entry price to all candles in the trade_id
    # (Simplified: In a continuous scaling strat, 'Avg Entry' is complex. 
    # We use 'Regime Start Open' as the anchor for the Stop Loss).
    
    # Get the index of the start of each trade
    _, start_indices = np.unique(trade_ids, return_index=True)
    entry_prices = open_p[start_indices] # Price at start of each trade
    
    # Map back to full array size
    # This creates an array where every bar has the EntryPrice of its current regime
    # We use searchsorted to map trade_ids to entry_prices indices efficiently
    # Since trade_ids is sorted (monotonic), this is valid.
    # Note: trade_ids starts at 1, so we adjust index
    current_entry_prices = entry_prices[trade_ids - 1]
    
    # 4. Check Stop Conditions
    # Long Stop: Low < Entry * (1 - SL)
    # Short Stop: High > Entry * (1 + SL)
    long_stop_mask = (regime == 1) & (low_p < current_entry_prices * (1 - sl_pct))
    short_stop_mask = (regime == -1) & (high_p > current_entry_prices * (1 + sl_pct))
    any_stop_mask = long_stop_mask | short_stop_mask
    
    # 5. Propagate Stops
    # If a stop is hit in a regime, ALL subsequent bars in that regime must be 0.
    # We use pandas groupby-transform (or numpy equivalent) to propagate "True" forward.
    # Since we are inside specific trade_ids, max() accumulation works.
    
    # For speed in pure numpy:
    # We need to set 'stopped' to True for the rest of the block if 'any_stop_mask' is True anywhere.
    # This is tricky in pure numpy without a loop over IDs.
    # Optimization: Use Pandas groupby (fast enough for 60k rows)
    
    df_temp = pd.DataFrame({'id': trade_ids, 'stopped': any_stop_mask})
    # cumul max propagates the True value down the group
    df_temp['stopped_prop'] = df_temp.groupby('id')['stopped'].cummax()
    
    final_mask = ~df_temp['stopped_prop'].values
    
    # Apply mask
    final_pos = raw_pos * final_mask
    return final_pos

def evaluate_individual(genome, data_dict):
    # Genome: [W1..W6, SL]
    weights = np.array(genome[:6])
    sl_pct = genome[6]
    
    signals = data_dict['signals']
    total_w = np.sum(weights)
    if total_w == 0: return -99.0
    
    # Raw Ensemble
    raw_pos = np.dot(signals, weights) / total_w
    
    # Apply SL
    clean_pos = apply_stop_loss_vectorized(
        raw_pos, sl_pct, 
        data_dict['open'], data_dict['low'], data_dict['high'], data_dict['close']
    )
    
    # Shift for execution (Signal at Close -> Trade at Open)
    final_pos = np.concatenate(([0], clean_pos[:-1]))
    
    # Returns
    returns = np.diff(data_dict['close']) / data_dict['close'][:-1]
    returns = np.concatenate(([0], returns))
    
    strat_rets = final_pos * returns
    
    # Sharpe
    mean = np.mean(strat_rets)
    std = np.std(strat_rets)
    if std < 1e-9: return -10.0
    return (mean / std) * np.sqrt(8760)

# --- GENETIC ALGORITHM ---
def create_individual():
    # 6 Weights (discrete steps) + 1 SL (continuous)
    weights = [random.choice(WEIGHT_STEPS) for _ in range(6)]
    sl = random.uniform(SL_MIN, SL_MAX)
    return weights + [sl]

def run_ga_thread():
    GLOBAL_STATE['is_optimizing'] = True
    GLOBAL_STATE['status'] = "Preprocessing Data..."
    GLOBAL_STATE['progress'] = 0
    
    data = get_aligned_data()
    
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_fitness = -999
    best_ind = None
    
    for gen in range(GENERATIONS):
        GLOBAL_STATE['status'] = f"Evolution: Gen {gen+1}/{GENERATIONS}"
        GLOBAL_STATE['progress'] = int((gen / GENERATIONS) * 100)
        
        scores = []
        for ind in population:
            f = evaluate_individual(ind, data)
            scores.append((ind, f))
            if f > best_fitness:
                best_fitness = f
                best_ind = ind[:]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        # Elitism
        next_gen = [x[0] for x in scores[:4]]
        
        while len(next_gen) < POPULATION_SIZE:
            # Tourney
            p1 = max(random.sample(scores, 4), key=lambda x:x[1])[0]
            p2 = max(random.sample(scores, 4), key=lambda x:x[1])[0]
            
            # Crossover
            pt = random.randint(1, 6)
            c1 = p1[:pt] + p2[pt:]
            
            # Mutate
            if random.random() < 0.3:
                idx = random.randint(0, 6)
                if idx < 6: # Weight
                    c1[idx] = random.choice(WEIGHT_STEPS)
                else: # SL
                    c1[idx] = random.uniform(SL_MIN, SL_MAX)
            
            next_gen.append(c1)
        population = next_gen

    # Finalize
    GLOBAL_STATE['best_params'] = {
        'weights': best_ind[:6],
        'sl': best_ind[6]
    }
    generate_report(best_ind, data)
    GLOBAL_STATE['status'] = "Optimization Completed"
    GLOBAL_STATE['progress'] = 100
    GLOBAL_STATE['is_optimizing'] = False

def generate_report(genome, data):
    weights = np.array(genome[:6])
    sl = genome[6]
    
    total_w = np.sum(weights)
    raw_pos = np.dot(data['signals'], weights) / total_w
    clean_pos = apply_stop_loss_vectorized(
        raw_pos, sl, data['open'], data['low'], data['high'], data['close']
    )
    
    final_pos = np.concatenate(([0], clean_pos[:-1]))
    returns = np.diff(data['close']) / data['close'][:-1]
    returns = np.concatenate(([0], returns))
    
    strat_rets = final_pos * returns
    cum_ret = np.cumprod(1 + strat_rets)
    
    # Metrics
    total_ret = (cum_ret[-1] - 1) * 100
    sharpe = (np.mean(strat_rets) / np.std(strat_rets)) * np.sqrt(8760)
    dd = (cum_ret - np.maximum.accumulate(cum_ret)) / np.maximum.accumulate(cum_ret)
    max_dd = np.min(dd) * 100
    
    GLOBAL_STATE['metrics'] = {
        'Total Return': f"{total_ret:,.2f}%",
        'Sharpe Ratio': f"{sharpe:.4f}",
        'Max Drawdown': f"{max_dd:.2f}%",
        'Optimal SL': f"{sl*100:.2f}%"
    }
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    idx = data['index']
    
    ax1.plot(idx, cum_ret, color='#00ff88', label=f'Optimized (SL: {sl*100:.1f}%)')
    ax1.set_yscale('log')
    ax1.set_title("Performance with Stop Loss")
    ax1.grid(True, alpha=0.1)
    
    ax2.plot(idx, final_pos, color='#00e5ff', linewidth=0.5)
    ax2.set_title("Position (0 = Stopped/Neutral)")
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
    if not GLOBAL_STATE['plot_url'] and not GLOBAL_STATE['is_optimizing']:
        threading.Thread(target=run_ga_thread).start()
        
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SL Ensemble Optimizer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <meta http-equiv="refresh" content="2">
        <style>
            body { background: #121212; color: #e0e0e0; font-family: 'Segoe UI'; padding: 20px; }
            .card { background: #1e1e1e; border: 1px solid #333; margin-bottom: 20px; }
            .live-step { border-left: 2px solid #00ff88; padding-left: 15px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="text-center text-success mb-4">🛡️ Ensemble + Stop Loss GA</h2>
            
            {% if state.is_optimizing %}
            <div class="alert alert-info text-center">
                <h4>Evolution in Progress... {{ state.progress }}%</h4>
                <small>Population: 100 | Generations: 30</small>
                <div class="progress mt-2" style="height: 5px;">
                    <div class="progress-bar" style="width: {{ state.progress }}%"></div>
                </div>
            </div>
            {% endif %}

            {% if state.plot_url %}
            <div class="row text-center mb-4">
                <div class="col-3"><h3>{{ state.metrics['Total Return'] }}</h3><small>Return</small></div>
                <div class="col-3"><h3 class="text-info">{{ state.metrics['Sharpe Ratio'] }}</h3><small>Sharpe</small></div>
                <div class="col-3"><h3 class="text-danger">{{ state.metrics['Max Drawdown'] }}</h3><small>Drawdown</small></div>
                <div class="col-3"><h3 class="text-warning">{{ state.metrics['Optimal SL'] }}</h3><small>Stop Loss</small></div>
            </div>

            <!-- Weights -->
            <div class="row mb-3">
                <div class="col-12"><h6 class="text-muted">Optimized Weights</h6></div>
                {% set labels = ['MACD 1H', 'MACD 4H', 'MACD 1D', 'SMA 1H', 'SMA 4H', 'SMA 1D'] %}
                {% for i in range(6) %}
                <div class="col-2">
                    <div class="card p-2 text-center">
                        <small>{{ labels[i] }}</small>
                        <div class="fw-bold" style="color: #00ff88">{{ state.best_params['weights'][i] }}</div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <div class="card p-2">
                <img src="data:image/png;base64,{{ state.plot_url }}" class="img-fluid">
            </div>

            <div class="card p-4 mt-4">
                <h3 class="text-white mb-4">🚀 Live Implementation with Stop Loss</h3>
                
                <div class="live-step">
                    <h5 class="text-success">Step 1: Calculate Target Position</h5>
                    <p>Every hour (e.g., 10:00:05), update signals and calculate the weighted target:</p>
                    <code class="bg-dark p-2 d-block rounded">
                        Target = SUM(Weight_i * Signal_i) / SUM(Weights)
                    </code>
                </div>

                <div class="live-step">
                    <h5 class="text-success">Step 2: Monitor Stop Loss (Continuous)</h5>
                    <p>Unlike the hourly signal, the Stop Loss must be monitored <strong>continuously</strong> or at least every minute.</p>
                    <p><strong>Logic:</strong></p>
                    <ul>
                        <li>Track the <strong>Regime Entry Price</strong>. This is the Open price of the hour when your position flipped from Short/Neutral to Long (or vice versa).</li>
                        <li><strong>Long Position:</strong> If <code>Current Price &lt; Entry * (1 - {{ state.metrics['Optimal SL'] }})</code> -> <strong>Close Position Immediately</strong>.</li>
                        <li><strong>Short Position:</strong> If <code>Current Price &gt; Entry * (1 + {{ state.metrics['Optimal SL'] }})</code> -> <strong>Close Position Immediately</strong>.</li>
                    </ul>
                </div>

                <div class="live-step">
                    <h5 class="text-success">Step 3: Re-Entry Logic</h5>
                    <p>If stopped out, your target override becomes <strong>0 (Flat)</strong>.</p>
                    <p>You remain flat until the <code>Target</code> calculated in Step 1 <strong>flips sign</strong> (e.g., goes from Long to Short) or crosses zero. This resets the "Regime" and allows a new entry.</p>
                </div>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """, state=GLOBAL_STATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
