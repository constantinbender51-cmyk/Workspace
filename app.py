# Your full file will be inserted here for editing.
# Please paste your entire app.py content (or confirm you want me to insert the previously provided version).

import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask
import time
import random
from threading import Thread

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' 
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

app = Flask(__name__)

# --- Global Storage ---
global_data = {
    'raw_data': None,
    'optimization_status': "Idle",
    'top_performers': [],
    'generation_info': [],
    'walk_forward_status': "Idle",
    'walk_forward_results': None,
    'walk_forward_equity_curve': None
}

# --- Data Fetching ---
def fetch_data(symbol, timeframe, start_str):
    print(f"Fetching {symbol} data from Binance starting {start_str}...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    
    ohlcv_list = []
    current_ts = start_ts
    now_ts = exchange.milliseconds()
    
    while current_ts < now_ts:
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts)
            if not ohlcvs: break
            current_ts = ohlcvs[-1][0] + 1
            ohlcv_list += ohlcvs
            time.sleep(0.05) 
        except Exception as e:
            print(f"Error fetching: {e}")
            break

    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"Data loaded: {len(df)} rows.")
    return df

# --- Optimized Strategy Engine ---
def run_strategy_dynamic(df, params):
    # Unpack Parameters
    p_sma_fast = int(params['sma_fast'])
    p_sma_slow = int(params['sma_slow'])
    p_w = int(params['w'])
    p_u = params['u']
    p_y = params['y']
    
    p_lev_th_low = params['lev_thresh_low']
    p_lev_th_high = params['lev_thresh_high']
    p_lev_low = params['lev_low']
    p_lev_mid = params['lev_mid']
    p_lev_high = params['lev_high']
    
    p_tp = params['tp_pct']
    p_sl = params['sl_pct']

    # Vectorized Calculations
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # 1. Indicators
    log_ret = np.log(closes[1:] / closes[:-1])
    log_ret = np.insert(log_ret, 0, 0)
    
    s_fast = df['close'].rolling(window=p_sma_fast).mean().values
    s_slow = df['close'].rolling(window=p_sma_slow).mean().values
    
    # 2. Efficiency Ratio (iii)
    roll_sum = pd.Series(log_ret).rolling(window=p_w).sum()
    roll_abs_sum = pd.Series(np.abs(log_ret)).rolling(window=p_w).sum()
    iii = (roll_sum.abs() / roll_abs_sum).fillna(0).values
    
    # 3. Signals Loop
    n = len(closes)
    returns = np.zeros(n)
    is_flat = False
    
    # Pre-calculate leverage array (shifted)
    iii_shifted = np.roll(iii, 1)
    iii_shifted[0] = 0
    
    # LEVERAGE ASSIGNMENT LOGIC:
    # Tier 1 (Lowest III): lev_low
    # Tier 2 (Middle III): lev_high
    # Tier 3 (Highest III): lev_mid (Default)
    lev_arr = np.full(n, p_lev_mid)
    lev_arr[iii_shifted < p_lev_th_high] = p_lev_high
    lev_arr[iii_shifted < p_lev_th_low] = p_lev_low
    
    start_idx = max(p_sma_slow, p_w) + 1
    
    for i in range(start_idx, n):
        # A. Regime Logic
        if iii_shifted[i] < p_u:
            is_flat = True
            
        if is_flat:
            prev_c = closes[i-1]
            prev_sf = s_fast[i-1]
            prev_ss = s_slow[i-1]
            
            if (abs(prev_c - prev_sf) <= prev_sf * p_y) or \
               (abs(prev_c - prev_ss) <= prev_ss * p_y):
                is_flat = False
        
        # B. Signal
        sig = 0
        if not is_flat:
            pc = closes[i-1]
            pf = s_fast[i-1]
            ps = s_slow[i-1]
            if pc > pf and pc > ps: sig = 1
            elif pc < pf and pc < ps: sig = -1
        
        # C. Returns
        if sig == 0: continue
            
        lev = lev_arr[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        
        # SL/TP Logic
        dr = 0.0
        if sig == 1:
            stop_price = o * (1 - p_sl)
            take_price = o * (1 + p_tp)
            if l <= stop_price: dr = -p_sl
            elif h >= take_price: dr = p_tp
            else: dr = (c - o) / o
        else:
            stop_price = o * (1 + p_sl)
            take_price = o * (1 - p_tp)
            if h >= stop_price: dr = -p_sl
            elif l <= take_price: dr = p_tp
            else: dr = (o - c) / o
            
        returns[i] = dr * lev

    # 4. Metrics
    if np.all(returns == 0): return 0, 0
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(365)) if np.std(returns) != 0 else 0
    total_ret = np.prod(1 + returns)
    return total_ret, sharpe

# --- Genetic Algorithm Components ---

class Individual:
    def __init__(self, params=None):
        if params:
            self.params = params.copy()
        else:
            self.params = {}
            self.random_init()
        self.sharpe = -999
        self.ret = 0
    
    def random_init(self):
        # BLIND INITIALIZATION - FULL UNBIASED RANGES
        self.params = {
            'sma_fast': random.randint(10, 80),
            'sma_slow': random.randint(60, 200),
            'w': random.randint(10, 100),
            'u': random.uniform(0.05, 0.40),
            'y': random.uniform(0.01, 0.10),
            'lev_thresh_low': random.uniform(0.05, 0.25),
            'lev_thresh_high': random.uniform(0.15, 0.40),
            
            # UNBIASED LEVERAGE: Any regime can have any leverage
            'lev_low': random.uniform(0.0, 5.0),  
            'lev_mid': random.uniform(0.0, 5.0),
            'lev_high': random.uniform(0.0, 5.0),
            
            'tp_pct': random.uniform(0.05, 0.30),
            'sl_pct': random.uniform(0.01, 0.10)
        }
        self.fix_constraints()

    def fix_constraints(self):
        if self.params['sma_fast'] >= self.params['sma_slow']:
            self.params['sma_fast'] = int(self.params['sma_slow'] * 0.5)
            
        if self.params['lev_thresh_low'] > self.params['lev_thresh_high']:
             self.params['lev_thresh_low'] = self.params['lev_thresh_high'] * 0.9

        self.params['sma_fast'] = int(self.params['sma_fast'])
        self.params['sma_slow'] = int(self.params['sma_slow'])
        self.params['w'] = int(self.params['w'])

    def mutate(self):
        key = random.choice(list(self.params.keys()))
        val = self.params[key]
        if isinstance(val, int):
            change = random.choice([-5, -1, 1, 5])
            self.params[key] = max(2, val + change)
        else:
            change = random.uniform(0.85, 1.15)
            self.params[key] = val * change
        self.fix_constraints()

def run_genetic_optimization(df):
    # --- HYPERPARAMETERS ---
    POP_SIZE = 300
    GENERATIONS = 50
    ELITISM_PCT = 0.2

    # SPLIT DATA (IS / OOS)
    cutoff = len(df) // 2
    train_df = df.iloc[:cutoff].copy()
    test_df = df.iloc[cutoff:].copy()
    
    print(f"Split Data: Training on {len(train_df)}, Testing on {len(test_df)}")

    global_data['optimization_status'] = "Initializing..."
    population = []
    
    for _ in range(POP_SIZE):
        population.append(Individual())
        
    for gen in range(GENERATIONS):
        global_data['optimization_status'] = f"Generation {gen+1}/{GENERATIONS}"
        print(f"--- Generation {gen+1} ---")
        
        for ind in population:
            if ind.sharpe == -999:
                ret, sharpe = run_strategy_dynamic(train_df, ind.params)
                ind.ret = ret
                ind.sharpe = sharpe
        
        population.sort(key=lambda x: x.sharpe, reverse=True)
        best = population[0]
        print(f"Gen {gen+1} Best Sharpe {best.sharpe:.3f}")

        global_data['generation_info'].append({
            'gen': gen+1, 'best_sharpe': best.sharpe
        })
        
        top_performers_snapshot = []
        for p in population[:5]:
             test_ret, test_sharpe = run_strategy_dynamic(test_df, p.params)
             top_performers_snapshot.append({
                 'params': p.params, 
                 'train_sharpe': p.sharpe, 
                 'train_ret': p.ret,
                 'test_sharpe': test_sharpe,
                 'test_ret': test_ret
             })
             
        global_data['top_performers'] = top_performers_snapshot
        
        next_gen = []
        elite_count = int(POP_SIZE * ELITISM_PCT)
        next_gen.extend(population[:elite_count])
        
        while len(next_gen) < POP_SIZE:
            p1 = random.choice(population[:POP_SIZE//2])
            p2 = random.choice(population[:POP_SIZE//2])
            child = Individual(p1.params)
            for k in child.params:
                if random.random() > 0.5:
                    child.params[k] = p2.params[k]
            if random.random() < 0.2:
                child.mutate()
            child.fix_constraints()
            next_gen.append(child)
            
        population = next_gen

    global_data['optimization_status'] = "Complete"
    print("Optimization Finished.")

def run_walk_forward_optimization(df):
    """Run walk-forward optimization and store results."""
    global_data['walk_forward_status'] = "Running..."
    print("Starting walk-forward optimization...")
    
    try:
        results_df, equity_curve = walk_forward_rolling(df, window=14)
        global_data['walk_forward_results'] = results_df
        global_data['walk_forward_equity_curve'] = equity_curve
        global_data['walk_forward_status'] = "Complete"
        print(f"Walk-forward optimization complete. Processed {len(results_df)} days.")
    except Exception as e:
        global_data['walk_forward_status'] = f"Error: {str(e)}"
        print(f"Walk-forward error: {e}")

# ============================================================
# === NEW: 14-DAY ROLLING WALK-FORWARD OPTIMIZATION ADDED HERE ===
# ============================================================

def walk_forward_rolling(df, window=14):
    """
    Rolling 14-day walk-forward optimization.
    For each day t:
        - optimize on [t-window : t-1]
        - trade on day t
    """
    results = []
    equity = 1.0
    equity_curve = [equity]

    for t in range(window, len(df)):
        train_df = df.iloc[t-window:t]
        trade_df = df.iloc[t:t+1]

        # small quick GA
        POP = 40
        GEN = 12
        population = [Individual() for _ in range(POP)]

        for _ in range(GEN):
            for ind in population:
                if ind.sharpe == -999:
                    ret, sharpe = run_strategy_dynamic(train_df, ind.params)
                    ind.sharpe = sharpe
                    ind.ret = ret

            population.sort(key=lambda x: x.sharpe, reverse=True)
            next_gen = population[:8]

            while len(next_gen) < POP:
                p1 = random.choice(population[:20])
                p2 = random.choice(population[:20])
                child = Individual(p1.params)
                for k in child.params:
                    if random.random() < 0.5:
                        child.params[k] = p2.params[k]
                if random.random() < 0.2:
                    child.mutate()
                child.fix_constraints()
                next_gen.append(child)

            population = next_gen

        best = population[0].params

        ret, _ = run_strategy_dynamic(trade_df, best)
        equity *= (ret if ret > 0 else 1 + ret)
        equity_curve.append(equity)

        results.append({
            "day": df.index[t],
            "params": best,
            "trade_return": ret,
            "equity": equity
        })

        print(f"[WFO] {df.index[t].date()}  ret={ret:.3f}  eq={equity:.3f}")

    return pd.DataFrame(results), equity_curve

# ============================================================


# --- Routes ---

@app.route('/')
def index():
    status = global_data['optimization_status']
    top = global_data['top_performers']
    gen_stats = global_data['generation_info']
    
    html = f"""
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        th {{ background-color: #4CAF50; color: white; }}
        .btn {{ padding: 10px 20px; background: blue; color: white; text-decoration: none; border-radius: 5px; }}
        .highlight {{ color: red; font-weight: bold; }}
    </style>
    <h1>Genetic Optimizer (IS/OOS Split)</h1>
    <p><strong>Training Data:</strong> First 50% | <strong>Test Data:</strong> Second 50%</p>
    <h3>Status: {status}</h3>
    <a href="/start" class="btn">START OPTIMIZATION</a>
    <br><br>
    <h2>Top 5 Results</h2>
    """
    if top:
        html += """
        <table>
            <tr>
                <th>Train Sharpe (IS)</th>
                <th>Test Sharpe (OOS)</th>
                <th>Train Return</th>
                <th>Test Return</th>
                <th>Parameters</th>
            </tr>
        """
        for t in top:
            test_style = "color: green; font-weight: bold;" if t['test_sharpe'] > 1.5 else ""
            p_formatted = ", ".join([f"{k}:{v:.3f}" if isinstance(v, float) else f"{k}:{v}" for k,v in t['params'].items()])
            
            html += f"""
            <tr>
                <td>{t['train_sharpe']:.3f}</td>
                <td style="{test_style}">{t['test_sharpe']:.3f}</td>
                <td>{t['train_ret']:.2f}x</td>
                <td>{t['test_ret']:.2f}x</td>
                <td style="font-size: 0.8em">{p_formatted}</td>
            </tr>
            """
        html += "</table>"
    else:
        html += "<p>No results yet.</p>"
    # Walk-forward section
    wf_status = global_data['walk_forward_status']
    wf_results = global_data['walk_forward_results']
    wf_equity = global_data['walk_forward_equity_curve']
    html += f"""
    <div class="section">
        <h1>Walk-Forward Optimization (14-Day Rolling)</h1>
        <p><strong>Method:</strong> For each day, optimize on previous 14 days, trade on current day</p>
        <h3>Walk-Forward Status: {wf_status}</h3>
        <a href="/start_walk_forward" class="btn btn-wf">START WALK-FORWARD OPTIMIZATION</a>
        <br><br>"""
    
    if wf_results is not None and not wf_results.empty:
        html += f"""
        <h2>Walk-Forward Results ({len(wf_results)} trading days)</h2>
        <p><strong>Final Equity:</strong> {wf_equity[-1]:.3f}x (from 1.0x starting)</p>
        <p><strong>Total Return:</strong> {(wf_equity[-1] - 1) * 100:.1f}%</p>
        <table>
            <tr>
                <th>Date</th>
                <th>Trade Return</th>
                <th>Cumulative Equity</th>
                <th>Parameters (Sample)</th>
            </tr>"""
        
        # Show first 10 and last 10 results
        for idx, row in wf_results.head(10).iterrows():
            params_sample = ", ".join([f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}" 
                                      for k, v in list(row['params'].items())[:3]]) + "..."
            html += f"""
            <tr>
                <td>{row['day'].date()}</td>
                <td>{row['trade_return']:.3f}</td>
                <td>{row['equity']:.3f}</td>
                <td style="font-size: 0.8em">{params_sample}</td>
            </tr>"""
        
        if len(wf_results) > 20:
            html += "<tr><td colspan='4' style='text-align: center;'>... (showing first 10 of " + str(len(wf_results)) + " days) ...</td></tr>"
        
        html += "</table>"
    elif wf_status != "Idle":
        html += f"<p>No walk-forward results yet. Status: {wf_status}</p>"
    else:
        html += "<p>No walk-forward results yet. Click the button above to start.</p>"
    
    html += "</div>"

    if gen_stats:
        html += "<h2>Training History</h2><ul>"
        for g in gen_stats:
            html += f"<li>Gen {g['gen']}: Best Train Sharpe {g['best_sharpe']:.3f}</li>"
        html += "</ul>"
    return html

@app.route('/start')
def start_opt():
    if global_data['raw_data'] is None: return "Data not loaded."
    if "Running" in global_data['optimization_status']: return "Running."
    t = Thread(target=run_genetic_optimization, args=(global_data['raw_data'],))
    t.start()
    return "Started! <a href='/'>Back</a>"
@app.route('/start_walk_forward')
def start_walk_forward():
    if global_data['raw_data'] is None: return "Data not loaded."
    if "Running" in global_data['walk_forward_status']: return "Walk-forward already running."
    t = Thread(target=run_walk_forward_optimization, args=(global_data['raw_data'],))
    t.start()
    return "Walk-forward optimization started! <a href='/'>Back</a>"

if __name__ == '__main__':
    raw_df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    global_data['raw_data'] = raw_df
    app.run(host='0.0.0.0', port=PORT)
