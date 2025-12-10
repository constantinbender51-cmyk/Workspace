import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, jsonify
import io
import time
import base64
import random
import copy
from threading import Thread

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' 
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

app = Flask(__name__)

# --- Global Storage ---
global_data = {
    'results': pd.DataFrame(),
    'best_params': {},
    'optimization_status': "Idle",
    'top_performers': []
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
    return df

# --- Dynamic Strategy Engine ---
def run_strategy_dynamic(df, params):
    """
    Runs the strategy using a dynamic parameter dictionary.
    """
    data = df.copy()
    
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

    # Pre-calc calculations
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    data['sma_fast'] = data['close'].rolling(window=p_sma_fast).mean()
    data['sma_slow'] = data['close'].rolling(window=p_sma_slow).mean()

    # Efficiency Ratio (iii)
    data['iii'] = (data['log_ret'].rolling(p_w).sum().abs() / 
                   data['log_ret'].abs().rolling(p_w).sum()).fillna(0)
    data['iii_shifted'] = data['iii'].shift(1)

    # Leverage Logic
    conditions = [
        (data['iii_shifted'] < p_lev_th_low),
        (data['iii_shifted'] < p_lev_th_high)
    ]
    choices = [p_lev_low, p_lev_high] # Note: High volatility (low iii) often implies high trend leverage in your logic? 
    # Logic Adjustment based on original script:
    # Original: < 0.13 -> 0.5 (Defensive), < 0.18 -> 4.5 (Aggressive), else 2.45
    # We maintain this structure:
    data['leverage'] = np.select(conditions, choices, default=p_lev_mid)
    
    # Numpy Optimization
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    sma_fasts = data['sma_fast'].values
    sma_slows = data['sma_slow'].values
    iii_shifted = data['iii_shifted'].values
    leverages = data['leverage'].values
    
    n = len(closes)
    signals = np.zeros(n)
    returns = np.zeros(n)
    is_flat = False
    
    start_idx = max(p_sma_slow, p_w) + 1
    
    for i in range(start_idx, n):
        # A. Regime Logic
        if iii_shifted[i] < p_u:
            is_flat = True
            
        if is_flat:
            prev_c = closes[i-1]
            prev_s_fast = sma_fasts[i-1]
            prev_s_slow = sma_slows[i-1]
            
            diff1 = abs(prev_c - prev_s_fast)
            diff2 = abs(prev_c - prev_s_slow)
            
            if diff1 <= (prev_s_fast * p_y) or diff2 <= (prev_s_slow * p_y):
                is_flat = False
        
        # B. Signal
        if is_flat:
            signals[i] = 0
        else:
            prev_c = closes[i-1]
            prev_s_fast = sma_fasts[i-1]
            prev_s_slow = sma_slows[i-1]
            if prev_c > prev_s_fast and prev_c > prev_s_slow:
                signals[i] = 1
            elif prev_c < prev_s_fast and prev_c < prev_s_slow:
                signals[i] = -1
            else:
                signals[i] = 0
                
        # C. PnL
        lev = leverages[i]
        sig = signals[i]
        
        if sig == 0 or np.isnan(lev):
            continue
            
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        daily_ret = 0.0
        
        if sig == 1:
            stop = o * (1 - p_sl)
            take = o * (1 + p_tp)
            if l <= stop: daily_ret = -p_sl
            elif h >= take: daily_ret = p_tp
            else: daily_ret = (c - o) / o
        else:
            stop = o * (1 + p_sl)
            take = o * (1 - p_tp)
            if h >= stop: daily_ret = -p_sl
            elif l <= take: daily_ret = p_tp
            else: daily_ret = (o - c) / o
            
        returns[i] = daily_ret * lev

    # Stats
    cum_ret = np.cumprod(1 + returns)[-1]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(365)) if np.std(returns) != 0 else 0
    
    return cum_ret, sharpe, returns

# --- Genetic Algorithm Logic ---

class Individual:
    def __init__(self):
        self.params = {}
        self.sharpe = -999
        self.ret = 0
    
    def random_init(self):
        self.params = {
            'sma_fast': random.randint(10, 100),
            'sma_slow': random.randint(50, 300),
            'w': random.randint(10, 100),
            'u': random.uniform(0.05, 0.60),
            'y': random.uniform(0.005, 0.20),
            'lev_thresh_low': random.uniform(0.05, 0.60),
            'lev_thresh_high': random.uniform(0.05, 0.60),
            'lev_low': random.uniform(0.0, 5.0),
            'lev_mid': random.uniform(0.0, 5.0),
            'lev_high': random.uniform(0.0, 5.0),
            'tp_pct': random.uniform(0.05, 0.50),
            'sl_pct': random.uniform(0.01, 0.20)
        }
        self.fix_constraints()

    def fix_constraints(self):
        # Ensure Logic holds
        if self.params['sma_fast'] >= self.params['sma_slow']:
            self.params['sma_fast'] = int(self.params['sma_slow'] * 0.5)
            
        if self.params['lev_thresh_low'] > self.params['lev_thresh_high']:
            self.params['lev_thresh_low'], self.params['lev_thresh_high'] = \
            self.params['lev_thresh_high'], self.params['lev_thresh_low']

    def mutate(self):
        key = random.choice(list(self.params.keys()))
        val = self.params[key]
        
        # Mutate by +/- 10-20%
        mutation_factor = random.uniform(0.8, 1.2)
        new_val = val * mutation_factor
        
        # Clamp ranges roughly
        if 'sma' in key or 'w' in key:
            self.params[key] = int(max(5, new_val))
        elif 'lev' in key and 'thresh' not in key:
            self.params[key] = min(5.0, max(0.0, new_val))
        else:
            self.params[key] = max(0.001, new_val)
            
        self.fix_constraints()

def run_genetic_optimization(df):
    POP_SIZE = 50
    GENERATIONS = 15
    TOP_K = 10
    
    global_data['optimization_status'] = "Initializing Population..."
    population = []
    
    # 1. Initialize
    for _ in range(POP_SIZE):
        ind = Individual()
        ind.random_init()
        population.append(ind)
        
    for gen in range(GENERATIONS):
        global_data['optimization_status'] = f"Running Generation {gen+1}/{GENERATIONS}..."
        print(f"--- Generation {gen+1} ---")
        
        # 2. Evaluate
        for ind in population:
            # Skip if already calculated
            if ind.sharpe != -999: continue 
            
            ret, sharpe, _ = run_strategy_dynamic(df, ind.params)
            ind.ret = ret
            ind.sharpe = sharpe
        
        # 3. Sort & Select
        population.sort(key=lambda x: x.sharpe, reverse=True)
        best_gen = population[0]
        print(f"Gen {gen+1} Best: Sharpe={best_gen.sharpe:.2f}, Ret={best_gen.ret:.2f}x")
        
        # Save top performers to global
        global_data['top_performers'] = [
            {'params': p.params, 'sharpe': p.sharpe, 'ret': p.ret} 
            for p in population[:5]
        ]
        
        # 4. Evolution (Crossover & Mutation)
        next_gen = population[:TOP_K] # Elitism: Keep top K
        
        while len(next_gen) < POP_SIZE:
            # Tournament Select
            parent1 = random.choice(population[:20])
            parent2 = random.choice(population[:20])
            
            child = Individual()
            # Crossover (Mix genes)
            for key in child.params:
                child.params[key] = parent1.params[key] if random.random() > 0.5 else parent2.params[key]
            
            # Mutation
            if random.random() < 0.4: # 40% chance to mutate
                child.mutate()
            
            child.fix_constraints()
            next_gen.append(child)
            
        population = next_gen
        
    global_data['optimization_status'] = "Complete"
    print("Optimization Complete.")

# --- Web Routes ---

@app.route('/')
def index():
    status = global_data['optimization_status']
    top = global_data['top_performers']
    
    html = f"""
    <h1>Optimization Status: {status}</h1>
    <a href="/start_opt"><button>START GENETIC OPTIMIZATION</button></a>
    <hr>
    <h3>Top 5 Found Parameter Sets (by Sharpe)</h3>
    """
    
    if top:
        html += "<table border='1' cellpadding='5'><tr><th>Sharpe</th><th>Return</th><th>Params</th></tr>"
        for t in top:
            p_str = "<br>".join([f"<b>{k}:</b> {v:.3f}" for k,v in t['params'].items()])
            html += f"<tr><td>{t['sharpe']:.2f}</td><td>{t['ret']:.2f}x</td><td>{p_str}</td></tr>"
        html += "</table>"
    else:
        html += "<p>No results yet.</p>"
        
    return html

@app.route('/start_opt')
def start_opt():
    # Run in background thread
    df = global_data.get('raw_data')
    if df is None: return "Error: No Data Loaded"
    
    t = Thread(target=run_genetic_optimization, args=(df,))
    t.start()
    return "Optimization Started! Go back to <a href='/'>Home</a> and refresh in a minute."

if __name__ == '__main__':
    # Initial Data Load
    raw_df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    global_data['raw_data'] = raw_df
    
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
