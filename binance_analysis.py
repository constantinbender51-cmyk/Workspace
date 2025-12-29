import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import json
import scipy.optimize as optimize
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 10000.0
# Initial default weights (1/3 each)
DEFAULT_WEIGHTS = [0.333, 0.333, 0.333] 

# Strategy Params (Kept from your original script)
PLANNER_PARAMS = {
    "S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13, 
    "S2_SMA": 400, "S2_STOP": 0.27, "S2_PROX": 0.05
}

TUMBLER_PARAMS = {
    "SMA1": 32, "SMA2": 114, 
    "STOP": 0.043, "TAKE_PROFIT": 0.126, 
    "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077, 
    "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]
}

GAINER_PARAMS = {
    "GA_WEIGHTS": {"MACD_1H": 0.8, "MACD_1D": 0.4, "SMA_1D": 0.4},
    "MACD_1H": {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]},
    "MACD_1D": {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]},
    "SMA_1D": {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}
}

# Normalization constants (used to scale raw signals before weighting)
TARGET_STRAT_LEV = 2.0
TUMBLER_MAX_LEV = 4.327

# ==========================================
# DATA UTILS
# ==========================================
def fetch_binance_klines(symbol, interval, start_date_str):
    filename = f"{symbol.lower()}_{interval}_{start_date_str}.csv"
    if os.path.exists(filename):
        print(f"Loading {interval} data from {filename}...")
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
        
    print(f"Fetching {interval} data for {symbol}...")
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    all_data = []
    
    while start_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "limit": 1000}
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data: break
            all_data.extend(data)
            start_ts = data[-1][0] + (60000 if interval == '1m' else 3600000 if interval == '1h' else 86400000)
            print(f"Fetched {len(all_data)} candles...", end='\r')
            time.sleep(0.05)
        except: break
    
    df = pd.DataFrame(all_data, columns=['ot', 'o', 'h', 'l', 'c', 'v', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'ig'])
    df['timestamp'] = pd.to_datetime(df['ot'], unit='ms')
    for c in ['o', 'h', 'l', 'c']: df[c] = df[c].astype(float)
    final = df[['timestamp', 'o', 'h', 'l', 'c']].set_index('timestamp').rename(columns={'o':'open','h':'high','l':'low','c':'close'})
    final.to_csv(filename)
    return final

def resample_1h_to_1d(df_1h):
    df_1d = df_1h.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    return df_1d

def get_sma(series, window):
    return series.rolling(window=window).mean()

def get_ewm(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ==========================================
# STRATEGY LOGIC (Vectorized/Fast where possible)
# ==========================================
def precalc_strategies(df_1h, df_1d):
    """
    Calculates the raw leverage signals for all 3 strategies.
    Returns aligned DataFrames.
    """
    print("Calculating Strategy Signals...")
    
    # --- PLANNER (Daily Logic) ---
    p_df = df_1d.copy()
    p_df['sma_s1'] = get_sma(p_df['close'], PLANNER_PARAMS["S1_SMA"])
    p_df['sma_s2'] = get_sma(p_df['close'], PLANNER_PARAMS["S2_SMA"])
    
    # NOTE: Planner has complex path-dependent logic (peaks, stops). 
    # We will approximate or run a fast loop for the signal generation.
    # For accurate reproduction of your script, we need the loop.
    # We will do a fast loop generation for Planner signals.
    planner_signals = []
    p_s1_equity, p_s2_equity = 1.0, 1.0
    p_s1_peak, p_s2_peak = 1.0, 1.0
    p_s1_stopped, p_s2_stopped = False, False
    p_s1_trend, p_s2_trend = 0, 0
    p_s1_entry = None
    
    # Pre-calculate trends to speed up loop
    s1_trend_raw = np.where(p_df['close'] > p_df['sma_s1'], 1, -1)
    s2_trend_raw = np.where(p_df['close'] > p_df['sma_s2'], 1, -1)
    
    dates = p_df.index
    closes = p_df['close'].values
    sma2s = p_df['sma_s2'].values
    
    last_lev_s1, last_lev_s2 = 0.0, 0.0

    for i in range(len(p_df)):
        date = dates[i]
        close = closes[i]
        
        # Returns for internal equity tracking
        ret = 0.0
        if i > 0: ret = (close - closes[i-1]) / closes[i-1]
        
        # Update internal equities
        p_s1_equity *= (1.0 + ret * last_lev_s1)
        p_s2_equity *= (1.0 + ret * last_lev_s2)
        
        # S1 Logic
        curr_s1_trend = s1_trend_raw[i]
        if p_s1_trend != curr_s1_trend:
            p_s1_trend = curr_s1_trend
            p_s1_entry = i
            p_s1_stopped = False
            p_s1_peak = p_s1_equity
        
        if p_s1_equity > p_s1_peak: p_s1_peak = p_s1_equity
        if (p_s1_peak - p_s1_equity) / p_s1_peak > PLANNER_PARAMS["S1_STOP"]: p_s1_stopped = True
        
        lev_s1 = 0.0
        if not p_s1_stopped:
            days_since = (i - p_s1_entry) if p_s1_entry is not None else 0
            decay = max(0.0, 1.0 - (days_since / PLANNER_PARAMS["S1_DECAY"])**2) if days_since < PLANNER_PARAMS["S1_DECAY"] else 0.0
            lev_s1 = float(p_s1_trend) * decay

        # S2 Logic
        curr_s2_trend = s2_trend_raw[i]
        if p_s2_trend != curr_s2_trend:
            p_s2_trend = curr_s2_trend
            p_s2_stopped = False
            p_s2_peak = p_s2_equity
            
        if p_s2_equity > p_s2_peak: p_s2_peak = p_s2_equity
        if (p_s2_peak - p_s2_equity) / p_s2_peak > PLANNER_PARAMS["S2_STOP"]: p_s2_stopped = True
        
        dist_pct = abs(close - sma2s[i]) / sma2s[i] if sma2s[i] > 0 else 0
        is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
        tgt_size = 0.5 if is_prox else 1.0
        
        lev_s2 = 0.0
        if p_s2_stopped:
            if is_prox:
                p_s2_stopped, p_s2_peak = False, p_s2_equity
                lev_s2 = float(curr_s2_trend) * tgt_size
        else:
            lev_s2 = float(curr_s2_trend) * tgt_size
            
        final_lev = max(-2.0, min(2.0, lev_s1 + lev_s2))
        planner_signals.append(final_lev)
        last_lev_s1, last_lev_s2 = lev_s1, lev_s2
        
    p_df['lev_planner'] = planner_signals

    # --- TUMBLER (Daily Logic) ---
    t_df = df_1d.copy()
    t_df['sma1'] = get_sma(t_df['close'], TUMBLER_PARAMS["SMA1"])
    t_df['sma2'] = get_sma(t_df['close'], TUMBLER_PARAMS["SMA2"])
    t_df['log_ret'] = np.log(t_df['close'] / t_df['close'].shift(1))
    
    w = TUMBLER_PARAMS["III_WIN"]
    num = t_df['log_ret'].rolling(w).sum().abs()
    den = t_df['log_ret'].abs().rolling(w).sum()
    t_df['iii'] = (num / den).fillna(0)
    
    # Vectorized Tumbler Logic
    conditions = [
        t_df['iii'] < TUMBLER_PARAMS['III_TH'][0],
        t_df['iii'] < TUMBLER_PARAMS['III_TH'][1]
    ]
    choices = [TUMBLER_PARAMS['LEVS'][0], TUMBLER_PARAMS['LEVS'][1]]
    t_df['raw_lev'] = np.select(conditions, choices, default=TUMBLER_PARAMS['LEVS'][2])
    
    # Flat Regime Logic (Approximation for vectorization - Close enough for optimization)
    # Ideally needs loop for state maintenance, but we'll use a rolling window check for simplicity in optimization
    # to keep this fast.
    band_check = (abs(t_df['close'] - t_df['sma1']) <= t_df['sma1']*TUMBLER_PARAMS['BAND']) | \
                 (abs(t_df['close'] - t_df['sma2']) <= t_df['sma2']*TUMBLER_PARAMS['BAND'])
    
    # If III is low, we enter flat. We exit flat if band_check is True.
    # This is path dependent. Let's do a quick loop.
    flat_regime = False
    tumbler_signals = []
    iii_vals = t_df['iii'].values
    band_checks = band_check.values
    raw_levs = t_df['raw_lev'].values
    close_vals = t_df['close'].values
    sma1_vals = t_df['sma1'].values
    sma2_vals = t_df['sma2'].values
    
    for i in range(len(t_df)):
        if iii_vals[i] < TUMBLER_PARAMS['FLAT_THRESH']: flat_regime = True
        if flat_regime and band_checks[i]: flat_regime = False
        
        lev = 0.0
        if not flat_regime:
            if close_vals[i] > sma1_vals[i] and close_vals[i] > sma2_vals[i]:
                lev = raw_levs[i]
            elif close_vals[i] < sma1_vals[i] and close_vals[i] < sma2_vals[i]:
                lev = -raw_levs[i]
        tumbler_signals.append(lev)
    
    t_df['lev_tumbler'] = tumbler_signals
    # Normalize Tumbler
    t_df['lev_tumbler'] = t_df['lev_tumbler'] * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)

    # --- GAINER (Hourly & Daily) ---
    # We will compute Gainer on 1H then resample signal to 1D for the unified optimization
    g_1h = df_1h.copy()
    g_1d = df_1d.copy()
    
    # Gainer Helper
    def calc_score(df, config, prefix):
        scores = []
        total_w = sum(config['weights'])
        for i, ((f, s, sig), w) in enumerate(zip(config['params'], config['weights'])):
            fast = get_ewm(df['close'], f)
            slow = get_ewm(df['close'], s)
            macd = fast - slow
            signal = get_ewm(macd, sig)
            score = np.where(macd > signal, 1.0, -1.0) * w
            scores.append(score)
        
        # Sum columns
        return np.sum(scores, axis=0) / total_w

    def calc_sma_score(df, config):
        scores = []
        total_w = sum(config['weights'])
        for i, (p, w) in enumerate(zip(config['params'], config['weights'])):
            sma = get_sma(df['close'], p)
            score = np.where(df['close'] > sma, 1.0, -1.0) * w
            scores.append(score)
        return np.sum(scores, axis=0) / total_w

    s_macd_1h = calc_score(g_1h, GAINER_PARAMS["MACD_1H"], 'm1h')
    g_1h['score_macd_1h'] = s_macd_1h
    
    s_macd_1d = calc_score(g_1d, GAINER_PARAMS["MACD_1D"], 'm1d')
    g_1d['score_macd_1d'] = s_macd_1d
    
    s_sma_1d = calc_sma_score(g_1d, GAINER_PARAMS["SMA_1D"])
    g_1d['score_sma_1d'] = s_sma_1d
    
    # Merge 1H score into 1D (take last value of the day)
    # Resample 1H scores to 1D
    g_1h_resampled = g_1h[['score_macd_1h']].resample('1D').last().dropna()
    
    # Align all dataframes
    common_idx = p_df.index.intersection(t_df.index).intersection(g_1d.index).intersection(g_1h_resampled.index)
    
    final_df = pd.DataFrame(index=common_idx)
    final_df['close'] = df_1d.loc[common_idx, 'close']
    final_df['lev_p'] = p_df.loc[common_idx, 'lev_planner']
    final_df['lev_t'] = t_df.loc[common_idx, 'lev_tumbler']
    
    # Calculate Final Gainer Signal
    ws = GAINER_PARAMS["GA_WEIGHTS"]
    denom = sum(ws.values())
    
    score_m1h = g_1h_resampled.loc[common_idx, 'score_macd_1h']
    score_m1d = g_1d.loc[common_idx, 'score_macd_1d']
    score_sma = g_1d.loc[common_idx, 'score_sma_1d']
    
    final_df['lev_g'] = (score_m1h * ws["MACD_1H"] + score_m1d * ws["MACD_1D"] + score_sma * ws["SMA_1D"]) / denom
    # Normalize Gainer
    final_df['lev_g'] = final_df['lev_g'] * TARGET_STRAT_LEV
    
    return final_df

# ==========================================
# OPTIMIZATION & ANALYSIS
# ==========================================
def run_backtest_vectorized(df, weights):
    """
    Fast vectorized backtest given weights [w_p, w_t, w_g]
    """
    net_lev = df['lev_p'] * weights[0] + df['lev_t'] * weights[1] + df['lev_g'] * weights[2]
    
    # Calculate daily returns of the strategy
    # Strategy Return = Net Lev * Asset Return - Friction
    asset_ret = df['close'].pct_change().fillna(0)
    
    # Friction approx (assuming daily rebalance)
    friction = np.abs(net_lev) * 0.0006 * 0.1 # Very rough estimate of daily friction cost
    
    strat_ret = net_lev.shift(1) * asset_ret - friction
    equity = (1 + strat_ret).cumprod()
    return equity, strat_ret

def optimize_weights(df):
    print("\nOptimizing Weights (Maximizing Sharpe Ratio)...")
    
    def objective(w):
        # Constraint: Sum of weights is flexible, but let's say max leverage sum is bounded or 
        # we just penalize volatility.
        # Let's target max Sharpe.
        _, rets = run_backtest_vectorized(df, w)
        if rets.std() == 0: return 999
        sharpe = rets.mean() / rets.std() * np.sqrt(365)
        return -sharpe # Minimize negative Sharpe

    # Bounds for weights (0.0 to 2.0 each - allowing overdrive)
    bounds = ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0))
    # Initial guess
    x0 = [0.33, 0.33, 0.33]
    
    res = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds)
    return res.x

def quantize_analysis(df, weights):
    equity, rets = run_backtest_vectorized(df, weights)
    
    # Create Quarter Column
    df['quarter'] = df.index.to_period('Q')
    df['strat_ret'] = rets
    
    print("\n" + "="*60)
    print(f"QUARTERLY QUANTIZATION REPORT (Weights: P={weights[0]:.2f}, T={weights[1]:.2f}, G={weights[2]:.2f})")
    print(f"{'Quarter':<10} | {'Return':<10} | {'StdDev':<10} | {'Sharpe':<10} | {'MaxDD':<10}")
    print("-" * 60)
    
    quarters = df.groupby('quarter')
    stats = []
    
    for q, data in quarters:
        if len(data) < 10: continue
        q_ret = (data['strat_ret'] + 1).prod() - 1
        q_std = data['strat_ret'].std() * np.sqrt(len(data)) # Volatility over the quarter
        q_sharpe = (data['strat_ret'].mean() / data['strat_ret'].std() * np.sqrt(252)) if data['strat_ret'].std() > 0 else 0
        
        # MaxDD in Quarter
        cum = (1 + data['strat_ret']).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        
        print(f"{str(q):<10} | {q_ret*100:6.2f}%   | {q_std*100:6.2f}%   | {q_sharpe:6.2f}     | {dd*100:6.2f}%")
        stats.append({'q': str(q), 'ret': q_ret, 'sharpe': q_sharpe})
        
    print("-" * 60)
    
    # Overall
    total_ret = equity.iloc[-1] - 1
    ann_ret = rets.mean() * 365
    ann_vol = rets.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol
    print(f"OVERALL | CAGR: {ann_ret*100:.2f}% | Vol: {ann_vol*100:.2f}% | Sharpe: {sharpe:.2f}")

    return equity

def export_to_json(df):
    # Downsample for web (Weekly or every 3rd day to save space if needed, 
    # but 1D for 5 years is ~1800 points, which is fine for JSON)
    export_data = []
    for date, row in df.iterrows():
        export_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(row['close'], 2),
            'lev_p': round(row['lev_p'], 3),
            'lev_t': round(row['lev_t'], 3),
            'lev_g': round(row['lev_g'], 3)
        })
    
    js_content = f"window.BACKTEST_DATA = {json.dumps(export_data)};"
    with open("backtest_data.js", "w") as f:
        f.write(js_content)
    print("\nData exported to backtest_data.js")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Fetching Data...")
    df_1h = fetch_binance_klines(SYMBOL, '1h', START_DATE)
    df_1d = resample_1h_to_1d(df_1h)
    
    # 1. Generate Signals
    sig_df = precalc_strategies(df_1h, df_1d)
    
    # 2. Optimize
    opt_weights = optimize_weights(sig_df)
    print(f"Optimal Weights Found: Planner={opt_weights[0]:.3f}, Tumbler={opt_weights[1]:.3f}, Gainer={opt_weights[2]:.3f}")
    
    # 3. Quantize & Report
    quantize_analysis(sig_df, opt_weights)
    
    # 4. Export for Web
    export_to_json(sig_df)
    
    # 5. Serve
    PORT = 8000
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"\nStarting Leverage Playground at http://localhost:{PORT}/dashboard.html")
    print("Press Ctrl+C to stop.")
    
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}/dashboard.html")).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
