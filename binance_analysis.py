import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import json
import scipy.optimize as optimize
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

# ==========================================
# CONFIGURATION (MATCHING ORIGINAL)
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 10000.0
# We calculate signals independently, so CAP_SPLIT is handled in the optimizer/UI
# But we use these constants for normalization:
TARGET_STRAT_LEV = 2.0
TUMBLER_MAX_LEV = 4.327

# Strategy Params
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

# ==========================================
# DATA UTILS
# ==========================================
def fetch_binance_klines(symbol, interval, start_date_str):
    filename = f"{symbol.lower()}_{interval}_{start_date_str}.csv"
    if os.path.exists(filename):
        print(f"Loading {interval} data from disk...")
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
        
    print(f"Fetching {interval} data from Binance...")
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
        except: break
    
    df = pd.DataFrame(all_data, columns=['ot', 'o', 'h', 'l', 'c', 'v', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'ig'])
    df['timestamp'] = pd.to_datetime(df['ot'], unit='ms')
    for c in ['o', 'h', 'l', 'c']: df[c] = df[c].astype(float)
    final = df[['timestamp', 'o', 'h', 'l', 'c']].set_index('timestamp').rename(columns={'o':'open','h':'high','l':'low','c':'close'})
    final.to_csv(filename)
    return final

def resample_1h_to_1d(df_1h):
    df_1d = df_1h.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    return df_1d

def get_sma(series, window):
    return series.rolling(window=window).mean()

def get_ewm(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ==========================================
# PRECALC HELPERS (EXACT COPIES)
# ==========================================
def precalc_planner(df_1d):
    df = df_1d.copy()
    df['sma_s1'] = get_sma(df['close'], PLANNER_PARAMS["S1_SMA"])
    df['sma_s2'] = get_sma(df['close'], PLANNER_PARAMS["S2_SMA"])
    return df

def precalc_tumbler(df_1d):
    df = df_1d.copy()
    df['sma1'] = get_sma(df['close'], TUMBLER_PARAMS["SMA1"])
    df['sma2'] = get_sma(df['close'], TUMBLER_PARAMS["SMA2"])
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    w = TUMBLER_PARAMS["III_WIN"]
    num = df['log_ret'].rolling(w).sum().abs()
    den = df['log_ret'].abs().rolling(w).sum()
    df['iii'] = (num / den).fillna(0)
    return df

def precalc_gainer(df_1h, df_1d):
    g_1h = df_1h.copy()
    g_1d = df_1d.copy()
    
    # MACD 1H
    config = GAINER_PARAMS["MACD_1H"]
    col_names = []
    for i, ((f, s, sig), w) in enumerate(zip(config['params'], config['weights'])):
        fast = get_ewm(g_1h['close'], f)
        slow = get_ewm(g_1h['close'], s)
        macd = fast - slow
        signal = get_ewm(macd, sig)
        col = f'macd_1h_{i}'
        g_1h[col] = np.where(macd > signal, 1.0, -1.0) * w
        col_names.append(col)
    g_1h['score_macd_1h'] = g_1h[col_names].sum(axis=1) / sum(config['weights'])
    
    # MACD 1D
    config = GAINER_PARAMS["MACD_1D"]
    col_names = []
    for i, ((f, s, sig), w) in enumerate(zip(config['params'], config['weights'])):
        fast = get_ewm(g_1d['close'], f)
        slow = get_ewm(g_1d['close'], s)
        macd = fast - slow
        signal = get_ewm(macd, sig)
        col = f'macd_1d_{i}'
        g_1d[col] = np.where(macd > signal, 1.0, -1.0) * w
        col_names.append(col)
    g_1d['score_macd_1d'] = g_1d[col_names].sum(axis=1) / sum(config['weights'])
    
    # SMA 1D
    config = GAINER_PARAMS["SMA_1D"]
    col_names = []
    for i, (p, w) in enumerate(zip(config['params'], config['weights'])):
        sma = get_sma(g_1d['close'], p)
        col = f'sma_1d_{i}'
        g_1d[col] = np.where(g_1d['close'] > sma, 1.0, -1.0) * w
        col_names.append(col)
    g_1d['score_sma_1d'] = g_1d[col_names].sum(axis=1) / sum(config['weights'])
    
    return g_1h, g_1d

# ==========================================
# SIGNAL GENERATION (HOURLY LOOP)
# ==========================================
def generate_signals(df_1h, df_1d):
    print("Pre-calculating Indicators...")
    df_1d_plan = precalc_planner(df_1d)
    df_1d_tumb = precalc_tumbler(df_1d)
    df_1h_gain, df_1d_gain = precalc_gainer(df_1h, df_1d)
    
    print("Running Exact Simulation Loop...")
    
    # Variables for Planner Path Dependence
    # We use a dummy capital to track the Planner's internal stops logic.
    # This ensures exact behavior match with the original script.
    dummy_cap = 10000.0
    p_s1_equity, p_s2_equity = dummy_cap, dummy_cap
    p_last_price, p_last_lev_s1, p_last_lev_s2 = 0.0, 0.0, 0.0
    p_s1_entry, p_s1_peak, p_s1_stopped, p_s1_trend = None, 0.0, False, 0
    p_s2_peak, p_s2_stopped, p_s2_trend = 0.0, False, 0
    t_flat_regime = False
    
    start_idx = 400 * 24 
    if start_idx >= len(df_1h): start_idx = 0
    
    results = []
    
    # Pre-align indices to avoid slow lookups inside loop
    # We will just look up by date key as in original, but wrap in try/except or check
    
    for i in range(start_idx, len(df_1h)):
        ts = df_1h.index[i]
        curr_price = df_1h['open'].iloc[i] # Using Open as per original script execution price
        
        yesterday_date = (ts - timedelta(days=1)).date()
        yesterday = pd.Timestamp(yesterday_date)
        
        if yesterday not in df_1d_plan.index:
            continue
            
        row_d_p = df_1d_plan.loc[yesterday]
        row_d_t = df_1d_tumb.loc[yesterday]
        row_d_g = df_1d_gain.loc[yesterday]
        
        # Gainer 1H row is the previous hour
        row_h_g = df_1h_gain.iloc[i-1]
        
        daily_close = row_d_p['close']
        
        # --- PLANNER LOGIC (Exact Copy) ---
        if p_last_price <= 0: p_last_price = daily_close
        
        last_exec_price = df_1h['open'].iloc[i-1]
        pct_change = (curr_price - last_exec_price) / last_exec_price
        
        # Update internal equity for stops tracking
        p_s1_equity *= (1.0 + pct_change * p_last_lev_s1)
        p_s2_equity *= (1.0 + pct_change * p_last_lev_s2)
        
        s1_trend_new = 1 if daily_close > row_d_p['sma_s1'] else -1
        if p_s1_trend != s1_trend_new:
            p_s1_trend, p_s1_entry, p_s1_stopped, p_s1_peak = s1_trend_new, ts, False, p_s1_equity
        if p_s1_equity > p_s1_peak: p_s1_peak = p_s1_equity
        if (p_s1_peak - p_s1_equity) / p_s1_peak > PLANNER_PARAMS["S1_STOP"]: p_s1_stopped = True
        
        lev_s1 = 0.0
        if not p_s1_stopped:
            days_since = (ts - p_s1_entry).total_seconds() / 86400 if p_s1_entry else 0
            decay = max(0.0, 1.0 - (days_since / PLANNER_PARAMS["S1_DECAY"])**2) if days_since < PLANNER_PARAMS["S1_DECAY"] else 0.0
            lev_s1 = float(p_s1_trend) * decay
            
        s2_trend_new = 1 if daily_close > row_d_p['sma_s2'] else -1
        if p_s2_trend != s2_trend_new:
            p_s2_trend, p_s2_stopped, p_s2_peak = s2_trend_new, False, p_s2_equity
        if p_s2_equity > p_s2_peak: p_s2_peak = p_s2_equity
        if (p_s2_peak - p_s2_equity) / p_s2_peak > PLANNER_PARAMS["S2_STOP"]: p_s2_stopped = True
        
        dist_pct = abs(daily_close - row_d_p['sma_s2']) / row_d_p['sma_s2']
        is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
        tgt_size = 0.5 if is_prox else 1.0
        
        lev_s2 = 0.0
        if p_s2_stopped:
            if is_prox:
                p_s2_stopped, p_s2_peak = False, p_s2_equity
                lev_s2 = float(s2_trend_new) * tgt_size
        else:
            lev_s2 = float(s2_trend_new) * tgt_size
            
        lev_planner = max(-2.0, min(2.0, lev_s1 + lev_s2))
        p_last_lev_s1, p_last_lev_s2 = lev_s1, lev_s2
        
        # --- TUMBLER LOGIC (Exact Copy) ---
        iii = row_d_t['iii']
        raw_lev_t = TUMBLER_PARAMS['LEVS'][2]
        if iii < TUMBLER_PARAMS['III_TH'][0]: raw_lev_t = TUMBLER_PARAMS['LEVS'][0]
        elif iii < TUMBLER_PARAMS['III_TH'][1]: raw_lev_t = TUMBLER_PARAMS['LEVS'][1]
        
        if iii < TUMBLER_PARAMS['FLAT_THRESH']: t_flat_regime = True
        if t_flat_regime:
            sma1, sma2 = row_d_t['sma1'], row_d_t['sma2']
            if abs(daily_close - sma1) <= sma1*TUMBLER_PARAMS['BAND'] or abs(daily_close - sma2) <= sma2*TUMBLER_PARAMS['BAND']:
                t_flat_regime = False
        
        lev_tumbler = 0.0
        if not t_flat_regime:
            if daily_close > row_d_t['sma1'] and daily_close > row_d_t['sma2']: lev_tumbler = raw_lev_t
            elif daily_close < row_d_t['sma1'] and daily_close < row_d_t['sma2']: lev_tumbler = -raw_lev_t
                
        # --- GAINER LOGIC (Exact Copy) ---
        ws = GAINER_PARAMS["GA_WEIGHTS"]
        lev_gainer = (row_h_g['score_macd_1h'] * ws["MACD_1H"] + row_d_g['score_macd_1d'] * ws["MACD_1D"] + row_d_g['score_sma_1d'] * ws["SMA_1D"]) / sum(ws.values())
        
        # --- PREPARE EXPORTS ---
        # Normalize as requested: 
        # Planner is already at scale
        # Tumbler needs scaling to Target
        # Gainer needs scaling to Target
        
        n_p = lev_planner
        n_t = lev_tumbler * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        n_g = lev_gainer * TARGET_STRAT_LEV
        
        results.append({
            'date': ts,
            'close': df_1h['close'].iloc[i], # Use Close for portfolio tracking
            'lev_p': n_p,
            'lev_t': n_t,
            'lev_g': n_g
        })
        
    return pd.DataFrame(results).set_index('date')

# ==========================================
# OPTIMIZATION
# ==========================================
def optimize_weights(df):
    print("Optimizing Weights...")
    
    # Calculate returns once for speed
    asset_ret = df['close'].pct_change().fillna(0)
    
    def objective(w):
        # Construct Net Leverage from the 3 independent streams
        # Note: In original, net_lev = (n_p + n_t + n_g) * CAP_SPLIT
        # Here 'w' represents the individual allocation to each strategy
        
        net_lev = df['lev_p']*w[0] + df['lev_t']*w[1] + df['lev_g']*w[2]
        
        # Simple friction model
        friction = np.abs(net_lev) * (0.0006 / 24) * 0.5 # Approx hourly friction
        
        strat_ret = net_lev.shift(1) * asset_ret - friction
        
        if strat_ret.std() == 0: return 999
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(365 * 24)
        return -sharpe
    
    # Start with 0.33 each
    res = optimize.minimize(objective, [0.33, 0.33, 0.33], method='SLSQP', bounds=((0, 3), (0, 3), (0, 3)))
    return res.x

def export_to_json(df):
    export_data = []
    # Downsample if too large (optional, but good for web perf)
    # We'll take every 4th hour to reduce JSON size while keeping enough detail
    for date, row in df.iloc[::4].iterrows():
        export_data.append({
            'date': date.strftime('%Y-%m-%d %H:%M'),
            'price': round(row['close'], 2),
            'lev_p': round(row['lev_p'], 3),
            'lev_t': round(row['lev_t'], 3),
            'lev_g': round(row['lev_g'], 3)
        })
    with open("backtest_data.js", "w") as f:
        f.write(f"window.BACKTEST_DATA = {json.dumps(export_data)};")
    print("Data exported to backtest_data.js")

# ==========================================
# SERVER MAIN
# ==========================================
if __name__ == "__main__":
    # 1. Prepare Data
    try:
        df_1h = fetch_binance_klines(SYMBOL, '1h', START_DATE)
        df_1d = resample_1h_to_1d(df_1h)
        
        # Generate Signals using EXACT Loop
        sig_df = generate_signals(df_1h, df_1d)
        
        # Optimize
        opt_weights = optimize_weights(sig_df)
        print(f"Optimization Complete: Planner={opt_weights[0]:.3f}, Tumbler={opt_weights[1]:.3f}, Gainer={opt_weights[2]:.3f}")
        
        # Export
        export_to_json(sig_df)
        
    except Exception as e:
        print(f"Error generating data: {e}")
        import traceback
        traceback.print_exc()
        with open("backtest_data.js", "w") as f: f.write("window.BACKTEST_DATA = [];")

    # 2. Start Web Server
    PORT = int(os.environ.get("PORT", 8080))
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    print(f"\nServer active on port {PORT}")
    sys.stdout.flush() 
    httpd.serve_forever()
