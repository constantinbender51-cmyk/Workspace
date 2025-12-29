import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

# ==========================================
# CONFIGURATION (MATCHING ORIGINAL)
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
# CAP_SPLIT is now handled in the Web UI
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
    return df_1h.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

def get_sma(series, window):
    return series.rolling(window=window).mean()

def get_ewm(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ==========================================
# PRECALC INDICATORS
# ==========================================
def precalc_indicators(df_1h, df_1d):
    print("Pre-calculating Indicators...")
    
    # Planner
    p_df = df_1d.copy()
    p_df['sma_s1'] = get_sma(p_df['close'], PLANNER_PARAMS["S1_SMA"])
    p_df['sma_s2'] = get_sma(p_df['close'], PLANNER_PARAMS["S2_SMA"])

    # Tumbler
    t_df = df_1d.copy()
    t_df['sma1'] = get_sma(t_df['close'], TUMBLER_PARAMS["SMA1"])
    t_df['sma2'] = get_sma(t_df['close'], TUMBLER_PARAMS["SMA2"])
    t_df['log_ret'] = np.log(t_df['close'] / t_df['close'].shift(1))
    w = TUMBLER_PARAMS["III_WIN"]
    num = t_df['log_ret'].rolling(w).sum().abs()
    den = t_df['log_ret'].abs().rolling(w).sum()
    t_df['iii'] = (num / den).fillna(0)
    
    # Gainer (1H & 1D)
    g_1h, g_1d = df_1h.copy(), df_1d.copy()
    
    # Gainer MACD Helper
    def calc_macd_score(df, config):
        scores = []
        for (f, s, sig), w in zip(config['params'], config['weights']):
            macd = get_ewm(df['close'], f) - get_ewm(df['close'], s)
            signal = get_ewm(macd, sig)
            scores.append(np.where(macd > signal, 1.0, -1.0) * w)
        return np.sum(scores, axis=0) / sum(config['weights'])

    g_1h['score_macd_1h'] = calc_macd_score(g_1h, GAINER_PARAMS["MACD_1H"])
    g_1d['score_macd_1d'] = calc_macd_score(g_1d, GAINER_PARAMS["MACD_1D"])
    
    # Gainer SMA Helper
    s_scores = []
    for p, w in zip(GAINER_PARAMS["SMA_1D"]['params'], GAINER_PARAMS["SMA_1D"]['weights']):
        s_scores.append(np.where(g_1d['close'] > get_sma(g_1d['close'], p), 1.0, -1.0) * w)
    g_1d['score_sma_1d'] = np.sum(s_scores, axis=0) / sum(GAINER_PARAMS["SMA_1D"]['weights'])
    
    return p_df, t_df, g_1h, g_1d

# ==========================================
# EXACT SIGNAL GENERATION LOOP
# ==========================================
def generate_signals(df_1h, df_1d):
    df_1d_plan, df_1d_tumb, df_1h_gain, df_1d_gain = precalc_indicators(df_1h, df_1d)
    
    print("Running Simulation Loop...")
    
    # Planner State
    dummy_cap = 10000.0
    p_s1_equity, p_s2_equity = dummy_cap, dummy_cap
    p_last_price, p_last_lev_s1, p_last_lev_s2 = 0.0, 0.0, 0.0
    p_s1_entry, p_s1_peak, p_s1_stopped, p_s1_trend = None, 0.0, False, 0
    p_s2_peak, p_s2_stopped, p_s2_trend = 0.0, False, 0
    
    # Tumbler State
    t_flat_regime = False
    
    start_idx = 400 * 24 
    if start_idx >= len(df_1h): start_idx = 0
    
    results = []
    
    for i in range(start_idx, len(df_1h)):
        ts = df_1h.index[i]
        curr_price = df_1h['open'].iloc[i]
        
        # Lookback to yesterday for daily signals
        yesterday = pd.Timestamp((ts - timedelta(days=1)).date())
        if yesterday not in df_1d_plan.index: continue
            
        row_d_p = df_1d_plan.loc[yesterday]
        row_d_t = df_1d_tumb.loc[yesterday]
        row_d_g = df_1d_gain.loc[yesterday]
        row_h_g = df_1h_gain.iloc[i-1]
        
        daily_close = row_d_p['close']
        
        # --- PLANNER ---
        if p_last_price <= 0: p_last_price = daily_close
        
        last_exec_price = df_1h['open'].iloc[i-1]
        pct_change = (curr_price - last_exec_price) / last_exec_price
        
        p_s1_equity *= (1.0 + pct_change * p_last_lev_s1)
        p_s2_equity *= (1.0 + pct_change * p_last_lev_s2)
        
        # S1
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
            
        # S2
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
        
        # --- TUMBLER ---
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
                
        # --- GAINER ---
        ws = GAINER_PARAMS["GA_WEIGHTS"]
        lev_gainer = (row_h_g['score_macd_1h'] * ws["MACD_1H"] + row_d_g['score_macd_1d'] * ws["MACD_1D"] + row_d_g['score_sma_1d'] * ws["SMA_1D"]) / sum(ws.values())
        
        # --- PREPARE ---
        n_p = lev_planner
        n_t = lev_tumbler * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        n_g = lev_gainer * TARGET_STRAT_LEV
        
        results.append({
            'date': ts,
            'close': df_1h['close'].iloc[i],
            'lev_p': n_p,
            'lev_t': n_t,
            'lev_g': n_g
        })
        
    return pd.DataFrame(results).set_index('date')

def export_to_json(df):
    export_data = []
    # Downsample for faster loading (every 4th hour)
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

if __name__ == "__main__":
    try:
        df_1h = fetch_binance_klines(SYMBOL, '1h', START_DATE)
        df_1d = resample_1h_to_1d(df_1h)
        sig_df = generate_signals(df_1h, df_1d)
        export_to_json(sig_df)
    except Exception as e:
        print(f"Error: {e}")
        with open("backtest_data.js", "w") as f: f.write("window.BACKTEST_DATA = [];")

    PORT = int(os.environ.get("PORT", 8080))
    httpd = HTTPServer(('0.0.0.0', PORT), SimpleHTTPRequestHandler)
    print(f"Server listening on port {PORT}")
    sys.stdout.flush()
    httpd.serve_forever()
