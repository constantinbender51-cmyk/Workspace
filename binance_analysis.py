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
# CONFIGURATION
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 10000.0

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

TARGET_STRAT_LEV = 2.0
TUMBLER_MAX_LEV = 4.327

# ==========================================
# DATA UTILS
# ==========================================
def fetch_binance_klines(symbol, interval, start_date_str):
    filename = f"{symbol.lower()}_{interval}_{start_date_str}.csv"
    # In cloud env, we might want to refetch or cache. For now, fetch if missing.
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
# LOGIC
# ==========================================
def precalc_strategies(df_1h, df_1d):
    print("Calculating Strategy Signals...")
    # --- PLANNER ---
    p_df = df_1d.copy()
    p_df['sma_s1'] = get_sma(p_df['close'], PLANNER_PARAMS["S1_SMA"])
    p_df['sma_s2'] = get_sma(p_df['close'], PLANNER_PARAMS["S2_SMA"])
    
    planner_signals = []
    p_s1_equity, p_s2_equity = 1.0, 1.0
    p_s1_peak, p_s2_peak = 1.0, 1.0
    p_s1_stopped, p_s2_stopped = False, False
    p_s1_trend, p_s2_trend = 0, 0
    p_s1_entry = None
    
    s1_trend_raw = np.where(p_df['close'] > p_df['sma_s1'], 1, -1)
    s2_trend_raw = np.where(p_df['close'] > p_df['sma_s2'], 1, -1)
    
    dates = p_df.index
    closes = p_df['close'].values
    sma2s = p_df['sma_s2'].values
    last_lev_s1, last_lev_s2 = 0.0, 0.0

    for i in range(len(p_df)):
        close = closes[i]
        ret = 0.0
        if i > 0: ret = (close - closes[i-1]) / closes[i-1]
        
        p_s1_equity *= (1.0 + ret * last_lev_s1)
        p_s2_equity *= (1.0 + ret * last_lev_s2)
        
        # S1 Logic
        curr_s1_trend = s1_trend_raw[i]
        if p_s1_trend != curr_s1_trend:
            p_s1_trend, p_s1_entry, p_s1_stopped, p_s1_peak = curr_s1_trend, i, False, p_s1_equity
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
            p_s2_trend, p_s2_stopped, p_s2_peak = curr_s2_trend, False, p_s2_equity
        if p_s2_equity > p_s2_peak: p_s2_peak = p_s2_equity
        if (p_s2_peak - p_s2_equity) / p_s2_peak > PLANNER_PARAMS["S2_STOP"]: p_s2_stopped = True
        
        dist_pct = abs(close - sma2s[i]) / sma2s[i] if sma2s[i] > 0 else 0
        is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
        tgt_size = 0.5 if is_prox else 1.0
        lev_s2 = float(curr_s2_trend) * tgt_size if not p_s2_stopped else (float(curr_s2_trend) * tgt_size if is_prox else 0.0)
        
        if p_s2_stopped and is_prox: p_s2_stopped, p_s2_peak = False, p_s2_equity

        planner_signals.append(max(-2.0, min(2.0, lev_s1 + lev_s2)))
        last_lev_s1, last_lev_s2 = lev_s1, lev_s2
        
    p_df['lev_planner'] = planner_signals

    # --- TUMBLER ---
    t_df = df_1d.copy()
    t_df['sma1'] = get_sma(t_df['close'], TUMBLER_PARAMS["SMA1"])
    t_df['sma2'] = get_sma(t_df['close'], TUMBLER_PARAMS["SMA2"])
    t_df['log_ret'] = np.log(t_df['close'] / t_df['close'].shift(1))
    
    w = TUMBLER_PARAMS["III_WIN"]
    num = t_df['log_ret'].rolling(w).sum().abs()
    den = t_df['log_ret'].abs().rolling(w).sum()
    t_df['iii'] = (num / den).fillna(0)
    
    conditions = [t_df['iii'] < TUMBLER_PARAMS['III_TH'][0], t_df['iii'] < TUMBLER_PARAMS['III_TH'][1]]
    choices = [TUMBLER_PARAMS['LEVS'][0], TUMBLER_PARAMS['LEVS'][1]]
    t_df['raw_lev'] = np.select(conditions, choices, default=TUMBLER_PARAMS['LEVS'][2])
    
    band_check = (abs(t_df['close'] - t_df['sma1']) <= t_df['sma1']*TUMBLER_PARAMS['BAND']) | \
                 (abs(t_df['close'] - t_df['sma2']) <= t_df['sma2']*TUMBLER_PARAMS['BAND'])
    
    tumbler_signals = []
    flat_regime = False
    for i in range(len(t_df)):
        if t_df['iii'].iloc[i] < TUMBLER_PARAMS['FLAT_THRESH']: flat_regime = True
        if flat_regime and band_check.iloc[i]: flat_regime = False
        
        lev = 0.0
        if not flat_regime:
            c, s1, s2 = t_df['close'].iloc[i], t_df['sma1'].iloc[i], t_df['sma2'].iloc[i]
            if c > s1 and c > s2: lev = t_df['raw_lev'].iloc[i]
            elif c < s1 and c < s2: lev = -t_df['raw_lev'].iloc[i]
        tumbler_signals.append(lev)
    t_df['lev_tumbler'] = np.array(tumbler_signals) * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)

    # --- GAINER ---
    g_1h, g_1d = df_1h.copy(), df_1d.copy()
    
    def calc_score(df, config):
        scores = []
        for (f, s, sig), w in zip(config['params'], config['weights']):
            macd = get_ewm(df['close'], f) - get_ewm(df['close'], s)
            signal = get_ewm(macd, sig)
            scores.append(np.where(macd > signal, 1.0, -1.0) * w)
        return np.sum(scores, axis=0) / sum(config['weights'])

    g_1h['score_macd_1h'] = calc_score(g_1h, GAINER_PARAMS["MACD_1H"])
    g_1d['score_macd_1d'] = calc_score(g_1d, GAINER_PARAMS["MACD_1D"])
    
    s_scores = []
    for p, w in zip(GAINER_PARAMS["SMA_1D"]['params'], GAINER_PARAMS["SMA_1D"]['weights']):
        s_scores.append(np.where(g_1d['close'] > get_sma(g_1d['close'], p), 1.0, -1.0) * w)
    g_1d['score_sma_1d'] = np.sum(s_scores, axis=0) / sum(GAINER_PARAMS["SMA_1D"]['weights'])
    
    g_1h_res = g_1h[['score_macd_1h']].resample('1D').last().dropna()
    common_idx = p_df.index.intersection(t_df.index).intersection(g_1d.index).intersection(g_1h_res.index)
    
    final_df = pd.DataFrame(index=common_idx)
    final_df['close'] = df_1d.loc[common_idx, 'close']
    final_df['lev_p'] = p_df.loc[common_idx, 'lev_planner']
    final_df['lev_t'] = t_df.loc[common_idx, 'lev_tumbler']
    
    ws = GAINER_PARAMS["GA_WEIGHTS"]
    denom = sum(ws.values())
    final_df['lev_g'] = (g_1h_res.loc[common_idx, 'score_macd_1h'] * ws["MACD_1H"] + 
                         g_1d.loc[common_idx, 'score_macd_1d'] * ws["MACD_1D"] + 
                         g_1d.loc[common_idx, 'score_sma_1d'] * ws["SMA_1D"]) / denom * TARGET_STRAT_LEV
    
    return final_df

def optimize_weights(df):
    print("Optimizing Weights...")
    def objective(w):
        net_lev = df['lev_p']*w[0] + df['lev_t']*w[1] + df['lev_g']*w[2]
        strat_ret = net_lev.shift(1) * df['close'].pct_change().fillna(0) - np.abs(net_lev)*0.00006
        if strat_ret.std() == 0: return 999
        return -(strat_ret.mean() / strat_ret.std() * np.sqrt(365))
    
    res = optimize.minimize(objective, [0.33, 0.33, 0.33], method='SLSQP', bounds=((0, 2), (0, 2), (0, 2)))
    return res.x

def export_to_json(df):
    export_data = []
    for date, row in df.iterrows():
        export_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(row['close'], 2),
            'lev_p': round(row['lev_p'], 3),
            'lev_t': round(row['lev_t'], 3),
            'lev_g': round(row['lev_g'], 3)
        })
    with open("backtest_data.js", "w") as f:
        f.write(f"window.BACKTEST_DATA = {json.dumps(export_data)};")
    print("Data exported.")

# ==========================================
# SERVER
# ==========================================
if __name__ == "__main__":
    # 1. Prepare Data
    try:
        df_1h = fetch_binance_klines(SYMBOL, '1h', START_DATE)
        df_1d = resample_1h_to_1d(df_1h)
        sig_df = precalc_strategies(df_1h, df_1d)
        opt_weights = optimize_weights(sig_df)
        print(f"Optimization Complete: {opt_weights}")
        export_to_json(sig_df)
    except Exception as e:
        print(f"Error generating data: {e}")
        # Create empty file if fail so server starts
        with open("backtest_data.js", "w") as f: f.write("window.BACKTEST_DATA = [];")

    # 2. Start Web Server
    PORT = int(os.environ.get("PORT", 8080))
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    print(f"\nServer active on port {PORT}")
    sys.stdout.flush() # Ensure logs appear in Railway console
    httpd.serve_forever()
