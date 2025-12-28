import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import json

# ==========================================
# CONFIGURATION (MATCHING MAIN (27).PY)
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 10000.0
CAP_SPLIT = 0.333
FEE_PCT = 0.0006 

# Normalization
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
    # Resample to Daily (Midnight UTC close)
    df_1d = df_1h.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    return df_1d

# ==========================================
# STRATEGY LOGIC
# ==========================================

# --- HELPER ---
def get_sma(series, window):
    return series.rolling(window=window).mean()

def get_ewm(series, span):
    return series.ewm(span=span, adjust=False).mean()

# --- PRE-CALCULATION FUNCTIONS (Vectorized for Speed) ---

def precalc_planner(df_1d):
    df = df_1d.copy()
    df['sma_s1'] = get_sma(df['close'], PLANNER_PARAMS["S1_SMA"])
    df['sma_s2'] = get_sma(df['close'], PLANNER_PARAMS["S2_SMA"])
    return df

def precalc_tumbler(df_1d):
    df = df_1d.copy()
    df['sma1'] = get_sma(df['close'], TUMBLER_PARAMS["SMA1"])
    df['sma2'] = get_sma(df['close'], TUMBLER_PARAMS["SMA2"])
    
    # III
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    w = TUMBLER_PARAMS["III_WIN"]
    num = df['log_ret'].rolling(w).sum().abs()
    den = df['log_ret'].abs().rolling(w).sum()
    df['iii'] = (num / den).fillna(0)
    return df

def precalc_gainer(df_1h, df_1d):
    # 1H MACDs
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
        # Store signal (+1/-1) directly
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
# MAIN SIMULATION
# ==========================================
def run_simulation(df_1h, df_1d):
    print("Pre-calculating Indicators...")
    df_1d_plan = precalc_planner(df_1d)
    df_1d_tumb = precalc_tumbler(df_1d)
    df_1h_gain, df_1d_gain = precalc_gainer(df_1h, df_1d)
    
    # Align Data:
    # 1. We iterate through df_1h (Hourly execution).
    # 2. For Daily data, we must access the row corresponding to "Yesterday" (Closed candle).
    #    e.g., if current time is 2023-01-05 14:00, latest daily candle is 2023-01-04.
    
    # Map 1H timestamps to previous date
    df_1h_dates = df_1h.index.date
    # Shift dates back by 1 day to find the relevant "closed" daily candle
    target_dates = df_1h_dates - timedelta(days=1)
    
    print("Running Loop...")
    
    # State Initialization
    cash = INITIAL_CAPITAL
    equity_curve = []
    
    # Planner State
    p_s1_equity = 0.0 # Will init to cap
    p_s2_equity = 0.0
    p_last_price = 0.0
    p_last_lev_s1 = 0.0
    p_last_lev_s2 = 0.0
    
    p_s1_entry = None
    p_s1_peak = 0.0
    p_s1_stopped = False
    p_s1_trend = 0
    
    p_s2_peak = 0.0
    p_s2_stopped = False
    p_s2_trend = 0
    
    # Tumbler State
    t_flat_regime = False
    
    # Start Loop
    # Need enough warmup for SMAs (400 days)
    start_idx = 400 * 24 
    if start_idx >= len(df_1h): start_idx = 0
    
    for i in range(start_idx, len(df_1h)):
        ts = df_1h.index[i]
        curr_price = df_1h['open'].iloc[i] # Exec at Open
        prev_close_1h = df_1h['close'].iloc[i-1] # Data for 1H indicators
        
        # Get Daily Data (Yesterday)
        yesterday = (ts - timedelta(days=1)).date()
        
        # Check if we have daily data for yesterday
        if yesterday not in df_1d_plan.index:
            equity_curve.append({'date': ts, 'equity': cash, 'net_lev': 0})
            continue
            
        # Extract Row Data
        row_d_p = df_1d_plan.loc[yesterday]
        row_d_t = df_1d_tumb.loc[yesterday]
        row_d_g = df_1d_gain.loc[yesterday]
        row_h_g = df_1h_gain.iloc[i-1] # Previous hour closed
        
        daily_close = row_d_p['close']
        
        # --- PLANNER LOGIC ---
        # Initialize virtual equity
        strat_cap = cash * CAP_SPLIT # Simplified: Assume strategy owns 33% of total
        # Actually master_trader logic: strat_cap = portfolio * 0.333.
        # But here we simulate the *whole* portfolio.
        # So effective capital for calculation is Total * 0.333
        
        effective_cap = cash * CAP_SPLIT 
        
        if p_s1_equity <= 0: p_s1_equity = effective_cap
        if p_s2_equity <= 0: p_s2_equity = effective_cap
        if p_last_price <= 0: p_last_price = daily_close
        
        # Update Virtual Equity (Daily Step Logic - approximated hourly)
        # Note: In main.py, this updates every cycle.
        # It uses (curr_price - last_price).
        # We need to simulate the state persisting.
        
        # Update based on price move from LAST EXECUTION (Hour i-1)
        # Last execution price was df_1h['open'].iloc[i-1]
        last_exec_price = df_1h['open'].iloc[i-1]
        
        pct_change = (curr_price - last_exec_price) / last_exec_price
        
        p_s1_equity *= (1.0 + pct_change * p_last_lev_s1)
        p_s2_equity *= (1.0 + pct_change * p_last_lev_s2)
        
        # S1 Logic
        s1_trend_new = 1 if daily_close > row_d_p['sma_s1'] else -1
        if p_s1_trend != s1_trend_new:
            p_s1_trend = s1_trend_new
            p_s1_entry = ts
            p_s1_stopped = False
            p_s1_peak = p_s1_equity
            
        if p_s1_equity > p_s1_peak: p_s1_peak = p_s1_equity
        dd_s1 = (p_s1_peak - p_s1_equity) / p_s1_peak if p_s1_peak > 0 else 0
        if dd_s1 > PLANNER_PARAMS["S1_STOP"]: p_s1_stopped = True
        
        lev_s1 = 0.0
        if not p_s1_stopped:
            # Decay
            days_since = 0
            if p_s1_entry:
                days_since = (ts - p_s1_entry).total_seconds() / 86400
            
            decay = 1.0
            if days_since < PLANNER_PARAMS["S1_DECAY"]:
                decay = 1.0 - (days_since / PLANNER_PARAMS["S1_DECAY"])**2
            else:
                decay = 0.0
            lev_s1 = float(p_s1_trend) * decay
            
        # S2 Logic
        s2_trend_new = 1 if daily_close > row_d_p['sma_s2'] else -1
        if p_s2_trend != s2_trend_new:
            p_s2_trend = s2_trend_new
            p_s2_stopped = False
            p_s2_peak = p_s2_equity
            
        if p_s2_equity > p_s2_peak: p_s2_peak = p_s2_equity
        dd_s2 = (p_s2_peak - p_s2_equity) / p_s2_peak if p_s2_peak > 0 else 0
        if dd_s2 > PLANNER_PARAMS["S2_STOP"]: p_s2_stopped = True
        
        dist_pct = abs(daily_close - row_d_p['sma_s2']) / row_d_p['sma_s2']
        is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
        tgt_size = 0.5 if is_prox else 1.0
        
        lev_s2 = 0.0
        if p_s2_stopped:
            if is_prox:
                p_s2_stopped = False
                p_s2_peak = p_s2_equity
                lev_s2 = float(s2_trend_new) * tgt_size
        else:
            lev_s2 = float(s2_trend_new) * tgt_size
            
        lev_planner = max(-2.0, min(2.0, lev_s1 + lev_s2))
        
        # Save state for next iter
        p_last_lev_s1 = lev_s1
        p_last_lev_s2 = lev_s2
        
        # --- TUMBLER LOGIC ---
        iii = row_d_t['iii']
        raw_lev_t = TUMBLER_PARAMS['LEVS'][2]
        if iii < TUMBLER_PARAMS['III_TH'][0]: raw_lev_t = TUMBLER_PARAMS['LEVS'][0]
        elif iii < TUMBLER_PARAMS['III_TH'][1]: raw_lev_t = TUMBLER_PARAMS['LEVS'][1]
        
        if iii < TUMBLER_PARAMS['FLAT_THRESH']: t_flat_regime = True
        
        if t_flat_regime:
            sma1, sma2 = row_d_t['sma1'], row_d_t['sma2']
            band = TUMBLER_PARAMS['BAND']
            if abs(daily_close - sma1) <= sma1*band or abs(daily_close - sma2) <= sma2*band:
                t_flat_regime = False
        
        lev_tumbler = 0.0
        if not t_flat_regime:
            if daily_close > row_d_t['sma1'] and daily_close > row_d_t['sma2']:
                lev_tumbler = raw_lev_t
            elif daily_close < row_d_t['sma1'] and daily_close < row_d_t['sma2']:
                lev_tumbler = -raw_lev_t
                
        # --- GAINER LOGIC ---
        # 1H Signal
        # score_macd_1h was calc based on close[i-1] (Previous Hour)
        s_m1h = row_h_g['score_macd_1h']
        s_m1d = row_d_g['score_macd_1d']
        s_s1d = row_d_g['score_sma_1d']
        
        ws = GAINER_PARAMS["GA_WEIGHTS"]
        lev_gainer = (s_m1h * ws["MACD_1H"] + s_m1d * ws["MACD_1D"] + s_s1d * ws["SMA_1D"]) / sum(ws.values())
        
        # --- COMBINATION ---
        # Normalization
        n_p = lev_planner
        n_t = lev_tumbler * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        n_g = lev_gainer * TARGET_STRAT_LEV
        
        # Combined Leverage on Total Capital
        # Each strat gets CAP_SPLIT (33%)
        net_lev = (n_p + n_t + n_g) * CAP_SPLIT
        
        # --- EXECUTION & PNL ---
        # We hold position for 1 hour
        # PnL = Leverage * %Change
        # Price Change: Open[i] to Open[i+1]? 
        # Backtest loop structure: We are at [i]. We decide `net_lev`.
        # We hold this until [i+1].
        # In this loop structure, we calculate PnL of PREVIOUS position first, then set NEW position.
        
        # Wait, simple way:
        # At index i, we set position.
        # At index i+1, we realize PnL.
        # So we need to store `net_lev` for next iteration.
        
        # Better: Calculate PnL from [i] to [i+1] using Close[i] / Open[i]?
        # Let's use Open[i] to Close[i] (Intraday hourly PnL)
        # Then next hour Open[i+1] ~ Close[i].
        
        hourly_ret = (df_1h['close'].iloc[i] - curr_price) / curr_price
        
        # Apply fees if leverage changed significantly?
        # Simplified: Just apply spread/fee proxy on hourly rebalance volume?
        # Main.py uses limit chaser, very efficient.
        # Let's subtract a tiny friction cost based on total leverage magnitude
        friction = abs(net_lev) * (0.0001 / 24) # Small drag for funding/slippage
        
        step_pnl = net_lev * hourly_ret
        cash *= (1.0 + step_pnl - friction)
        
        equity_curve.append({
            'date': ts, 
            'equity': cash, 
            'net_lev': net_lev,
            'lev_p': n_p,
            'lev_t': n_t,
            'lev_g': n_g
        })
        
    return pd.DataFrame(equity_curve).set_index('date')

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Fetching Data...")
    df_1h = fetch_binance_klines(SYMBOL, '1h', START_DATE)
    df_1d = resample_1h_to_1d(df_1h)
    
    print(f"Data range: {df_1h.index[0]} to {df_1h.index[-1]}")
    
    res = run_simulation(df_1h, df_1d)
    
    # Metrics
    final_eq = res['equity'].iloc[-1]
    ret = (final_eq / INITIAL_CAPITAL) - 1
    
    # BH
    bh = (df_1h['close'] / df_1h['close'].iloc[0]) * INITIAL_CAPITAL
    bh_ret = (bh.iloc[-1] / INITIAL_CAPITAL) - 1
    
    # Drawdown
    dd = (res['equity'] - res['equity'].cummax()) / res['equity'].cummax()
    max_dd = dd.min()
    
    print("\n" + "="*50)
    print("MASTER TRADER SYSTEM BACKTEST")
    print("="*50)
    print(f"Initial Cap:   ${INITIAL_CAPITAL:,.0f}")
    print(f"Final Equity:  ${final_eq:,.0f}")
    print(f"Total Return:  {ret*100:.2f}%")
    print(f"BH Return:     {bh_ret*100:.2f}%")
    print(f"Max Drawdown:  {max_dd*100:.2f}%")
    print("="*50)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(res.index, res['equity'], label='Master System')
    plt.plot(bh.index, bh, label='Buy & Hold', alpha=0.5)
    plt.yscale('log')
    plt.title('Master System Equity Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(res.index, res['lev_p'], label='Planner (Norm)', alpha=0.7)
    plt.plot(res.index, res['lev_t'], label='Tumbler (Norm)', alpha=0.7)
    plt.plot(res.index, res['lev_g'], label='Gainer (Norm)', alpha=0.7)
    plt.plot(res.index, res['net_lev'], label='NET LEVERAGE', color='black', linewidth=1.5)
    plt.title('Leverage Contribution (Normalized)')
    plt.legend(loc='lower left', ncol=4, fontsize='small')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("master_system_backtest.png")
    print("Saved plot to master_system_backtest.png")