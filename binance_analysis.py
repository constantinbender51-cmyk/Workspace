import os
import io
import base64
import logging
import requests
import json
import time
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MasterTraderBacktest")

# --- Configuration ---
SYMBOL = "BTCUSDT"
START_YEAR = 2018
CAP_SPLIT = 0.333

# --- STRATEGY PARAMETERS (Matches main (24).py) ---

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

# Gainer matches the "Ensemble Lite" from main (24).py
GAINER_PARAMS = {
    "GA_WEIGHTS": {"MACD_1H": 0.8, "MACD_1D": 0.4, "SMA_1D": 0.4},
    "MACD_1H": {
        'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 
        'weights': [0.45, 0.43, 0.01]
    },
    "MACD_1D": {
        'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 
        'weights': [0.87, 0.92, 0.73]
    },
    "SMA_1D": {
        'params': [40, 120, 390], 
        'weights': [0.6, 0.8, 0.4]
    }
}

# Normalization
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

GLOBAL_CACHE = {
    "stats": None,
    "plots": None,
    "is_updating": False,
    "progress": "Waiting..."
}

# --- Data Engine ---
def fetch_binance_data(symbol, interval, start_year):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    all_data = []
    current_start = start_ts
    
    logger.info(f"Fetching {interval} data from {start_year}...")
    
    while current_start < end_ts:
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_start, 'limit': 1000}
        try:
            r = requests.get(base_url, params=params, timeout=10)
            data = r.json()
            if not isinstance(data, list) or not data: break
            all_data.extend(data)
            current_start = data[-1][0] + 1
            if len(data) < 1000: break
            time.sleep(0.05)
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            time.sleep(1)
            
    if not all_data: return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['close']]

def get_sma(series, window):
    return series.rolling(window=window).mean()

# --- Sub-Strategy Logic ---

def calc_planner_signal(df_1d):
    """
    Vectorized Planner Logic approximation.
    Full virtual equity loop is hard to vectorize perfectly, so we use an iterative approach
    on the daily dataframe which is fast enough.
    """
    closes = df_1d['close'].values
    times = df_1d.index
    n = len(df_1d)
    
    sma120 = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"]).values
    sma400 = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"]).values
    
    levs = np.zeros(n)
    
    # State
    virtual_equity = 10000.0
    s1_peak = 10000.0
    s2_peak = 10000.0
    s1_stopped = False
    s2_stopped = False
    s1_entry_idx = -1
    
    last_lev = 0.0
    
    for i in range(1, n):
        price = closes[i]
        
        # 1. Update Virtual Equity
        # Using yesterday's leverage * today's return
        ret = (price - closes[i-1]) / closes[i-1]
        virtual_equity *= (1.0 + last_lev * ret)
        
        # 2. Stops
        if virtual_equity > s1_peak: s1_peak = virtual_equity
        if virtual_equity > s2_peak: s2_peak = virtual_equity
        
        # S1 Stop
        if s1_peak > 0:
            dd1 = (s1_peak - virtual_equity) / s1_peak
            if dd1 > PLANNER_PARAMS["S1_STOP"]:
                s1_stopped = True
                s1_entry_idx = -1
        
        # S2 Stop
        if s2_peak > 0:
            dd2 = (s2_peak - virtual_equity) / s2_peak
            if dd2 > PLANNER_PARAMS["S2_STOP"]:
                s2_stopped = True
                
        # 3. Logic
        # S1
        s1_lev = 0.0
        if price > sma120[i]:
            if not s1_stopped:
                if s1_entry_idx == -1: s1_entry_idx = i
                
                # Decay
                days = (times[i] - times[s1_entry_idx]).days
                decay = PLANNER_PARAMS["S1_DECAY"]
                weight = 0.0
                if days < decay:
                    weight = 1.0 * (1.0 - (days / decay)**2)
                s1_lev = max(0.0, weight)
        else:
            s1_stopped = False # Reset on cross under
            s1_peak = virtual_equity
            s1_entry_idx = -1
            s1_lev = -1.0
            
        # S2
        s2_lev = 0.0
        if price > sma400[i]:
            if not s2_stopped:
                s2_lev = 1.0
            else:
                # Re-entry check (Proximity)
                prox = (price - sma400[i]) / sma400[i]
                if prox < PLANNER_PARAMS["S2_PROX"]:
                    s2_stopped = False
                    s2_peak = virtual_equity
                    s2_lev = 0.5
        else:
            if s2_stopped: s2_stopped = False
            s2_lev = 0.0
            
        # Net
        net = max(-2.0, min(2.0, s1_lev + s2_lev))
        levs[i] = net
        last_lev = net
        
    return pd.Series(levs, index=df_1d.index)

def calc_tumbler_signal(df_1d):
    # Vectorized Tumbler
    close = df_1d['close']
    w = TUMBLER_PARAMS["III_WIN"]
    
    # III
    log_ret = np.log(close / close.shift(1))
    iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0)
    
    sma1 = get_sma(close, TUMBLER_PARAMS["SMA1"])
    sma2 = get_sma(close, TUMBLER_PARAMS["SMA2"])
    
    # Base Lev
    levs = np.full(len(close), TUMBLER_PARAMS["LEVS"][2])
    levs = np.where(iii < TUMBLER_PARAMS["III_TH"][1], TUMBLER_PARAMS["LEVS"][1], levs)
    levs = np.where(iii < TUMBLER_PARAMS["III_TH"][0], TUMBLER_PARAMS["LEVS"][0], levs)
    
    # Flat Regime (Iterative due to state)
    final_lev = np.zeros(len(close))
    flat_regime = False
    
    band = TUMBLER_PARAMS["BAND"]
    flat_th = TUMBLER_PARAMS["FLAT_THRESH"]
    
    p_arr = close.values
    s1_arr = sma1.values
    s2_arr = sma2.values
    iii_arr = iii.values
    lev_arr = levs
    
    for i in range(len(close)):
        if iii_arr[i] < flat_th:
            flat_regime = True
            
        if flat_regime:
            # Check exit
            d1 = abs(p_arr[i] - s1_arr[i])
            d2 = abs(p_arr[i] - s2_arr[i])
            if d1 <= s1_arr[i]*band or d2 <= s2_arr[i]*band:
                flat_regime = False
        
        if flat_regime:
            final_lev[i] = 0.0
        else:
            if p_arr[i] > s1_arr[i] and p_arr[i] > s2_arr[i]:
                final_lev[i] = lev_arr[i]
            elif p_arr[i] < s1_arr[i] and p_arr[i] < s2_arr[i]:
                final_lev[i] = -lev_arr[i]
            else:
                final_lev[i] = 0.0
                
    return pd.Series(final_lev, index=df_1d.index)

def calc_gainer_signal(df_1h, df_1d):
    # Helper for composite MACD/SMA
    def get_composite(prices, config, kind='macd'):
        composite = pd.Series(0.0, index=prices.index)
        params, weights = config['params'], config['weights']
        
        for p, w in zip(params, weights):
            sig = None
            if kind == 'macd':
                f, s, sig_p = p
                fast = prices.ewm(span=f, adjust=False).mean()
                slow = prices.ewm(span=s, adjust=False).mean()
                macd = fast - slow
                sl = macd.ewm(span=sig_p, adjust=False).mean()
                sig = np.where(macd > sl, 1.0, -1.0)
            else: # SMA
                ma = prices.rolling(window=p).mean()
                sig = np.where(prices > ma, 1.0, -1.0)
            
            composite += sig * w
        
        total = sum(weights)
        return composite / total if total > 0 else composite

    # 1H Signal
    m1h = get_composite(df_1h['close'], GAINER_PARAMS["MACD_1H"], 'macd')
    
    # 1D Signals
    m1d = get_composite(df_1d['close'], GAINER_PARAMS["MACD_1D"], 'macd')
    s1d = get_composite(df_1d['close'], GAINER_PARAMS["SMA_1D"], 'sma')
    
    # Combine (Align 1D to 1H)
    # Reindex 1D signals to 1H, shift by 1 day (trade 1H using yesterday's 1D signal)
    # The 'shift' logic in live trading matches 'yesterday close'.
    m1d_aligned = m1d.shift(1).reindex(df_1h.index).ffill().fillna(0)
    s1d_aligned = s1d.shift(1).reindex(df_1h.index).ffill().fillna(0)
    
    gw = GAINER_PARAMS["GA_WEIGHTS"]
    final = (m1h * gw["MACD_1H"] + m1d_aligned * gw["MACD_1D"] + s1d_aligned * gw["SMA_1D"])
    total_w = sum(gw.values())
    
    return final / total_w

def run_backtest():
    global GLOBAL_CACHE
    GLOBAL_CACHE['progress'] = "Fetching Data..."
    
    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    if df_1h.empty: return None
    
    # Resample 1D
    df_1d = df_1h.resample('1D').last().dropna()
    
    GLOBAL_CACHE['progress'] = "Running Planner..."
    # 1. Planner (Daily -> Aligned)
    raw_p = calc_planner_signal(df_1d)
    # Shift Planner by 1 day (trade today using yesterday's signal)
    p_aligned = raw_p.shift(1).reindex(df_1h.index).ffill().fillna(0)
    
    GLOBAL_CACHE['progress'] = "Running Tumbler..."
    # 2. Tumbler (Daily -> Aligned)
    raw_t = calc_tumbler_signal(df_1d)
    t_aligned = raw_t.shift(1).reindex(df_1h.index).ffill().fillna(0)
    
    GLOBAL_CACHE['progress'] = "Running Gainer..."
    # 3. Gainer (Mixed)
    g_aligned = calc_gainer_signal(df_1h, df_1d)
    
    # --- COMBINE ---
    GLOBAL_CACHE['progress'] = "Simulating Portfolio..."
    
    # Normalize
    # Planner is already -2 to 2
    # Tumbler needs scaling to Portfolio contribution
    # Gainer is -1 to 1, target 2.0
    
    w_p = p_aligned
    w_t = t_aligned * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
    w_g = g_aligned * TARGET_STRAT_LEV
    
    total_lev = w_p + w_t + w_g
    
    # Returns
    # Open positions at start of hour i, based on signal at end of i-1
    # Return is (Close_i - Close_i-1)/Close_i-1
    # We effectively trade the 'next' bar. 
    pos = total_lev.shift(1).fillna(0)
    ret = df_1h['close'].pct_change().fillna(0)
    
    strat_ret = pos * ret
    
    # Compile
    df_res = pd.DataFrame({
        'price': df_1h['close'],
        'pos': pos,
        'ret': strat_ret
    })
    
    df_res['cum_ret'] = (1 + df_res['ret']).cumprod()
    df_res['peak'] = df_res['cum_ret'].cummax()
    df_res['dd'] = (df_res['cum_ret'] - df_res['peak']) / df_res['peak']
    
    # Component Breakdown (Normalized to 100% cap for visualization)
    df_res['ret_p'] = (p_aligned.shift(1) * ret)
    df_res['ret_t'] = (t_aligned.shift(1) * ret) # Raw Tumbler
    df_res['ret_g'] = (g_aligned.shift(1) * ret) # Raw Gainer
    
    df_res['eq_p'] = (1 + df_res['ret_p']).cumprod()
    df_res['eq_t'] = (1 + df_res['ret_t']).cumprod()
    df_res['eq_g'] = (1 + df_res['ret_g']).cumprod()

    return df_res

def generate_analytics():
    global GLOBAL_CACHE
    if GLOBAL_CACHE['is_updating']: return
    GLOBAL_CACHE['is_updating'] = True
    
    try:
        df = run_backtest()
        if df is None: raise Exception("No Data")
        
        # Stats
        initial = 10000
        final = initial * df['cum_ret'].iloc[-1]
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (final / initial) ** (1/years) - 1
        
        hourly_std = df['ret'].std()
        sharpe = (df['ret'].mean() / hourly_std * np.sqrt(365*24)) if hourly_std > 0 else 0
        max_dd = df['dd'].min()
        
        stats = {
            "return": f"{(final - initial)/initial * 100:,.0f}%",
            "cagr": f"{cagr * 100:.2f}%",
            "equity": f"${final:,.0f}",
            "sharpe": f"{sharpe:.2f}",
            "max_dd": f"{max_dd * 100:.2f}%",
            "period": f"{df.index[0].year}-{df.index[-1].year}"
        }
        
        # Plots
        # 1. Main Equity
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['cum_ret'] * initial, label='Combined Strategy', color='#00ff88')
        ax.plot(df.index, (df['price'] / df['price'].iloc[0]) * initial, label='BTC Hold', color='gray', alpha=0.3)
        ax.set_yscale('log')
        ax.set_title("Master Trader Performance", color='white')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#121212')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#333', labelcolor='white')
        ax.grid(True, color='#333', linestyle=':')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plot_eq = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        
        # 2. Components
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df.index, df['eq_p'], label='Planner', color='#2196F3', linewidth=1)
        ax2.plot(df.index, df['eq_t'], label='Tumbler', color='#9C27B0', linewidth=1)
        ax2.plot(df.index, df['eq_g'], label='Gainer', color='#FF9800', linewidth=1)
        ax2.set_yscale('log')
        ax2.set_title("Component Performance (Raw Signals)", color='white')
        ax2.set_facecolor('#1e1e1e')
        fig2.patch.set_facecolor('#121212')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#333', labelcolor='white')
        ax2.grid(True, color='#333', linestyle=':')
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        plot_comp = base64.b64encode(buf2.getvalue()).decode()
        plt.close(fig2)
        
        GLOBAL_CACHE['stats'] = stats
        GLOBAL_CACHE['plots'] = {'equity': plot_eq, 'comp': plot_comp}
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        GLOBAL_CACHE['progress'] = f"Error: {str(e)}"
    finally:
        GLOBAL_CACHE['is_updating'] = False
        GLOBAL_CACHE['progress'] = "Done"

def bg_update():
    while True:
        time.sleep(21600)
        generate_analytics()

@app.route('/')
def index():
    if not GLOBAL_CACHE['stats']:
        return f"""
        <div style="background:#121212; color:white; height:100vh; display:flex; flex-direction:column; justify-content:center; align-items:center;">
            <h1>Running Master Trader Backtest...</h1>
            <p>{GLOBAL_CACHE['progress']}</p>
            <script>setTimeout(() => location.reload(), 5000)</script>
        </div>
        """
    return render_template('index.html', stats=GLOBAL_CACHE['stats'], plots=GLOBAL_CACHE['plots'])

threading.Thread(target=generate_analytics).start()
threading.Thread(target=bg_update, daemon=True).start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
