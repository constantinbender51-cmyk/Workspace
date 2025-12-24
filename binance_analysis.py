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
import seaborn as sns
from flask import Flask, render_template

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MasterTraderBacktest")

# --- Configuration ---
SYMBOL = "BTCUSDT"
START_YEAR = 2018
CAP_SPLIT = 0.333

# --- STRATEGY PARAMETERS ---
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

# Normalization Constants
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

GLOBAL_CACHE = {
    "stats": None,
    "plots": None,
    "heatmap": None,
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
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def get_sma(series, window):
    return series.rolling(window=window).mean()

# --- Simulation Logic ---

def calc_indicators(df_1h, df_1d):
    # Pre-calculate Indicators
    d = df_1d.copy()
    d['P_SMA120'] = get_sma(d['close'], PLANNER_PARAMS["S1_SMA"])
    d['P_SMA400'] = get_sma(d['close'], PLANNER_PARAMS["S2_SMA"])
    
    # --- Tumbler Indicators (Daily Logic applied to Daily DF) ---
    # NOTE: Tumbler logic in main.py runs on Daily data, but in backtest loop we access it per hour.
    # To be strictly safe, we compute on daily and align, OR compute on hourly if that was the intent.
    # Given main.py signature: run_tumbler(df_1d...), Tumbler is a DAILY strategy.
    # Therefore, we calculate on D and shift/align like Planner.
    
    d['T_SMA1'] = get_sma(d['close'], TUMBLER_PARAMS["SMA1"])
    d['T_SMA2'] = get_sma(d['close'], TUMBLER_PARAMS["SMA2"])
    
    w = TUMBLER_PARAMS["III_WIN"]
    log_ret = np.log(d['close'] / d['close'].shift(1))
    d['III'] = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0)
    
    # Gainer 1D
    def calc_composite(df, config, kind):
        comp = pd.Series(0.0, index=df.index)
        params, weights = config['params'], config['weights']
        for p, w in zip(params, weights):
            if kind == 'macd':
                f, s, sig_p = p
                fast = df['close'].ewm(span=f, adjust=False).mean()
                slow = df['close'].ewm(span=s, adjust=False).mean()
                macd = fast - slow
                sl = macd.ewm(span=sig_p, adjust=False).mean()
                comp += np.where(macd > sl, 1.0, -1.0) * w
            else:
                sma = df['close'].rolling(window=p).mean()
                comp += np.where(df['close'] > sma, 1.0, -1.0) * w
        return comp / sum(weights)

    d['G_MACD'] = calc_composite(d, GAINER_PARAMS["MACD_1D"], 'macd')
    d['G_SMA'] = calc_composite(d, GAINER_PARAMS["SMA_1D"], 'sma')
    
    # --- ALIGNMENT & SHIFT (CRITICAL FIX PART 1) ---
    # We shift Daily data by 1 so that at any hour of Day T, we see Day T-1's data.
    d_shifted = d.shift(1)
    aligned = d_shifted.reindex(df_1h.index).ffill()
    
    h = df_1h.copy()
    
    # --- GAINER 1H & SHIFT (CRITICAL FIX PART 2) ---
    # Calculate on current data, THEN shift by 1 hour to simulate trade at Open.
    gainer_1h_raw = calc_composite(h, GAINER_PARAMS["MACD_1H"], 'macd')
    h['G_MACD_1H'] = gainer_1h_raw.shift(1)
    
    # Join aligned Daily data (Planner, Tumbler, Gainer 1D) with Hourly data
    return h.join(aligned, rsuffix='_D').dropna()

def run_event_loop(df):
    """Event-Driven Loop replicating main (24).py"""
    n = len(df)
    times = df.index
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Planner & Tumbler indicators come from the ALIGNED (Daily) columns now
    p_sma120 = df['P_SMA120'].values
    p_sma400 = df['P_SMA400'].values
    
    # Tumbler inputs are now Daily inputs (shifted), matching main.py logic
    t_sma1 = df['T_SMA1'].values
    t_sma2 = df['T_SMA2'].values
    iii = df['III'].values
    
    g_macd_1h = df['G_MACD_1H'].values
    g_macd_1d = df['G_MACD'].values
    g_sma_1d = df['G_SMA'].values
    
    # State
    p_virt_eq = 10000.0
    p_s1_peak = 10000.0
    p_s2_peak = 10000.0
    p_s1_stopped = False
    p_s2_stopped = False
    p_s1_entry_idx = -1
    p_last_lev = 0.0
    
    t_flat_regime = False
    t_in_trade = False
    t_entry_price = 0.0
    t_trade_dir = 0
    
    # Equities
    eq_p = np.full(n, 10000.0)
    eq_t = np.full(n, 10000.0)
    eq_g = np.full(n, 10000.0)
    eq_total = np.full(n, 10000.0)
    
    # Params
    P_DECAY = PLANNER_PARAMS["S1_DECAY"]
    T_LEVS = TUMBLER_PARAMS["LEVS"]
    T_TH = TUMBLER_PARAMS["III_TH"]
    T_FLAT = TUMBLER_PARAMS["FLAT_THRESH"]
    T_BAND = TUMBLER_PARAMS["BAND"]
    T_TP = TUMBLER_PARAMS["TAKE_PROFIT"]
    T_STOP = TUMBLER_PARAMS["STOP"]

    for i in range(1, n):
        price_open = closes[i-1] # Entry price for this step is prev close (current open)
        price_curr = closes[i]   # Current close (for updating equity)
        price_high = highs[i]
        price_low = lows[i]
        
        # NOTE: logic decisions use Daily Shifted Data (p_sma120[i]) 
        # because [i] in aligned columns corresponds to Yesterday's Close.
        
        # --- PLANNER ---
        r_step = (price_curr - price_open) / price_open
        p_virt_eq *= (1.0 + p_last_lev * r_step)
        
        p_s1_peak = max(p_s1_peak, p_virt_eq)
        p_s2_peak = max(p_s2_peak, p_virt_eq)
        
        if p_s1_peak > 0 and (p_s1_peak - p_virt_eq)/p_s1_peak > PLANNER_PARAMS["S1_STOP"]:
            p_s1_stopped = True
            p_s1_entry_idx = -1
        
        if p_s2_peak > 0 and (p_s2_peak - p_virt_eq)/p_s2_peak > PLANNER_PARAMS["S2_STOP"]:
            p_s2_stopped = True
                
        lev_s1 = 0.0
        # Decision uses price_open (Open of current candle) vs Indicator (Yesterday/PrevHour)
        if price_open > p_sma120[i]:
            if not p_s1_stopped:
                if p_s1_entry_idx == -1: p_s1_entry_idx = i
                days = (i - p_s1_entry_idx) / 24.0
                if days < P_DECAY:
                    lev_s1 = 1.0 * (1.0 - (days/P_DECAY)**2)
        else:
            p_s1_stopped = False
            p_s1_peak = p_virt_eq
            p_s1_entry_idx = -1
            lev_s1 = -1.0
            
        lev_s2 = 0.0
        if price_open > p_sma400[i]:
            if not p_s2_stopped: lev_s2 = 1.0
            else:
                if (price_open - p_sma400[i])/p_sma400[i] < PLANNER_PARAMS["S2_PROX"]:
                    p_s2_stopped = False
                    p_s2_peak = p_virt_eq
                    lev_s2 = 0.5
        else:
            p_s2_stopped = False
            lev_s2 = 0.0
            
        final_lev_p = max(-2.0, min(2.0, lev_s1 + lev_s2))
        p_last_lev = final_lev_p
        
        # --- TUMBLER ---
        if iii[i] < T_FLAT: t_flat_regime = True
        
        if t_flat_regime:
            d1 = abs(price_open - t_sma1[i])
            d2 = abs(price_open - t_sma2[i])
            if d1 <= t_sma1[i]*T_BAND or d2 <= t_sma2[i]*T_BAND:
                t_flat_regime = False
        
        target_lev_t = 0.0
        if not t_flat_regime:
            base_lev = T_LEVS[2]
            if iii[i] < T_TH[0]: base_lev = T_LEVS[0]
            elif iii[i] < T_TH[1]: base_lev = T_LEVS[1]
            
            if price_open > t_sma1[i] and price_open > t_sma2[i]:
                target_lev_t = base_lev
            elif price_open < t_sma1[i] and price_open < t_sma2[i]:
                target_lev_t = -base_lev
                
        # Normalization
        norm_lev_t = target_lev_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        
        # TP/SL Logic
        realized_ret_t = 0.0
        if abs(norm_lev_t) > 0.001:
            if not t_in_trade:
                t_in_trade = True
                t_entry_price = price_open
                t_trade_dir = 1 if norm_lev_t > 0 else -1
            
            tp_price = t_entry_price * (1.0 + T_TP * t_trade_dir)
            sl_price = t_entry_price * (1.0 - T_STOP * t_trade_dir)
            
            hit_tp = False
            hit_sl = False
            
            if t_trade_dir == 1:
                if price_high >= tp_price: hit_tp = True
                if price_low <= sl_price: hit_sl = True
            else:
                if price_low <= tp_price: hit_tp = True
                if price_high >= sl_price: hit_sl = True
            
            # Conservative Backtest: If both hit in same candle, assume Stop Hit first (worst case)
            if hit_sl:
                step_ret = -T_STOP * abs(norm_lev_t)
                realized_ret_t = step_ret
                t_in_trade = False
            elif hit_tp:
                step_ret = T_TP * abs(norm_lev_t)
                realized_ret_t = step_ret
                t_in_trade = False
            else:
                step_ret = r_step * norm_lev_t
                realized_ret_t = step_ret
        else:
            t_in_trade = False
            realized_ret_t = 0.0
            
        # --- GAINER ---
        gw = GAINER_PARAMS["GA_WEIGHTS"]
        raw_g = (g_macd_1h[i]*gw["MACD_1H"] + g_macd_1d[i]*gw["MACD_1D"] + g_sma_1d[i]*gw["SMA_1D"]) / sum(gw.values())
        norm_lev_g = raw_g * TARGET_STRAT_LEV
        
        realized_ret_g = r_step * norm_lev_g
        
        # --- EQUITIES ---
        eq_p[i] = eq_p[i-1] * (1.0 + final_lev_p * r_step)
        eq_t[i] = eq_t[i-1] * (1.0 + realized_ret_t)
        eq_g[i] = eq_g[i-1] * (1.0 + realized_ret_g)
        
        # Combined Portfolio
        total_ret = ( (final_lev_p * r_step) + realized_ret_t + realized_ret_g ) / 3.0
        eq_total[i] = eq_total[i-1] * (1.0 + total_ret)
        
    return pd.DataFrame({
        'price': closes,
        'equity': eq_total,
        'eq_p': eq_p,
        'eq_t': eq_t,
        'eq_g': eq_g
    }, index=times)

def generate_heatmap(df):
    monthly_ret = df['equity'].resample('ME').last().pct_change().fillna(0) * 100
    years = monthly_ret.index.year.unique()
    months = range(1, 13)
    matrix = pd.DataFrame(index=years, columns=months)
    for date, ret in monthly_ret.items():
        matrix.at[date.year, date.month] = ret
    matrix = matrix.fillna(0.0)
    
    html = '<table style="width:100%; border-collapse: collapse; font-family: monospace; font-size: 0.9em;">'
    html += '<thead><tr style="color: #888;"><th>Year</th>'
    for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        html += f'<th>{m}</th>'
    html += '<th style="color:#fff;">Total</th></tr></thead><tbody>'
    
    for year, row in matrix.iterrows():
        html += f'<tr><td style="color:#fff; font-weight:bold;">{year}</td>'
        year_tot = 0.0
        for m in months:
            val = row[m]
            year_tot += val
            color = "#00ff88" if val > 0 else "#ff5252" if val < 0 else "#444"
            opacity = min(abs(val) / 20.0, 1.0) * 0.8 + 0.2
            bg = f"rgba({0 if val > 0 else 255}, {255 if val > 0 else 82}, {136 if val > 0 else 82}, {opacity * 0.3})"
            if val == 0: html += '<td style="color:#333;">-</td>'
            else: html += f'<td style="background:{bg}; color:{color}; padding:6px;">{val:+.1f}%</td>'
        tot_col = "#00ff88" if year_tot > 0 else "#ff5252"
        html += f'<td style="color:{tot_col}; font-weight:bold; border-left:1px solid #333;">{year_tot:+.1f}%</td></tr>'
    html += '</tbody></table>'
    return html

def run_backtest():
    global GLOBAL_CACHE
    GLOBAL_CACHE['progress'] = "Fetching Data..."
    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    if df_1h.empty: return None
    
    df_1d = df_1h.resample('1D').last().dropna()
    GLOBAL_CACHE['progress'] = "Calc Indicators..."
    full_df = calc_indicators(df_1h, df_1d)
    
    GLOBAL_CACHE['progress'] = "Running Event Loop..."
    res = run_event_loop(full_df)
    
    res['peak'] = res['equity'].cummax()
    res['dd'] = (res['equity'] - res['peak']) / res['peak']
    return res

def generate_analytics():
    global GLOBAL_CACHE
    if GLOBAL_CACHE['is_updating']: return
    GLOBAL_CACHE['is_updating'] = True
    
    try:
        df = run_backtest()
        if df is None: raise Exception("No Data")
        
        initial = df['equity'].iloc[0]
        final = df['equity'].iloc[-1]
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (final / initial) ** (1/years) - 1
        hourly_ret = df['equity'].pct_change().dropna()
        sharpe = (hourly_ret.mean() * 8760) / (hourly_ret.std() * np.sqrt(8760))
        max_dd = df['dd'].min()
        
        stats = {
            "return": f"{(final - initial)/initial * 100:,.0f}%",
            "cagr": f"{cagr * 100:.2f}%",
            "equity": f"${final:,.0f}",
            "sharpe": f"{sharpe:.2f}",
            "max_dd": f"{max_dd * 100:.2f}%",
            "period": f"{df.index[0].year}-{df.index[-1].year}"
        }
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['equity'], label='Master Trader', color='#00ff88')
        ax.plot(df.index, df['price']/df['price'].iloc[0]*10000, label='BTC', color='gray', alpha=0.3)
        ax.set_yscale('log')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#121212')
        ax.tick_params(colors='white')
        ax.grid(True, color='#333', linestyle=':')
        ax.legend(facecolor='#333', labelcolor='white')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plot_eq = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df.index, df['eq_p'], label='Planner', color='#2196F3', linewidth=1)
        ax2.plot(df.index, df['eq_t'], label='Tumbler', color='#9C27B0', linewidth=1)
        ax2.plot(df.index, df['eq_g'], label='Gainer', color='#FF9800', linewidth=1)
        ax2.set_yscale('log')
        ax2.set_facecolor('#1e1e1e')
        fig2.patch.set_facecolor('#121212')
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#333', linestyle=':')
        ax2.legend(facecolor='#333', labelcolor='white')
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        plot_comp = base64.b64encode(buf2.getvalue()).decode()
        plt.close(fig2)
        
        heatmap_html = generate_heatmap(df)
        
        GLOBAL_CACHE['stats'] = stats
        GLOBAL_CACHE['plots'] = {'equity': plot_eq, 'comp': plot_comp}
        GLOBAL_CACHE['heatmap'] = heatmap_html
        
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
            <h1>Running Backtest...</h1>
            <p>{GLOBAL_CACHE['progress']}</p>
            <script>setTimeout(() => location.reload(), 5000)</script>
        </div>
        """
    return render_template('index.html', stats=GLOBAL_CACHE['stats'], plots=GLOBAL_CACHE['plots'], heatmap=GLOBAL_CACHE['heatmap'])

threading.Thread(target=generate_analytics).start()
threading.Thread(target=bg_update, daemon=True).start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))