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

# --- Configuration ---
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"  # Expanded Range
CAP_SPLIT = 0.333

# Strategy Params
PLANNER_PARAMS = {"S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13, "S2_SMA": 400}
TUMBLER_PARAMS = {"SMA1": 32, "SMA2": 114, "STOP": 0.043, "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077, "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]}
GAINER_PARAMS = {"WEIGHTS": [0.8, 0.4], "MACD_1H": {'params': [(97, 366, 47)]}, "MACD_1D": {'params': [(52, 64, 61)]}}
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtester")

GLOBAL_CACHE = {
    "stats": None,
    "plots": None,
    "is_updating": False,
    "progress": "0%"
}

# --- Data Fetching ---
def fetch_binance_klines(symbol, interval, start_str):
    url = "https://api.binance.com/api/v3/klines"
    
    # Convert start string to ms timestamp
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    all_data = []
    current_start = start_ts
    
    logger.info(f"Fetching {interval} data for {symbol} from {start_str}...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if not isinstance(data, list) or len(data) == 0:
                break
            
            all_data.extend(data)
            last_time = data[-1][0]
            current_start = last_time + 1
            
            if len(data) < 1000 or current_start >= end_ts:
                break
                
            # Rate limit protection
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            time.sleep(1) # Backoff
            
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'vol', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df[['close']]

# --- Indicator Helpers ---
def get_sma(series, window):
    return series.rolling(window=window).mean()

def calc_macd_signal(series, params):
    f, s, sig = params
    fast = series.ewm(span=f, adjust=False).mean()
    slow = series.ewm(span=s, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    return np.where(macd > signal, 1.0, -1.0)

def calculate_indicators(df_1h, df_1d):
    # --- 1D Indicators (Shifted by 1 to prevent lookahead) ---
    # We calculate on the raw 1D series, then SHIFT(1), so that at any given day,
    # we are looking at "Yesterday's" closed indicators.
    
    d1 = df_1d.copy()
    
    # Planner
    d1['P_SMA120'] = get_sma(d1['close'], PLANNER_PARAMS["S1_SMA"])
    d1['P_SMA400'] = get_sma(d1['close'], PLANNER_PARAMS["S2_SMA"])
    
    # Tumbler
    d1['T_SMA1'] = get_sma(d1['close'], TUMBLER_PARAMS["SMA1"])
    d1['T_SMA2'] = get_sma(d1['close'], TUMBLER_PARAMS["SMA2"])
    
    # Tumbler III
    w = TUMBLER_PARAMS["III_WIN"]
    log_ret = np.log(d1['close'] / d1['close'].shift(1))
    d1['III'] = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0)
    
    # Gainer 1D MACD
    d1['G_MACD'] = calc_macd_signal(d1['close'], GAINER_PARAMS["MACD_1D"]['params'][0])
    
    # CRITICAL: Shift 1D features forward by 1 day.
    # When trading on Jan 2nd, we use Jan 1st's Daily SMA/MACD/III.
    d1_shifted = d1.shift(1)
    
    # --- 1H Indicators (No shift needed, we use closed candles) ---
    h1 = df_1h.copy()
    h1['G_MACD'] = calc_macd_signal(h1['close'], GAINER_PARAMS["MACD_1H"]['params'][0])
    
    # Align 1D (shifted) to 1H timeframe using ffill
    # This broadcasts "Yesterday's Daily Value" to every hour of "Today"
    combined = h1.join(d1_shifted, rsuffix='_daily', how='left')
    
    # Trim NaN start (due to long SMAs)
    combined.dropna(inplace=True)
    return combined

def run_backtest_pipeline():
    global GLOBAL_CACHE
    GLOBAL_CACHE['progress'] = "Fetching Data..."
    
    df_1h = fetch_binance_klines(SYMBOL, "1h", START_DATE)
    df_1d = fetch_binance_klines(SYMBOL, "1d", START_DATE)
    
    if df_1h.empty or df_1d.empty:
        logger.error("No data fetched")
        return pd.DataFrame()

    GLOBAL_CACHE['progress'] = "Calculating Indicators..."
    df = calculate_indicators(df_1h, df_1d)
    
    GLOBAL_CACHE['progress'] = "Running Simulation..."
    
    # Pre-convert columns to numpy arrays for speed
    closes = df['close'].values
    times = df.index
    
    # Planner Inputs
    p_sma120 = df['P_SMA120'].values
    p_sma400 = df['P_SMA400'].values
    
    # Tumbler Inputs
    t_sma1 = df['T_SMA1'].values
    t_sma2 = df['T_SMA2'].values
    t_iii = df['III'].values
    
    # Gainer Inputs
    g_macd_1h = df['G_MACD'].values
    g_macd_1d = df['G_MACD_daily'].values
    
    # State Variables
    planner_entry_idx = -1
    planner_stopped = False
    tumbler_flat = False
    
    # Results Arrays
    n = len(df)
    equity = np.zeros(n)
    eq_p = np.zeros(n)
    eq_t = np.zeros(n)
    eq_g = np.zeros(n)
    
    equity[0] = 10000.0
    eq_p[0] = 10000.0
    eq_t[0] = 10000.0
    eq_g[0] = 10000.0
    
    # Constants
    P_DECAY = PLANNER_PARAMS["S1_DECAY"]
    # Convert Decay Days to Hours (approx) or just track days. 
    # Since we iterate hourly, we'll track time diff.
    
    T_LEVS = TUMBLER_PARAMS["LEVS"]
    T_TH = TUMBLER_PARAMS["III_TH"]
    T_BAND = TUMBLER_PARAMS["BAND"]
    T_FLAT = TUMBLER_PARAMS["FLAT_THRESH"]
    
    G_W = GAINER_PARAMS["WEIGHTS"]
    G_WSUM = sum(G_W)
    
    # Loop
    for i in range(n - 1): # Stop 1 before end to calc return
        curr_price = closes[i]
        curr_time = times[i]
        
        # --- PLANNER LOGIC ---
        # Note: p_sma120[i] is yesterday's SMA120.
        # We compare Current Price vs Yesterday's SMA (Standard Trend Following)
        
        s1_lev = 0.0
        if curr_price > p_sma120[i]:
            if not planner_stopped:
                if planner_entry_idx == -1:
                    planner_entry_idx = i
                
                # Calc days since entry
                # (We use index difference approx or actual time)
                days_active = (curr_time - times[planner_entry_idx]).total_seconds() / 86400.0
                s1_lev = 1.0 * max(0.0, 1.0 - (days_active / P_DECAY)**2)
        else:
            planner_stopped = False
            planner_entry_idx = -1
            s1_lev = -1.0
            
        s2_lev = 1.0 if curr_price > p_sma400[i] else 0.0
        # Planner Raw Output
        raw_p = max(-2.0, min(2.0, s1_lev + s2_lev))
        
        # --- TUMBLER LOGIC ---
        iii_val = t_iii[i]
        lev_val = T_LEVS[2]
        if iii_val < T_TH[0]: lev_val = T_LEVS[0]
        elif iii_val < T_TH[1]: lev_val = T_LEVS[1]
        
        if iii_val < T_FLAT: tumbler_flat = True
        
        if tumbler_flat:
            # Check band exit (Yesterday's SMA1)
            dist = abs(curr_price - t_sma1[i])
            if dist <= t_sma1[i] * T_BAND:
                tumbler_flat = False
        
        raw_t = 0.0
        if not tumbler_flat:
            if curr_price > t_sma1[i] and curr_price > t_sma2[i]:
                raw_t = lev_val
            elif curr_price < t_sma1[i] and curr_price < t_sma2[i]:
                raw_t = -lev_val
                
        # --- GAINER LOGIC ---
        # G_MACD_daily is already shifted
        raw_g = (g_macd_1h[i] * G_W[0] + g_macd_1d[i] * G_W[1]) / G_WSUM
        
        # --- COMBINATION ---
        # Portfolio Weights
        w_p = raw_p
        w_t = raw_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        w_g = raw_g * TARGET_STRAT_LEV
        
        total_lev = w_p + w_t + w_g
        
        # --- RETURNS ---
        next_price = closes[i+1]
        mkt_ret = (next_price - curr_price) / curr_price
        
        # Main Portfolio
        equity[i+1] = equity[i] * (1.0 + total_lev * mkt_ret)
        
        # Individual Strategies (Normalized to 2.0x Leverage for fair comparison)
        # We use the SIGN of the raw strategy * 2.0 (except Planner which has magnitude)
        # Planner: Use raw_p (it targets 2.0 max anyway)
        eq_p[i+1] = eq_p[i] * (1.0 + raw_p * mkt_ret)
        
        # Tumbler: Normalize the variable leverage to a fixed 2.0x target for the 'Standalone' view?
        # Or just show the raw logic scaled to 2.0x max?
        # Let's show: (raw_t / TUMBLER_MAX_LEV) * 2.0 to match the portfolio contribution scale but on 100% equity
        eq_t[i+1] = eq_t[i] * (1.0 + (raw_t / TUMBLER_MAX_LEV * 2.0) * mkt_ret)
        
        # Gainer: raw_g is -1 to 1. Target 2.0x.
        eq_g[i+1] = eq_g[i] * (1.0 + (raw_g * 2.0) * mkt_ret)

    # Compile DataFrame
    res = pd.DataFrame({
        'price': closes,
        'equity': equity,
        'eq_p': eq_p,
        'eq_t': eq_t,
        'eq_g': eq_g
    }, index=times)
    
    # Calculate Drawdown
    res['peak'] = res['equity'].cummax()
    res['drawdown'] = (res['equity'] - res['peak']) / res['peak']
    
    return res

def generate_analytics():
    global GLOBAL_CACHE
    if GLOBAL_CACHE['is_updating']: return
    GLOBAL_CACHE['is_updating'] = True
    
    try:
        df = run_backtest_pipeline()
        
        if df.empty:
            logger.error("Backtest returned empty DF")
            return
            
        # Metrics
        initial = df['equity'].iloc[0]
        final = df['equity'].iloc[-1]
        years = (df.index[-1] - df.index[0]).days / 365.25
        
        # CAGR
        cagr = (final / initial) ** (1/years) - 1
        
        # Sharpe
        hourly_returns = df['equity'].pct_change().dropna()
        sharpe = (hourly_returns.mean() * 8760) / (hourly_returns.std() * np.sqrt(8760))
        
        stats = {
            "return": f"{(final - initial) / initial * 100:,.0f}%",
            "cagr": f"{cagr * 100:.1f}%",
            "equity": f"${final:,.0f}",
            "sharpe": f"{sharpe:.2f}",
            "max_dd": f"{df['drawdown'].min() * 100:.2f}%",
            "period": f"{df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}"
        }
        
        # --- Plotting ---
        # 1. Main Equity
        fig, ax1 = plt.subplots(figsize=(10, 5))
        df['benchmark'] = (df['price'] / df['price'].iloc[0]) * initial
        
        ax1.semilogy(df.index, df['equity'], label='Master Trader', color='#00ff00', linewidth=1.5)
        ax1.semilogy(df.index, df['benchmark'], label='BTC Buy & Hold', color='#555', linestyle='--', alpha=0.6)
        
        ax1.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#121212')
        ax1.tick_params(colors='white')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.set_title(f"Portfolio Performance (Log Scale)", color='white')
        plt.grid(True, color='#333333', linestyle=':', which='both')
        plt.legend(facecolor='#333333', labelcolor='white')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plot_eq = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # 2. Strategy Breakdown
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df.index, df['eq_p'], label='Planner (2.0x)', color='#2196F3', linewidth=1)
        ax3.plot(df.index, df['eq_t'], label='Tumbler (Norm)', color='#9C27B0', linewidth=1)
        ax3.plot(df.index, df['eq_g'], label='Gainer (2.0x)', color='#FF9800', linewidth=1)
        
        ax3.set_facecolor('#1e1e1e')
        fig3.patch.set_facecolor('#121212')
        ax3.tick_params(colors='white')
        ax3.set_yscale('log')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.set_title("Individual Strategies (Standalone Normalized)", color='white')
        plt.grid(True, color='#333333', linestyle=':', which='both')
        plt.legend(facecolor='#333333', labelcolor='white')
        
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', bbox_inches='tight')
        plot_strat = base64.b64encode(buf3.getvalue()).decode('utf-8')
        plt.close(fig3)
        
        # 3. Drawdown
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3)
        ax2.plot(df.index, df['drawdown'], color='red', linewidth=0.8)
        
        ax2.set_facecolor('#1e1e1e')
        fig2.patch.set_facecolor('#121212')
        ax2.tick_params(colors='white')
        ax2.set_title("Drawdown", color='white', fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.grid(True, color='#333333', linestyle=':')
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        plot_dd = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close(fig2)
        
        GLOBAL_CACHE['stats'] = stats
        GLOBAL_CACHE['plots'] = {'equity': plot_eq, 'drawdown': plot_dd, 'strategies': plot_strat}
        GLOBAL_CACHE['last_updated'] = datetime.now()
        
    except Exception as e:
        logger.error(f"Failed update: {e}", exc_info=True)
    finally:
        GLOBAL_CACHE['is_updating'] = False
        GLOBAL_CACHE['progress'] = "Done"

def background_updater():
    while True:
        time.sleep(43200) # 12 hours
        generate_analytics()

@app.route('/')
def dashboard():
    if not GLOBAL_CACHE['stats']:
        msg = GLOBAL_CACHE['progress']
        return f"""
        <div style="font-family:sans-serif; color:#fff; background:#121212; height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center;">
            <h1>Analytics Engine Starting...</h1>
            <p>Status: {msg}</p>
            <p style="color:#666">Fetching ~7 years of 1H data. This takes about 20-30 seconds.</p>
            <script>setTimeout(function(){{ location.reload(); }}, 5000);</script>
        </div>
        """
    return render_template('index.html', stats=GLOBAL_CACHE['stats'], 
                           plot_equity=GLOBAL_CACHE['plots']['equity'], 
                           plot_drawdown=GLOBAL_CACHE['plots']['drawdown'],
                           plot_strategies=GLOBAL_CACHE['plots']['strategies'])

# --- Start ---
threading.Thread(target=generate_analytics).start()
threading.Thread(target=background_updater, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
