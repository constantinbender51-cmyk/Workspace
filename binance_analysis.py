#!/usr/bin/env python3
"""
backtest_server.py
Standalone backtest engine for Planner, Tumbler, and Gainer strategies using Binance data.
Generates a report and serves it on port 8080.
"""

import os
import sys
import json
import time
import requests
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
import webbrowser

# --- Configuration ---
SYMBOL = "BTCUSDT"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
HISTORY_DAYS = 365  # Backtest duration
PORT = 8080

# Normalization Constants (Matched to uploaded file)
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0
CAP_SPLIT = 0.333

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("BACKTEST")

# --- Strategy Parameters (Copied from main.py) ---
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

# --- Data Fetching ---
def fetch_binance_data(symbol, interval, days):
    """Fetches historical klines from Binance."""
    log.info(f"Fetching {interval} data for {symbol} ({days} days)...")
    limit = 1000
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    
    while True:
        url = f"{BINANCE_API_URL}?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"
        try:
            resp = requests.get(url).json()
            if not isinstance(resp, list) or len(resp) == 0:
                break
            all_data.extend(resp)
            start_time = resp[-1][0] + 1
            if len(resp) < limit:
                break
            time.sleep(0.1) # Be nice to API
        except Exception as e:
            log.error(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('timestamp', inplace=True)
    return df[['close']]

# --- Helper Functions ---
def get_sma(prices, window):
    if len(prices) < window: return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def calculate_decay(entry_date_str, decay_days):
    if not entry_date_str: return 1.0
    entry_dt = datetime.fromisoformat(entry_date_str)
    if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    # Mock current time as entry time + decay calculation in backtest would need 'current backtest time'
    # For backtest simplicity, we pass the 'current backtest time' into this function or mock it.
    # To fix this properly for backtesting, we need to pass 'current_time' to this function.
    # See adaptation below in run_planner.
    return 1.0 

# Modified helper to accept current time for backtest accuracy
def calculate_decay_bt(entry_date_str, decay_days, current_time_dt):
    if not entry_date_str: return 1.0
    entry_dt = datetime.fromisoformat(entry_date_str)
    if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    if current_time_dt.tzinfo is None: current_time_dt = current_time_dt.replace(tzinfo=timezone.utc)
    
    days_since = (current_time_dt - entry_dt).total_seconds() / 86400
    if days_since >= decay_days: return 0.0
    weight = 1.0 - (days_since / decay_days) ** 2
    return max(0.0, weight)

# --- Strategy Logic (Extracted & Adapted) ---

def run_planner(df_1d, state, capital, current_time):
    p_state = state["planner"]
    price = df_1d['close'].iloc[-1]
    
    # Initialize
    if p_state["s1_equity"] <= 0.0: p_state["s1_equity"] = capital
    if p_state["s2_equity"] <= 0.0: p_state["s2_equity"] = capital
    if p_state["last_price"] <= 0.0: p_state["last_price"] = price

    # Update Virtual Equities
    last_p = p_state["last_price"]
    if last_p > 0:
        pct_change = (price - last_p) / last_p
        p_state["s1_equity"] *= (1.0 + pct_change * p_state["last_lev_s1"])
        p_state["s2_equity"] *= (1.0 + pct_change * p_state["last_lev_s2"])

    # Strategy 1
    sma120 = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"])
    s1_trend = 1 if price > sma120 else -1
    s1 = p_state["s1"]
    
    if s1.get("trend", 0) != s1_trend:
        s1["trend"] = s1_trend
        s1["entry_date"] = current_time.isoformat()
        s1["stopped"] = False
        s1["peak_equity"] = p_state["s1_equity"]
    
    if p_state["s1_equity"] > s1["peak_equity"]:
        s1["peak_equity"] = p_state["s1_equity"]
    
    dd_s1 = 0.0
    if s1["peak_equity"] > 0:
        dd_s1 = (s1["peak_equity"] - p_state["s1_equity"]) / s1["peak_equity"]
    
    if dd_s1 > PLANNER_PARAMS["S1_STOP"]:
        s1["stopped"] = True

    s1_lev = 0.0
    if not s1["stopped"]:
        decay_w = calculate_decay_bt(s1["entry_date"], PLANNER_PARAMS["S1_DECAY"], current_time)
        s1_lev = float(s1_trend) * decay_w
    
    # Strategy 2
    sma400 = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"])
    s2_trend = 1 if price > sma400 else -1
    s2 = p_state["s2"]
    
    if s2.get("trend", 0) != s2_trend:
        s2["trend"] = s2_trend
        s2["stopped"] = False
        s2["peak_equity"] = p_state["s2_equity"]

    if p_state["s2_equity"] > s2["peak_equity"]:
        s2["peak_equity"] = p_state["s2_equity"]
        
    dd_s2 = 0.0
    if s2["peak_equity"] > 0:
        dd_s2 = (s2["peak_equity"] - p_state["s2_equity"]) / s2["peak_equity"]
        
    if dd_s2 > PLANNER_PARAMS["S2_STOP"]:
        s2["stopped"] = True

    dist_pct = abs(price - sma400) / sma400
    is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
    tgt_size = 0.5 if is_prox else 1.0

    s2_lev = 0.0
    if s2["stopped"]:
        if is_prox:
            s2["stopped"] = False 
            s2["peak_equity"] = p_state["s2_equity"] 
            s2_lev = float(s2_trend) * tgt_size
    else:
        s2_lev = float(s2_trend) * tgt_size

    net_lev = max(-2.0, min(2.0, s1_lev + s2_lev))
    
    p_state["last_price"] = price
    p_state["last_lev_s1"] = s1_lev
    p_state["last_lev_s2"] = s2_lev
    
    return net_lev

def run_tumbler(df_1d, state, capital):
    s = state["tumbler"]
    w = TUMBLER_PARAMS["III_WIN"]
    if len(df_1d) < w+1: return 0.0
    # Calculate simple returns for III approx
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0).iloc[-1]
    
    lev = TUMBLER_PARAMS["LEVS"][2]
    if iii < TUMBLER_PARAMS["III_TH"][0]: lev = TUMBLER_PARAMS["LEVS"][0]
    elif iii < TUMBLER_PARAMS["III_TH"][1]: lev = TUMBLER_PARAMS["LEVS"][1]
    
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]: s["flat_regime"] = True
    
    if s["flat_regime"]:
        sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
        sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
        curr = df_1d['close'].iloc[-1]
        band1, band2 = sma1 * TUMBLER_PARAMS["BAND"], sma2 * TUMBLER_PARAMS["BAND"]
        if abs(curr - sma1) <= band1 or abs(curr - sma2) <= band2:
            s["flat_regime"] = False
            
    if s["flat_regime"]: return 0.0
    
    sma1, sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"]), get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    curr = df_1d['close'].iloc[-1]
    return lev if (curr > sma1 and curr > sma2) else (-lev if (curr < sma1 and curr < sma2) else 0.0)

def run_gainer(df_1h, df_1d):
    def calc_macd_pos(prices, config):
        params, weights = config['params'], config['weights']
        composite = 0.0
        for (f, s, sig_p), w in zip(params, weights):
            fast = prices.ewm(span=f, adjust=False).mean()
            slow = prices.ewm(span=s, adjust=False).mean()
            macd = fast - slow
            sig_line = macd.ewm(span=sig_p, adjust=False).mean()
            composite += (1.0 if macd.iloc[-1] > sig_line.iloc[-1] else -1.0) * w
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else composite

    def calc_sma_pos(prices, config):
        params, weights = config['params'], config['weights']
        composite = 0.0
        current = prices.iloc[-1]
        for p, w in zip(params, weights):
            composite += (1.0 if current > get_sma(prices, p) else -1.0) * w
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else composite

    # Ensure we have enough data
    if len(df_1h) < 400 or len(df_1d) < 400: return 0.0

    m1h = calc_macd_pos(df_1h['close'], GAINER_PARAMS["MACD_1H"]) * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1H"]
    m1d = calc_macd_pos(df_1d['close'], GAINER_PARAMS["MACD_1D"]) * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1D"]
    s1d = calc_sma_pos(df_1d['close'], GAINER_PARAMS["SMA_1D"]) * GAINER_PARAMS["GA_WEIGHTS"]["SMA_1D"]
    return (m1h + m1d + s1d) / sum(GAINER_PARAMS["GA_WEIGHTS"].values())

# --- Backtest Engine ---

def run_backtest():
    log.info("Starting Backtest...")
    
    # 1. Fetch Data
    df_1h_all = fetch_binance_data(SYMBOL, '1h', HISTORY_DAYS)
    
    # Init Results
    timestamps = []
    prices = []
    eq_planner = [10000.0]
    eq_tumbler = [10000.0]
    eq_gainer = [10000.0]
    lev_p_hist, lev_t_hist, lev_g_hist = [], [], []

    # Init State
    state = {
        "planner": {
            "s1_equity": 0.0, "s2_equity": 0.0,
            "last_price": 0.0, "last_lev_s1": 0.0, "last_lev_s2": 0.0,
            "s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False, "trend": 0},
            "s2": {"peak_equity": 0.0, "stopped": False, "trend": 0},
            "debug_levs": [0.0, 0.0] 
        },
        "tumbler": {"flat_regime": False}
    }

    # Warmup buffer for indicators (max SMA is 400 days, so we need 400 days? 
    # Actually we only have HISTORY_DAYS. We will step through.
    # To save time, we assume the beginning of data is "start".
    
    # Iterate Hourly
    # We need a rolling window. We'll slice pandas df for efficiency is slow in loop.
    # But for backtest correctness we must.
    
    # Optimization: We pre-calculate indicators on the full dataset? 
    # No, Planner/Tumbler are stateful path-dependent. Must loop.
    
    start_idx = 400 # Minimum warm up
    total_len = len(df_1h_all)
    
    if total_len < start_idx + 10:
        log.error("Not enough data fetched.")
        return {}

    log.info(f"Processing {total_len - start_idx} hours...")
    
    last_p_lev = 0.0
    last_t_lev = 0.0
    last_g_lev = 0.0
    
    for i in range(start_idx, total_len):
        curr_time = df_1h_all.index[i]
        curr_price = df_1h_all['close'].iloc[i]
        
        # Slices
        slice_1h = df_1h_all.iloc[:i+1]
        # Resample to 1D for daily strategies
        # Note: We include the current partial day as the last candle
        slice_1d = slice_1h.resample('1D').last()
        
        # Calculate Returns (Simulating holding from prev step)
        if i > start_idx:
            prev_price = df_1h_all['close'].iloc[i-1]
            pct_chg = (curr_price - prev_price) / prev_price
            
            eq_planner.append(eq_planner[-1] * (1.0 + last_p_lev * pct_chg))
            eq_tumbler.append(eq_tumbler[-1] * (1.0 + last_t_lev * pct_chg))
            eq_gainer.append(eq_gainer[-1] * (1.0 + last_g_lev * pct_chg))
        else:
            eq_planner.append(eq_planner[-1])
            eq_tumbler.append(eq_tumbler[-1])
            eq_gainer.append(eq_gainer[-1])

        # Run Strategies
        # Planner
        raw_p = run_planner(slice_1d, state, eq_planner[-1], curr_time)
        last_p_lev = raw_p # Fair 2.0 intrinsic
        
        # Tumbler
        raw_t = run_tumbler(slice_1d, state, eq_tumbler[-1])
        # Normalization: r_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        last_t_lev = raw_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
        
        # Gainer
        raw_g = run_gainer(slice_1h, slice_1d)
        # Normalization: r_g * TARGET_STRAT_LEV
        last_g_lev = raw_g * TARGET_STRAT_LEV
        
        timestamps.append(curr_time.strftime("%Y-%m-%d %H:%M"))
        prices.append(curr_price)
        lev_p_hist.append(last_p_lev)
        lev_t_hist.append(last_t_lev)
        lev_g_hist.append(last_g_lev)

        if i % 500 == 0:
            log.info(f"Progress: {i}/{total_len}")

    return {
        "dates": timestamps,
        "prices": prices,
        "planner": eq_planner[1:],
        "tumbler": eq_tumbler[1:],
        "gainer": eq_gainer[1:],
        "lev_p": lev_p_hist,
        "lev_t": lev_t_hist,
        "lev_g": lev_g_hist
    }

# --- HTML Generator ---
def generate_html(data):
    json_data = json.dumps(data)
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Master Trader Backtest</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: sans-serif; background: #1a1a1a; color: #ddd; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        h1 {{ color: #4CAF50; }}
        h2 {{ border-bottom: 1px solid #444; padding-bottom: 10px; }}
        canvas {{ max-height: 400px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .stat-box {{ background: #333; padding: 15px; border-radius: 5px; text-align: center; }}
        .val {{ font-size: 1.5em; font-weight: bold; margin: 10px 0; }}
        .lbl {{ font-size: 0.9em; color: #aaa; }}
        .p-color {{ color: #36a2eb; }}
        .t-color {{ color: #ff6384; }}
        .g-color {{ color: #ffcd56; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Master Trader Independent Backtest</h1>
            <p>Data Source: Binance ({SYMBOL}) | Interval: 1H | Duration: {HISTORY_DAYS} Days</p>
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="lbl">Planner Final Equity</div>
                <div class="val p-color" id="p_final">--</div>
            </div>
            <div class="stat-box">
                <div class="lbl">Tumbler Final Equity</div>
                <div class="val t-color" id="t_final">--</div>
            </div>
            <div class="stat-box">
                <div class="lbl">Gainer Final Equity</div>
                <div class="val g-color" id="g_final">--</div>
            </div>
        </div>

        <div class="card">
            <h2>Equity Curves (Normalized Base $10k)</h2>
            <canvas id="equityChart"></canvas>
        </div>

        <div class="card">
            <h2>Leverage Usage</h2>
            <canvas id="levChart"></canvas>
        </div>
    </div>

    <script>
        const data = {json_data};
        
        // Update Stats
        document.getElementById('p_final').innerText = '$' + Math.round(data.planner[data.planner.length-1]).toLocaleString();
        document.getElementById('t_final').innerText = '$' + Math.round(data.tumbler[data.tumbler.length-1]).toLocaleString();
        document.getElementById('g_final').innerText = '$' + Math.round(data.gainer[data.gainer.length-1]).toLocaleString();

        // Equity Chart
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: data.dates,
                datasets: [
                    {{ label: 'Planner', data: data.planner, borderColor: '#36a2eb', borderWidth: 2, radius: 0 }},
                    {{ label: 'Tumbler', data: data.tumbler, borderColor: '#ff6384', borderWidth: 2, radius: 0 }},
                    {{ label: 'Gainer', data: data.gainer, borderColor: '#ffcd56', borderWidth: 2, radius: 0 }},
                    {{ label: 'Buy & Hold', data: data.prices.map(p => p * (10000/data.prices[0])), borderColor: '#666', borderDash: [5,5], borderWidth: 1, radius: 0 }}
                ]
            }},
            options: {{
                responsive: true,
                interaction: {{ mode: 'index', intersect: false }},
                scales: {{
                    x: {{ ticks: {{ maxTicksLimit: 10 }} }},
                    y: {{ grid: {{ color: '#444' }} }}
                }}
            }}
        }});

        // Leverage Chart
        new Chart(document.getElementById('levChart'), {{
            type: 'line',
            data: {{
                labels: data.dates,
                datasets: [
                    {{ label: 'Planner Lev', data: data.lev_p, borderColor: '#36a2eb', borderWidth: 1, radius: 0 }},
                    {{ label: 'Tumbler Lev', data: data.lev_t, borderColor: '#ff6384', borderWidth: 1, radius: 0 }},
                    {{ label: 'Gainer Lev', data: data.lev_g, borderColor: '#ffcd56', borderWidth: 1, radius: 0 }}
                ]
            }},
            options: {{
                responsive: true,
                interaction: {{ mode: 'index', intersect: false }},
                scales: {{
                    x: {{ ticks: {{ maxTicksLimit: 10 }} }},
                    y: {{ min: -2.5, max: 2.5, grid: {{ color: '#444' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """
    return html

# --- Server Class ---
class BacktestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            super().do_GET()

def start_server():
    server = HTTPServer(('', PORT), BacktestHandler)
    log.info(f"Serving report at http://localhost:{PORT}")
    webbrowser.open(f"http://localhost:{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    # 1. Run Simulation
    results = run_backtest()
    if not results:
        log.error("Backtest failed or no data.")
        sys.exit(1)

    # 2. Generate HTML
    HTML_CONTENT = generate_html(results)
    
    # 3. Start Server
    start_server()
