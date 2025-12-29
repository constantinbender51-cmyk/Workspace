import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
import builtins
import http.server
import socketserver
import webbrowser
from datetime import datetime, timedelta
from threading import Thread

# --- Configuration ---
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
YEARS = 8
CACHE_FILE = "binance_data.csv"
PORT = 8080

# Strategy Constants
CAP_SPLIT = 0.333
TARGET_STRAT_LEV = 2.0
TUMBLER_MAX_LEV = 4.327

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

# --- Slow Print Configuration ---
def slow_print(*args, **kwargs):
    # Only slow print if not called from the server thread potentially
    builtins.print(*args, **kwargs)
    # Removing sleep to prevent server lag, user can add time.sleep(0.05) if strictly desired for effect
    # time.sleep(0.05) 

print = slow_print

# --- Data Fetching Engine ---
def fetch_binance_data(symbol, interval="1h", years=8):
    if os.path.exists(CACHE_FILE):
        print(f"[INFO] Loading cached data from {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return df

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=365 * years)
    start_ts = int(start_time.timestamp() * 1000)
    
    print(f"[INFO] Downloading {years} years of {interval} data for {symbol} from Binance...")
    
    klines = []
    current_ts = start_ts
    
    while True:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "startTime": current_ts, "limit": 1000}
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            if not data: break
            klines.extend(data)
            last_close_ts = data[-1][6]
            current_ts = last_close_ts + 1
            sys.stdout.write(f"\r[FETCH] Fetched {len(klines)} candles... Last: {datetime.fromtimestamp(last_close_ts/1000)}")
            sys.stdout.flush()
            time.sleep(0.05) # Rate limit kindness
            if len(data) < 1000: break
        except Exception as e:
            print(f"\n[ERROR] Fetch failed: {e}")
            break
            
    print("\n[INFO] Processing data...")
    df = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbav", "tbqav", "ignore"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("timestamp", inplace=True)
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)
    df[cols].to_csv(CACHE_FILE)
    return df[cols]

# --- Indicator Helpers ---
def get_sma(series, window): return series.rolling(window=window).mean()
def get_ema(series, span): return series.ewm(span=span, adjust=False).mean()

def calc_macd_pos(prices, config):
    params, weights = config['params'], config['weights']
    signals = []
    for (f, s, sig_p), w in zip(params, weights):
        fast, slow = get_ema(prices, f), get_ema(prices, s)
        macd = fast - slow
        sig_line = get_ema(macd, sig_p)
        signals.append(np.where(macd > sig_line, 1.0, -1.0) * w)
    return np.sum(signals, axis=0) / sum(weights)

def calc_sma_pos(prices, config):
    params, weights = config['params'], config['weights']
    signals = [np.where(prices > get_sma(prices, p), 1.0, -1.0) * w for p, w in zip(params, weights)]
    return np.sum(signals, axis=0) / sum(weights)

def calculate_decay(entry_idx, current_idx, decay_days, freq_per_day=24):
    if entry_idx is None: return 1.0
    days_since = (current_idx - entry_idx) / freq_per_day
    if days_since >= decay_days: return 0.0
    return max(0.0, 1.0 - (days_since / decay_days) ** 2)

# --- Backtest Class ---
class Backtester:
    def __init__(self, df_1h):
        self.df = df_1h.copy()
        self.df_1d = df_1h.resample('D').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        self.capital = 10000.0
        
    def precalculate_indicators(self):
        print("[INFO] Pre-calculating indicators (Fixing Lookahead)...")
        d = self.df_1d
        
        # Shift Daily indicators by 1 day
        self.df['d_sma120'] = get_sma(d['close'], PLANNER_PARAMS["S1_SMA"]).shift(1).reindex(self.df.index, method='ffill')
        self.df['d_sma400'] = get_sma(d['close'], PLANNER_PARAMS["S2_SMA"]).shift(1).reindex(self.df.index, method='ffill')
        self.df['d_sma32'] = get_sma(d['close'], TUMBLER_PARAMS["SMA1"]).shift(1).reindex(self.df.index, method='ffill')
        self.df['d_sma114'] = get_sma(d['close'], TUMBLER_PARAMS["SMA2"]).shift(1).reindex(self.df.index, method='ffill')
        
        log_ret = np.log(d['close'] / d['close'].shift(1))
        w = TUMBLER_PARAMS["III_WIN"]
        iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0)
        self.df['d_iii'] = iii.shift(1).reindex(self.df.index, method='ffill')
        
        d_g_macd = pd.Series(calc_macd_pos(d['close'], GAINER_PARAMS["MACD_1D"]), index=d.index)
        d_g_sma = pd.Series(calc_sma_pos(d['close'], GAINER_PARAMS["SMA_1D"]), index=d.index)
        
        self.df['d_gainer_macd'] = d_g_macd.shift(1).reindex(self.df.index, method='ffill')
        self.df['d_gainer_sma'] = d_g_sma.shift(1).reindex(self.df.index, method='ffill')
        
        # Shift Hourly MACD by 1 hour
        h_g_macd = pd.Series(calc_macd_pos(self.df['close'], GAINER_PARAMS["MACD_1H"]), index=self.df.index)
        self.df['h_gainer_macd'] = h_g_macd.shift(1)
        
        self.df.dropna(inplace=True)

    def run(self):
        p_state = {
            "s1_equity": self.capital * CAP_SPLIT, "s2_equity": self.capital * CAP_SPLIT,
            "last_price": self.df['close'].iloc[0], "last_lev_s1": 0.0, "last_lev_s2": 0.0,
            "s1": {"entry_idx": None, "peak_equity": 0.0, "stopped": False, "trend": 0},
            "s2": {"peak_equity": 0.0, "stopped": False, "trend": 0},
        }
        tumbler_flat, portfolio_curve = False, [self.capital]
        prev_total_lev, prev_price = 0.0, self.df['close'].iloc[0]
        gw = GAINER_PARAMS["GA_WEIGHTS"]
        gw_sum = sum(gw.values())

        print("[INFO] Starting simulation loop...")
        cnt = 0
        total_steps = len(self.df)
        
        for row in self.df.itertuples():
            cnt += 1
            curr_price = row.close
            
            # 1. Apply Returns
            if cnt > 1:
                step_ret = (curr_price - prev_price) / prev_price
                portfolio_curve.append(portfolio_curve[-1] * (1.0 + prev_total_lev * step_ret))
                
                # Update Planner
                p_state["s1_equity"] *= (1.0 + step_ret * p_state["last_lev_s1"])
                p_state["s2_equity"] *= (1.0 + step_ret * p_state["last_lev_s2"])
            
            # 2. Calculate Signals
            
            # Strategy 1
            s1_trend = 1 if curr_price > row.d_sma120 else -1
            s1 = p_state["s1"]
            if s1["trend"] != s1_trend:
                s1.update({"trend": s1_trend, "entry_idx": cnt, "stopped": False, "peak_equity": p_state["s1_equity"]})
            if p_state["s1_equity"] > s1["peak_equity"]: s1["peak_equity"] = p_state["s1_equity"]
            
            dd_s1 = (s1["peak_equity"] - p_state["s1_equity"]) / max(s1["peak_equity"], 1e-9)
            if dd_s1 > PLANNER_PARAMS["S1_STOP"]: s1["stopped"] = True
            
            s1_lev_out = float(s1_trend) * calculate_decay(s1["entry_idx"], cnt, PLANNER_PARAMS["S1_DECAY"]) if not s1["stopped"] else 0.0

            # Strategy 2
            s2_trend = 1 if curr_price > row.d_sma400 else -1
            s2 = p_state["s2"]
            if s2["trend"] != s2_trend:
                s2.update({"trend": s2_trend, "stopped": False, "peak_equity": p_state["s2_equity"]})
            if p_state["s2_equity"] > s2["peak_equity"]: s2["peak_equity"] = p_state["s2_equity"]
            
            dd_s2 = (s2["peak_equity"] - p_state["s2_equity"]) / max(s2["peak_equity"], 1e-9)
            if dd_s2 > PLANNER_PARAMS["S2_STOP"]: s2["stopped"] = True
            
            is_prox = (abs(curr_price - row.d_sma400) / row.d_sma400) < PLANNER_PARAMS["S2_PROX"]
            if s2["stopped"] and is_prox:
                s2.update({"stopped": False, "peak_equity": p_state["s2_equity"]})
            
            s2_lev_out = float(s2_trend) * (0.5 if is_prox else 1.0) if not s2["stopped"] else 0.0
            
            # Strategy 3
            if row.d_iii < TUMBLER_PARAMS["FLAT_THRESH"]: tumbler_flat = True
            if tumbler_flat and (abs(curr_price - row.d_sma32) <= row.d_sma32 * TUMBLER_PARAMS["BAND"] or 
                               abs(curr_price - row.d_sma114) <= row.d_sma114 * TUMBLER_PARAMS["BAND"]):
                tumbler_flat = False
            
            t_lev = 0.0
            if not tumbler_flat:
                base = TUMBLER_PARAMS["LEVS"][2]
                if row.d_iii < TUMBLER_PARAMS["III_TH"][0]: base = TUMBLER_PARAMS["LEVS"][0]
                elif row.d_iii < TUMBLER_PARAMS["III_TH"][1]: base = TUMBLER_PARAMS["LEVS"][1]
                if curr_price > row.d_sma32 and curr_price > row.d_sma114: t_lev = base
                elif curr_price < row.d_sma32 and curr_price < row.d_sma114: t_lev = -base
            
            # Store Signals
            p_state["last_lev_s1"] = s1_lev_out
            p_state["last_lev_s2"] = s2_lev_out
            
            n_p = max(-2.0, min(2.0, s1_lev_out + s2_lev_out))
            n_t = t_lev * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
            n_g = ((row.h_gainer_macd * gw["MACD_1H"] + row.d_gainer_macd * gw["MACD_1D"] + row.d_gainer_sma * gw["SMA_1D"]) / gw_sum) * TARGET_STRAT_LEV
            
            prev_total_lev = (n_p + n_t + n_g) * CAP_SPLIT
            prev_price = curr_price
            
            if cnt % 5000 == 0:
                sys.stdout.write(f"\r[SIM] Processed {cnt}/{total_steps} candles...")
                sys.stdout.flush()
        
        print("\n[INFO] Simulation complete.")
        return pd.Series(portfolio_curve, index=self.df.index[:len(portfolio_curve)])

    def report(self, equity_curve):
        returns = equity_curve.pct_change().dropna()
        total_ret = (equity_curve.iloc[-1] / self.capital) - 1
        cagr = (equity_curve.iloc[-1] / self.capital) ** (365*24 / len(equity_curve)) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24)
        max_dd = ((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min()
        
        results = {
            "final_equity": equity_curve.iloc[-1],
            "total_return": total_ret,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "years": YEARS,
            "symbol": SYMBOL
        }
        
        print("\n" + "="*40)
        print(f" REAL BACKTEST RESULTS (No Lookahead)")
        print(f" Period: {YEARS} Years | Asset: {SYMBOL}")
        print("="*40)
        print(f"Final Equity:   ${results['final_equity']:,.2f}")
        print(f"Total Return:   {results['total_return']*100:.2f}%")
        print(f"CAGR:           {results['cagr']*100:.2f}%")
        print(f"Sharpe Ratio:   {results['sharpe']:.4f}")
        print(f"Max Drawdown:   {results['max_dd']*100:.2f}%")
        print("="*40)
        
        return results

# --- Web Server Functions ---

def generate_web_files(df, equity_curve, stats):
    print("[WEB] Generating visualization data...")
    
    # 1. Prepare JSON Data
    # Downsample for web performance if too large (>10k points)
    factor = max(1, len(equity_curve) // 8000)
    
    eq_data = []
    price_data = []
    
    # Using unix timestamps for Lightweight Charts
    times = equity_curve.index[::factor]
    equities = equity_curve.values[::factor]
    prices = df['close'].reindex(equity_curve.index)[::factor].values
    
    for t, e, p in zip(times, equities, prices):
        ts = int(t.timestamp())
        eq_data.append({"time": ts, "value": float(e)})
        price_data.append({"time": ts, "value": float(p)})

    json_data = {
        "stats": {
            "final_equity": f"${stats['final_equity']:,.2f}",
            "total_return": f"{stats['total_return']*100:.2f}%",
            "cagr": f"{stats['cagr']*100:.2f}%",
            "sharpe": f"{stats['sharpe']:.2f}",
            "max_dd": f"{stats['max_dd']*100:.2f}%",
            "symbol": stats['symbol']
        },
        "series": {
            "equity": eq_data,
            "price": price_data
        }
    }
    
    with open("backtest_data.json", "w") as f:
        json.dump(json_data, f)
        
    # 2. Generate HTML
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Results</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, "Helvetica Neue", sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }
        .stat-card { background: #222; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #333; }
        .stat-val { font-size: 1.5em; font-weight: bold; color: #4caf50; }
        .stat-label { font-size: 0.9em; color: #888; margin-top: 5px; }
        .chart-container { height: 600px; width: 100%; border-radius: 8px; overflow: hidden; border: 1px solid #333; }
        .dd-val { color: #ff5252; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Backtest Results: <span id="sym"></span></h1>
            <div style="font-size: 0.8em; color: #666;">Strategy Visualization</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-val" id="final_equity">...</div><div class="stat-label">Final Equity</div></div>
            <div class="stat-card"><div class="stat-val" id="total_return">...</div><div class="stat-label">Total Return</div></div>
            <div class="stat-card"><div class="stat-val" id="cagr">...</div><div class="stat-label">CAGR</div></div>
            <div class="stat-card"><div class="stat-val" id="sharpe">...</div><div class="stat-label">Sharpe Ratio</div></div>
            <div class="stat-card"><div class="stat-val dd-val" id="max_dd">...</div><div class="stat-label">Max Drawdown</div></div>
        </div>

        <div id="chart" class="chart-container"></div>
    </div>

    <script>
        fetch('backtest_data.json')
            .then(response => response.json())
            .then(data => {
                // Populate Stats
                document.getElementById('sym').innerText = data.stats.symbol;
                document.getElementById('final_equity').innerText = data.stats.final_equity;
                document.getElementById('total_return').innerText = data.stats.total_return;
                document.getElementById('cagr').innerText = data.stats.cagr;
                document.getElementById('sharpe').innerText = data.stats.sharpe;
                document.getElementById('max_dd').innerText = data.stats.max_dd;

                // Create Chart
                const chartContainer = document.getElementById('chart');
                const chart = LightweightCharts.createChart(chartContainer, {
                    layout: { background: { color: '#111' }, textColor: '#DDD' },
                    grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } },
                    rightPriceScale: { borderColor: '#333' },
                    timeScale: { borderColor: '#333', timeVisible: true },
                    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                });

                // Equity Series (Green Area)
                const equitySeries = chart.addAreaSeries({
                    topColor: 'rgba(76, 175, 80, 0.56)',
                    bottomColor: 'rgba(76, 175, 80, 0.04)',
                    lineColor: 'rgba(76, 175, 80, 1)',
                    lineWidth: 2,
                    priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
                });
                equitySeries.setData(data.series.equity);

                // Price Series (Overlay Line, Right Scale with Margin)
                // We create a separate scale to compare relative moves or just overlay
                const priceSeries = chart.addLineSeries({
                    color: 'rgba(41, 98, 255, 0.6)',
                    lineWidth: 1,
                    priceScaleId: 'left', // Use left axis for price to separate from equity
                    title: 'Asset Price'
                });
                chart.priceScale('left').applyOptions({
                     visible: true,
                     borderColor: '#333'
                });
                
                priceSeries.setData(data.series.price);
                
                chart.timeScale().fitContent();
            })
            .catch(err => console.error("Error loading data:", err));
    </script>
</body>
</html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
    print(f"[WEB] Dashboard generated at index.html")

def run_server():
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Allow address reuse to prevent "Address already in use" errors on quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n[SERVER] Serving at http://localhost:{PORT}")
        print("[SERVER] Press Ctrl+C to stop.")
        # Try to open browser automatically
        try:
            webbrowser.open(f"http://localhost:{PORT}")
        except:
            pass
        httpd.serve_forever()

if __name__ == "__main__":
    # 1. Fetch
    df = fetch_binance_data(SYMBOL, INTERVAL, YEARS)
    
    # 2. Backtest
    bt = Backtester(df)
    bt.precalculate_indicators()
    equity = bt.run()
    stats = bt.report(equity)
    
    # 3. Generate Website
    generate_web_files(df, equity, stats)
    
    # 4. Start Server
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n[SERVER] Stopped.")
