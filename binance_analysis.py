import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import builtins
import http.server
import socketserver
import webbrowser
import urllib.parse
from datetime import datetime, timedelta

# --- Matplotlib Setup for Headless/Scientific Plotting ---
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt

# --- Configuration ---
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
YEARS = 8
CACHE_FILE = "binance_data.csv"
PORT = 8080

# Default Constants
DEFAULT_CAP_SPLIT = 0.333
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

# Global Data Cache
CACHED_DF = None

def slow_print(*args, **kwargs):
    builtins.print(*args, **kwargs)

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
            time.sleep(0.05)
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
        # print("[INFO] Pre-calculating indicators...")
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

    def run(self, cap_split):
        p_state = {
            "s1_equity": self.capital * cap_split, "s2_equity": self.capital * cap_split,
            "last_price": self.df['close'].iloc[0], "last_lev_s1": 0.0, "last_lev_s2": 0.0,
            "s1": {"entry_idx": None, "peak_equity": 0.0, "stopped": False, "trend": 0},
            "s2": {"peak_equity": 0.0, "stopped": False, "trend": 0},
        }
        
        # Individual Strategy Curves (Starting with 1/3 capital each * multiplier)
        c_blue = [self.capital * cap_split]  # Planner
        c_green = [self.capital * cap_split] # Tumbler
        c_brown = [self.capital * cap_split] # Gainer
        
        tumbler_flat = False
        prev_price = self.df['close'].iloc[0]
        gw = GAINER_PARAMS["GA_WEIGHTS"]
        gw_sum = sum(gw.values())

        # Previous leverages
        pl_blue = 0.0
        pl_green = 0.0
        pl_brown = 0.0

        print(f"[SIM] Running simulation with Split Factor: {cap_split}...")
        cnt = 0
        
        for row in self.df.itertuples():
            cnt += 1
            curr_price = row.close
            
            # 1. Apply Returns
            if cnt > 1:
                step_ret = (curr_price - prev_price) / prev_price
                
                c_blue.append(c_blue[-1] * (1.0 + pl_blue * step_ret))
                c_green.append(c_green[-1] * (1.0 + pl_green * step_ret))
                c_brown.append(c_brown[-1] * (1.0 + pl_brown * step_ret))
                
                # Update Planner Internal State
                # CRITICAL: This scales equity, but percentage drawdown logic remains invariant
                p_state["s1_equity"] *= (1.0 + step_ret * p_state["last_lev_s1"])
                p_state["s2_equity"] *= (1.0 + step_ret * p_state["last_lev_s2"])
            
            # 2. Calculate Signals
            
            # Strategy 1 & 2: Planner (BLUE)
            s1_trend = 1 if curr_price > row.d_sma120 else -1
            s1 = p_state["s1"]
            if s1["trend"] != s1_trend:
                s1.update({"trend": s1_trend, "entry_idx": cnt, "stopped": False, "peak_equity": p_state["s1_equity"]})
            if p_state["s1_equity"] > s1["peak_equity"]: s1["peak_equity"] = p_state["s1_equity"]
            
            # DRAWDOWN CHECK: (Peak - Current) / Peak
            # This is a ratio, so multiplying both Peak and Current by 'cap_split' cancels out.
            # The logic is preserved.
            dd_s1 = (s1["peak_equity"] - p_state["s1_equity"]) / max(s1["peak_equity"], 1e-9)
            if dd_s1 > PLANNER_PARAMS["S1_STOP"]: s1["stopped"] = True
            
            s1_lev_out = float(s1_trend) * calculate_decay(s1["entry_idx"], cnt, PLANNER_PARAMS["S1_DECAY"]) if not s1["stopped"] else 0.0

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
            
            # Strategy 3: Tumbler (GREEN)
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
            
            # Store Planner Internal
            p_state["last_lev_s1"] = s1_lev_out
            p_state["last_lev_s2"] = s2_lev_out
            
            # Calculate Next Step Leverages
            n_p = max(-2.0, min(2.0, s1_lev_out + s2_lev_out)) # Blue Leverage
            n_t = t_lev * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV) # Green Leverage
            n_g = ((row.h_gainer_macd * gw["MACD_1H"] + row.d_gainer_macd * gw["MACD_1D"] + row.d_gainer_sma * gw["SMA_1D"]) / gw_sum) * TARGET_STRAT_LEV # Brown Leverage
            
            # Update state for next loop
            pl_blue = n_p
            pl_green = n_t
            pl_brown = n_g
            prev_price = curr_price
        
        # Create DataFrame
        res_df = pd.DataFrame({
            'Blue': c_blue,
            'Green': c_green,
            'Brown': c_brown
        }, index=self.df.index[:len(c_blue)])
        
        return res_df

    def report(self, df_curves, cap_split):
        equity_curve = df_curves.sum(axis=1)
        
        # Helper for stats
        def get_stats(series, start_cap):
            ret = series.pct_change().dropna()
            # Handle edge case of flat returns
            if ret.std() == 0: sharpe = 0.0
            else: sharpe = ret.mean() / ret.std() * np.sqrt(365 * 24)
            return sharpe
            
        # Calculate Individual Sharpes
        # Note: Each strategy started with (capital * cap_split)
        start_seg = self.capital * cap_split
        sharpe_blue = get_stats(df_curves['Blue'], start_seg)
        sharpe_green = get_stats(df_curves['Green'], start_seg)
        sharpe_brown = get_stats(df_curves['Brown'], start_seg)

        # Total Stats
        returns = equity_curve.pct_change().dropna()
        total_ret = (equity_curve.iloc[-1] / self.capital) - 1
        cagr = (equity_curve.iloc[-1] / self.capital) ** (365*24 / len(equity_curve)) - 1
        sharpe_total = returns.mean() / returns.std() * np.sqrt(365 * 24)
        max_dd = ((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min()
        
        results = {
            "final_equity": equity_curve.iloc[-1],
            "total_return": total_ret,
            "cagr": cagr,
            "sharpe": sharpe_total,
            "max_dd": max_dd,
            "years": YEARS,
            "symbol": SYMBOL,
            "cap_split": cap_split,
            "sharpes": {
                "blue": sharpe_blue,
                "green": sharpe_green,
                "brown": sharpe_brown
            }
        }
        return results

# --- Viz & Report ---

def generate_scientific_report(df, df_curves, stats):
    print("[VIZ] Generating plot...")
    
    plt.style.use('classic')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_price = '#555555'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{stats["symbol"]} Price ($)', color=color_price)
    
    factor = max(1, len(df_curves) // 1000)
    p_series = df['close'].reindex(df_curves.index)[::factor]
    ax1.plot(p_series.index, p_series.values, color=color_price, linewidth=0.8, alpha=0.5, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.grid(True, linestyle=':', alpha=0.4)

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Segment Equity ($)', color='black')
    
    c_blue = df_curves['Blue'][::factor]
    c_green = df_curves['Green'][::factor]
    c_brown = df_curves['Brown'][::factor]
    
    ax2.plot(c_blue.index, c_blue.values, color='blue', linewidth=1.5, label=f'Blue (Sharpe: {stats["sharpes"]["blue"]:.2f})', alpha=0.8)
    ax2.plot(c_green.index, c_green.values, color='green', linewidth=1.5, label=f'Green (Sharpe: {stats["sharpes"]["green"]:.2f})', alpha=0.8)
    ax2.plot(c_brown.index, c_brown.values, color='brown', linewidth=1.5, label=f'Brown (Sharpe: {stats["sharpes"]["brown"]:.2f})', alpha=0.8)
    
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f"Performance (Split: {stats['cap_split']}) | Total Sharpe: {stats['sharpe']:.2f}")
    fig.tight_layout()  
    plt.savefig('chart.png', dpi=100)
    plt.close()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<title>Simulation: {stats['symbol']}</title>
<style>
  body {{ background-color: #f4f4f9; color: #333; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; margin: 0; padding: 20px; }}
  .container {{ max_width: 900px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-radius: 8px; }}
  h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; color: #2c3e50; }}
  .controls {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #bce8f1; }}
  .controls form {{ display: flex; align-items: center; gap: 10px; }}
  input[type="number"] {{ padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 100px; }}
  input[type="submit"] {{ padding: 8px 15px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }}
  input[type="submit"]:hover {{ background: #2980b9; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 12px; text-align: left; }}
  th {{ background-color: #f8f9fa; color: #2c3e50; }}
  .chart-container {{ text-align: center; margin-top: 20px; }}
  .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
  .strat-row td {{ font-weight: bold; }}
  .blue {{ color: blue; }}
  .green {{ color: green; }}
  .brown {{ color: brown; }}
</style>
</head>
<body>

<div class="container">
    <h1>Strategy Simulation: {stats['symbol']}</h1>
    
    <div class="controls">
        <form action="/" method="GET">
            <label for="split"><strong>Capital Split Factor (Leverage Scaling):</strong></label>
            <input type="number" id="split" name="split" value="{stats['cap_split']}" step="0.001" min="0.001" max="10.0">
            <input type="submit" value="Run Simulation">
        </form>
        <p><small>Default: 0.333. Increasing this scales exposure. <br><strong>Logic Check:</strong> Stop losses are based on % drawdown, so scaling capital here does <em>not</em> break the logic.</small></p>
    </div>

    <h2>Strategy Components (Sharpe Analysis)</h2>
    <table>
      <tr><th>Strategy</th><th>Sharpe Ratio</th></tr>
      <tr class="strat-row blue"><td>Blue (Planner)</td><td>{stats['sharpes']['blue']:.3f}</td></tr>
      <tr class="strat-row green"><td>Green (Tumbler)</td><td>{stats['sharpes']['green']:.3f}</td></tr>
      <tr class="strat-row brown"><td>Brown (Gainer)</td><td>{stats['sharpes']['brown']:.3f}</td></tr>
    </table>

    <h2>Portfolio Stats</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Final Equity</td><td><strong>${stats['final_equity']:,.2f}</strong></td></tr>
      <tr><td>Total Return</td><td>{stats['total_return']*100:.2f}%</td></tr>
      <tr><td>Combined Sharpe</td><td>{stats['sharpe']:.4f}</td></tr>
      <tr><td>Max Drawdown</td><td><span style="color: red">{stats['max_dd']*100:.2f}%</span></td></tr>
    </table>

    <div class="chart-container">
        <img src="chart.png" alt="Performance Chart">
    </div>
</div>

</body>
</html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
    print(f"[WEB] Report updated.")

# --- Server & Execution ---

def run_simulation(cap_split):
    global CACHED_DF
    if CACHED_DF is None:
        CACHED_DF = fetch_binance_data(SYMBOL, INTERVAL, YEARS)
    
    # Initialize backtester (indicators are pre-calc once ideally, but cheap enough to do here for safety)
    bt = Backtester(CACHED_DF)
    bt.precalculate_indicators()
    
    # Run with variable split
    equity_df = bt.run(cap_split)
    stats = bt.report(equity_df, cap_split)
    
    generate_scientific_report(CACHED_DF, equity_df, stats)

class InteractiveHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        
        # Check if 'split' parameter is present
        if 'split' in query_params:
            try:
                new_split = float(query_params['split'][0])
                print(f"\n[REQ] Received request to update Split to: {new_split}")
                run_simulation(new_split)
                # Redirect to root to clean URL, or just serve index
                self.path = '/index.html'
            except ValueError:
                print("[ERR] Invalid split value")
        
        # Default behavior: Serve static files
        return super().do_GET()

def start_server():
    # Initial Run
    run_simulation(DEFAULT_CAP_SPLIT)
    
    with socketserver.TCPServer(("", PORT), InteractiveHandler) as httpd:
        print(f"\n[SERVER] Interactive dashboard running at http://localhost:{PORT}")
        print("[SERVER] Change 'Capital Split' in the browser to update results.")
        try:
            webbrowser.open(f"http://localhost:{PORT}")
        except:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[SERVER] Stopping...")
            httpd.server_close()

if __name__ == "__main__":
    start_server()
