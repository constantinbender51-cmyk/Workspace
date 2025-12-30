import gdown
import pandas as pd
import numpy as np
import os
import time
import sys
import http.server
import socketserver
import base64
import urllib.parse
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# Global variable to store pre-loaded 1h and 1d data to speed up simulator
DATA_1H = None
DATA_1D = None
DATA_5M = None

# Gainer Configuration from main (28).py
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

def delayed_print(text):
    print(text)
    sys.stdout.flush()
    time.sleep(0.1)

def download_data(file_id, output_filename='ohlcv_data.csv'):
    url = f'https://drive.google.com/uc?id={file_id}'
    if os.path.exists(output_filename):
        return
    delayed_print("[INFO] Downloading dataset...")
    gdown.download(url, output_filename, quiet=True)

def prepare_data(csv_file):
    global DATA_1H, DATA_1D, DATA_5M
    delayed_print("[PROCESS] Resampling data for Gainer Ensemble...")
    df = pd.read_csv(csv_file)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    
    # Pre-calculate resampled data
    DATA_5M = df.resample('5min').agg({'close': 'last'}).dropna()
    DATA_1H = df.resample('1H').agg({'close': 'last'}).dropna()
    DATA_1D = df.resample('1D').agg({'close': 'last'}).dropna()

def calc_gainer_signal(df_1h, df_1d):
    """Implementation of Gainer Ensemble Logic from main (28).py"""
    def calc_macd_pos(prices, config):
        params, weights = config['params'], config['weights']
        composite = 0.0
        for (f, s, sig_p), w in zip(params, weights):
            fast = prices.ewm(span=f, adjust=False).mean()
            slow = prices.ewm(span=s, adjust=False).mean()
            macd = fast - slow
            sig_line = macd.ewm(span=sig_p, adjust=False).mean()
            # We align to the 1h/1d index
            val = 1.0 if macd.iloc[-1] > sig_line.iloc[-1] else -1.0
            composite += val * w
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else 0

    def calc_sma_pos(prices, config):
        params, weights = config['params'], config['weights']
        composite = 0.0
        current = prices.iloc[-1]
        for p, w in zip(params, weights):
            sma = prices.rolling(window=p).mean().iloc[-1]
            composite += (1.0 if current > sma else -1.0) * w
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else 0

    m1h = calc_macd_pos(df_1h['close'], GAINER_PARAMS["MACD_1H"]) * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1H"]
    m1d = calc_macd_pos(df_1d['close'], GAINER_PARAMS["MACD_1D"]) * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1D"]
    s1d = calc_sma_pos(df_1d['close'], GAINER_PARAMS["SMA_1D"]) * GAINER_PARAMS["GA_WEIGHTS"]["SMA_1D"]
    
    total_w = sum(GAINER_PARAMS["GA_WEIGHTS"].values())
    return (m1h + m1d + s1d) / total_w

def run_backtest(hold_days):
    global DATA_1H, DATA_1D, DATA_5M
    if DATA_5M is None: return None, None
    
    hold_period_5m = int(hold_days * 288) # Convert days to 5-min intervals
    
    # Calculate Gainer signal for every 5-minute bar
    # To optimize, we pre-calculate signals at 1h and 1d and forward fill to 5m
    delayed_print(f"[PROCESS] Running Gainer Backtest (Hold: {hold_days} days)...")
    
    # Note: For strict accuracy, we'd calculate signal at every bar using lookbacks.
    # To keep it interactive, we calculate it once per hour/day and broadcast.
    
    # MACD 1H signals
    macd_1h_sigs = []
    # Simplified vectorization for the simulator
    for t in DATA_1H.index:
        # Just use data up to time t
        temp_1h = DATA_1H.loc[:t]
        if len(temp_1h) < 400: macd_1h_sigs.append(0)
        else:
            def macd_val(prices, params, weights):
                comp = 0
                for (f, s, sp), w in zip(params, weights):
                    m = prices.ewm(span=f).mean() - prices.ewm(span=s).mean()
                    sig = m.ewm(span=sp).mean()
                    comp += (1.0 if m.iloc[-1] > sig.iloc[-1] else -1.0) * w
                return comp / sum(weights)
            macd_1h_sigs.append(macd_val(temp_1h, GAINER_PARAMS["MACD_1H"]['params'], GAINER_PARAMS["MACD_1H"]['weights']))
    
    s_1h = pd.Series(macd_1h_sigs, index=DATA_1H.index).reindex(DATA_5M.index, method='ffill').fillna(0)
    
    # SMA 1D signals (similarly broadcast)
    sma_1d_sigs = []
    for t in DATA_1D.index:
        temp_1d = DATA_1D.loc[:t]
        if len(temp_1d) < 400: sma_1d_sigs.append(0)
        else:
            def sma_val(prices, params, weights):
                comp = 0; curr = prices.iloc[-1]
                for p, w in zip(params, weights):
                    comp += (1.0 if curr > prices.rolling(p).mean().iloc[-1] else -1.0) * w
                return comp / sum(weights)
            sma_1d_sigs.append(sma_val(temp_1d, GAINER_PARAMS["SMA_1D"]['params'], GAINER_PARAMS["SMA_1D"]['weights']))
    
    s_1d = pd.Series(sma_1d_sigs, index=DATA_1D.index).reindex(DATA_5M.index, method='ffill').fillna(0)
    
    # Combine signals
    w = GAINER_PARAMS["GA_WEIGHTS"]
    # Simplification: we treat MACD_1D signal similarly to SMA_1D for this high-speed simulation
    combined_signal = (s_1h * w["MACD_1H"] + s_1d * (w["MACD_1D"] + w["SMA_1D"])) / sum(w.values())
    
    df = DATA_5M.copy()
    df['signal'] = combined_signal
    
    # Additive position: Sum of signals over the variable holding period
    df['net_exposure'] = df['signal'].rolling(window=hold_period_5m).sum()
    
    # Fees: 0.02% per unit of delta change
    fee_rate = 0.0002
    df['pos_delta'] = df['net_exposure'].diff().abs().fillna(0)
    df['fees'] = df['pos_delta'] * fee_rate
    
    # Returns
    future_close = df['close'].shift(-hold_period_5m)
    pct_change = (future_close - df['close']) / df['close']
    df['strategy_ret_raw'] = df['signal'] * pct_change
    df['strategy_ret_net'] = df['strategy_ret_raw'] - df['fees']
    
    results = df.dropna(subset=['strategy_ret_net']).copy()
    results['equity'] = results['strategy_ret_net'].cumsum()
    
    # Metrics
    daily_equity = results['equity'].resample('D').last().dropna()
    daily_diff = daily_equity.diff().dropna()
    sharpe = (daily_diff.mean() / daily_diff.std()) * np.sqrt(365) if len(daily_diff) > 1 and daily_diff.std() != 0 else 0
    max_dd = (results['equity'].cummax() - results['equity']).max() * 100

    metrics = {
        "hold_days": hold_days,
        "sharpe": sharpe,
        "total_return": results['equity'].iloc[-1] * 100,
        "total_fees": results['fees'].sum() * 100,
        "avg_trade_net": results['strategy_ret_net'].mean() * 100,
        "max_dd": max_dd,
        "max_exposure": results['net_exposure'].abs().max(),
    }

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(results.index, results['equity']*100, color='#1a73e8', lw=2)
    ax1.set_title(f'Gainer Ensemble Cumulative Net Return % (Hold: {hold_days} days)')
    ax1.set_ylabel('Net Return %')
    ax1.grid(True, alpha=0.2)
    
    ax2.fill_between(results.index, results['net_exposure'], color='#1a73e8', alpha=0.2)
    ax2.plot(results.index, results['net_exposure'], color='#1a73e8', lw=1)
    ax2.set_title('Additive Net Exposure (Units)')
    ax2.set_ylabel('Units')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png', dpi=90); plt.close()
    plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return metrics, plot_b64

class SimulatorHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        hold_val = float(params.get('hold', [1.0])[0])
        hold_val = max(1.0, min(hold_val, 20.0))
        
        metrics, plot_b64 = run_backtest(hold_val)
        
        html = f"""
        <!DOCTYPE html><html><head><title>Gainer Ensemble Simulator</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f4f6f9; margin:0; padding:20px; color: #202124; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 20px; margin-bottom: 25px; }}
            h1 {{ margin:0; font-size: 1.4rem; color: #1a73e8; }}
            .control-panel {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 25px; border: 1px solid #e8eaed; }}
            .slider-container {{ display: flex; align-items: center; gap: 20px; }}
            input[type=range] {{ flex-grow: 1; }}
            .display {{ font-weight: bold; color: #1a73e8; font-size: 1.2rem; min-width: 120px; text-align: right; }}
            .grid {{ display: grid; grid-template-columns: 280px 1fr; gap: 30px; }}
            .m-card {{ border-bottom: 1px solid #f1f3f4; padding: 12px 0; display: flex; justify-content: space-between; font-size: 0.9rem; }}
            .m-val {{ font-weight: 700; }}
            img {{ width: 100%; border-radius: 4px; border: 1px solid #eee; }}
            .hint {{ font-size: 0.75rem; color: #70757a; margin-top: 10px; line-height: 1.4; }}
        </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Gainer Ensemble Simulator</h1>
                    <div class="timestamp">{datetime.now().strftime("%H:%M:%S")}</div>
                </div>
                <div class="control-panel">
                    <div class="slider-container">
                        <label>Hold Duration:</label>
                        <input type="range" min="1" max="20" step="0.5" value="{hold_val}" oninput="updateVal(this.value)" onchange="applyVal(this.value)">
                        <div class="display" id="holdDisplay">{hold_val} Days</div>
                    </div>
                    <div class="hint">
                        <b>Strategy:</b> Weighted MACD(1h/1d) + SMA(1d). <br/>
                        <b>Hold Logic:</b> Each 5m signal creates a new trade held for X days. <b>Fees:</b> 0.02% on delta unit change.
                    </div>
                </div>
                <div class="grid">
                    <div class="sidebar">
                        <div class="m-card"><span>Daily Sharpe</span><span class="m-val" style="color:#1a73e8">{metrics['sharpe']:.2f}</span></div>
                        <div class="m-card"><span>Total Return (Net)</span><span class="m-val" style="color:#1e8e3e">{metrics['total_return']:.1f}%</span></div>
                        <div class="m-card"><span>Total Fees</span><span class="m-val" style="color:#d93025">{metrics['total_fees']:.1f}%</span></div>
                        <div class="m-card"><span>Avg Bar Net</span><span class="m-val">{metrics['avg_trade_net']:.4f}%</span></div>
                        <div class="m-card"><span>Max Drawdown</span><span class="m-val">{metrics['max_dd']:.1f}%</span></div>
                        <div class="m-card"><span>Max Exposure</span><span class="m-val">{metrics['max_exposure']:.0f} units</span></div>
                    </div>
                    <div class="main-content"><img src="data:image/png;base64,{plot_b64}"></div>
                </div>
            </div>
            <script>
                function updateVal(v) {{ document.getElementById('holdDisplay').innerText = v + " Days"; }}
                function applyVal(v) {{ window.location.href = "?hold=" + v; }}
            </script>
        </body></html>
        """
        self.send_response(200); self.send_header("Content-type", "text/html"); self.end_headers()
        self.wfile.write(html.encode())

if __name__ == "__main__":
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'; FILENAME = 'ohlcv.csv'
    download_data(FILE_ID, FILENAME)
    prepare_data(FILENAME)
    with socketserver.TCPServer(("", 8080), SimulatorHandler) as httpd:
        delayed_print("[INFO] Gainer Simulator live at http://localhost:8080")
        httpd.serve_forever()