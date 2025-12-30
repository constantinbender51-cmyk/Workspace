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

# Global variables for pre-loaded data
DATA_1H = None
DATA_1D = None
DATA_5M = None

# Gainer Ensemble Configuration from main (28).py
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

def download_data(file_id, output_filename='ohlcv.csv'):
    url = f'https://drive.google.com/uc?id={file_id}'
    if os.path.exists(output_filename):
        return
    delayed_print("[INFO] Downloading dataset...")
    gdown.download(url, output_filename, quiet=True)

def prepare_data(csv_file):
    global DATA_1H, DATA_1D, DATA_5M
    delayed_print("[PROCESS] Loading and slicing data...")
    
    df = pd.read_csv(csv_file)
    
    # User Request: Remove the first 12*24*365*4 entries
    skip_rows = 12 * 24 * 365 * 4
    delayed_print(f"[INFO] Skipping first {skip_rows} entries.")
    df = df.iloc[skip_rows:].reset_index(drop=True)
    
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    
    delayed_print("[PROCESS] Generating Resampled Frames...")
    DATA_5M = df.resample('5min').agg({'close': 'last'}).dropna()
    DATA_1H = df.resample('1H').agg({'close': 'last'}).dropna()
    DATA_1D = df.resample('1D').agg({'close': 'last'}).dropna()

def calc_vectorized_macd_signal(df, config):
    params = config['params']
    weights = config['weights']
    total_signal = pd.Series(0.0, index=df.index)
    
    for (f, s, sp), w in zip(params, weights):
        fast = df['close'].ewm(span=f, adjust=False).mean()
        slow = df['close'].ewm(span=s, adjust=False).mean()
        macd = fast - slow
        signal_line = macd.ewm(span=sp, adjust=False).mean()
        sub_signal = np.where(macd > signal_line, 1.0, -1.0)
        total_signal += sub_signal * w
        
    return total_signal / sum(weights)

def calc_vectorized_sma_signal(df, config):
    params = config['params']
    weights = config['weights']
    total_signal = pd.Series(0.0, index=df.index)
    
    for p, w in zip(params, weights):
        sma = df['close'].rolling(window=p).mean()
        sub_signal = np.where(df['close'] > sma, 1.0, -1.0)
        total_signal += sub_signal * w
        
    return total_signal / sum(weights)

def run_backtest(hold_days):
    global DATA_1H, DATA_1D, DATA_5M
    if DATA_5M is None: return None, None
    
    # hold_days can now be as low as 5 minutes (5 / 1440 days)
    hold_period_5m = max(1, int(round(hold_days * 288)))
    
    delayed_print(f"[PROCESS] Running Gainer Backtest (Hold: {hold_days:.4f} days / {hold_period_5m} bars)...")
    
    sig_1h = calc_vectorized_macd_signal(DATA_1H, GAINER_PARAMS["MACD_1H"])
    sig_1d_macd = calc_vectorized_macd_signal(DATA_1D, GAINER_PARAMS["MACD_1D"])
    sig_1d_sma = calc_vectorized_sma_signal(DATA_1D, GAINER_PARAMS["SMA_1D"])
    
    s_1h = sig_1h.reindex(DATA_5M.index, method='ffill').fillna(0)
    s_1d_macd = sig_1d_macd.reindex(DATA_5M.index, method='ffill').fillna(0)
    s_1d_sma = sig_1d_sma.reindex(DATA_5M.index, method='ffill').fillna(0)
    
    w = GAINER_PARAMS["GA_WEIGHTS"]
    raw_signal = (
        s_1h * w["MACD_1H"] + 
        s_1d_macd * w["MACD_1D"] + 
        s_1d_sma * w["SMA_1D"]
    ) / sum(w.values())
    
    df = DATA_5M.copy()
    df['signal'] = raw_signal.shift(1).fillna(0)
    
    df['net_exposure'] = df['signal'].rolling(window=hold_period_5m).sum()
    
    fee_rate = 0.0002
    df['pos_delta'] = df['net_exposure'].diff().abs().fillna(0)
    df['fees'] = df['pos_delta'] * fee_rate
    
    future_close = df['close'].shift(-hold_period_5m)
    pct_change = (future_close - df['close']) / df['close']
    
    df['strategy_ret_raw'] = df['signal'] * pct_change
    df['strategy_ret_net'] = df['strategy_ret_raw'] - df['fees']
    
    results = df.dropna(subset=['strategy_ret_net', 'close']).copy()
    results['equity'] = results['strategy_ret_net'].cumsum()
    
    daily_equity = results['equity'].resample('D').last().dropna()
    daily_diff = daily_equity.diff().dropna()
    
    if len(daily_diff) > 1 and daily_diff.std() != 0:
        sharpe = (daily_diff.mean() / daily_diff.std()) * np.sqrt(365)
    else:
        sharpe = 0
        
    max_dd = (results['equity'].cummax() - results['equity']).max() * 100

    metrics = {
        "hold_days": hold_days,
        "hold_bars": hold_period_5m,
        "sharpe": sharpe,
        "total_return": results['equity'].iloc[-1] * 100,
        "total_fees": results['fees'].sum() * 100,
        "max_dd": max_dd,
        "max_exposure": results['net_exposure'].abs().max(),
    }

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(results.index, results['equity']*100, color='#1a73e8', lw=2)
    ax1.set_title(f'Gainer Strategy | No Lookahead | Hold: {hold_period_5m} Bars')
    ax1.set_ylabel('Net Return %')
    ax1.grid(True, alpha=0.2)
    
    ax2.fill_between(results.index, results['net_exposure'], color='#1a73e8', alpha=0.2)
    ax2.plot(results.index, results['net_exposure'], color='#1a73e8', lw=1)
    ax2.set_title('Stacked Exposure (Units)')
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
        
        # hold is now a float (days)
        # 5 minutes is 1/288 days
        hold_val = float(params.get('hold', [1.0])[0])
        hold_val = max(1/288, min(hold_val, 20.0))
        
        metrics, plot_b64 = run_backtest(hold_val)
        
        # UI Formatting for the hold duration
        bars = metrics['hold_bars']
        if bars == 1:
            hold_str = "5 Minutes"
        elif bars < 12:
            hold_str = f"{bars*5} Minutes"
        elif bars < 288:
            hold_str = f"{(bars*5/60):.1f} Hours"
        else:
            hold_str = f"{(bars/288):.1f} Days"

        html = f"""
        <!DOCTYPE html><html><head><title>No-Lookahead Gainer Simulator</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f4f6f9; margin:0; padding:20px; color: #202124; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 15px; margin-bottom: 20px; }}
            h1 {{ margin:0; font-size: 1.3rem; color: #1a73e8; }}
            .control-panel {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 25px; border: 1px solid #e8eaed; }}
            .slider-container {{ display: flex; align-items: center; gap: 20px; }}
            input[type=range] {{ flex-grow: 1; cursor: pointer; }}
            .display {{ font-weight: bold; color: #1a73e8; font-size: 1.2rem; min-width: 150px; text-align: right; }}
            .grid {{ display: grid; grid-template-columns: 280px 1fr; gap: 30px; }}
            .m-card {{ border-bottom: 1px solid #f1f3f4; padding: 12px 0; display: flex; justify-content: space-between; font-size: 0.9rem; }}
            .m-val {{ font-weight: 700; }}
            img {{ width: 100%; border-radius: 4px; border: 1px solid #eee; }}
            .hint {{ font-size: 0.75rem; color: #70757a; margin-top: 10px; line-height: 1.4; border-top: 1px solid #eee; padding-top: 10px; }}
        </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Gainer Simulator (Anti-Lookahead)</h1>
                    <div class="timestamp">{datetime.now().strftime("%H:%M:%S")}</div>
                </div>
                <div class="control-panel">
                    <div class="slider-container">
                        <label>Hold Window:</label>
                        <!-- Range in days: 1/288 (5m) to 20 days -->
                        <input type="range" min="{1/288}" max="20" step="{1/288}" value="{hold_val}" oninput="updateVal(this.value)" onchange="applyVal(this.value)">
                        <div class="display" id="holdDisplay">{hold_str}</div>
                    </div>
                    <div class="hint">
                        <b>Min Hold:</b> 5 minutes (1 bar). <b>Max Hold:</b> 20 days (5760 bars).<br/>
                        <b>Logic:</b> Signal[T] = Ensemble_Calculation(Prices[:T-1]). Enter at Close(T), Exit at Close(T + Hold).
                    </div>
                </div>
                <div class="grid">
                    <div class="sidebar">
                        <div class="m-card"><span>Daily Sharpe</span><span class="m-val" style="color:#1a73e8">{metrics['sharpe']:.2f}</span></div>
                        <div class="m-card"><span>Total Return (Net)</span><span class="m-val" style="color:#1e8e3e">{metrics['total_return']:.1f}%</span></div>
                        <div class="m-card"><span>Total Fees Paid</span><span class="m-val" style="color:#d93025">{metrics['total_fees']:.1f}%</span></div>
                        <div class="m-card"><span>Max Drawdown</span><span class="m-val">{metrics['max_dd']:.1f}%</span></div>
                        <div class="m-card"><span>Max Net Units</span><span class="m-val">{metrics['max_exposure']:.0f}</span></div>
                    </div>
                    <div class="main-content"><img src="data:image/png;base64,{plot_b64}"></div>
                </div>
            </div>
            <script>
                function updateVal(v) {{ 
                    let bars = Math.round(v * 288);
                    let label = "";
                    if (bars == 1) label = "5 Minutes";
                    else if (bars < 12) label = (bars * 5) + " Minutes";
                    else if (bars < 288) label = (bars * 5 / 60).toFixed(1) + " Hours";
                    else label = (bars / 288).toFixed(1) + " Days";
                    document.getElementById('holdDisplay').innerText = label; 
                }}
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
        delayed_print("[INFO] Backtest Simulator live at http://localhost:8080")
        httpd.serve_forever()