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

# Global variable to store pre-loaded 5m data to speed up slider interaction
DATA_5M = None

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
    global DATA_5M
    delayed_print("[PROCESS] Pre-processing data into 5m intervals...")
    df = pd.read_csv(csv_file)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    DATA_5M = df.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

def run_backtest(sma_period):
    global DATA_5M
    if DATA_5M is None:
        return None, None
    
    df = DATA_5M.copy()
    df['sma'] = df['close'].rolling(window=sma_period).mean()
    hold_period = 288 # Constant 24h hold

    # Independent Streams
    df['long_sig'] = (df['close'] > df['sma']).astype(int)
    df['short_sig'] = (df['close'] < df['sma']).astype(int)
    
    df['pos_long'] = df['long_sig'].rolling(window=hold_period).sum()
    df['pos_short'] = df['short_sig'].rolling(window=hold_period).sum()
    df['net_exposure'] = df['pos_long'] - df['pos_short']

    # Fee Logic: 0.02% on delta change of net position size
    # We use abs(diff) because increasing or decreasing exposure incurs fees
    fee_rate = 0.0002
    df['pos_delta'] = df['net_exposure'].diff().abs().fillna(0)
    df['fees'] = df['pos_delta'] * fee_rate

    future_close = df['close'].shift(-hold_period)
    pct_change = (future_close - df['close']) / df['close']
    
    df['long_ret_raw'] = df['long_sig'] * pct_change
    df['short_ret_raw'] = df['short_sig'] * (-pct_change)
    
    # We subtract fees from the total return stream
    # Note: In an additive model, fees are usually attributed to the bar of execution
    df['total_ret_net'] = (df['long_ret_raw'] + df['short_ret_raw']) - df['fees']
    
    results = df.dropna(subset=['total_ret_net', 'sma']).copy()
    results['equity'] = results['total_ret_net'].cumsum()
    
    # Sharpe (using net equity)
    daily_equity = results['equity'].resample('D').last().dropna()
    daily_diff = daily_equity.diff().dropna()
    sharpe = (daily_diff.mean() / daily_diff.std()) * np.sqrt(365) if len(daily_diff) > 1 and daily_diff.std() != 0 else 0
            
    # Drawdown
    highmark = results['equity'].cummax()
    max_dd = (highmark - results['equity']).max() * 100

    metrics = {
        "sma_period": sma_period,
        "sharpe": sharpe,
        "avg_trade_ret": results['total_ret_net'].mean() * 100,
        "total_return": results['equity'].iloc[-1] * 100,
        "total_fees": results['fees'].sum() * 100,
        "max_dd": max_dd,
        "max_long_units": results['pos_long'].max(),
        "max_short_units": results['pos_short'].max(),
    }

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(results.index, results['equity']*100, color='#1a73e8', lw=2, label='Equity (Net of Fees)')
    ax1.set_title(f'Cumulative Net Return % (SMA: {sma_period})')
    ax1.set_ylabel('Net Return %')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left', fontsize='small')
    
    ax2.plot(results.index, results['pos_long'], color='#1e8e3e', lw=1, alpha=0.6, label='Long Units')
    ax2.plot(results.index, -results['pos_short'], color='#d93025', lw=1, alpha=0.6, label='Short Units')
    ax2.fill_between(results.index, results['net_exposure'], color='#1a73e8', alpha=0.2, label='Net Exposure')
    ax2.set_title('Exposure (Units)')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='lower left', fontsize='x-small')
    
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png', dpi=90); plt.close()
    plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return metrics, plot_b64

class StrategyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        sma_val = int(params.get('sma', [40])[0])
        sma_val = max(40, min(sma_val, 34560))
        
        metrics, plot_b64 = run_backtest(sma_val)
        
        if metrics is None:
            self.send_error(500, "Data not ready")
            return

        html = f"""
        <!DOCTYPE html><html><head><title>SMA Backtest Simulator</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f0f2f5; margin:0; padding:20px; color: #202124; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 20px; margin-bottom: 25px; }}
            h1 {{ margin:0; font-size: 1.4rem; color: #1a73e8; }}
            .control-panel {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 25px; border: 1px solid #e8eaed; }}
            .slider-container {{ display: flex; align-items: center; gap: 20px; }}
            input[type=range] {{ flex-grow: 1; height: 8px; border-radius: 5px; background: #ddd; outline: none; }}
            .sma-display {{ font-weight: bold; color: #1a73e8; font-size: 1.2rem; min-width: 180px; text-align: right; }}
            .grid {{ display: grid; grid-template-columns: 280px 1fr; gap: 30px; }}
            .m-card {{ border-bottom: 1px solid #f1f3f4; padding: 12px 0; display: flex; justify-content: space-between; font-size: 0.9rem; }}
            .m-val {{ font-weight: 700; }}
            .highlight {{ color: #1a73e8; font-size: 1.1rem; }}
            img {{ width: 100%; border-radius: 4px; }}
            .label-hint {{ font-size: 0.75rem; color: #70757a; margin-top: 4px; }}
        </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>SMA Adaptive Strategy Simulator</h1>
                    <div class="timestamp">{datetime.now().strftime("%H:%M:%S")}</div>
                </div>

                <div class="control-panel">
                    <div class="slider-container">
                        <label>SMA Window:</label>
                        <input type="range" id="smaSlider" min="40" max="34560" value="{sma_val}" oninput="updateVal(this.value)" onchange="applyVal(this.value)">
                        <div class="sma-display" id="smaValDisplay">{sma_val} Bars ({(sma_val/288):.1f} Days)</div>
                    </div>
                    <div class="label-hint">Range: 40 (5m) to 34560 (120 Days). Hold: 24h. Fees: 0.02% per delta unit.</div>
                </div>

                <div class="grid">
                    <div class="sidebar">
                        <div class="m-card"><span>Realistic Sharpe</span><span class="m-val highlight">{metrics['sharpe']:.2f}</span></div>
                        <div class="m-card"><span>Avg Trade (Net)</span><span class="m-val" style="color:#1a73e8">{metrics['avg_trade_ret']:.4f}%</span></div>
                        <div class="m-card"><span>Total Return (Net)</span><span class="m-val" style="color:#1e8e3e">{metrics['total_return']:.1f}%</span></div>
                        <div class="m-card"><span>Total Fees Paid</span><span class="m-val" style="color:#d93025">{metrics['total_fees']:.1f}%</span></div>
                        <div class="m-card"><span>Max Drawdown</span><span class="m-val">{metrics['max_dd']:.1f}%</span></div>
                        <div class="m-card"><span>Max Long Units</span><span class="m-val">{metrics['max_long_units']:.0f}</span></div>
                        <div class="m-card"><span>Max Short Units</span><span class="m-val">{metrics['max_short_units']:.0f}</span></div>
                    </div>
                    <div class="main-content">
                        <img src="data:image/png;base64,{plot_b64}" alt="Backtest Plot">
                    </div>
                </div>
            </div>

            <script>
                function updateVal(val) {{
                    let days = (val / 288).toFixed(1);
                    document.getElementById('smaValDisplay').innerText = val + " Bars (" + days + " Days)";
                }}
                function applyVal(val) {{
                    window.location.href = "?sma=" + val;
                }}
            </script>
        </body></html>
        """
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

if __name__ == "__main__":
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    FILENAME = 'ohlcv.csv'
    PORT = 8080

    download_data(FILE_ID, FILENAME)
    prepare_data(FILENAME)
    
    delayed_print(f"[SERVER] Starting interactive server on port {PORT}...")
    with socketserver.TCPServer(("", PORT), StrategyHandler) as httpd:
        delayed_print(f"[INFO] Simulator live at http://localhost:{PORT}")
        httpd.serve_forever()