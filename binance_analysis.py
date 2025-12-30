import gdown
import pandas as pd
import numpy as np
import os
import time
import sys
import http.server
import socketserver
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# Helper to handle the requested print delay
def delayed_print(text):
    print(text)
    sys.stdout.flush()
    time.sleep(0.3)

def download_data(file_id, output_filename='ohlcv_data.csv'):
    url = f'https://drive.google.com/uc?id={file_id}'
    if os.path.exists(output_filename):
        delayed_print(f"[INFO] File {output_filename} already exists. Skipping download.")
        return
    delayed_print("[INFO] Initializing download from Google Drive...")
    gdown.download(url, output_filename, quiet=False)
    delayed_print("[SUCCESS] Download complete.")

def run_strategy(csv_file):
    delayed_print("[PROCESS] Loading dataset...")
    try:
        df = pd.read_csv(csv_file)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.columns = [c.lower() for c in df.columns]
        
        delayed_print("[PROCESS] Resampling to 5-minute bars...")
        df_5m = df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        delayed_print("[PROCESS] Calculating signals for independent streams...")
        df_5m['sma_40'] = df_5m['close'].rolling(window=40).mean()
        hold_period = 288 # 24 hours
        
        # Binary signals
        df_5m['long_sig'] = (df_5m['close'] > df_5m['sma_40']).astype(int)
        df_5m['short_sig'] = (df_5m['close'] < df_5m['sma_40']).astype(int)
        
        # Independent additive positions
        df_5m['pos_long'] = df_5m['long_sig'].rolling(window=hold_period).sum()
        df_5m['pos_short'] = df_5m['short_sig'].rolling(window=hold_period).sum()
        df_5m['net_exposure'] = df_5m['pos_long'] - df_5m['pos_short']

        # Returns calculation
        future_close = df_5m['close'].shift(-hold_period)
        pct_change = (future_close - df_5m['close']) / df_5m['close']
        
        # PnL contribution from each stream
        df_5m['long_ret'] = df_5m['long_sig'] * pct_change
        df_5m['short_ret'] = df_5m['short_sig'] * (-pct_change)
        df_5m['total_ret'] = df_5m['long_ret'] + df_5m['short_ret']
        
        results = df_5m.dropna(subset=['total_ret', 'sma_40']).copy()
        results['equity'] = results['total_ret'].cumsum()
        results['cum_long'] = results['long_ret'].cumsum()
        results['cum_short'] = results['short_ret'].cumsum()
        
        delayed_print("[FINALIZE] Calculating Realistic Daily Sharpe...")
        
        # --- SHARPE CALCULATION ---
        daily_equity = results['equity'].resample('D').last().dropna()
        daily_diff = daily_equity.diff().dropna()
        if len(daily_diff) > 1:
            sharpe = (daily_diff.mean() / daily_diff.std()) * np.sqrt(365)
        else:
            sharpe = 0
            
        # Drawdown Fix (expressed as a positive percentage)
        highmark = results['equity'].cummax()
        drawdown_series = (highmark - results['equity'])
        max_dd_val = drawdown_series.max() * 100

        metrics = {
            "trades": len(results),
            "sharpe": sharpe,
            "total_return": results['equity'].iloc[-1] * 100,
            "long_return": results['cum_long'].iloc[-1] * 100,
            "short_return": results['cum_short'].iloc[-1] * 100,
            "max_dd": max_dd_val,
            "max_long_units": results['pos_long'].max(),
            "max_short_units": results['pos_short'].max(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Equity Curve with independent lines
        ax1.plot(results.index, results['equity']*100, color='#1a73e8', lw=2, label='Total Strategy')
        ax1.plot(results.index, results['cum_long']*100, color='#1e8e3e', lw=1, alpha=0.5, label='Long Only')
        ax1.plot(results.index, results['cum_short']*100, color='#d93025', lw=1, alpha=0.5, label='Short Only')
        ax1.set_title('Cumulative Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.2); ax1.legend(loc='upper left', fontsize='small')
        
        # Independent Position Sizes
        ax2.plot(results.index, results['pos_long'], color='#1e8e3e', lw=1, label='Long Units')
        ax2.plot(results.index, -results['pos_short'], color='#d93025', lw=1, label='Short Units')
        ax2.fill_between(results.index, results['net_exposure'], color='#1a73e8', alpha=0.2, label='Net Exposure')
        ax2.set_title('Independent Position Exposure (Additive)', fontsize=12)
        ax2.grid(True, alpha=0.2); ax2.legend(loc='lower left', fontsize='x-small')
        
        plt.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format='png', dpi=100); plt.close()
        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return metrics, plot_b64
    except Exception as e:
        delayed_print(f"[ERROR] {e}"); return None, None

def start_server(metrics, plot_data):
    PORT = 8080
    html = f"""
    <!DOCTYPE html><html><head><title>Backtest Dashboard</title><style>
    body {{ font-family: -apple-system, sans-serif; background: #f8f9fa; padding: 20px; color: #202124; }}
    .card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 1100px; margin: auto; }}
    h1 {{ font-size: 1.3rem; color: #1a73e8; border-bottom: 1px solid #eee; padding-bottom: 12px; margin-top:0; }}
    .grid {{ display: grid; grid-template-columns: 300px 1fr; gap: 25px; }}
    .m-row {{ display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f1f3f4; font-size: 0.9rem; }}
    .m-val {{ font-weight: 700; }}
    .highlight {{ color: #1a73e8; font-size: 1.1rem; }}
    img {{ width: 100%; border-radius: 4px; }}
    .note {{ font-size: 0.75rem; color: #70757a; margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px; line-height: 1.4; }}
    </style></head><body><div class="card">
    <h1>Strategy Dashboard: Independent Dual Streams</h1>
    <div class="grid"><div class="side">
    <div class="m-row"><span>Realistic Sharpe</span><span class="m-val highlight">{metrics['sharpe']:.2f}</span></div>
    <div class="m-row"><span>Total Return</span><span class="m-val" style="color:#1e8e3e">{metrics['total_return']:.1f}%</span></div>
    <div class="m-row"><span>Max Drawdown</span><span class="m-val" style="color:#d93025">{metrics['max_dd']:.1f}%</span></div>
    <div class="m-row"><span>Max Long Units</span><span class="m-val" style="color:#1e8e3e">{metrics['max_long_units']:.0f}</span></div>
    <div class="m-row"><span>Max Short Units</span><span class="m-val" style="color:#d93025">{metrics['max_short_units']:.0f}</span></div>
    <div class="m-row"><span>Long Contr.</span><span class="m-val">{metrics['long_return']:.1f}%</span></div>
    <div class="m-row"><span>Short Contr.</span><span class="m-val">{metrics['short_return']:.1f}%</span></div>
    <div class="note"><b>Dual Independent Positions:</b> Long and Short streams operate as additive 24-hour holds triggered every 5 minutes. Drawdown is magnitude from peak.</div>
    </div><div class="main"><img src="data:image/png;base64,{plot_data}"></div></div></div></body></html>
    """
    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-type", "text/html"); self.end_headers()
            self.wfile.write(html.encode())
    with socketserver.TCPServer(("", PORT), H) as httpd:
        delayed_print(f"[INFO] Server live at http://localhost:{PORT}"); httpd.serve_forever()

if __name__ == "__main__":
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'; FILENAME = 'ohlcv.csv'
    download_data(FILE_ID, FILENAME)
    res, plot = run_strategy(FILENAME)
    if res: start_server(res, plot)