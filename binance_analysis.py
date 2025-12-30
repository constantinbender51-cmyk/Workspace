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
        
        delayed_print("[PROCESS] Calculating 40 SMA and signals...")
        df_5m['sma_40'] = df_5m['close'].rolling(window=40).mean()
        hold_period = 288 # 24 hours
        
        # Signals
        df_5m['long_signal'] = (df_5m['close'] > df_5m['sma_40']).astype(int)
        df_5m['short_signal'] = (df_5m['close'] < df_5m['sma_40']).astype(int)
        
        # Additive positions
        df_5m['pos_long'] = df_5m['long_signal'].rolling(window=hold_period).sum()
        df_5m['pos_short'] = df_5m['short_signal'].rolling(window=hold_period).sum()
        df_5m['net_exposure'] = df_5m['pos_long'] - df_5m['pos_short']

        # Returns calculation
        future_close = df_5m['close'].shift(-hold_period)
        pct_change = (future_close - df_5m['close']) / df_5m['close']
        df_5m['long_ret'] = df_5m['long_signal'] * pct_change
        df_5m['short_ret'] = df_5m['short_signal'] * (-pct_change)
        df_5m['total_ret'] = df_5m['long_ret'] + df_5m['short_ret']
        
        results = df_5m.dropna(subset=['total_ret', 'sma_40']).copy()
        results['equity'] = results['total_ret'].cumsum()
        
        delayed_print("[FINALIZE] Calculating Realistic (Daily) Sharpe Ratio...")
        
        # --- THE CORRECT SHARPE CALCULATION ---
        # 1. Resample equity to Daily close
        daily_equity = results['equity'].resample('D').last().dropna()
        # 2. Calculate daily returns from the equity curve
        daily_diff = daily_equity.diff().dropna()
        # 3. Calculate Sharpe based on Daily standard deviation
        if len(daily_diff) > 1:
            daily_mean = daily_diff.mean()
            daily_std = daily_diff.std()
            # Annualize by multiplying by sqrt(365)
            sharpe = (daily_mean / daily_std) * np.sqrt(365) if daily_std != 0 else 0
        else:
            sharpe = 0
            
        # Profit Factor & Drawdown
        gross_p = results[results['total_ret'] > 0]['total_ret'].sum()
        gross_l = abs(results[results['total_ret'] < 0]['total_ret'].sum())
        pf = gross_p / gross_l if gross_l != 0 else 0
        
        highmark = results['equity'].cummax()
        dd = results['equity'] - highmark
        max_dd = dd.min()

        metrics = {
            "trades": len(results),
            "sharpe": sharpe,
            "total_return": results['equity'].iloc[-1] * 100,
            "pf": pf,
            "max_dd": max_dd * 100,
            "max_exp": results['net_exposure'].abs().max(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(results.index, results['equity']*100, color='#1a73e8', lw=1.5)
        ax1.set_title('Cumulative Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.2); ax1.set_ylabel('% Return')
        
        ax2.fill_between(results.index, results['net_exposure'], color='#1a73e8', alpha=0.2)
        ax2.plot(results.index, results['net_exposure'], color='#1a73e8', lw=1)
        ax2.set_title('Net Position Exposure (Additive Units)', fontsize=12)
        ax2.grid(True, alpha=0.2); ax2.set_ylabel('Units')
        
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
    .card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 1000px; margin: auto; }}
    h1 {{ font-size: 1.2rem; color: #1a73e8; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
    .grid {{ display: grid; grid-template-columns: 280px 1fr; gap: 20px; }}
    .m-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #f1f3f4; font-size: 0.9rem; }}
    .m-val {{ font-weight: 700; }}
    .sharpe-val {{ color: #1a73e8; font-size: 1.1rem; }}
    img {{ width: 100%; border-radius: 4px; }}
    .note {{ font-size: 0.75rem; color: #70757a; margin-top: 15px; font-style: italic; }}
    </style></head><body><div class="card">
    <h1>SMA Strategy: Corrected Realistic Sharpe</h1>
    <div class="grid"><div class="side">
    <div class="m-row"><span>Realistic Sharpe</span><span class="m-val sharpe-val">{metrics['sharpe']:.2f}</span></div>
    <div class="m-row"><span>Total Return</span><span class="m-val" style="color:#1e8e3e">{metrics['total_return']:.1f}%</span></div>
    <div class="m-row"><span>Profit Factor</span><span class="m-val">{metrics['pf']:.2f}</span></div>
    <div class="m-row"><span>Max Drawdown</span><span class="m-val" style="color:#d93025">{metrics['max_dd']:.1f}%</span></div>
    <div class="m-row"><span>Max Net Units</span><span class="m-val">{metrics['max_exp']:.0f}</span></div>
    <div class="note">Sharpe is calculated on <b>Daily Equity Delta</b> to remove 5-min autocorrelation bias. Annualized by sqrt(365).</div>
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