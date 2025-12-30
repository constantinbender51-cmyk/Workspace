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
    delayed_print("[PROCESS] Loading dataset into memory...")
    
    try:
        df = pd.read_csv(csv_file)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.columns = [c.lower() for c in df.columns]
        
        delayed_print(f"[STATS] Total 1-minute records: {len(df)}")
        
        delayed_print("[PROCESS] Resampling data to 5-minute intervals...")
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df_5m = df.resample('5min').agg({k: v for k, v in ohlc_dict.items() if k in df.columns}).dropna()
        
        delayed_print("[PROCESS] Calculating 40-period SMA...")
        df_5m['sma_40'] = df_5m['close'].rolling(window=40).mean()

        hold_period = 288 
        
        # 1. Generate individual trade signals
        df_5m['long_signal'] = (df_5m['close'] > df_5m['sma_40']).astype(int)
        df_5m['short_signal'] = (df_5m['close'] < df_5m['sma_40']).astype(int)

        # 2. Calculate additive position sizes
        df_5m['pos_long'] = df_5m['long_signal'].rolling(window=hold_period).sum()
        df_5m['pos_short'] = df_5m['short_signal'].rolling(window=hold_period).sum()
        df_5m['net_exposure'] = df_5m['pos_long'] - df_5m['pos_short']

        # 3. Calculate Returns
        future_close = df_5m['close'].shift(-hold_period)
        pct_change = (future_close - df_5m['close']) / df_5m['close']
        
        df_5m['long_return'] = df_5m['long_signal'] * pct_change
        df_5m['short_return'] = df_5m['short_signal'] * (-pct_change)
        df_5m['strategy_return'] = df_5m['long_return'] + df_5m['short_return']
        
        results = df_5m.dropna(subset=['strategy_return', 'sma_40', 'pos_long', 'pos_short']).copy()
        
        delayed_print("[FINALIZE] Calculating Annualized Sharpe Ratio...")
        
        # Calculate Risk Metrics
        avg_ret = results['strategy_return'].mean()
        std_ret = results['strategy_return'].std()
        
        # Annualization factor for 5-minute bars (24/7 market)
        # 12 bars/hr * 24 hr/day * 365 days/yr = 105,120
        ann_factor = np.sqrt(105120)
        sharpe_per_bar = avg_ret / std_ret if std_ret != 0 else 0
        annualized_sharpe = sharpe_per_bar * ann_factor
        
        # Equity curves
        results['cumulative_return'] = results['strategy_return'].cumsum()
        results['cum_ret_long'] = results['long_return'].cumsum()
        results['cum_ret_short'] = results['short_return'].cumsum()
        
        gross_profit = results[results['strategy_return'] > 0]['strategy_return'].sum()
        gross_loss = abs(results[results['strategy_return'] < 0]['strategy_return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        running_max = results['cumulative_return'].cummax()
        drawdown = results['cumulative_return'] - running_max
        max_drawdown = drawdown.min()

        metrics = {
            "total_trades": len(results),
            "win_rate": (len(results[results['strategy_return'] > 0]) / len(results)) * 100,
            "annualized_sharpe": annualized_sharpe,
            "sharpe_per_bar": sharpe_per_bar,
            "total_return": results['strategy_return'].sum() * 100,
            "long_return": results['long_return'].sum() * 100,
            "short_return": results['short_return'].sum() * 100,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "max_pos_long": results['pos_long'].max(),
            "max_pos_short": results['pos_short'].max(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(results.index, results['cumulative_return'] * 100, color='#1a73e8', linewidth=2, label='Total Strategy')
        ax1.plot(results.index, results['cum_ret_long'] * 100, color='#1e8e3e', linewidth=1, alpha=0.5, label='Long Only')
        ax1.plot(results.index, results['cum_ret_short'] * 100, color='#d93025', linewidth=1, alpha=0.5, label='Short Only')
        ax1.set_title('Cumulative Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize='small')
        
        ax2.plot(results.index, results['pos_long'], color='#1e8e3e', linewidth=1, alpha=0.7)
        ax2.plot(results.index, -results['pos_short'], color='#d93025', linewidth=1, alpha=0.7)
        ax2.fill_between(results.index, results['net_exposure'], color='#1a73e8', alpha=0.2, label='Net Exposure')
        ax2.set_title('Position Exposure (Units)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return metrics, plot_base64

    except Exception as e:
        delayed_print(f"[ERROR] {e}")
        return None, None

def start_server(metrics, plot_data):
    PORT = 8080
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Dashboard</title>
        <style>
            body {{ font-family: -apple-system, system-ui, sans-serif; background-color: #f8f9fa; color: #202124; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .card {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
            .grid {{ display: grid; grid-template-columns: 320px 1fr; gap: 30px; }}
            .metric-box {{ border-bottom: 1px solid #f1f3f4; padding: 12px 0; display: flex; justify-content: space-between; }}
            .metric-label {{ color: #5f6368; font-size: 0.85rem; }}
            .metric-value {{ font-weight: 700; }}
            .sharpe {{ color: #1a73e8; font-size: 1.2rem; }}
            .chart-area img {{ width: 100%; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Strategy Performance Dashboard</h1>
                <div class="grid">
                    <div class="sidebar">
                        <div class="metric-box"><span class="metric-label">Annualized Sharpe</span><span class="metric-value sharpe">{metrics['annualized_sharpe']:.2f}</span></div>
                        <div class="metric-box"><span class="metric-label">Per-Bar Sharpe</span><span class="metric-value">{metrics['sharpe_per_bar']:.4f}</span></div>
                        <div class="metric-box"><span class="metric-label">Win Rate</span><span class="metric-value">{metrics['win_rate']:.1f}%</span></div>
                        <div class="metric-box"><span class="metric-label">Profit Factor</span><span class="metric-value">{metrics['profit_factor']:.2f}</span></div>
                        <div class="metric-box"><span class="metric-label">Max Drawdown</span><span class="metric-value" style="color:#d93025">{metrics['max_drawdown']:.2f}%</span></div>
                        <div class="metric-box"><span class="metric-label">Total Return</span><span class="metric-value" style="color:#1e8e3e">{metrics['total_return']:.2f}%</span></div>
                    </div>
                    <div class="chart-area">
                        <img src="data:image/png;base64,{plot_data}">
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-type", "text/html"); self.end_headers()
            self.wfile.write(html_content.encode())
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        delayed_print(f"[INFO] Server live at http://localhost:{PORT}"); httpd.serve_forever()

if __name__ == "__main__":
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    FILENAME = 'downloaded_ohlcv.csv'
    download_data(FILE_ID, FILENAME)
    results, plot_b64 = run_strategy(FILENAME)
    if results: start_server(results, plot_b64)