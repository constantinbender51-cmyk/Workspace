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
        delayed_print(f"[STATS] Total 5-minute bars: {len(df_5m)}")

        delayed_print("[PROCESS] Calculating 40-period Simple Moving Average (SMA)...")
        df_5m['sma_40'] = df_5m['close'].rolling(window=40).mean()

        hold_period = 288 # 24 hours in 5-minute bars
        delayed_print(f"[PROCESS] Simulating independent Long and Short streams...")
        
        # 1. Generate individual trade signals
        df_5m['long_signal'] = (df_5m['close'] > df_5m['sma_40']).astype(int)
        df_5m['short_signal'] = (df_5m['close'] < df_5m['sma_40']).astype(int)

        # 2. Calculate additive position sizes for each stream
        # Each signal lasts for 288 bars.
        delayed_print("[PROCESS] Calculating independent additive exposures...")
        df_5m['pos_long'] = df_5m['long_signal'].rolling(window=hold_period).sum()
        df_5m['pos_short'] = df_5m['short_signal'].rolling(window=hold_period).sum()
        df_5m['net_exposure'] = df_5m['pos_long'] - df_5m['pos_short']

        # 3. Calculate Returns
        delayed_print("[PROCESS] Calculating performance for independent trades...")
        future_close = df_5m['close'].shift(-hold_period)
        pct_change = (future_close - df_5m['close']) / df_5m['close']
        
        # Long trades profit from price increase, Short trades profit from price decrease
        df_5m['long_return'] = df_5m['long_signal'] * pct_change
        df_5m['short_return'] = df_5m['short_signal'] * (-pct_change)
        
        # Total strategy return is the sum of both independent streams
        df_5m['strategy_return'] = df_5m['long_return'] + df_5m['short_return']
        
        results = df_5m.dropna(subset=['strategy_return', 'sma_40', 'pos_long', 'pos_short']).copy()
        
        delayed_print("[FINALIZE] Aggregating performance metrics...")
        
        # Calculate Equity Curves
        results['cum_ret_long'] = results['long_return'].cumsum()
        results['cum_ret_short'] = results['short_return'].cumsum()
        results['cumulative_return'] = results['strategy_return'].cumsum()
        
        # Metrics
        avg_ret = results['strategy_return'].mean()
        std_ret = results['strategy_return'].std()
        ret_std_ratio = avg_ret / std_ret if std_ret != 0 else 0
        
        gross_profit = results[results['strategy_return'] > 0]['strategy_return'].sum()
        gross_loss = abs(results[results['strategy_return'] < 0]['strategy_return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        running_max = results['cumulative_return'].cummax()
        drawdown = results['cumulative_return'] - running_max
        max_drawdown = drawdown.min()

        metrics = {
            "total_trades": len(results),
            "win_rate": (len(results[results['strategy_return'] > 0]) / len(results)) * 100,
            "avg_return": avg_ret * 100,
            "ret_std_ratio": ret_std_ratio,
            "total_return": results['strategy_return'].sum() * 100,
            "long_return": results['long_return'].sum() * 100,
            "short_return": results['short_return'].sum() * 100,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "max_pos_long": results['pos_long'].max(),
            "max_pos_short": results['pos_short'].max(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Generate Multi-Plot Graph
        delayed_print("[PROCESS] Generating performance and position graphs...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Equity Curve
        ax1.plot(results.index, results['cumulative_return'] * 100, color='#1a73e8', linewidth=2, label='Total Strategy')
        ax1.plot(results.index, results['cum_ret_long'] * 100, color='#1e8e3e', linewidth=1, alpha=0.6, label='Long Stream')
        ax1.plot(results.index, results['cum_ret_short'] * 100, color='#d93025', linewidth=1, alpha=0.6, label='Short Stream')
        ax1.set_title('Cumulative Strategy Return (%)', fontsize=12)
        ax1.set_ylabel('Return %')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper left', fontsize='small')
        
        # Plot 2: Independent Position Sizes
        ax2.plot(results.index, results['pos_long'], color='#1e8e3e', linewidth=1, label='Long Exposure')
        ax2.plot(results.index, -results['pos_short'], color='#d93025', linewidth=1, label='Short Exposure')
        ax2.fill_between(results.index, results['net_exposure'], color='#1a73e8', alpha=0.2, label='Net Exposure')
        ax2.set_title('Independent & Net Position Exposure (Units)', fontsize=12)
        ax2.set_ylabel('Units')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='lower left', fontsize='x-small')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        delayed_print("--------------------------------------------------")
        delayed_print(f"REPORT: Independent Additive Positions")
        delayed_print("--------------------------------------------------")
        delayed_print(f"Max Long Exposure:    {metrics['max_pos_long']:.0f} units")
        delayed_print(f"Max Short Exposure:   {metrics['max_pos_short']:.0f} units")
        delayed_print(f"Total Return (Sum):   {metrics['total_return']:.2f}%")
        delayed_print("--------------------------------------------------")
        
        return metrics, plot_base64

    except Exception as e:
        delayed_print(f"[ERROR] An error occurred: {e}")
        return None, None

def start_server(metrics, plot_data):
    PORT = 8080
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Dashboard - Independent Streams</title>
        <style>
            body {{ font-family: -apple-system, system-ui, sans-serif; background-color: #f8f9fa; color: #202124; margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .card {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
            h1 {{ font-size: 1.4rem; color: #1a73e8; margin-top: 0; margin-bottom: 25px; border-bottom: 1px solid #eee; padding-bottom: 15px; }}
            .dashboard-grid {{ display: grid; grid-template-columns: 320px 1fr; gap: 30px; }}
            .metric-box {{ border-bottom: 1px solid #f1f3f4; padding: 12px 0; display: flex; justify-content: space-between; }}
            .metric-label {{ color: #5f6368; font-size: 0.85rem; }}
            .metric-value {{ font-weight: 700; color: #202124; }}
            .chart-area {{ text-align: center; }}
            .chart-area img {{ width: 100%; border-radius: 4px; }}
            .info-panel {{ background: #fdf7e3; padding: 15px; border-radius: 6px; margin-top: 20px; font-size: 0.8rem; color: #856404; line-height: 1.4; border: 1px solid #ffeeba; }}
            .footer {{ text-align: center; margin-top: 30px; color: #70757a; font-size: 0.8rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Independent Long/Short Additive Strategy</h1>
                <div class="dashboard-grid">
                    <div class="metrics-sidebar">
                        <div class="metric-box"><span class="metric-label">Total Bars</span><span class="metric-value">{metrics['total_trades']}</span></div>
                        <div class="metric-box"><span class="metric-label">Return / STD</span><span class="metric-value" style="color:#1a73e8">{metrics['ret_std_ratio']:.4f}</span></div>
                        <div class="metric-box"><span class="metric-label">Profit Factor</span><span class="metric-value">{metrics['profit_factor']:.2f}</span></div>
                        <div class="metric-box"><span class="metric-label">Max Long Units</span><span class="metric-value" style="color:#1e8e3e">{metrics['max_pos_long']:.0f}</span></div>
                        <div class="metric-box"><span class="metric-label">Max Short Units</span><span class="metric-value" style="color:#d93025">{metrics['max_pos_short']:.0f}</span></div>
                        <div class="metric-box"><span class="metric-label">Long Return</span><span class="metric-value">{metrics['long_return']:.2f}%</span></div>
                        <div class="metric-box"><span class="metric-label">Short Return</span><span class="metric-value">{metrics['short_return']:.2f}%</span></div>
                        <div class="metric-box"><span class="metric-label">Total Return</span><span class="metric-value" style="color:#1a73e8; font-size:1.1rem">{metrics['total_return']:.2f}%</span></div>
                        
                        <div class="info-panel">
                            <strong>Logic:</strong> Long and Short streams are independent. 
                            If the price is above SMA, a 24h Long is opened. 
                            If the price is below SMA, a 24h Short is opened. 
                            These positions accumulate independently based on 5-minute entry signals.
                        </div>
                    </div>
                    <div class="chart-area">
                        <img src="data:image/png;base64,{plot_data}" alt="Equity and Position Curves">
                    </div>
                </div>
            </div>
            <div class="footer">Simulation generated at {metrics['timestamp']}</div>
        </div>
    </body>
    </html>
    """

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode())

    delayed_print(f"[SERVER] Starting web server on port {PORT}...")
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        delayed_print(f"[INFO] Server live at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    FILENAME = 'downloaded_ohlcv.csv'

    download_data(FILE_ID, FILENAME)
    results, plot_b64 = run_strategy(FILENAME)
    
    if results:
        start_server(results, plot_b64)