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

        hold_period = 288 
        delayed_print(f"[PROCESS] Applying strategy logic (Hold period: {hold_period} bars)...")
        
        df_5m['signal'] = 0
        df_5m.loc[df_5m['close'] > df_5m['sma_40'], 'signal'] = 1
        df_5m.loc[df_5m['close'] < df_5m['sma_40'], 'signal'] = -1

        delayed_print("[PROCESS] Calculating trade returns (24-hour forward window)...")
        future_close = df_5m['close'].shift(-hold_period)
        df_5m['trade_return'] = (future_close - df_5m['close']) / df_5m['close']
        df_5m['strategy_return'] = df_5m['signal'] * df_5m['trade_return']
        
        results = df_5m.dropna(subset=['strategy_return', 'sma_40']).copy()
        
        delayed_print("[FINALIZE] Aggregating performance metrics...")
        
        # Calculate Equity Curve
        results['cumulative_return'] = results['strategy_return'].cumsum()
        
        # Additional Metrics
        avg_ret = results['strategy_return'].mean()
        std_ret = results['strategy_return'].std()
        ret_std_ratio = avg_ret / std_ret if std_ret != 0 else 0
        
        gross_profit = results[results['strategy_return'] > 0]['strategy_return'].sum()
        gross_loss = abs(results[results['strategy_return'] < 0]['strategy_return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Max Drawdown
        running_max = results['cumulative_return'].cummax()
        drawdown = results['cumulative_return'] - running_max
        max_drawdown = drawdown.min()

        metrics = {
            "total_trades": len(results),
            "wins": len(results[results['strategy_return'] > 0]),
            "losses": len(results[results['strategy_return'] < 0]),
            "win_rate": (len(results[results['strategy_return'] > 0]) / len(results)) * 100,
            "avg_return": avg_ret * 100,
            "std_return": std_ret * 100,
            "ret_std_ratio": ret_std_ratio,
            "total_return": results['strategy_return'].sum() * 100,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Generate Plot
        delayed_print("[PROCESS] Generating performance graph...")
        plt.figure(figsize=(10, 5))
        plt.plot(results.index, results['cumulative_return'] * 100, color='#2980b9', linewidth=1.5)
        plt.title('Cumulative Strategy Return (%)', fontsize=14, pad=15)
        plt.ylabel('Return %')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Detailed Console Output
        delayed_print("--------------------------------------------------")
        delayed_print(f"STRATEGY REPORT: SMA 40 Cross + 24H Hold")
        delayed_print("--------------------------------------------------")
        delayed_print(f"Total Observations:   {metrics['total_trades']}")
        delayed_print(f"Win Rate:             {metrics['win_rate']:.2f}%")
        delayed_print(f"Return / STD Ratio:   {metrics['ret_std_ratio']:.4f}")
        delayed_print(f"Total Return (Sum):   {metrics['total_return']:.2f}%")
        delayed_print(f"Profit Factor:        {metrics['profit_factor']:.2f}")
        delayed_print(f"Max Drawdown:         {metrics['max_drawdown']:.2f}%")
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
        <title>Backtest Results</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; color: #1c1e21; margin: 0; padding: 40px; display: flex; flex-direction: column; align-items: center; }}
            .container {{ max-width: 1000px; width: 100%; }}
            .grid {{ display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }}
            .card {{ background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }}
            h1 {{ font-size: 1.5rem; margin-top: 0; color: #1a73e8; border-bottom: 1px solid #eee; padding-bottom: 12px; }}
            h2 {{ font-size: 1.1rem; color: #5f6368; margin-bottom: 20px; }}
            .stat-row {{ display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f8f9fa; }}
            .label {{ color: #70757a; font-weight: 500; }}
            .value {{ font-weight: 700; color: #202124; }}
            .plot-container {{ text-align: center; }}
            .plot-container img {{ max-width: 100%; border-radius: 8px; }}
            .footer {{ font-size: 0.85rem; color: #9aa0a6; text-align: center; margin-top: 30px; }}
            .positive {{ color: #1e8e3e; }}
            .negative {{ color: #d93025; }}
            .highlight {{ color: #1a73e8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>SMA Crossover Backtest Results</h1>
                <div class="grid">
                    <div class="stats-panel">
                        <h2>Key Metrics</h2>
                        <div class="stat-row"><span class="label">Total Trades</span><span class="value">{metrics['total_trades']}</span></div>
                        <div class="stat-row"><span class="label">Win Rate</span><span class="value">{metrics['win_rate']:.2f}%</span></div>
                        <div class="stat-row"><span class="label">Return / STD</span><span class="value highlight">{metrics['ret_std_ratio']:.4f}</span></div>
                        <div class="stat-row"><span class="label">Profit Factor</span><span class="value">{metrics['profit_factor']:.2f}</span></div>
                        <div class="stat-row"><span class="label">Avg Return</span><span class="value { 'positive' if metrics['avg_return'] > 0 else 'negative' }">{metrics['avg_return']:.4f}%</span></div>
                        <div class="stat-row"><span class="label">Total Return</span><span class="value { 'positive' if metrics['total_return'] > 0 else 'negative' }">{metrics['total_return']:.2f}%</span></div>
                        <div class="stat-row"><span class="label">Max Drawdown</span><span class="value negative">{metrics['max_drawdown']:.2f}%</span></div>
                    </div>
                    <div class="plot-container">
                        <h2>Equity Curve</h2>
                        <img src="data:image/png;base64,{plot_data}" alt="Strategy Plot">
                    </div>
                </div>
            </div>
            <div class="footer">Backtest completed at {metrics['timestamp']} | Hold Period: 24h | Intervals: 5m</div>
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
        delayed_print("[INFO] Press Ctrl+C to stop the script and server.")
        httpd.serve_forever()

if __name__ == "__main__":
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    FILENAME = 'downloaded_ohlcv.csv'

    download_data(FILE_ID, FILENAME)
    results, plot_b64 = run_strategy(FILENAME)
    
    if results:
        start_server(results, plot_b64)