import gdown
import pandas as pd
import numpy as np
import os
import time
import sys
import http.server
import socketserver
import threading
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
        
        results = df_5m.dropna(subset=['strategy_return', 'sma_40'])
        
        delayed_print("[FINALIZE] Aggregating performance metrics...")
        
        metrics = {
            "total_trades": len(results),
            "wins": len(results[results['strategy_return'] > 0]),
            "losses": len(results[results['strategy_return'] < 0]),
            "win_rate": (len(results[results['strategy_return'] > 0]) / len(results)) * 100,
            "avg_return": results['strategy_return'].mean() * 100,
            "total_return": results['strategy_return'].sum() * 100,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Detailed Console Output
        delayed_print("--------------------------------------------------")
        delayed_print(f"STRATEGY REPORT: SMA 40 Cross + 24H Hold")
        delayed_print("--------------------------------------------------")
        delayed_print(f"Total Observations:   {metrics['total_trades']}")
        delayed_print(f"Win Rate:             {metrics['win_rate']:.2f}%")
        delayed_print(f"Total Return (Sum):   {metrics['total_return']:.2f}%")
        delayed_print("--------------------------------------------------")
        
        return metrics

    except Exception as e:
        delayed_print(f"[ERROR] An error occurred: {e}")
        return None

def start_server(metrics):
    PORT = 8080
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Results</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f7f6; color: #333; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
            .card {{ background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 400px; }}
            h1 {{ font-size: 1.5rem; border-bottom: 2px solid #eee; padding-bottom: 1rem; margin-top: 0; color: #2c3e50; }}
            .stat-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #fafafa; }}
            .label {{ color: #7f8c8d; font-weight: 500; }}
            .value {{ font-weight: 700; color: #2980b9; }}
            .footer {{ font-size: 0.8rem; color: #bdc3c7; margin-top: 1.5rem; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Strategy Dashboard</h1>
            <div class="stat-row"><span class="label">Algorithm</span><span class="value">SMA 40 Cross</span></div>
            <div class="stat-row"><span class="label">Hold Period</span><span class="value">24 Hours</span></div>
            <div class="stat-row"><span class="label">Total Trades</span><span class="value">{metrics['total_trades']}</span></div>
            <div class="stat-row"><span class="label">Win Rate</span><span class="value">{metrics['win_rate']:.2f}%</span></div>
            <div class="stat-row"><span class="label">Avg Return</span><span class="value">{metrics['avg_return']:.4f}%</span></div>
            <div class="stat-row"><span class="label">Total Return</span><span class="value">{metrics['total_return']:.2f}%</span></div>
            <div class="footer">Backtest completed at {metrics['timestamp']}</div>
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
    results = run_strategy(FILENAME)
    
    if results:
        start_server(results)