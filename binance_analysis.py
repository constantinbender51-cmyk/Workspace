import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import http.server
import socketserver
import time
import os
import glob
import threading
import sys
# --- FIXED IMPORT BELOW ---
from datetime import datetime, timedelta, timezone
from collections import Counter

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTH = 5

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.1
GRID_STEPS = 100

# Ensemble Threshold
ENSEMBLE_ACC_THRESHOLD = 70.0

# Data Source
GITHUB_CSV_URL = "https://github.com/constantinbender51-cmyk/Models/raw/refs/heads/main/ohlc/ETHUSDT.csv"
TARGET_TIMEFRAME = '30min' 

# Global storage
GLOBAL_REF_PRICE = None
GLOBAL_HIGH_ACC_CONFIGS = []
LATEST_LIVE_RESULT = "Initializing..."

def fetch_and_process_github_data():
    """
    Downloads CSV, handles 2-column format (Timestamp, Close), and resamples to 30m.
    """
    print(f"Downloading data from GitHub: {GITHUB_CSV_URL}...")
    try:
        df = pd.read_csv(GITHUB_CSV_URL, header=None)
        
        if len(df.columns) == 2:
            print("Detected 2-column CSV (Timestamp, Close)")
            df.columns = ['timestamp', 'close']
        elif len(df.columns) >= 5:
            df = pd.read_csv(GITHUB_CSV_URL)
            df.columns = df.columns.str.strip().str.lower()
        else:
            print(f"Unknown CSV format. Defaulting to first 2 columns.")
            df = df.iloc[:, :2]
            df.columns = ['timestamp', 'close']

        first_val = df['timestamp'].iloc[0]
        try:
            if isinstance(first_val, (int, float, np.number)):
                unit = 'ms' if first_val > 10000000000 else 's'
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        except Exception as e:
            print(f"Timestamp parsing error: {e}")
            return pd.DataFrame()

        df = df.sort_values('timestamp').set_index('timestamp')
        
        print(f"Resampling to {TARGET_TIMEFRAME}...")
        agg_dict = {'close': 'last'}
        df_resampled = df.resample(TARGET_TIMEFRAME).agg(agg_dict)
        df_resampled = df_resampled.dropna().reset_index()
        
        print(f"Processing complete. Rows: {len(df)} -> {len(df_resampled)}")
        return df_resampled

    except Exception as e:
        print(f"Error processing GitHub data: {e}")
        return pd.DataFrame()

def get_grid_indices(df, step_size):
    close_array = df['close'].to_numpy()
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    abs_price_log = np.log(abs_price_raw)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    return grid_indices

def train_and_evaluate(df, step_size):
    grid_indices = get_grid_indices(df, step_size)
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_seq = grid_indices[:idx_80]
    val_seq = grid_indices[idx_80:idx_90]
    
    patterns = {}
    for i in range(len(train_seq) - SEQ_LENGTH):
        seq = tuple(train_seq[i : i + SEQ_LENGTH])
        target = train_seq[i + SEQ_LENGTH]
        if seq not in patterns: patterns[seq] = []
        patterns[seq].append(target)
        
    move_correct = 0
    move_total_valid = 0
    
    for i in range(len(val_seq) - SEQ_LENGTH):
        current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
        current_level = current_seq[-1]
        actual_next_level = val_seq[i + SEQ_LENGTH]
        
        if current_seq in patterns:
            history = patterns[current_seq]
            predicted_level = Counter(history).most_common(1)[0][0]
            pred_diff = predicted_level - current_level
            actual_diff = actual_next_level - current_level
            
            if pred_diff != 0 and actual_diff != 0:
                move_total_valid += 1
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    move_correct += 1
                    
    accuracy = (move_correct / move_total_valid * 100) if move_total_valid > 0 else 0.0
    
    return {
        'step_size': step_size,
        'accuracy': accuracy,
        'trade_count': move_total_valid,
        'patterns': patterns,
        'val_seq': val_seq 
    }

def run_combined_metric(high_acc_configs):
    if not high_acc_configs: return 0.0, 0
    ref_seq_len = len(high_acc_configs[0]['val_seq'])
    combined_correct = 0
    combined_total = 0
    
    for i in range(ref_seq_len - SEQ_LENGTH):
        up_votes = []
        down_votes = []
        for cfg in high_acc_configs:
            val_seq = cfg['val_seq']
            current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
            if current_seq in cfg['patterns']:
                history = cfg['patterns'][current_seq]
                predicted_level = Counter(history).most_common(1)[0][0]
                diff = predicted_level - current_seq[-1]
                if diff > 0: up_votes.append(cfg)
                elif diff < 0: down_votes.append(cfg)
        
        if len(up_votes) > 0 and len(down_votes) == 0:
            best_cfg = max(up_votes, key=lambda x: x['step_size'])
            chosen_val_seq = best_cfg['val_seq']
            curr_lvl = chosen_val_seq[i + SEQ_LENGTH - 1] 
            next_lvl = chosen_val_seq[i + SEQ_LENGTH]     
            actual_diff = next_lvl - curr_lvl
            if actual_diff != 0:
                combined_total += 1
                if actual_diff > 0: combined_correct += 1
                    
    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    return acc, combined_total

# --- LIVE PREDICTION (KRAKEN) ---

def kraken_live_loop(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count):
    global LATEST_LIVE_RESULT
    kraken = ccxt.kraken()
    symbol = 'ETH/USDT'
    timeframe = '30m'
    
    while True:
        try:
            # 1. SYNC WITH CLOCK (Wait for next 30m candle close)
            # Use pandas for easier time floor/ceil
            curr_ts = pd.Timestamp.now(tz='UTC')
            
            # Floor to current 30m start (e.g., 14:12 -> 14:00)
            current_candle_start = curr_ts.floor('30min') 
            
            # The next candle *close* is 30 mins after the current start (e.g., 14:00 -> 14:30)
            next_candle_close = current_candle_start + pd.Timedelta('30min')
            
            # Seconds to wait + 15 seconds buffer to ensure Kraken has the data
            sleep_seconds = (next_candle_close - curr_ts).total_seconds() + 15
            
            # Sanity check: if negative, we are slightly behind, just run now
            if sleep_seconds < 0: sleep_seconds = 0
            
            if sleep_seconds > 0:
                next_run_str = (curr_ts + pd.Timedelta(seconds=sleep_seconds)).strftime('%H:%M:%S UTC')
                print(f"[KRAKEN LIVE] Waiting {sleep_seconds:.0f}s for next candle close. (Next run: {next_run_str})")
                
                # Update HTML to show we are waiting
                LATEST_LIVE_RESULT = f"""
                <div style="border: 2px solid orange; padding: 15px; border-radius: 8px; background: #fff;">
                    <h2 style="color: orange; margin-top: 0;">WAITING FOR CANDLE CLOSE</h2>
                    <p><strong>Status:</strong> Sleeping until next 30m candle closes.</p>
                    <p><strong>Next Check:</strong> {next_run_str}</p>
                </div>
                """
                generate_html(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count)
                time.sleep(sleep_seconds)

            print(f"\n[KRAKEN LIVE] Fetching new completed candle...")
            
            # 2. FETCH DATA
            ohlcv = kraken.fetch_ohlcv(symbol, timeframe, limit=10)
            live_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms', utc=True)
            
            # 3. STRICT FILTERING
            # Recalculate time just to be sure
            now_check = pd.Timestamp.now(tz='UTC')
            current_forming_candle_start = now_check.floor('30min')
            
            # The last COMPLETED candle must have started 30 mins before the current forming one
            last_completed_candle_start = current_forming_candle_start - pd.Timedelta('30min')
            
            # Filter: Include only candles UP TO (and including) the last completed start time
            completed_df = live_df[live_df['timestamp'] <= last_completed_candle_start].copy()
            
            if len(completed_df) < SEQ_LENGTH:
                LATEST_LIVE_RESULT = f"Waiting for more data... (Have {len(completed_df)}/{SEQ_LENGTH} candles)"
            else:
                # 4. RUN PREDICTION
                target_df = completed_df.tail(SEQ_LENGTH).reset_index(drop=True)
                
                # Check timestamps
                last_candle_ts = target_df['timestamp'].iloc[-1]
                if last_candle_ts != last_completed_candle_start:
                    print(f"Warning: Expected last candle {last_completed_candle_start}, got {last_candle_ts}. Using latest available.")
                
                last_ts_str = last_candle_ts.strftime('%Y-%m-%d %H:%M UTC')
                current_price = target_df['close'].iloc[-1]
                
                up_votes = 0
                down_votes = 0
                
                if GLOBAL_REF_PRICE is None:
                    LATEST_LIVE_RESULT = "Error: Reference Price not set."
                else:
                    live_prices = target_df['close'].to_numpy()
                    qualified_configs = GLOBAL_HIGH_ACC_CONFIGS
                    
                    for cfg in qualified_configs:
                        step = cfg['step_size']
                        # Normalize relative to start
                        abs_price_log = np.log(live_prices / GLOBAL_REF_PRICE)
                        live_indices = np.floor(abs_price_log / step).astype(int)
                        
                        current_seq = tuple(live_indices)
                        
                        if current_seq in cfg['patterns']:
                            history = cfg['patterns'][current_seq]
                            predicted_level = Counter(history).most_common(1)[0][0]
                            diff = predicted_level - current_seq[-1]
                            if diff > 0: up_votes += 1
                            elif diff < 0: down_votes += 1
                    
                    decision = "NEUTRAL"
                    color = "gray"
                    if up_votes > 0 and down_votes == 0:
                        decision = "BULLISH (UP)"
                        color = "green"
                    elif down_votes > 0 and up_votes == 0:
                        decision = "BEARISH (DOWN)"
                        color = "red"
                    
                    LATEST_LIVE_RESULT = f"""
                    <div style="border: 2px solid {color}; padding: 15px; border-radius: 8px; background: #fff;">
                        <h2 style="color: {color}; margin-top: 0;">LIVE SIGNAL: {decision}</h2>
                        <p><strong>Source:</strong> Kraken Public API</p>
                        <p><strong>Last Closed Candle:</strong> {last_ts_str} @ ${current_price:.2f}</p>
                        <p><strong>Voters:</strong> {up_votes} UP / {down_votes} DOWN</p>
                    </div>
                    """
            generate_html(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count)
            
        except Exception as e:
            print(f"Kraken Error: {e}")
            LATEST_LIVE_RESULT = f"Kraken API Error: {e}"
            time.sleep(60)

def generate_html(image_filename, steps, accs, counts, cmb_acc, cmb_count):
    table_rows = ""
    for s, a, c in zip(steps, accs, counts):
        bg_style = "background-color: #e6fffa;" if a > ENSEMBLE_ACC_THRESHOLD else ""
        table_rows += f"<tr style='{bg_style}'><td>{s:.5f}</td><td>{a:.2f}%</td><td>{c}</td></tr>"

    html_content = f"""
    <html>
    <head>
        <title>Resampled Grid Search & Kraken Live</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f4; }}
            .container {{ background: white; padding: 20px; display: inline-block; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 900px; }}
            .metric-box {{ background: #2d3748; color: white; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            table {{ margin: 0 auto; border-collapse: collapse; width: 80%; }}
            th, td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Ensemble System (GitHub Data + Kraken Live)</h1>
            {LATEST_LIVE_RESULT}
            <div class="metric-box">
                <h2>Historical Performance (Resampled 30m)</h2>
                <p style="font-size: 24px; margin: 5px;">Accuracy: <strong>{cmb_acc:.2f}%</strong></p>
                <p>Total Trades: {cmb_count} | Threshold: >{ENSEMBLE_ACC_THRESHOLD}%</p>
            </div>
            <img src="{image_filename}" alt="Grid Search Plot">
            <h3>Individual Configs</h3>
            <table>
                <tr><th>Step Size</th><th>Accuracy</th><th>Trades</th></tr>
                {table_rows}
            </table>
        </div>
    </body>
    </html>
    """
    with open("index.html", "w") as f:
        f.write(html_content)

def run_system():
    global GLOBAL_REF_PRICE, GLOBAL_HIGH_ACC_CONFIGS
    
    df = fetch_and_process_github_data()
    if df.empty:
        print("Failed to load or process data.")
        return

    GLOBAL_REF_PRICE = df['close'].iloc[0]
    print(f"\nRef Price (Jan 2020): {GLOBAL_REF_PRICE}")

    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    results = []
    
    print(f"\n--- STARTING GRID SEARCH ({GRID_STEPS} Steps) ---")
    
    for step in step_sizes:
        res = train_and_evaluate(df, step)
        results.append(res)
        
    high_acc_configs = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
    GLOBAL_HIGH_ACC_CONFIGS = high_acc_configs
    
    cmb_acc, cmb_count = run_combined_metric(high_acc_configs)
    print(f"\nCombined Predictor Accuracy: {cmb_acc:.2f}% ({cmb_count} trades)")

    unique_id = int(time.time())
    filename = f"grid_search_{unique_id}.png"
    accuracies = [r['accuracy'] for r in results]
    trade_counts = [r['trade_count'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Log Step Size')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(step_sizes, accuracies, color='tab:blue')
    ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':')
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Trades', color='tab:orange')
    ax2.plot(step_sizes, trade_counts, color='tab:orange', linestyle='--')
    plt.savefig(filename)
    plt.close()
    
    live_thread = threading.Thread(
        target=kraken_live_loop, 
        args=(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count),
        daemon=True
    )
    live_thread.start()
    
    generate_html(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count)
    
    Handler = http.server.SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\n--- SERVER STARTED ---")
            print(f"View at: http://localhost:{PORT}")
            httpd.serve_forever()
    except OSError:
        print(f"Port {PORT} busy.")

if __name__ == "__main__":
    run_system()
