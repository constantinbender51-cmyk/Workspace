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
import sys  # Added for emergency exit
from datetime import datetime
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

# Global storage for live updates
GLOBAL_REF_PRICE = None
GLOBAL_HIGH_ACC_CONFIGS = []
LATEST_LIVE_RESULT = "Initializing..."

def get_exchange():
    """Returns a Binance instance with strict rate limiting enabled."""
    return ccxt.binance({
        'enableRateLimit': True,  # Critical: Internal throttling
        'options': {
            'defaultType': 'spot', 
        }
    })

def handle_binance_error(e):
    """Parses errors and kills the script immediately on Rate Limit/Ban."""
    err_str = str(e).lower()
    # Check for 429 (Too Many Requests) or 418 (IP Ban) or specific Binance code -1003
    if '429' in err_str or '418' in err_str or '-1003' in err_str or 'too many requests' in err_str:
        print(f"\n\n[CRITICAL] RATE LIMIT HIT OR IP BANNED.")
        print(f"Error details: {e}")
        print("STOPPING IMMEDIATELY TO PROTECT ACCOUNT.")
        os._exit(1) # Force kill everything including threads
    else:
        print(f"Network/Exchange Error: {e}")

def fetch_binance_data(symbol='ETH/USDT', timeframe='30m', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    print(f"Fetching {symbol} with RATE LIMIT PROTECTION...")
    exchange = get_exchange()
    
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    all_ohlcv = []
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000)}...", end='\r')
            
            if since >= end_ts: break
            
            # EXTRA SAFETY: Sleep 2 seconds between historical fetches to stay well under limits
            time.sleep(2) 
            
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.NetworkError) as e:
            handle_binance_error(e)
            # If not a critical ban, wait significantly before retrying (though handle_binance_error usually exits on 429)
            time.sleep(30)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

def get_grid_indices(df, step_size):
    """Converts price series to log-integer grid indices."""
    close_array = df['close'].to_numpy()
    
    # Absolute Price Index calculation
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # Log Rounding
    abs_price_log = np.log(abs_price_raw)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    return grid_indices

def train_and_evaluate(df, step_size):
    """
    Returns dictionary containing metrics and model data needed for the ensemble.
    """
    grid_indices = get_grid_indices(df, step_size)
    
    # Split
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_seq = grid_indices[:idx_80]
    val_seq = grid_indices[idx_80:idx_90]
    
    # Train (Build Patterns)
    patterns = {}
    for i in range(len(train_seq) - SEQ_LENGTH):
        seq = tuple(train_seq[i : i + SEQ_LENGTH])
        target = train_seq[i + SEQ_LENGTH]
        if seq not in patterns: patterns[seq] = []
        patterns[seq].append(target)
        
    # Evaluate on Validation Set
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
                same_direction = (pred_diff > 0 and actual_diff > 0) or \
                                 (pred_diff < 0 and actual_diff < 0)
                if same_direction:
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
    """
    Calculates the accuracy of the Ensemble Logic.
    """
    if not high_acc_configs:
        return 0.0, 0

    ref_seq_len = len(high_acc_configs[0]['val_seq'])
    
    combined_correct = 0
    combined_total = 0
    
    for i in range(ref_seq_len - SEQ_LENGTH):
        up_votes = []
        down_votes = []
        
        # Gather predictions from all qualified configs
        for cfg in high_acc_configs:
            val_seq = cfg['val_seq']
            current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
            
            if current_seq in cfg['patterns']:
                history = cfg['patterns'][current_seq]
                predicted_level = Counter(history).most_common(1)[0][0]
                current_level = current_seq[-1]
                diff = predicted_level - current_level
                
                if diff > 0:
                    up_votes.append(cfg)
                elif diff < 0:
                    down_votes.append(cfg)
        
        if len(up_votes) > 0 and len(down_votes) == 0:
            # Tie-breaker: Pick Maximum Step Size
            best_cfg = max(up_votes, key=lambda x: x['step_size'])
            
            chosen_val_seq = best_cfg['val_seq']
            
            curr_lvl = chosen_val_seq[i + SEQ_LENGTH - 1] 
            next_lvl = chosen_val_seq[i + SEQ_LENGTH]     
            
            actual_diff = next_lvl - curr_lvl
            
            if actual_diff != 0:
                combined_total += 1
                if actual_diff > 0: 
                    combined_correct += 1
                    
    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    return acc, combined_total

# --- LIVE PREDICTION LOGIC ---

def live_prediction_loop(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count):
    """Background thread that fetches fresh data and updates HTML every 30 mins."""
    global LATEST_LIVE_RESULT
    
    # Re-instantiate exchange inside thread with safety
    exchange = get_exchange()
    
    while True:
        try:
            print(f"\n[LIVE] Fetching fresh candles for prediction...")
            
            # 1. Fetch recent data with safety checks
            ohlcv = exchange.fetch_ohlcv('ETH/USDT', '30m', limit=20)
            
            # 2. Convert to DataFrame
            live_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms', utc=True)
            
            # 3. Filter: Exclude incomplete candle
            now = datetime.now(pd.Timestamp.now().tz).timestamp() * 1000
            last_ts = ohlcv[-1][0]
            
            if (now - last_ts) < (30 * 60 * 1000):
                completed_df = live_df.iloc[:-1].copy() 
            else:
                completed_df = live_df.copy() 
            
            if len(completed_df) < SEQ_LENGTH:
                LATEST_LIVE_RESULT = "Not enough data fetched."
            else:
                target_df = completed_df.tail(SEQ_LENGTH).reset_index(drop=True)
                last_ts_str = target_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')
                current_price = target_df['close'].iloc[-1]
                
                # --- ENSEMBLE VOTING ---
                up_votes = 0
                down_votes = 0
                
                if GLOBAL_REF_PRICE is None:
                    LATEST_LIVE_RESULT = "Error: Reference Price not set."
                else:
                    live_prices = target_df['close'].to_numpy()
                    qualified_configs = GLOBAL_HIGH_ACC_CONFIGS
                    
                    for cfg in qualified_configs:
                        step = cfg['step_size']
                        
                        # Normalize live prices relative to the 2020 Start Price
                        abs_price_log = np.log(live_prices / GLOBAL_REF_PRICE)
                        live_indices = np.floor(abs_price_log / step).astype(int)
                        
                        current_seq = tuple(live_indices)
                        
                        if current_seq in cfg['patterns']:
                            history = cfg['patterns'][current_seq]
                            predicted_level = Counter(history).most_common(1)[0][0]
                            current_level = current_seq[-1]
                            diff = predicted_level - current_level
                            
                            if diff > 0:
                                up_votes += 1
                            elif diff < 0:
                                down_votes += 1
                    
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
                        <p><strong>Last Candle Close:</strong> {last_ts_str} @ ${current_price:.2f}</p>
                        <p><strong>Voters:</strong> {up_votes} UP / {down_votes} DOWN (Total High-Acc Configs: {len(qualified_configs)})</p>
                        <small>Ref Price (2020): ${GLOBAL_REF_PRICE:.2f}</small>
                    </div>
                    """
            
            generate_html(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count)
            
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.NetworkError) as e:
            handle_binance_error(e) # This will exit the program
        except Exception as e:
            print(f"Live Loop Error: {e}")
            LATEST_LIVE_RESULT = f"Error in live loop: {e}"
        
        # Sleep 30 minutes
        time.sleep(30 * 60)

def generate_html(image_filename, steps, accs, counts, cmb_acc, cmb_count):
    """Generates the HTML file with current stats and live prediction."""
    table_rows = ""
    for s, a, c in zip(steps, accs, counts):
        bg_style = "background-color: #e6fffa;" if a > ENSEMBLE_ACC_THRESHOLD else ""
        table_rows += f"<tr style='{bg_style}'><td>{s:.5f}</td><td>{a:.2f}%</td><td>{c}</td></tr>"

    html_content = f"""
    <html>
    <head>
        <title>Grid Search & Live Prediction</title>
        <meta http-equiv="refresh" content="300"> <style>
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
            <h1>Crypto Ensemble System</h1>
            
            {LATEST_LIVE_RESULT}
            
            <div class="metric-box">
                <h2>Historical Combined Performance</h2>
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

def run_grid_search():
    global GLOBAL_REF_PRICE, GLOBAL_HIGH_ACC_CONFIGS
    
    df = fetch_binance_data()
    if df.empty: return

    # CAPTURE REFERENCE PRICE (First candle of 2020)
    GLOBAL_REF_PRICE = df['close'].iloc[0]
    print(f"\nTraining Reference Price (Jan 2020): {GLOBAL_REF_PRICE}")

    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    
    results = []
    print(f"\n\n--- STARTING GRID SEARCH ({GRID_STEPS} Steps) ---")
    
    for step in step_sizes:
        print(f"Testing Step Size: {step:.5f}...", end=" ")
        res = train_and_evaluate(df, step)
        results.append(res)
        print(f"Acc: {res['accuracy']:.2f}% | Trades: {res['trade_count']}")
        
    # --- COMBINED PREDICTOR ---
    print(f"\n--- CALCULATING COMBINED PREDICTOR ---")
    high_acc_configs = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
    GLOBAL_HIGH_ACC_CONFIGS = high_acc_configs # Store for live loop
    
    print(f"Qualifying Configs (> {ENSEMBLE_ACC_THRESHOLD}%): {len(high_acc_configs)}")
    
    cmb_acc, cmb_count = run_combined_metric(high_acc_configs)
    print(f"Combined Predictor Accuracy: {cmb_acc:.2f}%")
    print(f"Combined Predictor Trades:   {cmb_count}")

    # --- PLOTTING ---
    unique_id = int(time.time())
    filename = f"grid_search_{unique_id}.png"
    
    accuracies = [r['accuracy'] for r in results]
    trade_counts = [r['trade_count'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Log Step Size')
    ax1.set_ylabel('Directional Accuracy (%)', color='tab:blue')
    ax1.plot(step_sizes, accuracies, marker='o', color='tab:blue', label='Accuracy')
    ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':', label='Threshold (70%)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Number of Trades', color='tab:orange')
    ax2.plot(step_sizes, trade_counts, marker='x', linestyle='--', color='tab:orange', label='Trades')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title(f'Accuracy vs. Trade Volume\nCombined Predictor: {cmb_acc:.2f}% ({cmb_count} trades)')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"\nGrid search complete. Plot saved to {filename}")
    
    # Start Live Prediction Loop in background
    live_thread = threading.Thread(
        target=live_prediction_loop, 
        args=(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count),
        daemon=True
    )
    live_thread.start()
    
    # Generate initial HTML
    generate_html(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count)
    
    start_server()

def start_server():
    # Clean up old images
    current_imgs = glob.glob("grid_search_*.png")
    
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\n--- SERVER STARTED ---")
            print(f"View at: http://localhost:{PORT}")
            print(f"Press Ctrl+C to stop")
            httpd.serve_forever()
    except OSError:
        print(f"\nERROR: Port {PORT} is busy.")

if __name__ == "__main__":
    run_grid_search()
