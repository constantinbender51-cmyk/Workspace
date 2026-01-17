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
import sys
import pickle
from datetime import datetime
from collections import Counter

# --- NEW IMPORTS FOR GITHUB ---
from dotenv import load_dotenv
from github import Github

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTH = 5
MODEL_FILENAME = "/app/data/eth.pkl"

# GitHub Configuration
GITHUB_REPO = "constantinbender51-cmyk/Models"
GITHUB_FOLDER = "model2x"
GITHUB_BRANCH = "main"

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.1
GRID_STEPS = 100

# Ensemble Threshold
ENSEMBLE_ACC_THRESHOLD = 70.0

def fetch_binance_data(symbol='ETH/USDT', timeframe='30m', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    print(f"Fetching {symbol}...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    since = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)
    all_ohlcv = []
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            if not ohlcv:
                print("No more data received.")
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=None)}...", end='\r')
            
            if since >= end_ts:
                break
                
            time.sleep(exchange.rateLimit / 1000 * 1.1)

        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
            print(f"\n\nCRITICAL: Rate Limit Exceeded or DDoS Protection Triggered.")
            sys.exit(1)
            
        except ccxt.NetworkError as e:
            print(f"\nNetwork Error: {e}. Retrying in 10s...")
            time.sleep(10)
            
        except Exception as e:
            print(f"\nUnexpected Error: {e}")
            break
            
    print(f"\nData fetch complete. Total rows: {len(all_ohlcv)}")
    
    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

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

def upload_to_github(filename):
    """
    Uploads the specified file to the configured GitHub Repo and Folder.
    """
    print(f"\n--- GITHUB UPLOAD ---")
    
    # 1. Load Environment Variables
    load_dotenv()
    pat = os.getenv("PAT")
    
    if not pat:
        print("[ERROR] 'PAT' not found in .env file. Skipping upload.")
        return

    try:
        # 2. Authenticate
        g = Github(pat)
        repo = g.get_repo(GITHUB_REPO)
        
        # 3. Read the binary file
        with open(filename, 'rb') as f:
            content = f.read()
        
        # 4. Define Target Path (e.g., model2x/grid_ensemble_model.pkl)
        target_path = f"{GITHUB_FOLDER}/{filename}"
        
        # 5. Check if file exists (Update) or Create new
        try:
            contents = repo.get_contents(target_path, ref=GITHUB_BRANCH)
            print(f"File exists. Updating {target_path}...")
            repo.update_file(contents.path, f"Update model {datetime.now()}", content, contents.sha, branch=GITHUB_BRANCH)
            print("[SUCCESS] File Updated on GitHub.")
        except Exception:
            print(f"File does not exist. Creating {target_path}...")
            repo.create_file(target_path, f"Create model {datetime.now()}", content, branch=GITHUB_BRANCH)
            print("[SUCCESS] File Created on GitHub.")
            
    except Exception as e:
        print(f"[ERROR] GitHub Upload Failed: {e}")

def save_ensemble_model(high_acc_configs, initial_reference_price):
    if not high_acc_configs:
        print("No high accuracy configs to save.")
        return

    lean_configs = []
    for cfg in high_acc_configs:
        lean_configs.append({
            'step_size': cfg['step_size'],
            'patterns': cfg['patterns'],
            'accuracy': cfg['accuracy']
        })

    model_payload = {
        'timestamp': datetime.now().isoformat(),
        'initial_price': initial_reference_price,
        'ensemble_configs': lean_configs,
        'threshold_used': ENSEMBLE_ACC_THRESHOLD
    }

    try:
        # Save Locally
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(model_payload, f)
        print(f"\n[SUCCESS] Model saved locally to '{MODEL_FILENAME}'")
        
        # Upload to GitHub
        upload_to_github(MODEL_FILENAME)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to save model: {e}")

def run_grid_search():
    df = fetch_binance_data()
    if df.empty: return
    
    initial_price = df['close'].iloc[0]
    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    results = []
    
    print(f"\n\n--- STARTING GRID SEARCH ({GRID_STEPS} Steps) ---")
    for step in step_sizes:
        print(f"Testing Step Size: {step:.5f}...", end=" ")
        res = train_and_evaluate(df, step)
        results.append(res)
        print(f"Acc: {res['accuracy']:.2f}% | Trades: {res['trade_count']}")
        
    print(f"\n--- CALCULATING COMBINED PREDICTOR ---")
    high_acc_configs = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
    
    cmb_acc, cmb_count = run_combined_metric(high_acc_configs)
    print(f"Combined Accuracy: {cmb_acc:.2f}% | Trades: {cmb_count}")

    save_ensemble_model(high_acc_configs, initial_price)

    # --- PLOTTING ---
    unique_id = int(time.time())
    filename = f"grid_search_{unique_id}.png"
    
    accuracies = [r['accuracy'] for r in results]
    trade_counts = [r['trade_count'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Log Step Size')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(step_sizes, accuracies, marker='o', markersize=3, color='tab:blue')
    ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Trades', color='tab:orange')
    ax2.plot(step_sizes, trade_counts, marker='x', markersize=3, linestyle='--', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title(f'Accuracy vs. Trade Volume\nCombined: {cmb_acc:.2f}% ({cmb_count} trades)')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    start_server(filename, step_sizes, accuracies, trade_counts, cmb_acc, cmb_count)

def start_server(image_filename, steps, accs, counts, cmb_acc, cmb_count):
    for f in glob.glob("grid_search_*.png"):
        if f != image_filename:
            try: os.remove(f)
            except: pass

    table_rows = ""
    for s, a, c in zip(steps, accs, counts):
        bg = "background-color: #e6fffa;" if a > ENSEMBLE_ACC_THRESHOLD else ""
        wt = "font-weight: bold;" if a > ENSEMBLE_ACC_THRESHOLD else ""
        table_rows += f"<tr style='{bg} {wt}'><td>{s:.5f}</td><td>{a:.2f}%</td><td>{c}</td></tr>"

    html_content = f"""
    <html>
    <head><title>Results</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f4; }}
        .container {{ background: white; padding: 30px; border-radius: 12px; margin: 0 auto; max-width: 900px; }}
        table {{ margin: 20px auto; border-collapse: collapse; width: 100%; }}
        th {{ background: #4a5568; color: white; padding: 12px; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
    </style>
    </head>
    <body>
        <div class="container">
            <h1>Grid Strategy Results</h1>
            <h2>Ensemble: {cmb_acc:.2f}% ({cmb_count} trades)</h2>
            <img src="{image_filename}" style="max-width:100%">
            <table><tr><th>Step</th><th>Acc</th><th>Trades</th></tr>{table_rows}</table>
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
        
    Handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\nServer at: http://localhost:{PORT}")
            httpd.serve_forever()
    except: pass

if __name__ == "__main__":
    try:
        run_grid_search()
    except KeyboardInterrupt:
        print("\nStopped.")
