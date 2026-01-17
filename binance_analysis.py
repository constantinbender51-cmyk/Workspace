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

# --- NEW IMPORTS FOR HUGGING FACE ---
from dotenv import load_dotenv
from huggingface_hub import HfApi

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTHS = [5, 6, 7, 8, 9, 10]  # Grid search over these lengths
MODEL_FILENAME = "/app/data/eth.pkl"

# Hugging Face Configuration
# IMPORTANT: Change this to your actual Hugging Face "username/repo_name"
HF_REPO_ID = "constantinbender51/Models" 
HF_FOLDER = "model2x"

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.05
GRID_STEPS = 20 

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

def train_and_evaluate(df, step_size, seq_len):
    """
    Trains a pattern matcher for a specific step_size AND sequence length.
    """
    grid_indices = get_grid_indices(df, step_size)
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    train_seq = grid_indices[:idx_80]
    val_seq = grid_indices[idx_80:idx_90]
    
    patterns = {}
    
    # Train
    for i in range(len(train_seq) - seq_len):
        seq = tuple(train_seq[i : i + seq_len])
        target = train_seq[i + seq_len]
        if seq not in patterns: patterns[seq] = []
        patterns[seq].append(target)
        
    move_correct = 0
    move_total_valid = 0
    
    # Validate
    for i in range(len(val_seq) - seq_len):
        current_seq = tuple(val_seq[i : i + seq_len])
        current_level = current_seq[-1]
        actual_next_level = val_seq[i + seq_len]
        
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
        'seq_len': seq_len,
        'accuracy': accuracy,
        'trade_count': move_total_valid,
        'patterns': patterns,
        'val_seq': val_seq
    }

def run_combined_metric(high_acc_configs):
    if not high_acc_configs: return 0.0, 0
    
    # All val_seqs have the same timestamps length, but different integer values.
    ref_len = len(high_acc_configs[0]['val_seq'])
    
    # We must start prediction at an index where ALL models have enough history.
    max_seq_len = max(cfg['seq_len'] for cfg in high_acc_configs)
    
    combined_correct = 0
    combined_total = 0
    
    # Iterate through time by TARGET index
    for target_idx in range(max_seq_len, ref_len):
        up_votes = []
        down_votes = []
        
        for cfg in high_acc_configs:
            s_len = cfg['seq_len']
            val_seq = cfg['val_seq']
            
            # Extract the sequence ending right before target_idx
            # Input: [target_idx - s_len ... target_idx - 1] -> Predicts: target_idx
            current_seq = tuple(val_seq[target_idx - s_len : target_idx])
            current_level = current_seq[-1] # The level at target_idx - 1
            
            if current_seq in cfg['patterns']:
                history = cfg['patterns'][current_seq]
                predicted_level = Counter(history).most_common(1)[0][0]
                diff = predicted_level - current_level
                
                if diff > 0: up_votes.append(cfg)
                elif diff < 0: down_votes.append(cfg)
        
        # Ensemble Voting Logic (Unanimity of active voters)
        if len(up_votes) > 0 and len(down_votes) == 0:
            # Check correctness using the "best" config's grid (highest step size typically filters noise)
            best_cfg = max(up_votes, key=lambda x: x['step_size'])
            chosen_val_seq = best_cfg['val_seq']
            
            curr_lvl = chosen_val_seq[target_idx - 1]
            next_lvl = chosen_val_seq[target_idx]
            actual_diff = next_lvl - curr_lvl
            
            if actual_diff != 0:
                combined_total += 1
                if actual_diff > 0: combined_correct += 1
        
        elif len(down_votes) > 0 and len(up_votes) == 0:
            # Check correctness for Short
            best_cfg = max(down_votes, key=lambda x: x['step_size'])
            chosen_val_seq = best_cfg['val_seq']
            
            curr_lvl = chosen_val_seq[target_idx - 1]
            next_lvl = chosen_val_seq[target_idx]
            actual_diff = next_lvl - curr_lvl
            
            if actual_diff != 0:
                combined_total += 1
                if actual_diff < 0: combined_correct += 1

    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    return acc, combined_total

def upload_to_huggingface(filename):
    print(f"\n--- HUGGING FACE UPLOAD ---")
    load_dotenv()
    # Looking for 'HFT' in .env
    token = os.getenv("HFT")
    
    if not token:
        print("[ERROR] 'HFT' token not found in .env file. Skipping upload.")
        return

    try:
        api = HfApi()
        # path_in_repo keeps the folder structure or just the filename if you prefer
        # Here we use the global GITHUB_FOLDER (now acting as HF_FOLDER)
        path_in_repo = f"{HF_FOLDER}/{os.path.basename(filename)}"
        
        print(f"Uploading {filename} to {HF_REPO_ID} at {path_in_repo}...")
        
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
            commit_message=f"Update model {datetime.now()}"
        )
        print("[SUCCESS] File Updated on Hugging Face.")
            
    except Exception as e:
        print(f"[ERROR] Hugging Face Upload Failed: {e}")

def save_ensemble_model(high_acc_configs, initial_reference_price):
    if not high_acc_configs:
        print("No high accuracy configs to save.")
        return

    lean_configs = []
    for cfg in high_acc_configs:
        lean_configs.append({
            'step_size': cfg['step_size'],
            'seq_len': cfg['seq_len'],
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
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(model_payload, f)
        print(f"\n[SUCCESS] Model saved locally to '{MODEL_FILENAME}'")
        upload_to_huggingface(MODEL_FILENAME)
    except Exception as e:
        print(f"\n[ERROR] Failed to save model: {e}")

def run_grid_search():
    df = fetch_binance_data()
    if df.empty: return
    
    initial_price = df['close'].iloc[0]
    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    results = []
    
    print(f"\n\n--- STARTING GRID SEARCH (Steps: {GRID_STEPS}, SeqLens: {SEQ_LENGTHS}) ---")
    
    # Nested Grid Search: Step Size x Sequence Length
    for s_len in SEQ_LENGTHS:
        print(f"\nTesting Sequence Length: {s_len}")
        for step in step_sizes:
            res = train_and_evaluate(df, step, s_len)
            results.append(res)
            # Minimal logging to avoid console spam
            if res['accuracy'] > 60:
                print(f"  > Step: {step:.4f} | Acc: {res['accuracy']:.2f}% | Trades: {res['trade_count']}")
        
    print(f"\n--- CALCULATING COMBINED PREDICTOR ---")
    high_acc_configs = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
    
    print(f"Found {len(high_acc_configs)} configurations above {ENSEMBLE_ACC_THRESHOLD}% accuracy.")
    cmb_acc, cmb_count = run_combined_metric(high_acc_configs)
    print(f"Combined Accuracy: {cmb_acc:.2f}% | Trades: {cmb_count}")

    save_ensemble_model(high_acc_configs, initial_price)

    # --- PLOTTING ---
    unique_id = int(time.time())
    filename = f"grid_search_{unique_id}.png"
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Step Size (Log Scale Proxy)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':', label='Threshold')

    # Color map for different sequence lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(SEQ_LENGTHS)))
    
    for idx, s_len in enumerate(SEQ_LENGTHS):
        # Filter results for this sequence length
        subset = [r for r in results if r['seq_len'] == s_len]
        if not subset: continue
        
        steps_sub = [r['step_size'] for r in subset]
        accs_sub = [r['accuracy'] for r in subset]
        
        ax1.plot(steps_sub, accs_sub, marker='o', markersize=3, label=f'Seq Len {s_len}', color=colors[idx], alpha=0.8)

    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    plt.title(f'Accuracy vs. Step Size per Sequence Length\nEnsemble (All Lens): {cmb_acc:.2f}% ({cmb_count} trades)')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    start_server(filename, results, cmb_acc, cmb_count)

def start_server(image_filename, results, cmb_acc, cmb_count):
    for f in glob.glob("grid_search_*.png"):
        if f != image_filename:
            try: os.remove(f)
            except: pass

    # Sort results by Accuracy Descending for the table
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    top_results = sorted_results[:50] # Show top 50

    table_rows = ""
    for r in top_results:
        bg = "background-color: #e6fffa;" if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD else ""
        wt = "font-weight: bold;" if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD else ""
        table_rows += f"<tr style='{bg} {wt}'><td>{r['seq_len']}</td><td>{r['step_size']:.5f}</td><td>{r['accuracy']:.2f}%</td><td>{r['trade_count']}</td></tr>"

    html_content = f"""
    <html>
    <head><title>Results</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; padding: 20px; background: #f4f4f4; }}
        .container {{ background: white; padding: 30px; border-radius: 12px; margin: 0 auto; max-width: 1000px; }}
        table {{ margin: 20px auto; border-collapse: collapse; width: 100%; font-size: 14px; }}
        th {{ background: #4a5568; color: white; padding: 10px; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        .stats {{ font-size: 1.2em; margin-bottom: 20px; color: #2d3748; }}
    </style>
    </head>
    <body>
        <div class="container">
            <h1>Grid & Sequence Search Results</h1>
            <div class="stats"><strong>Ensemble Performance:</strong> {cmb_acc:.2f}% Accuracy | {cmb_count} Trades</div>
            <img src="{image_filename}" style="max-width:100%; border: 1px solid #ddd; margin-bottom: 20px;">
            <h3>Top 50 Configurations</h3>
            <table><tr><th>Seq Len</th><th>Step Size</th><th>Accuracy</th><th>Trades</th></tr>{table_rows}</table>
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
