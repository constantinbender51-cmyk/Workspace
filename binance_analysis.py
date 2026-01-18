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
import gc
from datetime import datetime
from collections import Counter

from dotenv import load_dotenv
from huggingface_hub import HfApi

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTHS = [4, 5, 7, 10]

ASSETS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 
    'AVAX/USDT', 'DOT/USDT', 'LTC/USDT', 'BCH/USDT',
    'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'NEAR/USDT', 
    'FIL/USDT', 'ALGO/USDT', 'XLM/USDT', 'EOS/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'SAND/USDT'
]

DATA_DIR = "/app/data/"
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"
GRID_MIN = 0.005
GRID_MAX = 0.1
GRID_STEPS = 50
ENSEMBLE_ACC_THRESHOLD = 70.0

def fetch_binance_data(symbol, timeframe='30m', start_date='2020-01-01T00:00:00Z', end_date='2026-01-01T00:00:00Z'):
    print(f"\n--- Processing Data for {symbol} ---")
    
    safe_symbol = symbol.replace('/', '_')
    file_path = os.path.join(DATA_DIR, f"{safe_symbol}_{timeframe}.csv")
    
    if os.path.exists(file_path):
        print(f"Loading cached data from {file_path}...")
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        except Exception as e:
            print(f"Error reading cache: {e}. Re-fetching...")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        print(f"Cache miss. Fetching from Binance...")
        exchange = ccxt.binance({'enableRateLimit': True})
        
        since = exchange.parse8601(start_date)
        end_ts = exchange.parse8601(end_date)
        all_ohlcv = []
        
        while since < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=None)}...", end='\r')
                
                if since >= end_ts:
                    break
                    
                time.sleep(exchange.rateLimit / 1000 * 1.1)

            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection):
                print(f"\nRate Limit. Sleeping 60s.")
                time.sleep(60)
            except ccxt.NetworkError as e:
                print(f"\nNetwork Error: {e}. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                print(f"\nUnexpected Error: {e}")
                break
                
        print(f"\n{symbol} Data fetch complete. Total rows: {len(all_ohlcv)}")
        
        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            print(f"Saved data to {file_path}")
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")

    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')
    
    if 'timestamp' in df.columns and not df.empty:
        return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)
    
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

def train_and_evaluate(df, step_size, seq_len):
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
        if seq not in patterns:
            patterns[seq] = []
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
    if not high_acc_configs:
        return 0.0, 0
    
    ref_len = len(high_acc_configs[0]['val_seq'])
    max_seq_len = max(cfg['seq_len'] for cfg in high_acc_configs)
    
    combined_correct = 0
    combined_total = 0
    
    for target_idx in range(max_seq_len, ref_len):
        up_votes = []
        down_votes = []
        
        for cfg in high_acc_configs:
            s_len = cfg['seq_len']
            val_seq = cfg['val_seq']
            
            current_seq = tuple(val_seq[target_idx - s_len : target_idx])
            current_level = current_seq[-1]
            
            if current_seq in cfg['patterns']:
                history = cfg['patterns'][current_seq]
                predicted_level = Counter(history).most_common(1)[0][0]
                diff = predicted_level - current_level
                
                if diff > 0:
                    up_votes.append(cfg)
                elif diff < 0:
                    down_votes.append(cfg)
        
        if len(up_votes) > 0 and len(down_votes) == 0:
            best_cfg = max(up_votes, key=lambda x: x['step_size'])
            chosen_val_seq = best_cfg['val_seq']
            curr_lvl = chosen_val_seq[target_idx - 1]
            next_lvl = chosen_val_seq[target_idx]
            actual_diff = next_lvl - curr_lvl
            
            if actual_diff != 0:
                combined_total += 1
                if actual_diff > 0:
                    combined_correct += 1
        
        elif len(down_votes) > 0 and len(up_votes) == 0:
            best_cfg = max(down_votes, key=lambda x: x['step_size'])
            chosen_val_seq = best_cfg['val_seq']
            curr_lvl = chosen_val_seq[target_idx - 1]
            next_lvl = chosen_val_seq[target_idx]
            actual_diff = next_lvl - curr_lvl
            
            if actual_diff != 0:
                combined_total += 1
                if actual_diff < 0:
                    combined_correct += 1

    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    return acc, combined_total

def upload_to_huggingface(filename):
    print(f"--- HUGGING FACE UPLOAD: {filename} ---")
    load_dotenv()
    token = os.getenv("HFT")
    
    if not token:
        print("[ERROR] 'HFT' token not found in .env file. Skipping upload.")
        return

    try:
        api = HfApi()
        path_in_repo = f"{HF_FOLDER}/{os.path.basename(filename)}"
        
        print(f"Uploading {filename} to {HF_REPO_ID} at {path_in_repo}...")
        
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
            commit_message=f"Update model {os.path.basename(filename)} {datetime.now()}"
        )
        print("[SUCCESS] File Updated on Hugging Face.")
            
    except Exception as e:
        print(f"[ERROR] Hugging Face Upload Failed: {e}")

def save_ensemble_model(high_acc_configs, initial_reference_price, model_filename):
    if not high_acc_configs:
        print(f"No high accuracy configs to save for {model_filename}.")
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
        with open(model_filename, 'wb') as f:
            pickle.dump(model_payload, f)
        print(f"[SUCCESS] Model saved locally to '{model_filename}'")
        upload_to_huggingface(model_filename)
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")

def run_analysis_for_asset(symbol):
    """
    MEMORY-OPTIMIZED: Processes one asset with aggressive cleanup
    """
    clean_name = symbol.split('/')[0].lower()
    model_filename = f"{clean_name}.pkl"
    image_filename = f"grid_{clean_name}_{int(time.time())}.png"

    # Fetch data
    df = fetch_binance_data(symbol)
    if df.empty:
        print(f"Skipping {symbol} due to empty data.")
        gc.collect()
        return None
    
    initial_price = df['close'].iloc[0]
    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    
    # CRITICAL: Store only lightweight results
    lightweight_results = []
    high_acc_configs = []
    
    print(f"--- PROCESSING {symbol} (Steps: {GRID_STEPS}, SeqLens: {SEQ_LENGTHS}) ---")
    
    for s_len in SEQ_LENGTHS:
        for step in step_sizes:
            res = train_and_evaluate(df, step, s_len)
            
            # Store lightweight version for plotting
            lightweight_results.append({
                'seq_len': res['seq_len'],
                'step_size': res['step_size'],
                'accuracy': res['accuracy'],
                'trade_count': res['trade_count']
            })
            
            # Keep full config only if high accuracy
            if res['accuracy'] > ENSEMBLE_ACC_THRESHOLD:
                high_acc_configs.append(res)
            else:
                # CRITICAL: Delete heavy data immediately
                del res['patterns']
                del res['val_seq']
            
            del res
            
            # Periodic cleanup every 10 iterations
            if len(lightweight_results) % 10 == 0:
                gc.collect()
    
    print(f"Found {len(high_acc_configs)} configurations above {ENSEMBLE_ACC_THRESHOLD}% for {symbol}.")
    
    # Run ensemble metric
    cmb_acc, cmb_count = run_combined_metric(high_acc_configs)
    print(f"{symbol} Ensemble: {cmb_acc:.2f}% | Trades: {cmb_count}")

    # Save Model
    save_ensemble_model(high_acc_configs, initial_price, model_filename)

    # Generate Plot using lightweight results
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Step Size')
    ax1.set_ylabel('Accuracy (%)')
    ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':', label='Threshold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(SEQ_LENGTHS)))
    for idx, s_len in enumerate(SEQ_LENGTHS):
        subset = [r for r in lightweight_results if r['seq_len'] == s_len]
        if not subset:
            continue
        steps_sub = [r['step_size'] for r in subset]
        accs_sub = [r['accuracy'] for r in subset]
        ax1.plot(steps_sub, accs_sub, marker='o', markersize=3, label=f'Seq {s_len}', color=colors[idx], alpha=0.8)

    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.title(f'{symbol} - Accuracy vs Step Size\nEnsemble: {cmb_acc:.2f}% ({cmb_count} trades)')
    fig.tight_layout()
    plt.savefig(image_filename, dpi=100)
    plt.close(fig)
    plt.clf()
    
    # Prepare summary with only top 20 lightweight results
    sorted_results = sorted(lightweight_results, key=lambda x: x['accuracy'], reverse=True)[:20]
    
    summary = {
        'symbol': symbol,
        'image': image_filename,
        'results': sorted_results,
        'cmb_acc': cmb_acc,
        'cmb_count': cmb_count
    }
    
    # AGGRESSIVE CLEANUP
    del df
    del lightweight_results
    del high_acc_configs
    del sorted_results
    del step_sizes
    plt.close('all')
    gc.collect()
    
    print(f"[MEMORY] Cleaned up {symbol}. Forcing garbage collection...")
    
    return summary

def start_server_combined(all_asset_data):
    # Clean up old pngs
    current_images = [d['image'] for d in all_asset_data]
    for f in glob.glob("grid_*.png"):
        if f not in current_images:
            try:
                os.remove(f)
            except:
                pass

    sections_html = ""
    
    for data in all_asset_data:
        table_rows = ""
        for r in data['results']:
            bg = "background-color: #e6fffa;" if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD else ""
            wt = "font-weight: bold;" if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD else ""
            table_rows += f"<tr style='{bg} {wt}'><td>{r['seq_len']}</td><td>{r['step_size']:.5f}</td><td>{r['accuracy']:.2f}%</td><td>{r['trade_count']}</td></tr>"

        sections_html += f"""
        <div class="asset-section">
            <h2>{data['symbol']}</h2>
            <div class="stats"><strong>Ensemble:</strong> {data['cmb_acc']:.2f}% Accuracy | {data['cmb_count']} Trades</div>
            <img src="{data['image']}" class="chart">
            <div class="table-container">
                <table>
                    <tr><th>Seq Len</th><th>Step Size</th><th>Accuracy</th><th>Trades</th></tr>
                    {table_rows}
                </table>
            </div>
        </div>
        <hr>
        """

    html_content = f"""
    <html>
    <head><title>Multi-Asset Grid Search Results</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding: 20px; background: #f0f2f5; color: #333; }}
        .container {{ background: white; padding: 40px; border-radius: 15px; margin: 0 auto; max-width: 1100px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .asset-section {{ margin-bottom: 40px; }}
        h1 {{ color: #1a202c; margin-bottom: 30px; }}
        h2 {{ color: #2d3748; border-left: 5px solid #3182ce; padding-left: 10px; text-align: left; margin-left: 5%; }}
        table {{ margin: 10px auto; border-collapse: collapse; width: 90%; font-size: 13px; }}
        th {{ background: #4a5568; color: white; padding: 10px; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        .stats {{ font-size: 1.1em; margin-bottom: 15px; color: #2b6cb0; font-weight: bold; }}
        .chart {{ max-width: 90%; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 20px; }}
        hr {{ border: 0; height: 1px; background: #cbd5e0; margin: 40px 0; }}
    </style>
    </head>
    <body>
        <div class="container">
            <h1>Crypto Pattern Matcher Dashboard</h1>
            {sections_html}
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
            print(f"\n=================================================")
            print(f"All assets processed.")
            print(f"Server active at: http://localhost:{PORT}")
            print(f"=================================================")
            httpd.serve_forever()
    except Exception as e:
        print(f"Server error: {e}")

def run_multi_asset_search():
    final_report_data = []
    
    print(f"Starting Multi-Asset Analysis for: {ASSETS}")
    
    for idx, symbol in enumerate(ASSETS):
        try:
            print(f"\n[{idx+1}/{len(ASSETS)}] Processing {symbol}...")
            asset_data = run_analysis_for_asset(symbol)
            
            if asset_data:
                final_report_data.append(asset_data)
            
            # Force cleanup after each asset
            gc.collect()
            print(f"[MEMORY] Post-cleanup for {symbol} complete.\n")
            
        except Exception as e:
            print(f"CRITICAL ERROR processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()  # Cleanup even on error
            
    if final_report_data:
        start_server_combined(final_report_data)
    else:
        print("No results generated.")

if __name__ == "__main__":
    try:
        run_multi_asset_search()
    except KeyboardInterrupt:
        print("\nStopped.")