import ccxt.async_support as ccxt  # Async support
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
import asyncio 
import concurrent.futures
from datetime import datetime
from collections import Counter, defaultdict
from numba import njit, prange # JIT Compilation

# --- NEW IMPORTS FOR HUGGING FACE ---
from dotenv import load_dotenv
from huggingface_hub import HfApi

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTHS = [5, 6, 7, 8, 9, 10]
ASSETS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
DATA_DIR = "/app/data/"
MAX_WORKERS = os.cpu_count() # Automatically use all available cores

# Hugging Face Configuration
HF_REPO_ID = "Llama26051996/Models"
HF_FOLDER = "model2x"

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.05
GRID_STEPS = 20
ENSEMBLE_ACC_THRESHOLD = 70.0

# --- JIT COMPILED FUNCTIONS (NUMBA) ---

@njit(fastmath=True)
def get_grid_indices_numba(close_array, step_size):
    """
    Optimized calculation of grid indices using Log-Price grids.
    """
    n = len(close_array)
    grid_indices = np.zeros(n, dtype=np.int32)
    
    # Pre-calculate log prices for speed
    # We can do the cumprod/log approach, or simpler log(price) / step
    # The original logic used percentage change cumprod, we replicate that exactly:
    
    pct_change = np.zeros(n)
    # diff/close approach
    for i in range(1, n):
        pct_change[i] = (close_array[i] - close_array[i-1]) / close_array[i-1]
        
    # Reconstruct normalized price path
    current_price = 1.0
    for i in range(n):
        if i > 0:
            current_price = current_price * (1.0 + pct_change[i])
        
        # log(price) / step
        grid_indices[i] = int(np.floor(np.log(current_price) / step_size))
        
    return grid_indices

def train_patterns_optimized(grid_indices, seq_len, train_end_idx):
    """
    Trains patterns using a nested dictionary structure for memory efficiency.
    Returns: Dict[Tuple, Dict[int, int]] -> {sequence: {next_level: count}}
    """
    patterns = defaultdict(lambda: defaultdict(int))
    
    # We loop in python here because dictionary manipulation with tuples 
    # is complex in Numba. However, we only store COUNTS, not lists.
    for i in range(train_end_idx - seq_len):
        # Create tuple key (fast in Python)
        seq = tuple(grid_indices[i : i + seq_len])
        target = grid_indices[i + seq_len]
        patterns[seq][target] += 1
        
    return patterns

def evaluate_patterns_optimized(grid_indices, patterns, seq_len, train_end_idx, val_end_idx):
    """
    Validates the patterns.
    """
    move_correct = 0
    move_total_valid = 0
    
    for i in range(train_end_idx, val_end_idx - seq_len):
        current_seq = tuple(grid_indices[i : i + seq_len])
        current_level = current_seq[-1]
        actual_next_level = grid_indices[i + seq_len]
        
        if current_seq in patterns:
            # Get most common next level from counts
            outcome_counts = patterns[current_seq]
            if not outcome_counts:
                continue
                
            # Find key with max value
            predicted_level = max(outcome_counts, key=outcome_counts.get)
            
            pred_diff = predicted_level - current_level
            actual_diff = actual_next_level - current_level
            
            if pred_diff != 0 and actual_diff != 0:
                move_total_valid += 1
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    move_correct += 1
                    
    return move_correct, move_total_valid

# --- ASYNC DATA FETCHING ---

async def fetch_binance_data_async(symbol, timeframe, start_date, end_date):
    print(f"\n--- Processing Data for {symbol} [{timeframe}] ---")
    
    # Cache using PARQUET (Faster I/O, Compression)
    safe_symbol = symbol.replace('/', '_')
    file_path = os.path.join(DATA_DIR, f"{safe_symbol}_{timeframe}.parquet")
    
    if os.path.exists(file_path):
        print(f"Loading cached parquet from {file_path}...")
        try:
            df = pd.read_parquet(file_path)
            # Ensure proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df
        except Exception as e:
            print(f"Error reading parquet: {e}. Re-fetching...")
    
    # Async Fetch
    print(f"Cache miss. Fetching from Binance (Async)...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    try:
        since = exchange.parse8601(start_date)
        end_ts = exchange.parse8601(end_date)
        all_ohlcv = []
        
        while since < end_ts:
            # Fetch
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            # Simple rate limit wait (non-blocking)
            await asyncio.sleep(exchange.rateLimit / 1000 * 1.1)
            
        await exchange.close()
        
        if not all_ohlcv: return pd.DataFrame()

        # Create DF
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Optimize Types (Memory)
        df = df.astype({
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'volume': 'float32'
        })
        
        # Save Parquet
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_parquet(file_path, index=False)
        print(f"Saved data to {file_path}")
        
        # Filter Dates
        start_dt = pd.Timestamp(start_date, tz='UTC')
        end_dt = pd.Timestamp(end_date, tz='UTC')
        return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)

    except Exception as e:
        print(f"Fetch Error: {e}")
        await exchange.close()
        return pd.DataFrame()

# --- WORKER FUNCTION FOR MULTIPROCESSING ---

def process_grid_config(args):
    """
    Worker function executed in parallel. 
    Receives: (close_array (numpy), step_size, seq_len)
    """
    close_array, step_size, seq_len = args
    
    # 1. Numba Calculation of indices
    grid_indices = get_grid_indices_numba(close_array, step_size)
    
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    # 2. Train (Optimized Dictionary)
    patterns = train_patterns_optimized(grid_indices, seq_len, idx_80)
    
    # 3. Evaluate
    correct, total = evaluate_patterns_optimized(grid_indices, patterns, seq_len, idx_80, idx_90)
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    # Return strict minimal data to avoid serialization overhead
    # We return 'val_seq' purely for the ensemble step later
    return {
        'step_size': step_size,
        'seq_len': seq_len,
        'accuracy': accuracy,
        'trade_count': total,
        'patterns': dict(patterns), # Convert defaultdict to dict for pickling
        'val_seq': grid_indices[idx_80:idx_90] 
    }

# --- MAIN LOGIC ---

def run_combined_metric(high_acc_configs):
    if not high_acc_configs: return 0.0, 0
    
    # All val_seqs should be roughly aligned in time, 
    # but grid indices differ. We compare by index.
    ref_len = len(high_acc_configs[0]['val_seq'])
    max_seq_len = max(cfg['seq_len'] for cfg in high_acc_configs)
    
    combined_correct = 0
    combined_total = 0
    
    # Vectorize this loop? Hard due to different patterns.
    # We keep it imperative but clean.
    
    for target_idx in range(max_seq_len, ref_len):
        up_votes = []
        down_votes = []
        
        for cfg in high_acc_configs:
            s_len = cfg['seq_len']
            val_seq = cfg['val_seq']
            
            # Bounds check
            if target_idx < s_len: continue

            current_seq = tuple(val_seq[target_idx - s_len : target_idx])
            current_level = current_seq[-1]
            
            if current_seq in cfg['patterns']:
                outcome_counts = cfg['patterns'][current_seq]
                predicted_level = max(outcome_counts, key=outcome_counts.get)
                
                diff = predicted_level - current_level
                if diff > 0: up_votes.append(cfg)
                elif diff < 0: down_votes.append(cfg)
        
        # Ensemble Logic
        vote_diff = 0
        chosen_cfg = None
        
        if len(up_votes) > 0 and len(down_votes) == 0:
            chosen_cfg = max(up_votes, key=lambda x: x['step_size'])
            expected_dir = 1
        elif len(down_votes) > 0 and len(up_votes) == 0:
            chosen_cfg = max(down_votes, key=lambda x: x['step_size'])
            expected_dir = -1
        else:
            continue # Conflict or no votes

        if chosen_cfg:
            curr_lvl = chosen_cfg['val_seq'][target_idx - 1]
            next_lvl = chosen_cfg['val_seq'][target_idx]
            actual_diff = next_lvl - curr_lvl
            
            if actual_diff != 0:
                combined_total += 1
                if (expected_dir > 0 and actual_diff > 0) or (expected_dir < 0 and actual_diff < 0):
                    combined_correct += 1

    acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
    return acc, combined_total

def upload_to_huggingface(filename):
    print(f"--- HUGGING FACE UPLOAD: {filename} ---")
    load_dotenv()
    token = os.getenv("HFT")
    if not token: return

    try:
        api = HfApi()
        path_in_repo = f"{HF_FOLDER}/{os.path.basename(filename)}"
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
            commit_message=f"Update {datetime.now()}"
        )
        print("[SUCCESS] Uploaded.")
    except Exception as e:
        print(f"[ERROR] Upload Failed: {e}")

def save_ensemble_model(high_acc_configs, initial_reference_price, model_filename):
    if not high_acc_configs: return

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
        'initial_price': float(initial_reference_price),
        'ensemble_configs': lean_configs,
        'threshold_used': ENSEMBLE_ACC_THRESHOLD
    }

    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model_payload, f)
        upload_to_huggingface(model_filename)
    except Exception as e:
        print(f"Save Error: {e}")

async def calculate_timeframe_metrics(symbol, timeframe, executor):
    """
    Orchestrates the parallel processing for a single asset/timeframe.
    """
    df = await fetch_binance_data_async(symbol, timeframe, '2020-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
    
    if df.empty: return None
    
    initial_price = df['close'].iloc[0]
    # Ensure close_array is contiguous in memory for Numba
    close_array = np.ascontiguousarray(df['close'].values)
    
    step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
    
    # Prepare arguments for parallel execution
    tasks = []
    for s_len in SEQ_LENGTHS:
        for step in step_sizes:
            tasks.append((close_array, step, s_len))
    
    print(f"--- ANALYZING {symbol} [{timeframe}] with {len(tasks)} tasks on {MAX_WORKERS} cores ---")
    
    # Run Grid Search in Parallel
    loop = asyncio.get_running_loop()
    results = []
    
    # We use executor.map implicitly via loop.run_in_executor for individual tasks 
    # OR we block briefly to run the whole batch map.
    # Since these are CPU bound, running them in a ProcessPool is best.
    
    # Helper wrapper to unpack args
    # Note: process_grid_config must be picklable (top-level function)
    
    # Using run_in_executor for the whole batch wait
    results_gen = await loop.run_in_executor(executor, _map_helper, tasks)
    results = list(results_gen)
        
    high_acc_configs = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
    
    cmb_acc, cmb_count = run_combined_metric(high_acc_configs)
    metric_score = ((cmb_acc / 100.0) - 0.5) * cmb_count
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'initial_price': initial_price,
        'results': results,
        'high_acc_configs': high_acc_configs,
        'cmb_acc': cmb_acc,
        'cmb_count': cmb_count,
        'metric_score': metric_score
    }

def _map_helper(tasks):
    # This runs inside the ProcessPool
    # We map the worker function over the tasks
    # Re-create a pool here? No, we are already IN a pool worker if we used run_in_executor?
    # Actually, loop.run_in_executor runs a single callable. 
    # To use all cores, we should use the pool's map method.
    
    # We need a separate ProcessPoolExecutor instance inside here? No.
    # Correct pattern for asyncio + process pool map:
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return list(pool.map(process_grid_config, tasks))

def process_winning_timeframe(best_result):
    symbol = best_result['symbol']
    timeframe = best_result['timeframe']
    clean_name = symbol.split('/')[0].lower()
    
    model_filename = f"{clean_name}.pkl"
    image_filename = f"grid_{clean_name}_{int(time.time())}.png"
    
    print(f"\n>>> WINNER: {symbol} {timeframe} (Score: {best_result['metric_score']:.4f})")
    
    save_ensemble_model(best_result['high_acc_configs'], best_result['initial_price'], model_filename)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Step Size')
    ax1.set_ylabel('Accuracy (%)')
    ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(SEQ_LENGTHS)))
    for idx, s_len in enumerate(SEQ_LENGTHS):
        subset = [r for r in best_result['results'] if r['seq_len'] == s_len]
        if not subset: continue
        steps_sub = [r['step_size'] for r in subset]
        accs_sub = [r['accuracy'] for r in subset]
        ax1.plot(steps_sub, accs_sub, marker='o', markersize=3, label=f'Seq {s_len}', color=colors[idx], alpha=0.8)

    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    plt.title(f'{symbol} ({timeframe}) - Ensemble: {best_result["cmb_acc"]:.2f}% ({best_result["cmb_count"]} trades)')
    plt.tight_layout()
    plt.savefig(image_filename)
    plt.close()
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'image': image_filename,
        'results': best_result['results'],
        'cmb_acc': best_result['cmb_acc'],
        'cmb_count': best_result['cmb_count']
    }

def start_server_combined(all_asset_data):
    # Clean old pngs
    current_images = [d['image'] for d in all_asset_data]
    for f in glob.glob("grid_*.png"):
        if f not in current_images:
            try: os.remove(f)
            except: pass

    sections_html = ""
    for data in all_asset_data:
        sorted_results = sorted(data['results'], key=lambda x: x['accuracy'], reverse=True)
        top_results = sorted_results[:15]
        
        table_rows = ""
        for r in top_results:
            bg = "background-color: #e6fffa;" if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD else ""
            table_rows += f"<tr style='{bg}'><td>{r['seq_len']}</td><td>{r['step_size']:.5f}</td><td>{r['accuracy']:.2f}%</td><td>{r['trade_count']}</td></tr>"

        sections_html += f"""
        <div class="asset-section">
            <h2>{data['symbol']} <span style="font-size:0.8em; color:#666;">({data['timeframe']})</span></h2>
            <div class="stats">Ensemble: {data['cmb_acc']:.2f}% | Trades: {data['cmb_count']}</div>
            <img src="{data['image']}" class="chart">
            <table><tr><th>Seq</th><th>Step</th><th>Acc</th><th>Trades</th></tr>{table_rows}</table>
        </div><hr>"""

    html = f"""<html><head><title>Grid Results</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #f4f4f9; }}
        .asset-section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
        td, th {{ padding: 6px; border-bottom: 1px solid #eee; text-align: center; }}
        .chart {{ max-width: 100%; margin: 10px 0; }}
    </style></head><body><h1>Analysis Dashboard</h1>{sections_html}</body></html>"""
    
    with open("index.html", "w") as f: f.write(html)
    
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\nServer active at http://localhost:{PORT}")
        httpd.serve_forever()

async def main():
    final_report_data = []
    
    # Create a ProcessPoolExecutor to reuse across timeframes
    # This avoids spinning up new processes constantly
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        for symbol in ASSETS:
            best_score = -float('inf')
            best_result = None
            
            print(f"\n=== Evaluating {symbol} ===")
            
            for tf in TIMEFRAMES:
                try:
                    # Pass the executor to the helper
                    # We utilize a workaround because we cannot pickle the executor itself easily
                    # inside the loop.run_in_executor if we nest them deeply.
                    # But _map_helper creates its own pool? No, that's inefficient.
                    # BETTER APPROACH: Use the executor we created here.
                    
                    # Refactored call:
                    # To run the map in parallel, we need to offload the mapping logic to a thread
                    # or just run it here if it's blocking.
                    # Since 'process_grid_config' is CPU heavy, we want it in processes.
                    
                    df = await fetch_binance_data_async(symbol, tf, '2020-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                    if df.empty: continue
                    
                    close_arr = np.ascontiguousarray(df['close'].values)
                    start_p = df['close'].iloc[0]
                    
                    tasks = [(close_arr, s, l) for l in SEQ_LENGTHS for s in np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)]
                    
                    # Blocking call to map (runs in parallel processes)
                    # We run this in a separate thread so we don't block the async event loop
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, lambda: list(executor.map(process_grid_config, tasks)))
                    
                    high_acc = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
                    acc, count = run_combined_metric(high_acc)
                    score = ((acc / 100.0) - 0.5) * count
                    
                    print(f"-> {tf}: Acc={acc:.2f}%, Trades={count} | Score={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'symbol': symbol, 'timeframe': tf, 'initial_price': start_p,
                            'results': results, 'high_acc_configs': high_acc,
                            'cmb_acc': acc, 'cmb_count': count, 'metric_score': score
                        }
                        
                except Exception as e:
                    print(f"Error {symbol} {tf}: {e}")
                    import traceback
                    traceback.print_exc()

            if best_result:
                final_report_data.append(process_winning_timeframe(best_result))
    
    if final_report_data:
        start_server_combined(final_report_data)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
