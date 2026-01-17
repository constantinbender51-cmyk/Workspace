import ccxt.async_support as ccxt
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
import asyncio 
import concurrent.futures
import pickle
from datetime import datetime
from collections import Counter, defaultdict
from numba import njit
from dotenv import load_dotenv
from huggingface_hub import HfApi

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTHS = [5, 6, 7, 8, 9, 10]
ASSETS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
# We fetch '1m' ONLY, then derive the rest locally
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'] 
DATA_DIR = "/app/data/"
MAX_WORKERS = os.cpu_count()

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.05
GRID_STEPS = 20 
ENSEMBLE_ACC_THRESHOLD = 70.0

# Hugging Face
HF_REPO_ID = "Llama26051996/Models" 
HF_FOLDER = "model2x"

# --- JIT COMPILED MATH (NUMBA) ---

@njit(fastmath=True)
def get_grid_indices_numba(close_array, step_size):
    n = len(close_array)
    grid_indices = np.zeros(n, dtype=np.int32)
    pct_change = np.zeros(n)
    
    # Calculate percentage changes
    for i in range(1, n):
        pct_change[i] = (close_array[i] - close_array[i-1]) / close_array[i-1]
        
    current_price = 1.0
    for i in range(n):
        if i > 0:
            current_price = current_price * (1.0 + pct_change[i])
        grid_indices[i] = int(np.floor(np.log(current_price) / step_size))
        
    return grid_indices

def train_patterns_optimized(grid_indices, seq_len, train_end_idx):
    patterns = defaultdict(lambda: defaultdict(int))
    for i in range(train_end_idx - seq_len):
        seq = tuple(grid_indices[i : i + seq_len])
        target = grid_indices[i + seq_len]
        patterns[seq][target] += 1
    return patterns

def evaluate_patterns_optimized(grid_indices, patterns, seq_len, train_end_idx, val_end_idx):
    move_correct = 0
    move_total_valid = 0
    for i in range(train_end_idx, val_end_idx - seq_len):
        current_seq = tuple(grid_indices[i : i + seq_len])
        current_level = current_seq[-1]
        actual_next_level = grid_indices[i + seq_len]
        
        if current_seq in patterns:
            outcome_counts = patterns[current_seq]
            if not outcome_counts: continue
            predicted_level = max(outcome_counts, key=outcome_counts.get)
            
            pred_diff = predicted_level - current_level
            actual_diff = actual_next_level - current_level
            
            if pred_diff != 0 and actual_diff != 0:
                move_total_valid += 1
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    move_correct += 1
    return move_correct, move_total_valid

# --- RESAMPLING & DATA ---

def resample_data(df, timeframe_str):
    """
    Resamples 1m data to target timeframe.
    Timeframe mapping: '5m'->'5T', '1h'->'1H', '1d'->'1D'
    """
    if timeframe_str == '1m':
        return df.copy()

    # Convert common CCXT timeframe strings to Pandas aliases
    tf_map = {
        'm': 'T', 'h': 'H', 'd': 'D', 'w': 'W'
    }
    unit = timeframe_str[-1]
    val = timeframe_str[:-1]
    if unit in tf_map:
        rule = f"{val}{tf_map[unit]}"
    else:
        print(f"Unknown timeframe format {timeframe_str}, returning original.")
        return df

    print(f"   -> Resampling 1m data to {timeframe_str}...")
    
    # Ensure index is datetime
    df_res = df.set_index('timestamp').copy()
    
    # Resample Logic: OHLCV aggregation
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    resampled = df_res.resample(rule).agg(agg_dict)
    resampled = resampled.dropna().reset_index()
    return resampled

async def fetch_base_data(symbol, start_date, end_date):
    """
    Fetches ONLY the 1m data (Base Data).
    """
    timeframe = '1m'
    print(f"\n--- Fetching BASE DATA (1m) for {symbol} ---")
    
    safe_symbol = symbol.replace('/', '_')
    # Using Parquet for speed
    file_path = os.path.join(DATA_DIR, f"{safe_symbol}_{timeframe}.parquet")
    
    # 1. Try Cache
    if os.path.exists(file_path):
        print(f"Loading cached 1m parquet from {file_path}...")
        try:
            df = pd.read_parquet(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            # Filter Date Range locally
            start_dt = pd.Timestamp(start_date, tz='UTC')
            end_dt = pd.Timestamp(end_date, tz='UTC')
            return df.loc[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)].reset_index(drop=True)
        except Exception as e:
            print(f"Error reading cache: {e}. Re-fetching...")

    # 2. Fetch from Binance
    print(f"Cache miss. Downloading 1m history (this may take time)...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    try:
        since = exchange.parse8601(start_date)
        end_ts = exchange.parse8601(end_date)
        all_ohlcv = []
        
        while since < end_ts:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            # Slight buffer to prevent rate limits
            await asyncio.sleep(exchange.rateLimit / 1000 * 1.05)
            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
            
        await exchange.close()
        print(f"\nDownload complete. Total 1m rows: {len(all_ohlcv)}")
        
        if not all_ohlcv: return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Optimize types
        df = df.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'})
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_parquet(file_path, index=False)
        return df

    except Exception as e:
        print(f"Fetch Error: {e}")
        await exchange.close()
        return pd.DataFrame()

# --- WORKER & ANALYSIS ---

def process_grid_config(args):
    # Worker: Receives numpy array, calculates grid, trains, validates
    close_array, step_size, seq_len = args
    
    grid_indices = get_grid_indices_numba(close_array, step_size)
    total_len = len(grid_indices)
    idx_80 = int(total_len * 0.80)
    idx_90 = int(total_len * 0.90)
    
    patterns = train_patterns_optimized(grid_indices, seq_len, idx_80)
    correct, total = evaluate_patterns_optimized(grid_indices, patterns, seq_len, idx_80, idx_90)
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return {
        'step_size': step_size, 'seq_len': seq_len, 'accuracy': accuracy,
        'trade_count': total, 'patterns': dict(patterns), 
        'val_seq': grid_indices[idx_80:idx_90] 
    }

def run_combined_metric(high_acc_configs):
    if not high_acc_configs: return 0.0, 0
    ref_len = len(high_acc_configs[0]['val_seq'])
    max_seq_len = max(cfg['seq_len'] for cfg in high_acc_configs)
    
    combined_correct = 0
    combined_total = 0
    
    for target_idx in range(max_seq_len, ref_len):
        up_votes, down_votes = [], []
        
        for cfg in high_acc_configs:
            s_len = cfg['seq_len']
            val_seq = cfg['val_seq']
            if target_idx < s_len: continue
            
            current_seq = tuple(val_seq[target_idx - s_len : target_idx])
            
            if current_seq in cfg['patterns']:
                outcome = cfg['patterns'][current_seq]
                pred = max(outcome, key=outcome.get)
                diff = pred - current_seq[-1]
                if diff > 0: up_votes.append(cfg)
                elif diff < 0: down_votes.append(cfg)
        
        chosen = None
        exp_dir = 0
        if up_votes and not down_votes:
            chosen = max(up_votes, key=lambda x: x['step_size'])
            exp_dir = 1
        elif down_votes and not up_votes:
            chosen = max(down_votes, key=lambda x: x['step_size'])
            exp_dir = -1
            
        if chosen:
            real_diff = chosen['val_seq'][target_idx] - chosen['val_seq'][target_idx-1]
            if real_diff != 0:
                combined_total += 1
                if (exp_dir > 0 and real_diff > 0) or (exp_dir < 0 and real_diff < 0):
                    combined_correct += 1

    return (combined_correct / combined_total * 100) if combined_total > 0 else 0.0, combined_total

def save_and_upload_model(best_result):
    clean_name = best_result['symbol'].split('/')[0].lower()
    fname = f"{clean_name}.pkl"
    
    print(f"Saving model to {fname}...")
    lean = [{'step_size': c['step_size'], 'seq_len': c['seq_len'], 
             'patterns': c['patterns'], 'accuracy': c['accuracy']} 
            for c in best_result['high_acc_configs']]
            
    payload = {
        'timestamp': datetime.now().isoformat(),
        'initial_price': float(best_result['initial_price']),
        'ensemble_configs': lean,
        'threshold': ENSEMBLE_ACC_THRESHOLD
    }
    
    with open(fname, 'wb') as f: pickle.dump(payload, f)
    
    # Upload
    load_dotenv()
    token = os.getenv("HFT")
    if token:
        try:
            api = HfApi()
            api.upload_file(path_or_fileobj=fname, path_in_repo=f"{HF_FOLDER}/{fname}", 
                            repo_id=HF_REPO_ID, repo_type="model", token=token)
            print("Uploaded to Hugging Face.")
        except Exception as e: print(f"Upload failed: {e}")

# --- MAIN EXECUTION ---

async def main():
    final_report = []
    
    # Create Process Pool once
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        for symbol in ASSETS:
            # 1. Fetch 1m Base Data ONCE
            base_df = await fetch_base_data(symbol, '2020-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            if base_df.empty: 
                print(f"Skipping {symbol} (No data)")
                continue
                
            best_score = -float('inf')
            best_result = None
            
            print(f"\n=== Evaluating Timeframes for {symbol} ===")
            
            # 2. Iterate Timeframes and Resample locally
            for tf in TIMEFRAMES:
                try:
                    # Resample from base_df
                    df_tf = resample_data(base_df, tf)
                    if len(df_tf) < 500: # Skip if too few candles
                        print(f"Skipping {tf} (Not enough data: {len(df_tf)})")
                        continue
                        
                    close_arr = np.ascontiguousarray(df_tf['close'].values)
                    start_p = df_tf['close'].iloc[0]
                    
                    # Prepare tasks
                    tasks = [(close_arr, s, l) for l in SEQ_LENGTHS for s in np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)]
                    
                    # Run Parallel Grid Search
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, lambda: list(executor.map(process_grid_config, tasks)))
                    
                    # Ensemble Logic
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

            # 3. Process Winner for this Asset
            if best_result:
                save_and_upload_model(best_result)
                
                # Generate Chart
                img_name = f"grid_{symbol.split('/')[0].lower()}_{int(time.time())}.png"
                fig, ax = plt.subplots(figsize=(10, 6))
                for s_len in SEQ_LENGTHS:
                    sub = [r for r in best_result['results'] if r['seq_len'] == s_len]
                    if sub:
                        ax.plot([r['step_size'] for r in sub], [r['accuracy'] for r in sub], 
                                marker='o', markersize=3, alpha=0.7, label=f'Seq {s_len}')
                ax.axhline(ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':')
                ax.legend()
                ax.set_title(f"{symbol} {best_result['timeframe']} (Ensemble: {best_result['cmb_acc']:.1f}%)")
                plt.savefig(img_name)
                plt.close()
                
                best_result['image'] = img_name
                final_report.append(best_result)

    # 4. Start Server
    if final_report:
        start_server(final_report)

def start_server(data_list):
    html = "<html><body style='font-family:sans-serif; padding:20px; background:#f0f0f0'>"
    for d in data_list:
        top = sorted(d['results'], key=lambda x: x['accuracy'], reverse=True)[:10]
        rows = "".join([f"<tr><td>{r['seq_len']}</td><td>{r['step_size']:.4f}</td><td>{r['accuracy']:.1f}%</td></tr>" for r in top])
        html += f"""
        <div style='background:white; padding:20px; margin-bottom:20px; border-radius:10px'>
            <h2>{d['symbol']} ({d['timeframe']})</h2>
            <p><strong>Ensemble:</strong> {d['cmb_acc']:.2f}% ({d['cmb_count']} trades)</p>
            <img src='{d['image']}' style='max-width:100%'>
            <table border=1 cellspacing=0 cellpadding=5 style='width:100%'>
                <tr><th>Seq</th><th>Step</th><th>Acc</th></tr>
                {rows}
            </table>
        </div>"""
    html += "</body></html>"
    with open("index.html", "w") as f: f.write(html)
    
    with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"\nServer at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
