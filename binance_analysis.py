import json
import time
import requests
import urllib.request
import threading
import http.server
import socketserver
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/"
BASE_INTERVAL = "15m"
LATENCY_BUFFER = 2 
HTTP_PORT = 8080

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT", "BNBUSDT"
]

TIMEFRAMES = ["15m", "30m", "60m", "240m", "1d"]

# GLOBAL STATE
LATEST_PREDICTIONS = {}
GLOBAL_TRADE_HISTORY = [] # Stores backtest results for the /history endpoint
CACHED_MODELS = {} # Stores pre-parsed models

# --- UTILS ---

def delayed_print(msg):
    print(msg)

def get_bucket(price, bucket_size):
    return int(price // bucket_size)

def get_sleep_time_to_next_candle(interval_str="15m"):
    now = datetime.now()
    if interval_str.endswith("m"):
        minutes = int(interval_str[:-1])
        next_minute = (now.minute // minutes + 1) * minutes
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)
    elif interval_str.endswith("h"):
        hours = int(interval_str[:-1])
        next_hour = (now.hour // hours + 1) * hours
        next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=next_hour)
    elif interval_str == "1d":
        next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        minutes = 15
        next_minute = (now.minute // minutes + 1) * minutes
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)

    sleep_seconds = (next_time - now).total_seconds()
    if sleep_seconds < 0: sleep_seconds += 60
    return sleep_seconds + LATENCY_BUFFER

def fetch_recent_binance_data(symbol, days=30):
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    data_points = []
    current_start = start_ts
    base_url = "https://api.binance.com/api/v3/klines"
    
    while current_start < end_ts:
        url = f"{base_url}?symbol={symbol}&interval={BASE_INTERVAL}&startTime={current_start}&endTime={end_ts}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                batch = json.loads(response.read().decode())
                if not batch: break
                parsed_batch = [(int(c[6]), float(c[4])) for c in batch]
                data_points.extend(parsed_batch)
                last_close_time = batch[-1][6]
                current_start = last_close_time + 1
                if last_close_time >= end_ts - 1000: break
        except Exception as e:
            # Silent fail for cleaner logs
            break
            
    unique_data = {x[0]: x[1] for x in data_points}
    return sorted([(k, v) for k, v in unique_data.items()])

# --- OPTIMIZED MODEL LOADING ---

def fetch_and_parse_model(asset, timeframe):
    """
    Fetches model and PRE-PARSES string keys to tuples once.
    This saves massive CPU time during the backtest loop.
    """
    cache_key = f"{asset}_{timeframe}"
    if cache_key in CACHED_MODELS:
        return CACHED_MODELS[cache_key]

    url = f"{GITHUB_RAW_URL}{asset}_{timeframe}.json"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            # OPTIMIZATION: Parse keys immediately
            for strat in data.get("strategy_union", []):
                params = strat['trained_parameters']
                
                # Convert abs_map keys
                new_abs = {}
                for k, v in params['abs_map'].items():
                    key_tuple = tuple(map(int, k.split('|')))
                    new_abs[key_tuple] = Counter({int(pred): freq for pred, freq in v.items()})
                params['abs_map'] = new_abs

                # Convert der_map keys
                new_der = {}
                for k, v in params['der_map'].items():
                    key_tuple = tuple(map(int, k.split('|')))
                    new_der[key_tuple] = Counter({int(pred): freq for pred, freq in v.items()})
                params['der_map'] = new_der
            
            CACHED_MODELS[cache_key] = data
            return data
    except Exception:
        pass
    return None

def preload_all_models():
    """ Fetches all models in parallel to speed up startup """
    print(">>> PRE-LOADING MODELS (Parallel fetch)...")
    tasks = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for asset in ASSETS:
            for tf in TIMEFRAMES:
                tasks.append(executor.submit(fetch_and_parse_model, asset, tf))
    # Wait for all
    for t in tasks: t.result()
    print(f">>> Cached {len(CACHED_MODELS)} models.")

# --- INFERENCE ENGINE (Optimized) ---

def get_prediction(strategies, prices):
    votes = []
    min_bucket = float('inf')
    
    # Pre-calc len to avoid repeated calls
    n_prices = len(prices)

    for strat in strategies:
        params = strat['trained_parameters']
        s_len = params['seq_len']
        if n_prices < s_len: continue

        b_size = params['bucket_size']
        if b_size < min_bucket: min_bucket = b_size
        
        # Slicing is fast in python
        seq_prices = prices[-s_len:]
        
        # Optimization: List comp is faster than loop
        buckets = [int(p // b_size) for p in seq_prices]
        
        a_seq = tuple(buckets)
        last_bucket = buckets[-1]
        
        # Maps are already parsed to tuples! Direct lookup.
        abs_map = params['abs_map']
        der_map = params['der_map']
        m_type = strat['config']['model_type']

        pred_bucket = None
        
        if m_type == "Absolute":
            if a_seq in abs_map:
                pred_bucket = abs_map[a_seq].most_common(1)[0][0]
        elif m_type == "Derivative":
            if s_len > 1:
                d_seq = tuple(buckets[k] - buckets[k-1] for k in range(1, len(buckets)))
                if d_seq in der_map:
                    pred_bucket = last_bucket + der_map[d_seq].most_common(1)[0][0]
        elif m_type == "Combined":
            # Only calc d_seq if needed
            d_seq = tuple(buckets[k] - buckets[k-1] for k in range(1, len(buckets))) if s_len > 1 else ()
            abs_cand = abs_map.get(a_seq, Counter())
            der_cand = der_map.get(d_seq, Counter())
            
            # Fast set union
            candidates = set(abs_cand).union({last_bucket + c for c in der_cand})
            if candidates:
                pred_bucket = max(candidates, key=lambda v: abs_cand[v] + der_cand[v - last_bucket])

        if pred_bucket is not None:
            diff = int(pred_bucket) - last_bucket
            if diff > 0: votes.append(1)
            elif diff < 0: votes.append(-1)
            
    if not votes: return 0, min_bucket
    
    # Fast majority vote
    s = sum(votes)
    if s > 0: return 1, min_bucket
    if s < 0: return -1, min_bucket
    return 0, min_bucket

def get_latest_signal(asset_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return 0, 0.0

    tf_name = model_payload['timeframe']
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[tf_name]
    
    df = pd.DataFrame(asset_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)
    prices_series = df['price'] if not pandas_tf else df['price'].resample(pandas_tf).last().dropna()
    full_prices = prices_series.tolist()
    
    if len(full_prices) > 1:
        completed_prices = full_prices[:-1]
    else:
        return 0, 0.0

    sig, min_bucket = get_prediction(strategies, completed_prices)
    return sig, min_bucket

def calculate_backtest_logic(asset_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return None, []
    
    asset_name = model_payload['asset']
    tf_name = model_payload['timeframe']
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[tf_name]

    df = pd.DataFrame(asset_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)
    prices_series = df['price'] if not pandas_tf else df['price'].resample(pandas_tf).last().dropna()
    full_prices = prices_series.tolist()
    timestamps = prices_series.index.tolist()
    
    if len(full_prices) < 20: return None, []
    
    test_window_size = min(len(full_prices) - 10, 7 * (24 if tf_name == "60m" else 96 if tf_name == "15m" else 1))
    if tf_name == "1d": test_window_size = min(len(full_prices) - 8, 30)
    if tf_name == "240m": test_window_size = min(len(full_prices) - 8, 42)

    stats = {"wins": 0, "losses": 0, "noise": 0, "total_pnl_pct": 0.0}
    trade_log = []
    
    # Pre-calculate min bucket size once
    min_b_size = min([s['trained_parameters']['bucket_size'] for s in strategies])
    start_idx = len(full_prices) - test_window_size

    for i in range(start_idx, len(full_prices) - 1):
        hist_prices = full_prices[:i+1]
        current_price = full_prices[i]
        
        signal, _ = get_prediction(strategies, hist_prices)
        if signal != 0:
            actual_next_price = full_prices[i+1]
            bucket_diff = int(actual_next_price // min_b_size) - int(current_price // min_b_size)
            
            # Fast PnL
            pct_change = ((actual_next_price - current_price) / current_price) * 100
            trade_pnl_pct = pct_change * signal
            
            outcome = "NOISE"
            if signal == 1:
                if bucket_diff > 0: outcome = "WIN"
                elif bucket_diff < 0: outcome = "LOSS"
            elif signal == -1:
                if bucket_diff < 0: outcome = "WIN"
                elif bucket_diff > 0: outcome = "LOSS"
            
            stats["total_pnl_pct"] += trade_pnl_pct
            if outcome == "WIN": stats["wins"] += 1
            elif outcome == "LOSS": stats["losses"] += 1
            else: stats["noise"] += 1
            
            trade_log.append({
                "time": timestamps[i], "asset": asset_name, "tf": tf_name, 
                "signal": "BUY" if signal == 1 else "SELL", "price": current_price, 
                "pnl": trade_pnl_pct, "outcome": outcome
            })
    return stats, trade_log

# --- EXECUTION FUNCTIONS ---

def process_single_asset_backtest(asset):
    """ Worker function to process one asset entirely (all timeframes) """
    raw_data = fetch_recent_binance_data(asset, days=30)
    if not raw_data: return [], 0, 0, 0, 0.0, 0
    
    asset_trades = []
    w, l, n, p, t = 0, 0, 0, 0.0, 0
    
    for tf in TIMEFRAMES:
        model = fetch_and_parse_model(asset, tf) # Fast lookup
        if model:
            stats, trades = calculate_backtest_logic(raw_data, model)
            if stats:
                asset_trades.extend(trades)
                w += stats['wins']
                l += stats['losses']
                n += stats['noise']
                p += stats['total_pnl_pct']
                t += (stats['wins'] + stats['losses'] + stats['noise'])
    
    return asset_trades, w, l, n, p, t

def backtest():
    """ Runs parallel backtest """
    global GLOBAL_TRADE_HISTORY # Access global variable
    
    print("\n" + "="*80)
    print(f"STARTING OPTIMIZED BACKTEST (Scanning {len(ASSETS)} Assets)")
    print("="*80)
    
    all_trades = []
    g_wins, g_losses, g_noise, g_pnl, g_total_trades = 0, 0, 0, 0.0, 0

    # Parallel Execution
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_single_asset_backtest, ASSETS)
        
        for res in results:
            trades, w, l, n, p, t = res
            all_trades.extend(trades)
            g_wins += w
            g_losses += l
            g_noise += n
            g_pnl += p
            g_total_trades += t

    # Save to Global for Endpoint
    all_trades.sort(key=lambda x: x['time'], reverse=True)
    GLOBAL_TRADE_HISTORY = all_trades

    # Results
    print("\n" + "="*50)
    print("BACKTEST RESULTS (Combined)")
    print("="*50)
    g_total_valid = g_wins + g_losses
    g_strict_acc = (g_wins / g_total_trades * 100) if g_total_trades > 0 else 0
    g_app_acc = (g_wins / g_total_valid * 100) if g_total_valid > 0 else 0
    
    print(f"Combined PnL    : {g_pnl:+.2f}%")
    print(f"Strict Accuracy : {g_strict_acc:.2f}% (Includes Noise)")
    print(f"App Accuracy    : {g_app_acc:.2f}% (Excludes Noise)")
    print(f"Total Trades    : {g_total_trades}")
    print("="*50)

    print("\n" + "="*80)
    print("FULL TRADE HISTORY")
    print("="*80)
    
    for t in all_trades[:50]:
        time_str = t['time'].strftime("%Y-%m-%d %H:%M")
        print(f"{time_str:<20} {t['asset']:<10} {t['tf']:<5} {t['signal']:<5} {t['price']:<12.4f} {t['pnl']:+.2f}% {t['outcome']}")

def process_single_asset_live(asset):
    """ Worker for live loop """
    raw_data = fetch_recent_binance_data(asset, days=30)
    if not raw_data: return asset, 0, [0] * len(TIMEFRAMES)
    
    total_score = 0
    comp_signals = []
    
    for tf in TIMEFRAMES:
        model = fetch_and_parse_model(asset, tf)
        if model:
            sig, _ = get_latest_signal(raw_data, model)
            total_score += sig
            comp_signals.append(sig)
        else:
            comp_signals.append(0)
            
    return asset, total_score, comp_signals

def run_live_loop():
    global LATEST_PREDICTIONS
    print(">>> LIVE STRATEGY ENGINE STARTED IN BACKGROUND")
    
    while True:
        temp_preds = {}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] REFRESHING LIVE SIGNALS (Parallel)...")
        update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Parallel Execution for Live Loop too
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(process_single_asset_live, ASSETS)
            for asset, score, comp in results:
                temp_preds[asset] = {
                    "sum": score,
                    "comp": comp,
                    "upd": update_time
                }

        LATEST_PREDICTIONS = temp_preds
        print(f"UPDATED PREDICTIONS: {json.dumps(LATEST_PREDICTIONS, indent=None)}") # compact print
        
        sleep_sec = get_sleep_time_to_next_candle("15m")
        print(f">>> SLEEPING {int(sleep_sec)}s")
        time.sleep(sleep_sec)

# --- LIGHTWEIGHT HTTP SERVER ---

class JSONRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/predictions':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') # CORS
            self.end_headers()
            self.wfile.write(json.dumps(LATEST_PREDICTIONS).encode('utf-8'))
            
        elif self.path == '/history':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*') # CORS
            self.end_headers()
            
            # Helper to serialize datetime objects in the trade list
            def date_handler(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return str(obj)
                
            self.wfile.write(json.dumps(GLOBAL_TRADE_HISTORY, default=date_handler).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return # Suppress default logging to keep console clean

def run_server():
    server_address = ('', HTTP_PORT)
    httpd = socketserver.TCPServer(server_address, JSONRequestHandler)
    print(f">>> JSON SERVER RUNNING ON PORT {HTTP_PORT}")
    httpd.serve_forever()

def main():
    # 1. Pre-load Models (Huge speed boost)
    preload_all_models()

    # 2. Run Backtest (Parallelized)
    backtest()
    
    # 3. Start Server Thread
    server_t = threading.Thread(target=run_server)
    server_t.daemon = True
    server_t.start()

    # 4. Start Live Loop (Main Thread)
    run_live_loop()

if __name__ == "__main__":
    main()
