import json
import time
import requests
import urllib.request
import threading
import http.server
import socketserver
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/"
BASE_INTERVAL = "15m"
LATENCY_BUFFER = 3
HTTP_PORT = 8080
BACKTEST_DAYS = 7  # <--- Controls how far back the history goes (Fixes "Starting a month early")

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT", "BNBUSDT"
]

TIMEFRAMES = ["15m", "30m", "60m", "240m", "1d"]

# GLOBAL STATE
LATEST_PREDICTIONS = {}
GLOBAL_TRADE_HISTORY = [] 
CACHED_MODELS = {} 

# --- UTILS ---

def get_sleep_time_to_next_candle(interval_str="15m"):
    """
    Calculates strict UTC sleep time to the start of the next candle + buffer.
    """
    now = datetime.now(timezone.utc)
    
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
    """
    Fetches data safely respecting Binance Rate Limits (prevents HTTP 429/403).
    """
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    
    data_points = []
    current_start = start_ts
    base_url = "https://api.binance.com/api/v3/klines"
    
    # 1. User-Agent to prevent generic 403 blocks
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    while current_start < end_ts:
        url = f"{base_url}?symbol={symbol}&interval={BASE_INTERVAL}&startTime={current_start}&endTime={end_ts}&limit=1000"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                batch = json.loads(response.read().decode())
                if not batch: break
                
                parsed_batch = [(int(c[0]), float(c[4])) for c in batch]
                data_points.extend(parsed_batch)
                
                last_close_time = batch[-1][0]
                current_start = last_close_time + 1
                
                if last_close_time >= end_ts - 1000: break
                
                # 2. CRITICAL: Sleep 0.2s between batches to respect IP weight limits
                time.sleep(0.2)
                
        except Exception as e:
            # Soft fail: print error but return what we have so far
            print(f"!! Error fetching {symbol}: {e}")
            time.sleep(1) # Backoff
            break
            
    unique_data = {x[0]: x[1] for x in data_points}
    return sorted([(k, v) for k, v in unique_data.items()])

def process_data_structure(raw_data, timeframe):
    """
    Standardizes data and removes the incomplete (forming) candle.
    """
    if not raw_data: return [], []
    
    df = pd.DataFrame(raw_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)
    
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[timeframe]
    
    if pandas_tf:
        prices_series = df['price'].resample(pandas_tf).last().dropna()
    else:
        prices_series = df['price']
        
    # Drop the last candle if it's the currently forming one
    if len(prices_series) > 1:
        valid_history = prices_series.iloc[:-1]
    else:
        valid_history = prices_series
        
    return valid_history.tolist(), valid_history.index.tolist()

# --- OPTIMIZED MODEL LOADING ---

def fetch_and_parse_model(asset, timeframe):
    cache_key = f"{asset}_{timeframe}"
    if cache_key in CACHED_MODELS:
        return CACHED_MODELS[cache_key]

    url = f"{GITHUB_RAW_URL}{asset}_{timeframe}.json"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Convert string keys back to tuples/ints for the strategy logic
            for strat in data.get("strategy_union", []):
                params = strat['trained_parameters']
                
                new_abs = {}
                for k, v in params['abs_map'].items():
                    key_tuple = tuple(map(int, k.split('|')))
                    new_abs[key_tuple] = Counter({int(pred): freq for pred, freq in v.items()})
                params['abs_map'] = new_abs

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
    print(">>> PRE-LOADING MODELS (Parallel fetch)...")
    tasks = []
    # Limit workers to avoid network congestion
    with ThreadPoolExecutor(max_workers=5) as executor:
        for asset in ASSETS:
            for tf in TIMEFRAMES:
                tasks.append(executor.submit(fetch_and_parse_model, asset, tf))
    for t in tasks: t.result()
    print(f">>> Cached {len(CACHED_MODELS)} models.")

# --- INFERENCE ENGINE ---

def get_prediction(strategies, prices):
    votes = []
    min_bucket = float('inf')
    n_prices = len(prices)

    for strat in strategies:
        params = strat['trained_parameters']
        s_len = params['seq_len']
        if n_prices < s_len: continue

        b_size = params['bucket_size']
        if b_size < min_bucket: min_bucket = b_size
        
        seq_prices = prices[-s_len:]
        buckets = [int(p // b_size) for p in seq_prices]
        
        a_seq = tuple(buckets)
        last_bucket = buckets[-1]
        
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
            d_seq = tuple(buckets[k] - buckets[k-1] for k in range(1, len(buckets))) if s_len > 1 else ()
            abs_cand = abs_map.get(a_seq, Counter())
            der_cand = der_map.get(d_seq, Counter())
            
            candidates = set(abs_cand).union({last_bucket + c for c in der_cand})
            if candidates:
                pred_bucket = max(candidates, key=lambda v: abs_cand[v] + der_cand[v - last_bucket])

        if pred_bucket is not None:
            diff = int(pred_bucket) - last_bucket
            if diff > 0: votes.append(1)
            elif diff < 0: votes.append(-1)
            
    if not votes: return 0, min_bucket
    
    s = sum(votes)
    if s > 0: return 1, min_bucket
    if s < 0: return -1, min_bucket
    return 0, min_bucket

def get_latest_signal(raw_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return 0, 0.0

    tf_name = model_payload['timeframe']
    full_prices, _ = process_data_structure(raw_data, tf_name)
    
    if len(full_prices) < 2: return 0, 0.0

    sig, min_bucket = get_prediction(strategies, full_prices)
    return sig, min_bucket

def calculate_backtest_logic(raw_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return None, []
    
    asset_name = model_payload['asset']
    tf_name = model_payload['timeframe']
    
    full_prices, timestamps = process_data_structure(raw_data, tf_name)
    
    if len(full_prices) < 20: return None, []
    
    # --- LOGIC FIX: UNIFIED WINDOW CALCULATION ---
    # Calculates window based on BACKTEST_DAYS for ALL timeframes
    # This prevents '1d' from stretching back 30 days while others are 7 days
    
    # 1. Map Timeframe to minutes
    tf_minutes = {
        "15m": 15, "30m": 30, "60m": 60, "240m": 240, "1d": 1440
    }.get(tf_name, 15)
    
    # 2. Calculate how many candles fit in BACKTEST_DAYS
    # e.g., 7 days * 24 hours * 60 mins / 15 mins = 672 candles
    total_minutes = BACKTEST_DAYS * 24 * 60
    required_candles = int(total_minutes // tf_minutes)
    
    # 3. Determine safe window size
    test_window_size = min(len(full_prices) - 10, required_candles)

    stats = {"wins": 0, "losses": 0, "noise": 0, "total_pnl_pct": 0.0}
    trade_log = []
    
    min_b_size = min([s['trained_parameters']['bucket_size'] for s in strategies])
    
    # Start Index
    start_idx = max(0, len(full_prices) - test_window_size)

    for i in range(start_idx, len(full_prices) - 1):
        hist_prices = full_prices[:i+1]
        current_price = full_prices[i]
        
        signal, _ = get_prediction(strategies, hist_prices)
        if signal != 0:
            actual_next_price = full_prices[i+1]
            bucket_diff = int(actual_next_price // min_b_size) - int(current_price // min_b_size)
            
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
                "time": timestamps[i], 
                "asset": asset_name, 
                "tf": tf_name, 
                "signal": "BUY" if signal == 1 else "SELL", 
                "price": current_price, 
                "pnl": trade_pnl_pct, 
                "outcome": outcome,
                "th": min_b_size
            })
    return stats, trade_log

# --- EXECUTION FUNCTIONS ---

def process_single_asset_backtest(asset):
    # Fetch enough data to cover BACKTEST_DAYS + buffer
    # We fetch a bit more (e.g. +5 days) to ensure we have the lead-in sequence
    fetch_days = BACKTEST_DAYS + 5
    raw_data = fetch_recent_binance_data(asset, days=fetch_days)
    
    if not raw_data: return [], 0, 0, 0, 0.0, 0
    
    asset_trades = []
    w, l, n, p, t = 0, 0, 0, 0.0, 0
    
    for tf in TIMEFRAMES:
        model = fetch_and_parse_model(asset, tf)
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
    global GLOBAL_TRADE_HISTORY
    
    print("\n" + "="*80)
    print(f"STARTING BACKTEST (Scanning {len(ASSETS)} Assets for last {BACKTEST_DAYS} Days)")
    print("="*80)
    
    all_trades = []
    g_wins, g_losses, g_noise, g_pnl, g_total_trades = 0, 0, 0, 0.0, 0

    # Limit workers to 5 to prevent HTTP 429
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_single_asset_backtest, ASSETS)
        
        for res in results:
            trades, w, l, n, p, t = res
            all_trades.extend(trades)
            g_wins += w
            g_losses += l
            g_noise += n
            g_pnl += p
            g_total_trades += t

    all_trades.sort(key=lambda x: x['time'], reverse=True)
    GLOBAL_TRADE_HISTORY = all_trades

    print("\n" + "="*50)
    print("BACKTEST RESULTS (Combined)")
    print("="*50)
    g_total_valid = g_wins + g_losses
    g_app_acc = (g_wins / g_total_valid * 100) if g_total_valid > 0 else 0
    
    print(f"Combined PnL    : {g_pnl:+.2f}%")
    print(f"App Accuracy    : {g_app_acc:.2f}% (Excludes Noise)")
    print(f"Total Trades    : {g_total_trades}")
    print("="*50)

def process_single_asset_live(asset):
    # Only need very recent data for live signals
    raw_data = fetch_recent_binance_data(asset, days=3) 
    if not raw_data: return asset, 0, [0] * len(TIMEFRAMES), [0.0] * len(TIMEFRAMES)
    
    total_score = 0
    comp_signals = []
    comp_th = []
    
    for tf in TIMEFRAMES:
        model = fetch_and_parse_model(asset, tf)
        if model:
            sig, bucket = get_latest_signal(raw_data, model)
            total_score += sig
            comp_signals.append(sig)
            comp_th.append(bucket)
        else:
            comp_signals.append(0)
            comp_th.append(0.0)
            
    return asset, total_score, comp_signals, comp_th

def run_live_loop():
    global LATEST_PREDICTIONS
    print(">>> LIVE STRATEGY ENGINE STARTED IN BACKGROUND (UTC Mode)")
    
    while True:
        sleep_sec = get_sleep_time_to_next_candle("15m")
        print(f">>> SLEEPING {int(sleep_sec)}s")
        time.sleep(sleep_sec)

        temp_preds = {}
        print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] REFRESHING LIVE SIGNALS...")
        update_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(process_single_asset_live, ASSETS)
            for asset, score, comp, th in results:
                temp_preds[asset] = {
                    "sum": score,
                    "comp": comp,
                    "th": th,
                    "upd": update_time
                }

        LATEST_PREDICTIONS = temp_preds
        print(f"UPDATED PREDICTIONS: {json.dumps(LATEST_PREDICTIONS)}") 

# --- SERVER & AUTO-UPDATER ---

class JSONRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/predictions':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(LATEST_PREDICTIONS).encode('utf-8'))
            
        elif self.path == '/history':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            def date_handler(obj):
                return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
            self.wfile.write(json.dumps(GLOBAL_TRADE_HISTORY, default=date_handler).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        return

def run_server():
    server_address = ('', HTTP_PORT)
    httpd = socketserver.TCPServer(server_address, JSONRequestHandler)
    print(f">>> JSON SERVER RUNNING ON PORT {HTTP_PORT}")
    httpd.serve_forever()

def run_history_updater():
    print(">>> HISTORY AUTO-UPDATER STARTED")
    while True:
        # Sync to run at :05, :20, :35, :50
        now = datetime.now(timezone.utc)
        minutes = now.minute
        targets = [5, 20, 35, 50]
        next_min = next((t for t in targets if minutes < t), 5)
        
        target_time = now.replace(minute=next_min, second=30, microsecond=0)
        if minutes >= 50 and next_min == 5:
            target_time += timedelta(hours=1)
            
        sleep_sec = (target_time - now).total_seconds()
        if sleep_sec < 0: sleep_sec = 10
        
        time.sleep(sleep_sec)
        print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] REFRESHING HISTORY...")
        backtest()

def main():
    preload_all_models()
    backtest()
    
    server_t = threading.Thread(target=run_server)
    server_t.daemon = True
    server_t.start()

    hist_t = threading.Thread(target=run_history_updater)
    hist_t.daemon = True
    hist_t.start()

    run_live_loop()

if __name__ == "__main__":
    main()
