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
import traceback

# --- CONFIGURATION ---
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/"
BASE_INTERVAL = "15m"
LATENCY_BUFFER = 5
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
GLOBAL_TRADE_HISTORY = [] 
CACHED_MODELS = {} 

# --- UTILS ---

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
    
    for attempt in range(3):
        try:
            while current_start < end_ts:
                url = f"{base_url}?symbol={symbol}&interval={BASE_INTERVAL}&startTime={current_start}&endTime={end_ts}&limit=1000"
                with urllib.request.urlopen(url, timeout=10) as response:
                    batch = json.loads(response.read().decode())
                    if not batch: break
                    # Use OpenTime (c[0]) for index
                    parsed_batch = [(int(c[0]), float(c[4])) for c in batch]
                    data_points.extend(parsed_batch)
                    last_close_time = batch[-1][6] 
                    current_start = last_close_time + 1
                    if last_close_time >= end_ts - 1000: break
            break 
        except Exception as e:
            if attempt == 2:
                print(f"Failed to fetch data for {symbol}: {e}")
                return []
            time.sleep(1)
            
    unique_data = {x[0]: x[1] for x in data_points}
    return sorted([(k, v) for k, v in unique_data.items()])

# --- OPTIMIZED MODEL LOADING ---

def fetch_and_parse_model(asset, timeframe):
    cache_key = f"{asset}_{timeframe}"
    if cache_key in CACHED_MODELS:
        return CACHED_MODELS[cache_key]

    url = f"{GITHUB_RAW_URL}{asset}_{timeframe}.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
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
    with ThreadPoolExecutor(max_workers=20) as executor:
        for asset in ASSETS:
            for tf in TIMEFRAMES:
                tasks.append(executor.submit(fetch_and_parse_model, asset, tf))
    for t in tasks: 
        try: t.result()
        except: pass
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

def get_latest_signal(asset_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return 0, 0.0

    asset_name = model_payload['asset'] # Get name for debug
    tf_name = model_payload['timeframe']
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[tf_name]
    
    df = pd.DataFrame(asset_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)
    
    # Resample
    if pandas_tf:
        resampled = df['price'].resample(pandas_tf).last().dropna()
        prices_series = resampled
    else:
        prices_series = df['price']

    full_prices = prices_series.tolist()
    full_timestamps = prices_series.index.tolist()
    
    # LIVE: Strip last incomplete candle
    if len(full_prices) > 1:
        completed_prices = full_prices[:-1]
        completed_timestamps = full_timestamps[:-1]
        
        # --- DEBUG PRINT FOR BTCUSDT ---
        if asset_name == "BTCUSDT" and tf_name == "15m":
            print(f"\n[DEBUG LIVE] {asset_name} {tf_name} Last 5 Candles Used:")
            for t, p in zip(completed_timestamps[-5:], completed_prices[-5:]):
                print(f"   Time: {t} | Price: {p}")
        # -------------------------------
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

    # --- INTELLIGENT STRIP LOGIC ---
    if len(timestamps) > 0:
        last_ts = timestamps[-1]
        
        if tf_name == "15m": span = timedelta(minutes=15)
        elif tf_name == "30m": span = timedelta(minutes=30)
        elif tf_name == "60m": span = timedelta(hours=1)
        elif tf_name == "240m": span = timedelta(hours=4)
        elif tf_name == "1d": span = timedelta(days=1)
        else: span = timedelta(minutes=15)
        
        candle_close_time = last_ts + span
        
        # Strip if forming
        if candle_close_time > datetime.now() - timedelta(seconds=10):
            full_prices = full_prices[:-1]
            timestamps = timestamps[:-1]

    # --- DEBUG PRINT FOR BTCUSDT ---
    if asset_name == "BTCUSDT" and tf_name == "15m":
        print(f"\n[DEBUG BACKTEST] {asset_name} {tf_name} Last 5 Candles Used:")
        for t, p in zip(timestamps[-5:], full_prices[-5:]):
            print(f"   Time: {t} | Price: {p}")
    # -------------------------------
    
    if len(full_prices) < 20: return None, []
    
    test_window_size = min(len(full_prices) - 10, 7 * (24 if tf_name == "60m" else 96 if tf_name == "15m" else 1))
    if tf_name == "1d": test_window_size = min(len(full_prices) - 8, 30)
    if tf_name == "240m": test_window_size = min(len(full_prices) - 8, 42)

    stats = {"wins": 0, "losses": 0, "noise": 0, "total_pnl_pct": 0.0}
    trade_log = []
    
    min_b_size = min([s['trained_parameters']['bucket_size'] for s in strategies])
    start_idx = len(full_prices) - test_window_size

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
                "time": timestamps[i], "asset": asset_name, "tf": tf_name, 
                "signal": "BUY" if signal == 1 else "SELL", "price": current_price, 
                "pnl": trade_pnl_pct, "outcome": outcome
            })
    return stats, trade_log

# --- EXECUTION FUNCTIONS ---

def process_single_asset_backtest(asset):
    try:
        raw_data = fetch_recent_binance_data(asset, days=30)
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
    except Exception as e:
        print(f"!!! Error backtesting {asset}: {e}")
        return [], 0, 0, 0, 0.0, 0

def backtest():
    global GLOBAL_TRADE_HISTORY
    
    print("\n" + "="*80)
    print(f"STARTING OPTIMIZED BACKTEST (Scanning {len(ASSETS)} Assets)")
    print("="*80)
    
    all_trades = []
    g_wins, g_losses, g_noise, g_pnl, g_total_trades = 0, 0, 0, 0.0, 0

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

    all_trades.sort(key=lambda x: x['time'], reverse=True)
    GLOBAL_TRADE_HISTORY = all_trades

    g_total_valid = g_wins + g_losses
    g_strict_acc = (g_wins / g_total_trades * 100) if g_total_trades > 0 else 0
    print(f"BACKTEST DONE | PnL: {g_pnl:+.2f}% | WinRate: {g_strict_acc:.2f}% | Trades: {g_total_trades}")

def process_single_asset_live(asset):
    try:
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
    except Exception:
        return asset, 0, [0] * len(TIMEFRAMES)

def run_live_loop():
    global LATEST_PREDICTIONS
    print(">>> LIVE STRATEGY ENGINE STARTED IN BACKGROUND")
    
    while True:
        try:
            temp_preds = {}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] REFRESHING LIVE SIGNALS (Parallel)...")
            update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = executor.map(process_single_asset_live, ASSETS)
                for asset, score, comp in results:
                    temp_preds[asset] = {
                        "sum": score,
                        "comp": comp,
                        "upd": update_time
                    }

            LATEST_PREDICTIONS = temp_preds
            print(">>> LIVE SIGNALS UPDATED")
            
            sleep_sec = get_sleep_time_to_next_candle("15m")
            print(f">>> NEXT CYCLE IN {int(sleep_sec)}s")
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"CRITICAL ERROR IN LIVE LOOP: {e}")
            time.sleep(60)

# --- LIGHTWEIGHT HTTP SERVER ---

class JSONRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.end_headers()
        
        if self.path == '/predictions':
            self.wfile.write(json.dumps(LATEST_PREDICTIONS).encode('utf-8'))
            
        elif self.path == '/history':
            def date_handler(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return str(obj)
            self.wfile.write(json.dumps(GLOBAL_TRADE_HISTORY, default=date_handler).encode('utf-8'))
            
        else:
            self.wfile.write(b'{}')
    
    def log_message(self, format, *args):
        return 

def run_server():
    server_address = ('', HTTP_PORT)
    httpd = socketserver.TCPServer(server_address, JSONRequestHandler)
    print(f">>> JSON SERVER RUNNING ON PORT {HTTP_PORT}")
    httpd.serve_forever()

def run_history_updater():
    print(">>> HISTORY AUTO-UPDATER STARTED (Sync :01)")
    while True:
        try:
            now = datetime.now()
            targets = [1, 16, 31, 46]
            next_minute = 1
            target_time = None
            
            for t in targets:
                if now.minute < t:
                    next_minute = t
                    target_time = now.replace(minute=next_minute, second=0, microsecond=0)
                    break
            
            if target_time is None:
                target_time = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
                
            sleep_sec = (target_time - now).total_seconds()
            if sleep_sec < 0: sleep_sec = 0
            
            time.sleep(sleep_sec)
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] REFRESHING HISTORY...")
            backtest()
        except Exception as e:
            print(f"CRITICAL ERROR IN HISTORY UPDATER: {e}")
            traceback.print_exc()
            time.sleep(60)

def main():
    preload_all_models()
    try: backtest()
    except Exception as e: print(f"Initial backtest failed: {e}")

    server_t = threading.Thread(target=run_server)
    server_t.daemon = True
    server_t.start()

    hist_t = threading.Thread(target=run_history_updater)
    hist_t.daemon = True
    hist_t.start()

    run_live_loop()

if __name__ == "__main__":
    main()
