import json
import time
import requests
import urllib.request
import threading
from flask import Flask, jsonify
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/"
BASE_INTERVAL = "15m"
LATENCY_BUFFER = 2 

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT", "BNBUSDT"
]

TIMEFRAMES = ["15m", "30m", "60m", "240m", "1d"]

# Global Storage for Flask to read
LATEST_PREDICTIONS = {}

app = Flask(__name__)

# --- UTILS ---

def delayed_print(msg):
    print(msg)
    time.sleep(0.01)

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
            delayed_print(f"Error fetching Binance data for {symbol}: {e}")
            break
    unique_data = {x[0]: x[1] for x in data_points}
    return sorted([(k, v) for k, v in unique_data.items()])

def get_model_from_github(asset, timeframe):
    url = f"{GITHUB_RAW_URL}{asset}_{timeframe}.json"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass 
    return None

# --- INFERENCE ENGINE ---

def get_prediction(strategies, prices):
    votes = []
    min_bucket = float('inf')

    for strat in strategies:
        params = strat['trained_parameters']
        b_size = params['bucket_size']
        if b_size < min_bucket: min_bucket = b_size
        s_len = params['seq_len']
        m_type = strat['config']['model_type']
        
        if len(prices) < s_len: continue
        
        seq_prices = prices[-s_len:]
        buckets = [get_bucket(p, b_size) for p in seq_prices]
        a_seq = tuple(buckets)
        d_seq = tuple(buckets[k] - buckets[k-1] for k in range(1, len(a_seq))) if s_len > 1 else ()
        last_bucket = buckets[-1]
        
        abs_map = {}
        for k, v in params['abs_map'].items():
            key_tuple = tuple(map(int, k.split('|')))
            abs_map[key_tuple] = Counter({int(pred): freq for pred, freq in v.items()})

        der_map = {}
        for k, v in params['der_map'].items():
            key_tuple = tuple(map(int, k.split('|')))
            der_map[key_tuple] = Counter({int(pred): freq for pred, freq in v.items()})

        pred_bucket = None
        if m_type == "Absolute" and a_seq in abs_map:
            pred_bucket = abs_map[a_seq].most_common(1)[0][0]
        elif m_type == "Derivative" and d_seq in der_map:
            pred_bucket = last_bucket + der_map[d_seq].most_common(1)[0][0]
        elif m_type == "Combined":
            abs_cand = abs_map.get(a_seq, Counter())
            der_cand = der_map.get(d_seq, Counter())
            candidates = set(abs_cand.keys()).union({last_bucket + c for c in der_cand.keys()})
            if candidates:
                pred_bucket = max(candidates, key=lambda v: abs_cand[v] + der_cand[v - last_bucket])

        if pred_bucket is not None:
            pred_diff = int(pred_bucket) - last_bucket
            if pred_diff > 0: votes.append(1)
            elif pred_diff < 0: votes.append(-1)
            
    if not votes: return 0, min_bucket
    
    up = votes.count(1)
    down = votes.count(-1)
    final_sig = 0
    if up > down: final_sig = 1
    elif down > up: final_sig = -1
    return final_sig, min_bucket

def get_latest_signal(asset_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return "WAIT", 0.0

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
        return "WAIT", 0.0

    sig, min_bucket = get_prediction(strategies, completed_prices)
    if sig == 1: return 1, min_bucket # Return int directly for summing
    if sig == -1: return -1, min_bucket
    return 0, min_bucket

def calculate_backtest_logic(asset_data, model_payload):
    # (Stripped down backtest helper logic if you ever need to re-enable detailed history logging)
    # Re-using the logic from previous steps to maintain compatibility
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
    min_b_size = min([s['trained_parameters']['bucket_size'] for s in strategies])
    start_idx = len(full_prices) - test_window_size

    for i in range(start_idx, len(full_prices) - 1):
        hist_prices = full_prices[:i+1]
        current_price = full_prices[i]
        current_time = timestamps[i]
        signal, _ = get_prediction(strategies, hist_prices)
        if signal != 0:
            actual_next_price = full_prices[i+1]
            bucket_diff = get_bucket(actual_next_price, min_b_size) - get_bucket(current_price, min_b_size)
            trade_pnl_pct = (((actual_next_price - current_price) / current_price) * 100) * signal
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
                "time": current_time, "asset": asset_name, "tf": tf_name, 
                "signal": "BUY" if signal == 1 else "SELL", "price": current_price, 
                "pnl": trade_pnl_pct, "outcome": outcome
            })
    return stats, trade_log

# --- EXECUTION FUNCTIONS ---

def backtest():
    """ Runs once at startup """
    print("\n" + "="*80)
    print(f"STARTING FULL BACKTEST (Scanning {len(ASSETS)} Assets)")
    print("="*80)
    
    all_trades = []
    for asset in ASSETS:
        raw_data = fetch_recent_binance_data(asset, days=30)
        if not raw_data: continue
        for tf in TIMEFRAMES:
            model = get_model_from_github(asset, tf)
            if model:
                stats, trades = calculate_backtest_logic(raw_data, model)
                if stats: all_trades.extend(trades)

    print("\n" + "="*80)
    print("FULL TRADE HISTORY")
    print("="*80)
    all_trades.sort(key=lambda x: x['time'], reverse=True)
    for t in all_trades[:50]:
        time_str = t['time'].strftime("%Y-%m-%d %H:%M")
        print(f"{time_str:<20} {t['asset']:<10} {t['tf']:<5} {t['signal']:<5} {t['price']:<12.4f} {t['pnl']:+.2f}% {t['outcome']}")

def run_live_loop():
    """ Infinite loop running in a background thread """
    global LATEST_PREDICTIONS
    print(">>> LIVE STRATEGY ENGINE STARTED IN BACKGROUND")
    
    while True:
        temp_predictions = {}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] REFRESHING LIVE SIGNALS...")
        
        for asset in ASSETS:
            raw_data = fetch_recent_binance_data(asset, days=30)
            if not raw_data: continue
            
            asset_total_score = 0
            # signal_breakdown = {}  # Optional: if you want details per timeframe

            for tf in TIMEFRAMES:
                model = get_model_from_github(asset, tf)
                if model:
                    sig_val, _ = get_latest_signal(raw_data, model)
                    asset_total_score += sig_val
                    # signal_breakdown[tf] = sig_val 

            # Aggregated result per asset
            temp_predictions[asset] = asset_total_score

        # Update global state atomically (Python dict assignment is atomic)
        LATEST_PREDICTIONS = temp_predictions
        print(f"UPDATED PREDICTIONS: {json.dumps(LATEST_PREDICTIONS, indent=2)}")
        
        sleep_sec = get_sleep_time_to_next_candle("15m")
        print(f">>> SLEEPING {int(sleep_sec)}s")
        time.sleep(sleep_sec)

# --- FLASK ROUTE ---

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """ Serves the aggregated sums as JSON """
    return jsonify(LATEST_PREDICTIONS)

def main():
    # 1. Run Backtest
    backtest()
    
    # 2. Start Live Loop in Background Thread
    t = threading.Thread(target=run_live_loop)
    t.daemon = True
    t.start()
    
    # 3. Start Flask Server (Blocking)
    print(">>> STARTING WEB SERVER ON PORT 8080...")
    app.run(host='0.0.0.0', port=8080)

if __name__ == "__main__":
    main()
