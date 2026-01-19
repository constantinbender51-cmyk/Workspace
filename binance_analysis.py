import os
import json
import time
import requests
import urllib.request
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter

# --- CONFIGURATION ---
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/"
BASE_INTERVAL = "15m"

# --- UTILS ---

def delayed_print(msg):
    """Custom print with a 1-second delay for readability."""
    print(msg)
    time.sleep(1)

def get_bucket(price, bucket_size):
    return int(price // bucket_size)

def fetch_recent_binance_data(symbol, days=9):
    """Fetches enough data for a 7-day backtest plus leading sequences."""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={start_ts}&endTime={end_ts}&limit=1000"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            return [(int(c[0]), float(c[4])) for c in data]
    except Exception as e:
        delayed_print(f"Error fetching Binance data for {symbol}: {e}")
        return []

def get_model_from_github(asset, timeframe):
    url = f"{GITHUB_RAW_URL}{asset}_{timeframe}.json"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        delayed_print(f"Could not load model {asset}_{timeframe}: {e}")
    return None

# --- BACKTEST ENGINE ---

def run_backtest(asset_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies:
        return None

    tf_name = model_payload['timeframe']
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[tf_name]
    
    df = pd.DataFrame(asset_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)
    
    if pandas_tf:
        prices_series = df['price'].resample(pandas_tf).last().dropna()
    else:
        prices_series = df['price']

    full_prices = prices_series.tolist()
    # Target approximately the last 7 days
    test_window_size = min(len(full_prices) - 10, 7 * (24 if tf_name == "60m" else 96 if tf_name == "15m" else 1))
    
    total_trades = 0
    correct_trades = 0
    total_pnl_buckets = 0.0

    delayed_print(f"\n>>> BACKTESTING {model_payload['asset']} [{tf_name}]")

    for strat in strategies:
        params = strat['trained_parameters']
        b_size = params['bucket_size']
        s_len = params['seq_len']
        m_type = strat['config']['model_type']
        
        delayed_print(f"Model: {m_type} | BucketSize: {b_size:.4f} | SeqLen: {s_len}")

        # --- FIX: Explicitly cast keys AND values to integers ---
        abs_map = {}
        for k, v in params['abs_map'].items():
            key_tuple = tuple(map(int, k.split('|')))
            # Cast the prediction key (bucket) to int, keep frequency as int
            abs_map[key_tuple] = Counter({int(pred): freq for pred, freq in v.items()})

        der_map = {}
        for k, v in params['der_map'].items():
            key_tuple = tuple(map(int, k.split('|')))
            der_map[key_tuple] = Counter({int(pred): freq for pred, freq in v.items()})

        start_idx = len(full_prices) - test_window_size
        for i in range(start_idx, len(full_prices) - 1):
            seq_prices = full_prices[i - s_len + 1 : i + 1]
            if len(seq_prices) < s_len: continue
            
            buckets = [get_bucket(p, b_size) for p in seq_prices]
            a_seq = tuple(buckets)
            d_seq = tuple(buckets[k] - buckets[k-1] for k in range(1, len(buckets)))
            
            last_bucket = buckets[-1]
            actual_next_price = full_prices[i+1]
            actual_next_bucket = get_bucket(actual_next_price, b_size)
            actual_diff = actual_next_bucket - last_bucket
            
            pred_bucket = None
            if m_type == "Absolute" and a_seq in abs_map:
                pred_bucket = abs_map[a_seq].most_common(1)[0][0]
            elif m_type == "Derivative" and d_seq in der_map:
                # der_map stores the CHANGE, so we add it to last_bucket
                pred_bucket = last_bucket + der_map[d_seq].most_common(1)[0][0]
            elif m_type == "Combined":
                abs_cand = abs_map.get(a_seq, Counter())
                der_cand = der_map.get(d_seq, Counter())
                candidates = set(abs_cand.keys()).union({last_bucket + c for c in der_cand.keys()})
                if candidates:
                    # Score based on frequencies in both maps
                    pred_bucket = max(candidates, key=lambda v: abs_cand[v] + der_cand[v - last_bucket])

            # Final safety check: ensure pred_bucket is int before math
            if pred_bucket is not None:
                pred_diff = int(pred_bucket) - last_bucket
                if pred_diff != 0:
                    total_trades += 1
                    # PnL in normalized bucket units
                    trade_pnl = (actual_next_price - full_prices[i]) / b_size
                    total_pnl_buckets += trade_pnl
                    
                    if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                        correct_trades += 1

    accuracy = (correct_trades / total_trades * 100) if total_trades > 0 else 0
    return accuracy, total_pnl_buckets, total_trades

def main():
    # Assets from your deployment list
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"] 
    timeframes = ["15m", "60m"]

    for asset in assets:
        raw_data = fetch_recent_binance_data(asset)
        if not raw_data: continue
        
        for tf in timeframes:
            model = get_model_from_github(asset, tf)
            if model:
                acc, pnl, count = run_backtest(raw_data, model)
                delayed_print(f"--- RESULTS {asset} {tf} ---")
                delayed_print(f"Accuracy: {acc:.2f}%")
                delayed_print(f"PnL (Last Week): {pnl:.2f} buckets")
                delayed_print(f"Total Trades: {count}")
                delayed_print("-" * 40)

if __name__ == "__main__":
    main()
