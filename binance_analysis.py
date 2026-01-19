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
    """Custom print with a slight delay for readability."""
    print(msg)
    time.sleep(0.5)

def get_bucket(price, bucket_size):
    return int(price // bucket_size)

def fetch_recent_binance_data(symbol, days=20):
    """
    Fetches data using CLOSE TIME to match the Web App logic.
    Increased days default to 20 to ensure enough history for resampling.
    """
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={start_ts}&endTime={end_ts}&limit=1000"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            # FIX 1: Use c[6] (Close Time) instead of c[0] (Open Time)
            # This aligns the backtest with the live scheduler logic
            return [(int(c[6]), float(c[4])) for c in data]
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
    
    # Resample logic identical to app
    if pandas_tf:
        prices_series = df['price'].resample(pandas_tf).last().dropna()
    else:
        prices_series = df['price']

    full_prices = prices_series.tolist()
    
    # Backtest Window: Last 7 days approx
    test_window_size = min(len(full_prices) - 10, 7 * (24 if tf_name == "60m" else 96 if tf_name == "15m" else 1))
    
    # Stats
    stats = {
        "wins": 0,
        "losses": 0,
        "noise": 0,
        "total_pnl_pct": 0.0,
        "total_pnl_buckets": 0.0
    }

    delayed_print(f"\n>>> BACKTESTING {model_payload['asset']} [{tf_name}]")

    for strat in strategies:
        params = strat['trained_parameters']
        b_size = params['bucket_size']
        s_len = params['seq_len']
        m_type = strat['config']['model_type']
        
        # Parse Maps
        abs_map = {}
        for k, v in params['abs_map'].items():
            key_tuple = tuple(map(int, k.split('|')))
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
            current_price = full_prices[i]
            
            # --- PREDICTION LOGIC ---
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
                pred_val = int(pred_bucket)
                pred_diff = pred_val - last_bucket
                
                # Signal Generation
                if pred_diff != 0:
                    direction = 1 if pred_diff > 0 else -1
                    
                    # Actual Outcome
                    actual_next_price = full_prices[i+1]
                    actual_next_bucket = get_bucket(actual_next_price, b_size)
                    actual_diff = actual_next_bucket - last_bucket
                    
                    # FIX 2: Correct PnL Calculation (Multiply by direction)
                    # Web App uses percentage PnL
                    pct_change = ((actual_next_price - current_price) / current_price) * 100
                    trade_pnl_pct = pct_change * direction
                    
                    # Bucket PnL (Legacy metric)
                    bucket_pnl = (actual_next_price - current_price) / b_size * direction

                    # FIX 3: Noise Filtering Logic
                    if actual_diff == 0:
                        stats["noise"] += 1
                        # Noise does NOT count towards PnL in the Web App metrics usually,
                        # but technically the money is invested.
                        # We will track PnL but exclude from "Forgiving Accuracy"
                        stats["total_pnl_pct"] += trade_pnl_pct
                    elif (direction == 1 and actual_diff > 0) or (direction == -1 and actual_diff < 0):
                        stats["wins"] += 1
                        stats["total_pnl_pct"] += trade_pnl_pct
                        stats["total_pnl_buckets"] += bucket_pnl
                    else:
                        stats["losses"] += 1
                        stats["total_pnl_pct"] += trade_pnl_pct
                        stats["total_pnl_buckets"] += bucket_pnl

    return stats

def main():
    # Expanded list to include assets that are actually trading in your logs
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "ZECUSDT", "TONUSDT", "XLMUSDT"] 
    timeframes = ["15m", "60m"]

    for asset in assets:
        raw_data = fetch_recent_binance_data(asset)
        if not raw_data: continue
        
        for tf in timeframes:
            model = get_model_from_github(asset, tf)
            if model:
                stats = run_backtest(raw_data, model)
                if not stats: continue
                
                wins = stats['wins']
                losses = stats['losses']
                noise = stats['noise']
                total_valid = wins + losses
                total_all = total_valid + noise
                
                # Metric 1: Strict Accuracy (Includes Noise as Failure)
                strict_acc = (wins / total_all * 100) if total_all > 0 else 0
                
                # Metric 2: Forgiving Accuracy (Matches Web App "Noise Filtering")
                forgiving_acc = (wins / total_valid * 100) if total_valid > 0 else 0
                
                if total_all > 0:
                    print(f"--- RESULTS {asset} {tf} ---")
                    print(f"Strict Acc   : {strict_acc:.2f}%  (Wins: {wins} | Loss: {losses} | Noise: {noise})")
                    print(f"Forgiving Acc: {forgiving_acc:.2f}%  (Matches Web Dashboard)")
                    print(f"Total PnL %  : {stats['total_pnl_pct']:.2f}%")
                    print("-" * 40)

if __name__ == "__main__":
    main()
