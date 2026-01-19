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
DATA_DIR = "backtest_data"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- UTILS ---

def delayed_print(msg):
    """Custom print with a 1-second delay as requested."""
    print(msg)
    time.sleep(1)

def get_bucket(price, bucket_size):
    return int(price // bucket_size)

def fetch_recent_binance_data(symbol, days=14):
    """Fetches enough data to cover the last week plus padding for seq_len."""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={start_ts}&endTime={end_ts}&limit=1000"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            # Return list of (timestamp, close_price)
            return [(int(c[0]), float(c[4])) for c in data]
    except Exception as e:
        delayed_print(f"Error fetching Binance data for {symbol}: {e}")
        return []

def get_model_from_github(asset, timeframe):
    filename = f"{asset}_{timeframe}.json"
    url = GITHUB_RAW_URL + filename
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

# --- BACKTEST ENGINE ---

def run_backtest(asset_data, model_payload):
    """
    Backtests the 'Strategy Union' (top 5 configs) against the last 7 days.
    """
    strategies = model_payload.get("strategy_union", [])
    if not strategies:
        return None

    # Filter data for the last 7 days
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    
    # We need a dataframe to handle resampling
    df = pd.DataFrame(asset_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)

    # Dictionary to store results per timeframe
    tf_name = model_payload['timeframe']
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[tf_name]
    
    if pandas_tf:
        prices_series = df['price'].resample(pandas_tf).last().dropna()
    else:
        prices_series = df['price']

    test_prices = prices_series[prices_series.index >= seven_days_ago].tolist()
    # We also need leading prices for the sequence buffer
    full_prices = prices_series.tolist()
    
    total_trades = 0
    correct_trades = 0
    total_pnl = 0.0

    delayed_print(f"--- Backtesting {model_payload['asset']} [{tf_name}] ---")
    
    for idx, strat in enumerate(strategies):
        params = strat['trained_parameters']
        b_size = params['bucket_size']
        s_len = params['seq_len']
        m_type = strat['config']['model_type']
        
        # Reconstruct maps from serialized strings
        abs_map = {tuple(map(int, k.split('|'))): Counter(v) for k, v in params['abs_map'].items()}
        der_map = {tuple(map(int, k.split('|'))): Counter(v) for k, v in params['der_map'].items()}

        delayed_print(f"  Model {idx+1}: {m_type} | BucketSize: {b_size:.6f} | SeqLen: {s_len}")

        # Slide through the test window
        # We start looking back from the first 'test_price'
        start_offset = len(full_prices) - len(test_prices)
        
        for i in range(start_offset, len(full_prices) - 1):
            seq_prices = full_prices[i - s_len + 1 : i + 1]
            if len(seq_prices) < s_len: continue
            
            buckets = [get_bucket(p, b_size) for p in seq_prices]
            a_seq = tuple(buckets)
            d_seq = tuple(buckets[k] - buckets[k-1] for k in range(1, len(buckets)))
            
            last_val = buckets[-1]
            actual_next_val = get_bucket(full_prices[i+1], b_size)
            actual_diff = actual_next_val - last_val
            
            # Prediction Logic
            pred_val = None
            if m_type == "Absolute":
                if a_seq in abs_map: pred_val = abs_map[a_seq].most_common(1)[0][0]
            elif m_type == "Derivative":
                if d_seq in der_map: pred_val = last_val + der_map[d_seq].most_common(1)[0][0]
            elif m_type == "Combined":
                # Simplified Combined logic
                abs_cand = abs_map.get(a_seq, Counter())
                der_cand = der_map.get(d_seq, Counter())
                all_c = set(abs_cand.keys()).union({last_val + c for c in der_cand.keys()})
                if all_c:
                    pred_val = max(all_c, key=lambda v: abs_cand[v] + der_cand[v - last_val])

            if pred_val is not None:
                pred_diff = pred_val - last_val
                if pred_diff != 0:
                    total_trades += 1
                    # PnL in terms of bucket moves
                    # (actual price change / bucket size)
                    trade_pnl = (full_prices[i+1] - full_prices[i]) / b_size
                    total_pnl += trade_pnl
                    
                    if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                        correct_trades += 1

    accuracy = (correct_trades / total_trades * 100) if total_trades > 0 else 0
    return accuracy, total_pnl, total_trades

def main():
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] # Add others from your ASSETS list
    timeframes = ["15m", "1h"]

    for asset in assets:
        raw_data = fetch_recent_binance_data(asset)
        if not raw_data: continue
        
        for tf in timeframes:
            model_payload = get_model_from_github(asset, tf)
            if not model_payload:
                delayed_print(f"No model found for {asset} {tf} on GitHub.")
                continue
            
            results = run_backtest(raw_data, model_payload)
            if results:
                acc, pnl, count = results
                delayed_print(f"RESULT [{asset} {tf}]:")
                delayed_print(f" >> Last Week Accuracy: {acc:.2f}%")
                delayed_print(f" >> Last Week PnL (Buckets): {pnl:.2f}")
                delayed_print(f" >> Total Trades: {count}")
                delayed_print("-" * 30)

if __name__ == "__main__":
    main()
