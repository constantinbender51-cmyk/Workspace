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

# Full Asset List from app.py
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT", "BNBUSDT"
]

# --- UTILS ---

def delayed_print(msg):
    print(msg)
    time.sleep(0.1)

def get_bucket(price, bucket_size):
    return int(price // bucket_size)

def fetch_recent_binance_data(symbol, days=14):
    """
    Fetches data using CLOSE TIME (c[6]) to match Web App logic.
    """
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={start_ts}&endTime={end_ts}&limit=1000"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            # Use c[6] (Close Time) 
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
        return None, []

    asset_name = model_payload['asset']
    tf_name = model_payload['timeframe']
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[tf_name]
    
    df = pd.DataFrame(asset_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)
    
    # Resample
    if pandas_tf:
        prices_series = df['price'].resample(pandas_tf).last().dropna()
    else:
        prices_series = df['price']

    full_prices = prices_series.tolist()
    timestamps = prices_series.index.tolist()
    
    # Backtest Window: Approx last 7 days
    test_window_size = min(len(full_prices) - 10, 7 * (24 if tf_name == "60m" else 96 if tf_name == "15m" else 1))
    
    stats = {"wins": 0, "losses": 0, "noise": 0, "total_pnl_pct": 0.0}
    trade_log = []

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
            current_time = timestamps[i]
            
            # Prediction
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
                
                if pred_diff != 0:
                    direction = 1 if pred_diff > 0 else -1
                    signal_str = "BUY" if direction == 1 else "SELL"
                    
                    actual_next_price = full_prices[i+1]
                    actual_next_bucket = get_bucket(actual_next_price, b_size)
                    actual_diff = actual_next_bucket - last_bucket
                    
                    # PnL Calc
                    pct_change = ((actual_next_price - current_price) / current_price) * 100
                    trade_pnl_pct = pct_change * direction
                    
                    outcome = "NOISE"
                    if actual_diff == 0:
                        stats["noise"] += 1
                        stats["total_pnl_pct"] += trade_pnl_pct
                    elif (direction == 1 and actual_diff > 0) or (direction == -1 and actual_diff < 0):
                        stats["wins"] += 1
                        outcome = "WIN"
                        stats["total_pnl_pct"] += trade_pnl_pct
                    else:
                        stats["losses"] += 1
                        outcome = "LOSS"
                        stats["total_pnl_pct"] += trade_pnl_pct
                    
                    # Log Trade
                    trade_log.append({
                        "time": current_time,
                        "asset": asset_name,
                        "tf": tf_name,
                        "signal": signal_str,
                        "price": current_price,
                        "pnl": trade_pnl_pct,
                        "outcome": outcome
                    })

    return stats, trade_log

def main():
    all_trades = []
    timeframes = ["15m", "60m"]
    
    print(f"{'ASSET':<10} {'TF':<5} | {'STRICT ACC':<10} | {'APP ACC':<10} | {'PNL %':<10} | {'TRADES'}")
    print("-" * 75)

    for asset in ASSETS:
        raw_data = fetch_recent_binance_data(asset)
        if not raw_data: continue
        
        for tf in timeframes:
            model = get_model_from_github(asset, tf)
            if model:
                stats, trades = run_backtest(raw_data, model)
                if not stats: continue
                
                all_trades.extend(trades)
                
                wins = stats['wins']
                losses = stats['losses']
                noise = stats['noise']
                total_valid = wins + losses
                total_all = total_valid + noise
                
                if total_all > 0:
                    strict_acc = (wins / total_all * 100)
                    forgiving_acc = (wins / total_valid * 100) if total_valid > 0 else 0.0
                    pnl = stats['total_pnl_pct']
                    
                    print(f"{asset:<10} {tf:<5} | {strict_acc:6.2f}%    | {forgiving_acc:6.2f}%    | {pnl:+6.2f}%    | {total_all}")

    # --- PRINT ALL TRADES LIST ---
    print("\n" + "="*80)
    print("FULL TRADE HISTORY (Last 7 Days)")
    print("="*80)
    print(f"{'TIME':<20} {'ASSET':<10} {'TF':<5} {'SIGNAL':<5} {'PRICE':<12} {'PNL %':<8} {'OUTCOME'}")
    print("-" * 80)
    
    # Sort trades by time (Newest First)
    all_trades.sort(key=lambda x: x['time'], reverse=True)
    
    for t in all_trades:
        time_str = t['time'].strftime("%Y-%m-%d %H:%M")
        pnl_str = f"{t['pnl']:+.2f}%"
        print(f"{time_str:<20} {t['asset']:<10} {t['tf']:<5} {t['signal']:<5} {t['price']:<12.4f} {pnl_str:<8} {t['outcome']}")

if __name__ == "__main__":
    main()
