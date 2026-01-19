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

# Full Asset List
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT", "BNBUSDT"
]

# Updated Timeframes List
TIMEFRAMES = ["15m", "30m", "60m", "240m", "1d"]

# --- UTILS ---

def delayed_print(msg):
    print(msg)
    time.sleep(0.05)

def get_bucket(price, bucket_size):
    return int(price // bucket_size)

def fetch_recent_binance_data(symbol, days=30):
    """
    Fetches ~30 days of 15m data using pagination to ensure enough history
    for higher timeframes (4h, 1d) to function correctly.
    """
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
                
                # Use c[6] (Close Time) to match App Logic
                # c[6] is close time, c[4] is close price
                parsed_batch = [(int(c[6]), float(c[4])) for c in batch]
                data_points.extend(parsed_batch)
                
                # Update pointer to fetch next batch
                last_close_time = batch[-1][6]
                current_start = last_close_time + 1
                
                # Safety break if we are close to end
                if last_close_time >= end_ts - 1000: break
        except Exception as e:
            delayed_print(f"Error fetching Binance data segment for {symbol}: {e}")
            break
            
    # Remove duplicates if any (dict comprehension preserves order in Python 3.7+)
    unique_data = {x[0]: x[1] for x in data_points}
    return sorted([(k, v) for k, v in unique_data.items()])

def get_model_from_github(asset, timeframe):
    url = f"{GITHUB_RAW_URL}{asset}_{timeframe}.json"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        # Some asset/timeframe combos might not exist in the repo
        pass 
    return None

# --- CORE INFERENCE LOGIC ---

def get_prediction(strategies, prices):
    """
    Generates a signal (-1, 0, 1) based on a sequence of prices.
    Used for both Backtesting and Live Prediction.
    """
    votes = []
    
    for strat in strategies:
        params = strat['trained_parameters']
        b_size = params['bucket_size']
        s_len = params['seq_len']
        m_type = strat['config']['model_type']
        
        # We need enough data for the sequence
        if len(prices) < s_len: continue
        
        seq_prices = prices[-s_len:]
        buckets = [get_bucket(p, b_size) for p in seq_prices]
        
        a_seq = tuple(buckets)
        if s_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()
            
        last_bucket = buckets[-1]
        
        # Parse Maps
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
            
    if not votes: return 0
    
    # Majority Vote
    up = votes.count(1)
    down = votes.count(-1)
    
    if up > down: return 1
    elif down > up: return -1
    return 0

# --- BACKTEST & LIVE ENGINES ---

def run_backtest(asset_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return None, []

    asset_name = model_payload['asset']
    tf_name = model_payload['timeframe']
    
    # Updated mapping to include 240m (4h) and 1d
    pandas_tf = {"15m": None, "30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}[tf_name]
    
    df = pd.DataFrame(asset_data, columns=['timestamp', 'price'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('dt', inplace=True)
    
    if pandas_tf:
        prices_series = df['price'].resample(pandas_tf).last().dropna()
    else:
        prices_series = df['price']

    full_prices = prices_series.tolist()
    timestamps = prices_series.index.tolist()
    
    # Ensure enough data exists after resampling
    if len(full_prices) < 20: 
        return None, []

    # Backtest Window: Approx last 7 days or max available
    # For daily, 7 days is just 7 candles, so we take what we can get
    test_window_size = min(len(full_prices) - 10, 7 * (24 if tf_name == "60m" else 96 if tf_name == "15m" else 1))
    if tf_name == "1d": test_window_size = min(len(full_prices) - 8, 30) # Test last 30 days for Daily
    if tf_name == "240m": test_window_size = min(len(full_prices) - 8, 42) # Test last 7 days (6 * 7)

    stats = {"wins": 0, "losses": 0, "noise": 0, "total_pnl_pct": 0.0}
    trade_log = []

    start_idx = len(full_prices) - test_window_size
    min_b_size = min([s['trained_parameters']['bucket_size'] for s in strategies])

    for i in range(start_idx, len(full_prices) - 1):
        hist_prices = full_prices[:i+1]
        current_price = full_prices[i]
        current_time = timestamps[i]
        
        signal = get_prediction(strategies, hist_prices)
        
        if signal != 0:
            actual_next_price = full_prices[i+1]
            
            curr_b = get_bucket(current_price, min_b_size)
            next_b = get_bucket(actual_next_price, min_b_size)
            bucket_diff = next_b - curr_b
            
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
                "time": current_time,
                "asset": asset_name,
                "tf": tf_name,
                "signal": "BUY" if signal == 1 else "SELL",
                "price": current_price,
                "pnl": trade_pnl_pct,
                "outcome": outcome
            })

    return stats, trade_log

def get_latest_signal(asset_data, model_payload):
    strategies = model_payload.get("strategy_union", [])
    if not strategies: return "WAIT"

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
    
    if len(full_prices) > 1:
        completed_prices = full_prices[:-1]
    else:
        return "WAIT"

    sig = get_prediction(strategies, completed_prices)
    if sig == 1: return "BUY"
    if sig == -1: return "SELL"
    return "WAIT"

# --- MAIN ---

def main():
    all_trades = []
    current_signals = []
    
    # Global Aggregators
    g_wins = 0
    g_losses = 0
    g_noise = 0
    g_pnl = 0.0
    g_total_trades = 0

    print(f"{'ASSET':<10} {'TF':<5} | {'STRICT ACC':<10} | {'APP ACC':<10} | {'PNL %':<10} | {'TRADES'}")
    print("-" * 75)

    for asset in ASSETS:
        raw_data = fetch_recent_binance_data(asset, days=30)
        if not raw_data: continue
        
        for tf in TIMEFRAMES:
            model = get_model_from_github(asset, tf)
            if model:
                # 1. Backtest
                stats, trades = run_backtest(raw_data, model)
                if stats:
                    all_trades.extend(trades)
                    
                    wins = stats['wins']
                    losses = stats['losses']
                    noise = stats['noise']
                    pnl = stats['total_pnl_pct']
                    total_valid = wins + losses
                    total_all = total_valid + noise
                    
                    g_wins += wins
                    g_losses += losses
                    g_noise += noise
                    g_pnl += pnl
                    g_total_trades += total_all
                    
                    strict_acc = (wins / total_all * 100) if total_all > 0 else 0
                    forgiving_acc = (wins / total_valid * 100) if total_valid > 0 else 0
                    
                    if total_all > 0:
                        print(f"{asset:<10} {tf:<5} | {strict_acc:6.2f}%    | {forgiving_acc:6.2f}%    | {pnl:+6.2f}%    | {total_all}")

                # 2. Live Prediction
                live_sig = get_latest_signal(raw_data, model)
                if live_sig != "WAIT":
                    current_signals.append({
                        "asset": asset,
                        "tf": tf,
                        "signal": live_sig
                    })

    # --- OUTPUT: COMBINED PERFORMANCE ---
    print("\n" + "="*40)
    print("COMBINED PERFORMANCE (All Assets/Timeframes)")
    print("="*40)
    
    g_total_valid = g_wins + g_losses
    g_strict_acc = (g_wins / g_total_trades * 100) if g_total_trades > 0 else 0
    g_app_acc = (g_wins / g_total_valid * 100) if g_total_valid > 0 else 0
    
    print(f"Total Trades    : {g_total_trades}")
    print(f"Total Wins      : {g_wins}")
    print(f"Total Losses    : {g_losses}")
    print(f"Total Noise     : {g_noise}")
    print("-" * 40)
    print(f"Combined PnL    : {g_pnl:+.2f}%")
    print(f"Strict Accuracy : {g_strict_acc:.2f}% (Includes Noise)")
    print(f"App Accuracy    : {g_app_acc:.2f}% (Excludes Noise)")
    print("="*40)

    # --- OUTPUT: CURRENT SIGNALS ---
    print("\n" + "="*40)
    print("CURRENT SIGNALS (Latest Completed Candle)")
    print("="*40)
    if current_signals:
        for s in current_signals:
            print(f"{s['asset']:<10} {s['tf']:<5} : {s['signal']}")
    else:
        print("No active signals currently.")
    print("="*40)

    # --- OUTPUT: FULL TRADE HISTORY ---
    print("\n" + "="*80)
    print("FULL TRADE HISTORY (Last 7 Days / 30 Days for Daily)")
    print("="*80)
    print(f"{'TIME':<20} {'ASSET':<10} {'TF':<5} {'SIGNAL':<5} {'PRICE':<12} {'PNL %':<8} {'OUTCOME'}")
    print("-" * 80)
    
    all_trades.sort(key=lambda x: x['time'], reverse=True)
    
    for t in all_trades:
        time_str = t['time'].strftime("%Y-%m-%d %H:%M")
        pnl_str = f"{t['pnl']:+.2f}%"
        print(f"{time_str:<20} {t['asset']:<10} {t['tf']:<5} {t['signal']:<5} {t['price']:<12.4f} {pnl_str:<8} {t['outcome']}")

if __name__ == "__main__":
    main()
