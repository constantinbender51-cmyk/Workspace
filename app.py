import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask
import io
import time
import base64

# --- Configuration ---
SYMBOL = 'ETH/USDT' # Changed to ETH
TIMEFRAME = '1d' 
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

app = Flask(__name__)

# --- Global Storage ---
global_data = {
    'df': pd.DataFrame(),
    'best_params': {},
    'results': pd.DataFrame()
}

def fetch_data(symbol, timeframe, start_str):
    """Fetches historical OHLCV data from Binance."""
    print(f"Fetching {symbol} data from Binance starting {start_str}...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    
    ohlcv_list = []
    current_ts = start_ts
    now_ts = exchange.milliseconds()
    
    while current_ts < now_ts:
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts)
            if not ohlcvs:
                break
            current_ts = ohlcvs[-1][0] + 1
            ohlcv_list += ohlcvs
            time.sleep(0.05) 
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"Fetched {len(df)} rows.")
    return df

def prepare_arrays(df):
    """
    Pre-calculates everything possible outside the loops.
    Returns a dictionary of numpy arrays.
    """
    data = df.copy()

    # 1. Log Returns
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    # 2. SMAs
    data['sma_40'] = data['close'].rolling(window=40).mean()
    data['sma_120'] = data['close'].rolling(window=120).mean()

    # 3. Leverage Control (Fixed 35-day window as per original spec)
    w35 = 35
    iii_35 = (data['log_ret'].rolling(w35).sum().abs() / 
              data['log_ret'].abs().rolling(w35).sum()).fillna(0)
    
    conditions = [(iii_35 < 0.13), (iii_35 < 0.18)]
    choices = [0.5, 4.5]
    data['base_leverage'] = np.select(conditions, choices, default=2.45)
    data['base_leverage'] = data['base_leverage'].shift(1) # Shift to avoid leakage

    # 4. Pre-calculate 'iii' for ALL windows 1 to 36
    # We store these in a dictionary: { window_size: numpy_array }
    iii_dict = {}
    for w in range(1, 37):
        iii_val = (data['log_ret'].rolling(w).sum().abs() / 
                   data['log_ret'].abs().rolling(w).sum()).fillna(0)
        # Shift by 1 because the decision at day 'i' must be based on 'iii' from 'i-1'
        iii_dict[w] = iii_val.shift(1).values

    # Convert columns to numpy for speed
    arrays = {
        'open': data['open'].values,
        'high': data['high'].values,
        'low': data['low'].values,
        'close': data['close'].values,
        'sma40': data['sma_40'].values,
        'sma120': data['sma_120'].values,
        'leverage': data['base_leverage'].values,
        'iii_dict': iii_dict,
        'dates': data.index,
        'log_ret': data['log_ret'].values
    }
    return arrays

def run_backtest_fast(arrays, w, u, y):
    """
    Optimized backtest loop using numpy arrays.
    """
    opens = arrays['open']
    highs = arrays['high']
    lows = arrays['low']
    closes = arrays['close']
    sma40s = arrays['sma40']
    sma120s = arrays['sma120']
    leverages = arrays['leverage']
    
    # Get the specific pre-shifted iii array for this window w
    iii_array = arrays['iii_dict'][w]
    
    n = len(closes)
    returns = np.zeros(n)
    
    # Constants
    sl_pct = 0.02
    tp_pct = 0.16
    
    # State
    is_flat_regime = False
    
    # Loop
    # Start at 120 to ensure SMAs are valid
    for i in range(120, n):
        # 1. Update Flat Regime State
        # Trigger: iii drops below u
        # Note: iii_array is already shifted by 1 in prepare_arrays
        current_iii = iii_array[i] 
        
        # We need PREVIOUS price/sma for band check to avoid lookahead on the "release" trigger
        prev_c = closes[i-1]
        prev_s40 = sma40s[i-1]
        prev_s120 = sma120s[i-1]
        
        if current_iii < u:
            is_flat_regime = True
            
        # Release: enters band y
        if is_flat_regime:
            # Check bands
            diff1 = abs(prev_c - prev_s40)
            diff2 = abs(prev_c - prev_s120)
            threshold1 = prev_s40 * y
            threshold2 = prev_s120 * y
            
            if diff1 <= threshold1 or diff2 <= threshold2:
                is_flat_regime = False
        
        # 2. Determine Signal (if not flat)
        signal = 0
        if not is_flat_regime:
            if prev_c > prev_s40 and prev_c > prev_s120:
                signal = 1
            elif prev_c < prev_s40 and prev_c < prev_s120:
                signal = -1
        
        # 3. Calculate Return
        lev = leverages[i]
        
        # If no signal or invalid leverage/data
        if signal == 0 or np.isnan(lev):
            continue
            
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        
        if signal == 1:
            stop = o * (1 - sl_pct)
            take = o * (1 + tp_pct)
            if l <= stop: 
                ret = -sl_pct
            elif h >= take: 
                ret = tp_pct
            else: 
                ret = (c - o) / o
        else:
            stop = o * (1 + sl_pct)
            take = o * (1 - tp_pct)
            if h >= stop: 
                ret = -sl_pct
            elif l <= take: 
                ret = tp_pct
            else: 
                ret = (o - c) / o
                
        returns[i] = ret * lev

    # Calculate Sharpe
    std_dev = np.std(returns)
    if std_dev == 0:
        sharpe = 0.0
    else:
        sharpe = (np.mean(returns) / std_dev) * np.sqrt(365)
        
    return sharpe, returns

def optimize_strategy(df):
    """3D Grid search for w, u, and y."""
    arrays = prepare_arrays(df)
    
    print("Starting 3D Grid Search (w, u, y)...")
    
    # Parameter Space
    w_values = range(1, 37) # 1 to 36
    u_values = np.arange(0.0, 1.01, 0.02) # Step 0.02
    y_values = np.arange(0.0, 0.051, 0.005) # 11 steps
    
    best_sharpe = -999
    best_params = {'w': 0, 'u': 0, 'y': 0}
    best_returns = None
    
    start_time = time.time()
    
    for w in w_values:
        # Print progress every window iteration
        print(f"Scanning Window {w}/36...")
        for u in u_values:
            for y in y_values:
                sharpe, returns = run_backtest_fast(arrays, w, u, y)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'w': w, 'u': u, 'y': y}
                    best_returns = returns

    elapsed = time.time() - start_time
    print(f"Optimization finished in {elapsed:.2f}s")
    print(f"Best Sharpe: {best_sharpe:.4f}")
    print(f"Best Params: {best_params}")
    
    # Reconstruct DataFrame for plotting
    res_df = df.copy()
    res_df['strategy_ret'] = best_returns
    res_df['cumulative_ret'] = (1 + res_df['strategy_ret']).cumprod()
    res_df['log_ret'] = np.log(res_df['close'] / res_df['close'].shift(1))
    res_df['bnh_ret'] = (1 + res_df['log_ret']).cumprod()
    
    return best_params, res_df

# --- Main Execution ---
try:
    raw_df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    
    if not raw_df.empty:
        b_params, res_df = optimize_strategy(raw_df)
        
        global_data['df'] = raw_df
        global_data['best_params'] = b_params
        global_data['results'] = res_df
    else:
        print("No data fetched.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# --- Web Server ---

@app.route('/')
def index():
    df_res = global_data['results']
    if df_res is None or df_res.empty:
        return "Error: No results available."
    
    p = global_data['best_params']
    
    # Create Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_res.index, df_res['cumulative_ret'], label=f'Strategy (w={p["w"]}, u={p["u"]:.2f}, y={p["y"]:.1%})', color='blue')
    plt.plot(df_res.index, df_res['bnh_ret'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.title(f'Optimized Strategy: {SYMBOL} (Sharpe Optimized)')
    plt.ylabel('Cumulative Return (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    total_ret = df_res['cumulative_ret'].iloc[-1]
    max_dd = (df_res['cumulative_ret'] / df_res['cumulative_ret'].cummax() - 1).min()
    
    returns = df_res['strategy_ret']
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() != 0 else 0

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimized Strategy Results</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; background-color: #f4f4f4; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }}
            img {{ max-width: 100%; height: auto; }}
            .stats {{ margin-top: 20px; font-size: 1.2em; display: flex; justify-content: space-around; }}
            .stat-box {{ padding: 10px; border: 1px solid #eee; border-radius: 5px; min-width: 150px; }}
            .highlight {{ color: #2ecc71; font-weight: bold; }}
            .params {{ margin: 20px 0; background: #f9f9f9; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Results (3D Optimized: ETH/USDT)</h1>
            
            <div class="params">
                <h3>Optimal Parameters Found:</h3>
                <p><strong>iii Window (w):</strong> {p['w']} days</p>
                <p><strong>iii Threshold (u):</strong> {p['u']:.2f}</p>
                <p><strong>Band Width (y):</strong> {p['y']:.1%}</p>
            </div>

            <img src="data:image/png;base64,{plot_url}">
            
            <div class="stats">
                <div class="stat-box"><strong>Total Return:</strong><br>{total_ret:.2f}x</div>
                <div class="stat-box"><strong>Max Drawdown:</strong><br>{max_dd:.2%}</div>
                <div class="stat-box"><strong>Sharpe Ratio:</strong><br>{sharpe:.2f}</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
