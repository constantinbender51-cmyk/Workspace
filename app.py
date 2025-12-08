import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
from flask import Flask
import io
import time
import base64
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' 
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

app = Flask(__name__)

# --- Global Storage ---
# We store processed data here to avoid re-fetching on every request
global_data = {
    'df': pd.DataFrame(),
    'best_u': 0.0,
    'best_y': 0.0,
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

def calculate_indicators(df):
    """Calculates static indicators (SMAs, iii) to speed up grid search."""
    data = df.copy()

    # 1. Log Returns
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    # 2. SMAs
    data['sma_40'] = data['close'].rolling(window=40).mean()
    data['sma_120'] = data['close'].rolling(window=120).mean()

    # 3. iii (Efficiency Ratio) - 35 day window
    w35 = 35
    data['iii_35'] = (data['log_ret'].rolling(w35).sum().abs() / 
                      data['log_ret'].abs().rolling(w35).sum()).fillna(0)

    # 4. iii (Efficiency Ratio) - 7 day window (For new rule)
    w7 = 7
    data['iii_7'] = (data['log_ret'].rolling(w7).sum().abs() / 
                     data['log_ret'].abs().rolling(w7).sum()).fillna(0)

    # 5. Base Leverage (Shifted)
    conditions = [
        (data['iii_35'] < 0.13),
        (data['iii_35'] < 0.18)
    ]
    choices = [0.5, 4.5]
    data['base_leverage'] = np.select(conditions, choices, default=2.45)
    data['base_leverage'] = data['base_leverage'].shift(1) # Prevent leakage

    return data

def run_backtest(df, u, y):
    """
    Runs the backtest loop with specific u and y parameters.
    Returns (sharpe_ratio, result_dataframe)
    """
    data = df.copy()
    
    # Pre-calculate data needed for loop to avoid DataFrame overhead in loop
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    sma40s = data['sma_40'].values
    sma120s = data['sma_120'].values
    iii7s = data['iii_7'].values
    leverages = data['base_leverage'].values
    
    # Initialize arrays
    signals = np.zeros(len(data))
    returns = np.zeros(len(data))
    
    # State for the new rule
    is_flat_regime = False
    
    sl_pct = 0.02
    tp_pct = 0.16
    
    # Start loop (skip first 120 for SMAs)
    for i in range(120, len(data)):
        # 1. Update Regime State (based on PREVIOUS day i-1)
        prev_iii7 = iii7s[i-1]
        prev_close = closes[i-1]
        prev_sma40 = sma40s[i-1]
        prev_sma120 = sma120s[i-1]
        
        # Trigger: iii_7 drops below u
        if prev_iii7 < u:
            is_flat_regime = True
            
        # Release: enters band y around SMA 1 or 2
        # Band check: abs(price - sma) <= sma * y
        if is_flat_regime:
            in_band_1 = abs(prev_close - prev_sma40) <= (prev_sma40 * y)
            in_band_2 = abs(prev_close - prev_sma120) <= (prev_sma120 * y)
            if in_band_1 or in_band_2:
                is_flat_regime = False
        
        # 2. Determine Signal
        if is_flat_regime:
            signals[i] = 0
        else:
            # Standard Logic (based on PREVIOUS day)
            if prev_close > prev_sma40 and prev_close > prev_sma120:
                signals[i] = 1
            elif prev_close < prev_sma40 and prev_close < prev_sma120:
                signals[i] = -1
            else:
                signals[i] = 0
        
        # 3. Calculate Return (Intra-day)
        if signals[i] == 0 or np.isnan(leverages[i]):
            returns[i] = 0.0
            continue
            
        signal = signals[i]
        lev = leverages[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        
        daily_ret = 0.0
        
        if signal == 1:
            stop = o * (1 - sl_pct)
            take = o * (1 + tp_pct)
            if l <= stop: daily_ret = -sl_pct
            elif h >= take: daily_ret = tp_pct
            else: daily_ret = (c - o) / o
        else: # signal -1
            stop = o * (1 + sl_pct)
            take = o * (1 - tp_pct)
            if h >= stop: daily_ret = -sl_pct
            elif l <= take: daily_ret = tp_pct
            else: daily_ret = (o - c) / o
            
        returns[i] = daily_ret * lev

    data['strategy_ret'] = returns
    data['cumulative_ret'] = (1 + data['strategy_ret']).cumprod()
    
    # Calculate Sharpe
    if np.std(returns) == 0:
        sharpe = 0
    else:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365)
        
    return sharpe, data

def optimize_strategy(df):
    """Grid search for u and y."""
    print("Starting Grid Search optimization...")
    
    # Parameters ranges
    u_values = np.arange(0.0, 1.01, 0.05) # Reduced step to 0.05 for speed (20 steps)
    # Note: User asked for 0.01 steps. 
    # If we do 0.01 steps (100) * 0.005 steps (10) = 1000 iterations.
    # Let's try to honor the 0.01 request but keep y step reasonable.
    u_values = np.arange(0.0, 1.01, 0.01) 
    y_values = np.arange(0.0, 0.051, 0.005) # 0% to 5% in 0.5% steps (11 steps)
    
    best_sharpe = -999
    best_u = 0
    best_y = 0
    best_df = None
    
    total_iter = len(u_values) * len(y_values)
    print(f"Total iterations: {total_iter}")
    
    start_time = time.time()
    
    # Pre-calc indicators once
    df_ind = calculate_indicators(df)
    
    count = 0
    for u in u_values:
        for y in y_values:
            sharpe, res_df = run_backtest(df_ind, u, y)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_u = u
                best_y = y
                best_df = res_df
            
            count += 1
            if count % 200 == 0:
                print(f"Processed {count}/{total_iter}...")

    print(f"Optimization finished in {time.time() - start_time:.2f}s")
    print(f"Best Sharpe: {best_sharpe:.4f} | u: {best_u:.2f} | y: {best_y:.3f}")
    
    return best_u, best_y, best_df

# --- Main Execution ---
try:
    raw_df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    
    # Check if we have data
    if not raw_df.empty:
        # Run optimization
        b_u, b_y, res_df = optimize_strategy(raw_df)
        
        # Store in global
        global_data['df'] = raw_df
        global_data['best_u'] = b_u
        global_data['best_y'] = b_y
        global_data['results'] = res_df
        
        # Calculate BnH for comparison
        global_data['results']['bnh_ret'] = (1 + global_data['results']['log_ret']).cumprod()
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
    
    u = global_data['best_u']
    y = global_data['best_y']
    
    # Create Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_res.index, df_res['cumulative_ret'], label=f'Strategy (u={u:.2f}, y={y:.1%})', color='blue')
    plt.plot(df_res.index, df_res['bnh_ret'], label='Buy & Hold', color='gray', alpha=0.5)
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
    
    # Recalculate sharpe for display
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
            .stat-box {{ padding: 10px; border: 1px solid #eee; border-radius: 5px; }}
            .highlight {{ color: #2ecc71; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Results (Optimized)</h1>
            <p><strong>Optimization:</strong> Grid search found best Sharpe at:</p>
            <p class="highlight">u (iii threshold) = {u:.2f} | y (Band width) = {y:.1%}</p>
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
