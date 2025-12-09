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
SYMBOL = 'SOL/USDT'
TIMEFRAME = '1d' 
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

# --- Strategy Parameters (Fixed) ---
PARAM_W = 35      # iii window
PARAM_U = 0.16    # iii threshold for flat regime
PARAM_Y = 0.045   # Bandwidth (4.5%)

app = Flask(__name__)

# --- Global Storage ---
global_data = {
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

def run_strategy(df):
    """
    Runs the strategy with fixed parameters:
    W=35, U=0.16, Y=0.045
    """
    data = df.copy()

    # 1. Log Returns
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    # 2. SMAs
    data['sma_40'] = data['close'].rolling(window=40).mean()
    data['sma_120'] = data['close'].rolling(window=120).mean()

    # 3. Calculate iii (Efficiency Ratio) for window W=35
    # Used for BOTH leverage calculation AND flat regime trigger
    w = PARAM_W
    data['iii'] = (data['log_ret'].rolling(w).sum().abs() / 
                   data['log_ret'].abs().rolling(w).sum()).fillna(0)
    
    # SHIFT iii by 1: We decide today's trade using yesterday's efficiency
    data['iii_shifted'] = data['iii'].shift(1)

    # 4. Leverage Logic
    # < 0.13 -> 0.5, < 0.18 -> 4.5, else 2.45
    # Uses the shifted iii
    conditions = [
        (data['iii_shifted'] < 0.13),
        (data['iii_shifted'] < 0.18)
    ]
    choices = [0.5, 4.5]
    data['leverage'] = np.select(conditions, choices, default=2.45)
    
    # Prepare Numpy Arrays for loop
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    sma40s = data['sma_40'].values
    sma120s = data['sma_120'].values
    iii_shifted = data['iii_shifted'].values
    leverages = data['leverage'].values
    
    n = len(closes)
    signals = np.zeros(n)
    returns = np.zeros(n)
    
    # Strategy Loop State
    is_flat_regime = False
    sl_pct = 0.02
    tp_pct = 0.16
    
    # Start at 120 to allow SMAs to populate
    for i in range(120, n):
        # --- A. Flat Regime Logic ---
        
        # 1. Trigger: If iii (yesterday) < U, enter flat regime
        if iii_shifted[i] < PARAM_U:
            is_flat_regime = True
            
        # 2. Release: If in flat regime, check if price enters band Y around SMAs
        if is_flat_regime:
            # Check bands using YESTERDAY'S price vs SMAs (to avoid lookahead on release)
            prev_c = closes[i-1]
            prev_s40 = sma40s[i-1]
            prev_s120 = sma120s[i-1]
            
            diff1 = abs(prev_c - prev_s40)
            diff2 = abs(prev_c - prev_s120)
            thresh1 = prev_s40 * PARAM_Y
            thresh2 = prev_s120 * PARAM_Y
            
            # If price is within Y% of EITHER SMA
            if diff1 <= thresh1 or diff2 <= thresh2:
                is_flat_regime = False
        
        # --- B. Signal Generation ---
        
        if is_flat_regime:
            signals[i] = 0
        else:
            # Standard Trend Logic (Yesterday's close vs SMAs)
            prev_c = closes[i-1]
            prev_s40 = sma40s[i-1]
            prev_s120 = sma120s[i-1]
            
            if prev_c > prev_s40 and prev_c > prev_s120:
                signals[i] = 1 # Long
            elif prev_c < prev_s40 and prev_c < prev_s120:
                signals[i] = -1 # Short
            else:
                signals[i] = 0
        
        # --- C. PnL Calculation ---
        
        lev = leverages[i]
        signal = signals[i]
        
        if signal == 0 or np.isnan(lev):
            returns[i] = 0.0
            continue
            
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        daily_ret = 0.0
        
        if signal == 1:
            stop = o * (1 - sl_pct)
            take = o * (1 + tp_pct)
            if l <= stop: 
                daily_ret = -sl_pct
            elif h >= take: 
                daily_ret = tp_pct
            else: 
                daily_ret = (c - o) / o
        else: # Short
            stop = o * (1 + sl_pct)
            take = o * (1 - tp_pct)
            if h >= stop: 
                daily_ret = -sl_pct
            elif l <= take: 
                daily_ret = tp_pct
            else: 
                daily_ret = (o - c) / o
                
        returns[i] = daily_ret * lev

    # Store results
    data['strategy_ret'] = returns
    data['cumulative_ret'] = (1 + data['strategy_ret']).cumprod()
    
    # Buy & Hold for comparison
    data['bnh_ret'] = (1 + data['log_ret']).cumprod()
    
    return data

# --- Main Execution ---
try:
    raw_df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    
    if not raw_df.empty:
        print("Running strategy with fixed parameters...")
        res_df = run_strategy(raw_df)
        global_data['results'] = res_df
        print("Strategy calculation complete.")
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
    
    # Create Plot
    plt.figure(figsize=(12, 6))
    label_str = f'Strategy (w={PARAM_W}, u={PARAM_U}, y={PARAM_Y:.1%})'
    plt.plot(df_res.index, df_res['cumulative_ret'], label=label_str, color='blue')
    plt.plot(df_res.index, df_res['bnh_ret'], label=f'Buy & Hold ({SYMBOL})', color='gray', alpha=0.5)
    plt.title(f'Strategy Result: {SYMBOL} (Fixed Parameters)')
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
        <title>Strategy Results</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; background-color: #f4f4f4; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }}
            img {{ max-width: 100%; height: auto; }}
            .stats {{ margin-top: 20px; font-size: 1.2em; display: flex; justify-content: space-around; }}
            .stat-box {{ padding: 10px; border: 1px solid #eee; border-radius: 5px; min-width: 150px; }}
            .params {{ margin: 20px 0; background: #e8f4f8; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Results: {SYMBOL}</h1>
            
            <div class="params">
                <h3>Fixed Parameters:</h3>
                <p><strong>iii Window (w):</strong> {PARAM_W} days</p>
                <p><strong>iii Threshold (u):</strong> {PARAM_U}</p>
                <p><strong>Band Width (y):</strong> {PARAM_Y:.1%}</p>
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
