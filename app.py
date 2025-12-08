import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
from flask import Flask, Response
import io
import time
import base64 # Added missing import
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' # Daily candles for the 35-day window calculation
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

app = Flask(__name__)

def fetch_data(symbol, timeframe, start_str):
    """Fetches historical OHLCV data from Binance."""
    print(f"Fetching {symbol} data from Binance starting {start_str}...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    
    ohlcv_list = []
    current_ts = start_ts
    now_ts = exchange.milliseconds()
    
    # Loop to fetch all data since limit is usually 500/1000 per call
    while current_ts < now_ts:
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts)
            if not ohlcvs:
                break
            current_ts = ohlcvs[-1][0] + 1  # Move just past the last fetched candle
            ohlcv_list += ohlcvs
            # Small sleep to respect rate limits
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Remove duplicates just in case
    df = df[~df.index.duplicated(keep='first')]
    print(f"Fetched {len(df)} rows.")
    return df

def apply_strategy(df):
    """Calculates indicators and runs the backtest logic."""
    data = df.copy()

    # 1. Calculate Log Returns
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    # 2. Calculate SMAs
    data['sma_40'] = data['close'].rolling(window=40).mean()
    data['sma_120'] = data['close'].rolling(window=120).mean()

    # 3. Calculate 'iii' (Efficiency Ratio)
    # iii = abs(sum(log returns)) / sum(abs(log returns)) over 35 days
    window_35 = 35
    sum_log_ret = data['log_ret'].rolling(window=window_35).sum()
    sum_abs_log_ret = data['log_ret'].abs().rolling(window=window_35).sum()
    
    # Avoid division by zero
    data['iii'] = (sum_log_ret.abs() / sum_abs_log_ret).fillna(0)

    # 4. Determine Leverage
    # < 0.13 -> 0.5
    # < 0.18 (implied >= 0.13) -> 4.5
    # Otherwise -> 2.45
    conditions = [
        (data['iii'] < 0.13),
        (data['iii'] < 0.18)
    ]
    choices = [0.5, 4.5]
    data['leverage'] = np.select(conditions, choices, default=2.45)
    
    # --- FIX LEAKAGE ---
    # Shift leverage by 1 so we rely on YESTERDAY'S efficiency ratio for TODAY'S trade
    data['leverage'] = data['leverage'].shift(1)

    # 5. Determine Trend Signals (Based on Previous Close)
    prev_close = data['close'].shift(1)
    prev_sma40 = data['sma_40'].shift(1)
    prev_sma120 = data['sma_120'].shift(1)

    data['signal'] = 0 # 0: Flat, 1: Long, -1: Short
    
    long_cond = (prev_close > prev_sma40) & (prev_close > prev_sma120)
    short_cond = (prev_close < prev_sma40) & (prev_close < prev_sma120)
    
    data.loc[long_cond, 'signal'] = 1
    data.loc[short_cond, 'signal'] = -1

    # 6. Calculate PnL with SL/TP logic
    strategy_returns = []
    
    # Parameters
    sl_pct = 0.02
    tp_pct = 0.16
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        # Skip if no signal or data not ready
        # Also check if leverage is NaN (due to shift)
        if pd.isna(row['sma_120']) or pd.isna(row['iii']) or pd.isna(row['leverage']) or row['signal'] == 0:
            strategy_returns.append(0.0)
            continue

        signal = row['signal']
        leverage = row['leverage']
        
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        # Intra-day Logic
        daily_ret = 0.0
        
        if signal == 1: # Long
            stop_price = open_price * (1 - sl_pct)
            take_profit_price = open_price * (1 + tp_pct)
            
            if low_price <= stop_price:
                daily_ret = -sl_pct
            elif high_price >= take_profit_price:
                daily_ret = tp_pct
            else:
                daily_ret = (close_price - open_price) / open_price
                
        elif signal == -1: # Short
            stop_price = open_price * (1 + sl_pct)
            take_profit_price = open_price * (1 - tp_pct)
            
            if high_price >= stop_price:
                daily_ret = -sl_pct
            elif low_price <= take_profit_price:
                daily_ret = tp_pct
            else:
                daily_ret = (open_price - close_price) / open_price

        # Apply Leverage
        strategy_returns.append(daily_ret * leverage)

    data['strategy_ret'] = strategy_returns
    
    # Cumulative Returns
    data['cumulative_ret'] = (1 + data['strategy_ret']).cumprod()
    
    # Buy and Hold (for comparison)
    data['bnh_ret'] = (1 + data['log_ret']).cumprod()
    
    return data

# --- Data Loading (Run once on startup) ---
try:
    df_raw = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    df_results = apply_strategy(df_raw)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    df_results = pd.DataFrame()

# --- Web Server ---

@app.route('/')
def index():
    if df_results.empty:
        return "Error fetching data or calculating strategy."
    
    # Create Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Strategy vs Buy & Hold
    plt.plot(df_results.index, df_results['cumulative_ret'], label='Strategy (Leveraged)', color='blue')
    plt.plot(df_results.index, df_results['bnh_ret'], label='Buy & Hold (BTC)', color='gray', alpha=0.5)
    
    plt.title(f'Strategy Backtest: {SYMBOL} (Start: {START_DATE_STR})')
    plt.ylabel('Cumulative Return (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save to IO buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Calculate Stats
    total_ret = df_results['cumulative_ret'].iloc[-1] if not df_results.empty else 0
    max_drawdown = (df_results['cumulative_ret'] / df_results['cumulative_ret'].cummax() - 1).min()
    
    # Sharpe Ratio Calculation
    if not df_results.empty and df_results['strategy_ret'].std() != 0:
        # Assuming 365 trading days for crypto
        sharpe_ratio = (df_results['strategy_ret'].mean() / df_results['strategy_ret'].std()) * np.sqrt(365)
    else:
        sharpe_ratio = 0.0

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
            .stat-box {{ padding: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Algorithmic Trading Strategy Results</h1>
            <p>Parameters: SMA(40, 120) | SL 2% | TP 16% | Dynamic Leverage (0.5x, 2.45x, 4.5x)</p>
            <img src="data:image/png;base64,{plot_url}">
            <div class="stats">
                <div class="stat-box"><strong>Total Return:</strong><br>{total_ret:.2f}x</div>
                <div class="stat-box"><strong>Max Drawdown:</strong><br>{max_drawdown:.2%}</div>
                <div class="stat-box"><strong>Sharpe Ratio:</strong><br>{sharpe_ratio:.2f}</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
