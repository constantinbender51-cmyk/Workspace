import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server usage
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
from datetime import datetime
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_YEAR = 2018
SMA_PERIOD = 120
PORT = 8080

app = Flask(__name__)

# --- Data Fetching ---
def fetch_data():
    print(f"Fetching {SYMBOL} data from Binance starting {START_YEAR}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
    all_ohlcv = []
    
    # Fetch in loops to get full history
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Move pointer forward
            
            # Break if we reached current time
            if since > exchange.milliseconds():
                break
            
            # Rate limit sleep
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    # Remove potential duplicates
    df = df[~df.index.duplicated(keep='first')]
    print(f"Data fetched: {len(df)} rows.")
    return df

# --- Analysis & Strategy ---
def run_strategy(df):
    data = df.copy()
    data['SMA'] = data['close'].rolling(window=SMA_PERIOD).mean()
    
    # 1 = Long (Price > SMA), -1 = Short (Price < SMA)
    # Using shift(1) to avoid lookahead bias (trading on close of same day)
    data['Signal'] = np.where(data['close'] > data['SMA'], 1, -1)
    data['Position'] = data['Signal'].shift(1) # We enter on the next open/close based on prev signal
    
    # Calculate returns
    data['Log_Ret'] = np.log(data['close'] / data['close'].shift(1))
    data['Strat_Ret'] = data['Position'] * data['Log_Ret']
    
    return data.dropna()

def analyze_trades(df):
    # Identify trade starts: where position changes
    df['Trade_ID'] = (df['Position'] != df['Position'].shift(1)).cumsum()
    
    trades = []
    
    # Event Study: Average price curve after signal
    # We will track price performance for up to 60 days after a signal switch
    look_forward_window = 60 
    curves = []

    grouped = df.groupby('Trade_ID')
    
    for trade_id, trade_data in grouped:
        if len(trade_data) < 1: continue
        
        start_date = trade_data.index[0]
        end_date = trade_data.index[-1]
        position = trade_data['Position'].iloc[0] # 1 or -1
        
        # Entry price (Close of the first candle of the trade - simplifying assumption)
        entry_price = trade_data['close'].iloc[0] 
        exit_price = trade_data['close'].iloc[-1]
        
        # Trade duration
        duration_days = (end_date - start_date).days
        
        # Return
        if position == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price
            
        # Success?
        is_win = raw_return > 0
        
        trades.append({
            'Trade_ID': trade_id,
            'Position': 'Long' if position == 1 else 'Short',
            'Start': start_date,
            'Duration': duration_days,
            'Return': raw_return,
            'Win': is_win
        })
        
        # --- Event Study Logic ---
        # Get price curve normalized to entry
        # We look at the actual Close prices relative to Entry
        # If Short, we invert the curve to show "profitability" trajectory
        
        # Get next N records from full dataframe starting at trade start
        subset = df.loc[start_date:].iloc[:look_forward_window].copy()
        if not subset.empty:
            normalized_prices = subset['close'] / entry_price
            if position == -1:
                # For shorts, if price goes down (0.9), that's a 1.1 gain approx in equity
                # Simple inversion for visualization: 2 - price_ratio (so 0.9 becomes 1.1)
                normalized_prices = 2 - normalized_prices
            
            # Reindex to 0..N
            normalized_prices = normalized_prices.reset_index(drop=True)
            curves.append(normalized_prices)

    trades_df = pd.DataFrame(trades)
    
    # Average Curve Calculation
    if curves:
        curve_df = pd.concat(curves, axis=1)
        avg_curve = curve_df.mean(axis=1)
    else:
        avg_curve = pd.Series()
        
    return trades_df, avg_curve

# --- Web Server & Plotting ---
@app.route('/')
def dashboard():
    # 1. Pipeline
    df_raw = fetch_data()
    df_strat = run_strategy(df_raw)
    trades_df, avg_curve = analyze_trades(df_strat)
    
    # 2. Statistics
    total_trades = len(trades_df)
    win_rate = trades_df['Win'].mean() * 100
    avg_duration = trades_df['Duration'].mean()
    expectancy = trades_df['Return'].mean() * 100
    
    # 3. Meticulous Plotting
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    
    # Ax1: Price & SMA & Backgrounds
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title(f'{SYMBOL} Price vs SMA {SMA_PERIOD} (Red=Long, Blue=Short)', fontsize=14, fontweight='bold')
    ax1.plot(df_strat.index, df_strat['close'], color='black', label='Price', linewidth=1)
    ax1.plot(df_strat.index, df_strat['SMA'], color='orange', label=f'SMA {SMA_PERIOD}', linewidth=1.5)
    
    # Background Logic
    # We find segments where Position is 1 or -1
    # Resample slightly to reduce fill operations or just iterate changes
    # Vectorized fill_between approach
    ax1.fill_between(df_strat.index, df_strat['close'].min(), df_strat['close'].max(), 
                     where=(df_strat['Position'] == 1), color='red', alpha=0.15, label='Long Zone')
    ax1.fill_between(df_strat.index, df_strat['close'].min(), df_strat['close'].max(), 
                     where=(df_strat['Position'] == -1), color='blue', alpha=0.15, label='Short Zone')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price (USDT)')

    # Ax2: Cumulative Returns
    ax2 = fig.add_subplot(gs[1, 0])
    df_strat['Cum_Ret'] = df_strat['Strat_Ret'].cumsum()
    ax2.set_title('Cumulative Strategy Log Returns', fontsize=12)
    ax2.plot(df_strat.index, df_strat['Cum_Ret'], color='green')
    ax2.fill_between(df_strat.index, df_strat['Cum_Ret'], 0, alpha=0.1, color='green')
    ax2.grid(True, alpha=0.3)

    # Ax3: Event Study (Avg Price Curve)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(f'Avg Performance After Signal (First {len(avg_curve)} Days)', fontsize=12)
    ax3.plot(avg_curve.index, avg_curve.values, color='purple', linewidth=2)
    ax3.axhline(1.0, color='gray', linestyle='--')
    ax3.set_xlabel('Days after Signal')
    ax3.set_ylabel('Normalized Value (1.0 = Entry)')
    ax3.grid(True, alpha=0.3)
    
    # Ax4: Trade Duration Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(trades_df['Duration'], bins=20, color='gray', edgecolor='black', alpha=0.7)
    ax4.set_title('Trade Duration Distribution (Days)', fontsize=12)
    ax4.set_xlabel('Days')
    
    # Ax5: Returns Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(trades_df['Return'] * 100, bins=20, color='teal', edgecolor='black', alpha=0.7)
    ax5.set_title('Trade Return Distribution (%)', fontsize=12)
    ax5.axvline(0, color='red', linestyle='--')
    ax5.set_xlabel('Return %')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    # HTML Template
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Binance SMA Backtest</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f4f4f4; text-align: center; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
            .stat-card {{ background: #eee; padding: 15px; border-radius: 5px; }}
            .stat-val {{ font-size: 1.5em; font-weight: bold; color: #333; }}
            .stat-label {{ font-size: 0.9em; color: #666; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Binance SMA 120 Strategy Analysis ({SYMBOL})</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-val">{win_rate:.2f}%</div>
                    <div class="stat-label">Win Rate (Signal to Desired Outcome)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{avg_duration:.1f} days</div>
                    <div class="stat-label">Avg Trade Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{expectancy:.2f}%</div>
                    <div class="stat-label">Expectation per Signal</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{total_trades}</div>
                    <div class="stat-label">Total Signals</div>
                </div>
            </div>

            <img src="data:image/png;base64,{data}" />
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting analysis server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
