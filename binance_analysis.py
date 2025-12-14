import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_YEAR = 2018
SMA_PERIOD = 120
DECAY_DAYS = 40
PORT = 8080

app = Flask(__name__)

# --- Data Fetching ---
def fetch_data():
    print(f"Fetching {SYMBOL} data from Binance starting {START_YEAR}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds():
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Pre-calculate indicators
    df['SMA'] = df['close'].rolling(window=SMA_PERIOD).mean()
    df['Asset_Ret'] = df['close'].pct_change().fillna(0.0)
    
    # Pre-calculate Trend to speed up loop
    df['Trend_Base'] = np.where(df['close'] > df['SMA'], 1, -1)
    
    df.dropna(inplace=True)
    print(f"Data ready: {len(df)} rows.")
    return df

# --- Backtest Function ---
def backtest_sma120(df, stop_pct):
    """
    Runs SMA 120 + Decay + Variable Stop Loss
    Returns: Annualized Sharpe Ratio, Total Return %
    """
    trends = df['Trend_Base'].values
    asset_rets = df['Asset_Ret'].values
    n = len(df)
    
    strat_daily_rets = np.zeros(n)
    
    # State
    current_trend = 0
    last_signal_idx = -9999
    
    trade_equity = 1.0
    max_trade_equity = 1.0
    is_stopped_out = False
    
    stop_threshold = 1.0 - stop_pct
    current_pos = 0.0
    
    # Pre-calc weights for speed
    # weights[d] = 1 - (d/40)^2
    decay_weights = np.zeros(DECAY_DAYS + 1)
    for d in range(1, DECAY_DAYS + 1):
        decay_weights[d] = 1 - (d / float(DECAY_DAYS))**2
    
    for i in range(n):
        # 1. PnL from yesterday's pos
        todays_pnl = current_pos * asset_rets[i]
        strat_daily_rets[i] = todays_pnl
        
        # Track Trade Equity
        if current_pos != 0 and np.sign(current_pos) == current_trend:
             trade_equity *= (1 + todays_pnl)
             if trade_equity > max_trade_equity:
                 max_trade_equity = trade_equity
             
             # Check Stop
             if not is_stopped_out and trade_equity < (max_trade_equity * stop_threshold):
                 is_stopped_out = True
        
        # 2. Logic for Tomorrow
        trend_now = trends[i]
        
        if trend_now != current_trend:
            # New Trend -> Reset
            current_trend = trend_now
            last_signal_idx = i
            trade_equity = 1.0
            max_trade_equity = 1.0
            is_stopped_out = False
            
            # Day 0 (technically day 1 of trade execution tomorrow)
            # We use weight for d=0 which is 1.0 in formula logic, or logic below
            current_pos = current_trend * 1.0
            
        else:
            # Same Trend
            d_trade = i - last_signal_idx
            
            if is_stopped_out:
                current_pos = 0.0
            elif 1 <= d_trade <= DECAY_DAYS:
                weight = decay_weights[d_trade]
                current_pos = current_trend * weight
            else:
                current_pos = 0.0
                
    # Metrics
    mean_ret = np.mean(strat_daily_rets)
    std_ret = np.std(strat_daily_rets)
    
    if std_ret == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_ret / std_ret) * np.sqrt(365)
        
    total_ret_equity = (np.prod(1 + strat_daily_rets) - 1) * 100
    
    return sharpe, total_ret_equity

@app.route('/')
def run_optimization():
    df = fetch_data()
    
    # Parameter Range: 1% to 50%
    stop_range = range(1, 51)
    
    results = []
    
    print("Starting optimization loop...")
    start_time = time.time()
    
    for stop_val in stop_range:
        s_pct = stop_val / 100.0
        sharpe, tot_ret = backtest_sma120(df, s_pct)
        results.append({
            'Stop %': stop_val,
            'Sharpe': sharpe,
            'Total Return': tot_ret
        })
        
    print(f"Optimization complete in {time.time() - start_time:.2f}s")
    
    res_df = pd.DataFrame(results)
    
    # Find Best
    best_sharpe_row = res_df.loc[res_df['Sharpe'].idxmax()]
    best_return_row = res_df.loc[res_df['Total Return'].idxmax()]
    
    # --- Visualization ---
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot Sharpe (Primary Y)
    line1, = ax1.plot(res_df['Stop %'], res_df['Sharpe'], color='blue', marker='o', markersize=4, label='Sharpe Ratio')
    ax1.set_xlabel('Trailing Stop Loss (%)', fontsize=12)
    ax1.set_ylabel('Annualized Sharpe Ratio', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # Plot Return (Secondary Y)
    ax2 = ax1.twinx()
    line2, = ax2.plot(res_df['Stop %'], res_df['Total Return'], color='orange', linestyle='--', alpha=0.7, label='Total Return %')
    ax2.set_ylabel('Total Equity Return (%)', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Highlight Peak Sharpe
    ax1.axvline(best_sharpe_row['Stop %'], color='green', linestyle=':', alpha=0.8)
    ax1.text(best_sharpe_row['Stop %'] + 1, best_sharpe_row['Sharpe'], 
             f"Best Sharpe: {best_sharpe_row['Sharpe']:.2f} @ {best_sharpe_row['Stop %']:.0f}%", 
             color='green', fontweight='bold')

    plt.title(f'SMA 120 Optimization: Trailing Stop Sensitivity', fontsize=14, fontweight='bold')
    
    # Combined Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SMA 120 Optimization</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; padding: 20px; text-align: center; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .highlight {{ color: #2e7d32; font-weight: bold; font-size: 1.2em; }}
            .stat-box {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .card {{ background: #e3f2fd; padding: 15px; border-radius: 8px; border: 1px solid #90caf9; width: 40%; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Optimization Results: SMA 120 Strategy</h1>
            <p>Scanning Trailing Stop Loss from 1% to 50%</p>
            
            <div class="stat-box">
                <div class="card">
                    <h3>🏆 Best Risk-Adjusted (Sharpe)</h3>
                    <p>Stop Loss: <span class="highlight">{best_sharpe_row['Stop %']:.0f}%</span></p>
                    <p>Sharpe Ratio: {best_sharpe_row['Sharpe']:.2f}</p>
                    <p>Total Return: {best_sharpe_row['Total Return']:.0f}%</p>
                </div>
                <div class="card" style="background: #fff3e0; border-color: #ffcc80;">
                    <h3>💰 Best Absolute Return</h3>
                    <p>Stop Loss: <span class="highlight">{best_return_row['Stop %']:.0f}%</span></p>
                    <p>Sharpe Ratio: {best_return_row['Sharpe']:.2f}</p>
                    <p>Total Return: {best_return_row['Total Return']:.0f}%</p>
                </div>
            </div>
            
            <img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;" />
            <p style="color: #666; font-size: 0.9em; margin-top: 15px;">Note: Very tight stops (1-5%) may choke the trade before the 40-day decay can capture the move.</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting SMA 120 optimization on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
