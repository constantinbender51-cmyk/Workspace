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
import seaborn as sns

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_YEAR = 2018
SMA_PERIOD = 400
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
    
    # Pre-calculate inputs
    df['Trend_Base'] = np.where(df['close'] > df['SMA'], 1, -1)
    df['Dist_Pct'] = (df['close'] - df['SMA']).abs() / df['SMA']
    
    df.dropna(inplace=True)
    
    print(f"Data ready: {len(df)} rows.")
    return df

# --- Optimized Backtest Function (Sharpe) ---
def backtest_strategy(df, prox_threshold, stop_threshold_pct):
    """
    Runs the logic for a single parameter set.
    Returns: Annualized Sharpe Ratio
    """
    trends = df['Trend_Base'].values
    dists = df['Dist_Pct'].values
    asset_rets = df['Asset_Ret'].values
    n = len(df)
    
    # Arrays to store daily strategy returns for Sharpe calc
    strat_daily_rets = np.zeros(n)
    
    # State
    current_trend = 0
    trade_equity = 1.0
    max_trade_equity = 1.0
    is_stopped_out = False
    
    stop_mult = 1.0 - stop_threshold_pct
    current_pos = 0.0 
    
    for i in range(n):
        # 1. Calculate PnL for today based on YESTERDAY's decision
        todays_pnl = current_pos * asset_rets[i]
        strat_daily_rets[i] = todays_pnl
        
        # Update Trade Internal Equity
        if current_pos != 0 and np.sign(current_pos) == current_trend:
             trade_equity *= (1 + todays_pnl)
             if trade_equity > max_trade_equity:
                 max_trade_equity = trade_equity
             
             # Check Stop
             if not is_stopped_out and trade_equity < (max_trade_equity * stop_mult):
                 is_stopped_out = True
        
        # 2. Determine Logic for TOMORROW (i+1)
        trend_now = trends[i]
        is_proximal = dists[i] < prox_threshold
        target_weight = 0.5 if is_proximal else 1.0
        
        # New Trend?
        if trend_now != current_trend:
            current_trend = trend_now
            trade_equity = 1.0
            max_trade_equity = 1.0
            is_stopped_out = False
            current_pos = current_trend * target_weight
            
        else:
            # Same Trend
            if is_stopped_out:
                if is_proximal:
                    # Re-enter
                    is_stopped_out = False
                    trade_equity = 1.0
                    max_trade_equity = 1.0
                    current_pos = current_trend * target_weight
                else:
                    current_pos = 0.0
            else:
                current_pos = current_trend * target_weight
                
    # --- Calculate Sharpe Ratio ---
    mean_ret = np.mean(strat_daily_rets)
    std_ret = np.std(strat_daily_rets)
    
    if std_ret == 0:
        return 0.0
        
    # Annualized Sharpe (assuming 365 days for crypto)
    sharpe = (mean_ret / std_ret) * np.sqrt(365)
    return sharpe

# --- Server & Logic ---
@app.route('/')
def run_grid_search():
    df = fetch_data()
    
    # Ranges: 1% to 30%
    stop_range = range(1, 31)
    prox_range = range(1, 31)
    
    results = []
    
    print("Starting Grid Search (900 iterations) for Sharpe Ratio...")
    start_time = time.time()
    
    for stop_pct in stop_range:
        row = []
        for prox_pct in prox_range:
            s_val = stop_pct / 100.0
            p_val = prox_pct / 100.0
            
            final_sharpe = backtest_strategy(df, p_val, s_val)
            row.append(final_sharpe)
        results.append(row)
        
    print(f"Grid Search Complete in {time.time() - start_time:.2f}s")
    
    # Convert to DataFrame
    res_df = pd.DataFrame(results, index=stop_range, columns=prox_range)
    
    # Find Best Params
    stacked = res_df.stack()
    best_coords = stacked.idxmax() # (Stop, Prox)
    best_sharpe = stacked.max()
    
    best_stop = best_coords[0]
    best_prox = best_coords[1]
    
    # --- Visualization ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    
    sns.heatmap(res_df, ax=ax, cmap='viridis', annot=False, fmt=".2f", 
                cbar_kws={'label': 'Annualized Sharpe Ratio'})
    
    # Highlight the Max
    ax.add_patch(plt.Rectangle((best_prox - 1, best_stop - 1), 1, 1, fill=False, edgecolor='red', lw=3, clip_on=False))
    
    ax.set_title(f'Grid Search: SMA 400 Sharpe Optimization\nBest Sharpe: {best_sharpe:.2f} @ Stop: {best_stop}%, Proximity: {best_prox}%', fontsize=14, fontweight='bold')
    ax.set_xlabel('Re-entry Proximity Threshold (%)', fontsize=12)
    ax.set_ylabel('Trailing Stop Loss (%)', fontsize=12)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Strategy Grid Search (Sharpe)</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; padding: 20px; text-align: center; }}
            .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .highlight {{ color: #2e7d32; font-weight: bold; font-size: 1.2em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Optimization Results: SMA 400 (Target: Sharpe Ratio)</h1>
            <p>Analyzed 900 parameter combinations (1% to 30% for both settings).</p>
            
            <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 20px 0; border: 1px solid #c8e6c9;">
                <h2>🏆 Optimal Parameters Found</h2>
                <p>Trailing Stop Loss: <span class="highlight">{best_stop}%</span></p>
                <p>Re-entry Proximity: <span class="highlight">{best_prox}%</span></p>
                <p>Sharpe Ratio: <span class="highlight">{best_sharpe:.2f}</span></p>
            </div>
            
            <img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;" />
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting Grid Search server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
