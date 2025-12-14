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
SMA_PERIOD_1 = 120
SMA_PERIOD_2 = 400
PORT = 8080
TRAILING_STOP_PCT = 0.15  # Updated to 15% Max Loss from Peak

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
    print(f"Data fetched: {len(df)} rows.")
    return df

# --- Strategy 1: SMA 120 Weighted Decay ---
# Trade 40 days, Position = 1 - (d/40)^2
def run_strategy_1(df):
    data = df.copy()
    data['SMA_120'] = data['close'].rolling(window=SMA_PERIOD_1).mean()
    
    closes = data['close'].values
    smas = data['SMA_120'].values
    n = len(data)
    position_arr = np.zeros(n)
    
    current_trend = 0
    last_signal_idx = -9999
    
    for i in range(SMA_PERIOD_1, n):
        trend_now = 1 if closes[i] > smas[i] else -1
        
        if trend_now != current_trend:
            current_trend = trend_now
            last_signal_idx = i 
            
        d_trade = i - last_signal_idx
        
        if 1 <= d_trade <= 40:
            weight = 1 - (d_trade / 40.0)**2
            position_arr[i] = current_trend * weight
        else:
            position_arr[i] = 0.0
            
    data['Pos_1'] = position_arr
    data['Active_Pos_1'] = data['Pos_1'].shift(1).fillna(0.0)
    data['Asset_Ret'] = data['close'].pct_change().fillna(0.0)
    data['Strat_1_Daily_Ret'] = data['Active_Pos_1'] * data['Asset_Ret']
    data['Equity_1'] = (1 + data['Strat_1_Daily_Ret']).cumprod()
    
    return data

# --- Strategy 2: SMA 400 + Proximity + Re-entry (15% Stop) ---
def run_strategy_2(df):
    data = df.copy()
    data['SMA_400'] = data['close'].rolling(window=SMA_PERIOD_2).mean()
    
    closes = data['close'].values
    smas = data['SMA_400'].values
    asset_rets = data['Asset_Ret'].values
    n = len(data)
    
    position_arr = np.zeros(n)
    strat_daily_ret = np.zeros(n)
    
    # State Variables
    current_trend = 0
    
    # Trade Management State
    trade_equity = 1.0      # Current trade's equity
    max_trade_equity = 1.0  # Current trade's High Water Mark
    is_stopped_out = False  # Flag for trailing stop
    
    # Threshold for Trailing Stop (0.85 means 15% max loss from peak)
    STOP_THRESHOLD = 1.0 - TRAILING_STOP_PCT 
    
    for i in range(SMA_PERIOD_2, n):
        # 1. Determine Inputs
        trend_now = 1 if closes[i] > smas[i] else -1
        
        # Proximity Check: < 5% distance
        dist_pct = abs(closes[i] - smas[i]) / smas[i]
        is_proximal = dist_pct < 0.05
        
        # Target Weight Logic
        target_weight = 0.5 if is_proximal else 1.0
        
        # 2. Check for Signal Flip (Hard Reset)
        if trend_now != current_trend:
            current_trend = trend_now
            # Reset Trade State for new trend
            trade_equity = 1.0
            max_trade_equity = 1.0
            is_stopped_out = False
            
            # Enter new position immediately (Close of day i)
            position_arr[i] = current_trend * target_weight
        
        else:
            # SAME TREND: Manage Active Trade or Check Re-entry
            
            # --- PnL Calculation from Yesterday's Position ---
            prev_pos = position_arr[i-1] if i > 0 else 0
            todays_pnl = prev_pos * asset_rets[i]
            strat_daily_ret[i] = todays_pnl
            
            # Update Equity Tracking if we were active
            if prev_pos != 0 and np.sign(prev_pos) == current_trend:
                trade_equity *= (1 + todays_pnl)
                if trade_equity > max_trade_equity:
                    max_trade_equity = trade_equity
                
                # Check Trailing Stop (15% Drawdown)
                if not is_stopped_out and trade_equity < (max_trade_equity * STOP_THRESHOLD):
                    is_stopped_out = True
            
            # --- Position Sizing for Today ---
            
            if is_stopped_out:
                # We are in "Waiting" mode. Check Re-entry condition.
                if is_proximal:
                    # RE-ENTER: Reset the stop, reset equity tracker for this new leg
                    is_stopped_out = False
                    trade_equity = 1.0
                    max_trade_equity = 1.0
                    position_arr[i] = current_trend * target_weight # (0.5)
                else:
                    # Stay Flat
                    position_arr[i] = 0.0
            else:
                # We are Active. Just update weight based on proximity.
                position_arr[i] = current_trend * target_weight

    data['Pos_2'] = position_arr
    data['Active_Pos_2'] = data['Pos_2'].shift(1).fillna(0.0) 
    data['Strat_2_Daily_Ret'] = strat_daily_ret
    data['Equity_2'] = (1 + data['Strat_2_Daily_Ret']).cumprod()
    
    return data

def analyze_stats(df):
    s1_total_ret = (df['Equity_1'].iloc[-1] - 1) * 100
    s2_total_ret = (df['Equity_2'].iloc[-1] - 1) * 100
    
    ann_1 = df['Strat_1_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1).mean() * 100
    ann_2 = df['Strat_2_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1).mean() * 100
    
    return {
        'S1_Total': s1_total_ret,
        'S2_Total': s2_total_ret,
        'S1_Ann': ann_1,
        'S2_Ann': ann_2
    }

@app.route('/')
def dashboard():
    df_raw = fetch_data()
    df_s1 = run_strategy_1(df_raw)
    df_final = run_strategy_2(df_s1)
    stats = analyze_stats(df_final)
    
    # Plotting
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    
    # 1. Price + SMAs + S2 Activity
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.set_title(f'{SYMBOL}: S1 (Decay) vs S2 (SMA 400 Proximity + 15% Stop)', fontsize=14, fontweight='bold')
    ax1.plot(df_final.index, df_final['close'], color='black', alpha=0.5, label='Price', linewidth=0.8)
    ax1.plot(df_final.index, df_final['SMA_120'], color='orange', linestyle='--', label='SMA 120', linewidth=1)
    ax1.plot(df_final.index, df_final['SMA_400'], color='blue', linestyle='-', label='SMA 400', linewidth=1.5)
    
    # Highlight S2 Trades (Blue Zones)
    ax1.fill_between(df_final.index, df_final['close'].min(), df_final['close'].max(), 
                     where=(df_final['Active_Pos_2'] > 0.4), color='blue', alpha=0.1, label='S2 Long')
    ax1.fill_between(df_final.index, df_final['close'].min(), df_final['close'].max(), 
                     where=(df_final['Active_Pos_2'] < -0.4), color='purple', alpha=0.1, label='S2 Short')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_yscale('log')
    ax1.set_ylabel('Price (Log)')
    
    # 2. Equity Curves
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title(f"Equity Curves (Daily Compounding)", fontsize=12, fontweight='bold')
    ax2.plot(df_final.index, df_final['Equity_1'], color='orange', linewidth=2, label=f'S1: SMA 120 (40d Decay) | Total: {stats["S1_Total"]:.0f}%')
    ax2.plot(df_final.index, df_final['Equity_2'], color='blue', linewidth=2, label=f'S2: SMA 400 (15% Stop) | Total: {stats["S2_Total"]:.0f}%')
    ax2.axhline(1.0, color='black', linestyle='--')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Equity Multiple')
    
    # 3. Position Weights Visualization
    ax3 = fig.add_subplot(gs[3, 0])
    ax3.plot(df_final.index, df_final['Pos_1'].abs(), color='orange', label='S1 Weight')
    ax3.set_title('S1 Weight (Decay)')
    ax3.set_ylabel('Weight')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.plot(df_final.index, df_final['Pos_2'].abs(), color='blue', label='S2 Weight')
    ax4.set_title('S2 Weight (Dynamic: 0.5 or 1.0)')
    ax4.set_ylabel('Weight')
    ax4.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Strategy Comparison</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; padding: 20px; text-align: center; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }}
            .card {{ padding: 15px; border-radius: 8px; border: 1px solid #ddd; }}
            .card.s1 {{ background: #fff8e1; border-left: 5px solid orange; }}
            .card.s2 {{ background: #e3f2fd; border-left: 5px solid blue; }}
            .val {{ font-size: 24px; font-weight: bold; color: #333; }}
            .lbl {{ font-size: 14px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Comparison</h1>
            <div class="stats">
                <div class="card s1">
                    <h3 style="color:#e65100">Strategy 1 (SMA 120)</h3>
                    <div class="val">{stats['S1_Total']:.0f}%</div>
                    <div class="lbl">Total Equity Return</div>
                    <div class="lbl">Decay 40d</div>
                </div>
                <div class="card s2">
                    <h3 style="color:#0d47a1">Strategy 2 (SMA 400)</h3>
                    <div class="val">{stats['S2_Total']:.0f}%</div>
                    <div class="lbl">Total Equity Return</div>
                    <div class="lbl">Proximity Weighting + 15% Trailing Stop/Re-entry</div>
                </div>
            </div>
            <img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;" />
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting comparison server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
