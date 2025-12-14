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
PORT = 8080

# --- Strategy Constants ---
# Strategy 1 (The Challenger)
S1_SMA = 120
S1_DECAY_DAYS = 40

# Strategy 2 (The Winner)
S2_SMA = 400
S2_PROX_PCT = 0.05   # 5% Proximity
S2_STOP_PCT = 0.27   # 27% Trailing Stop

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
    
    # Pre-calc Indicators
    df['Asset_Ret'] = df['close'].pct_change().fillna(0.0)
    df['SMA_120'] = df['close'].rolling(window=S1_SMA).mean()
    df['SMA_400'] = df['close'].rolling(window=S2_SMA).mean()
    
    # Pre-calc Distance for S2 speed
    df['Dist_Pct_400'] = (df['close'] - df['SMA_400']).abs() / df['SMA_400']
    
    df.dropna(inplace=True)
    print(f"Data ready: {len(df)} rows.")
    return df

# --- Strategy 1: SMA 120 Weighted Decay ---
def run_strategy_1(df):
    data = df.copy()
    closes = data['close'].values
    smas = data['SMA_120'].values
    n = len(data)
    position_arr = np.zeros(n)
    
    current_trend = 0
    last_signal_idx = -9999
    
    for i in range(n):
        trend_now = 1 if closes[i] > smas[i] else -1
        
        if trend_now != current_trend:
            current_trend = trend_now
            last_signal_idx = i 
            
        d_trade = i - last_signal_idx
        
        if 1 <= d_trade <= S1_DECAY_DAYS:
            weight = 1 - (d_trade / float(S1_DECAY_DAYS))**2
            position_arr[i] = current_trend * weight
        else:
            position_arr[i] = 0.0
            
    data['Pos_1'] = position_arr
    data['Active_Pos_1'] = data['Pos_1'].shift(1).fillna(0.0)
    data['Strat_1_Ret'] = data['Active_Pos_1'] * data['Asset_Ret']
    data['Equity_1'] = (1 + data['Strat_1_Ret']).cumprod()
    return data

# --- Strategy 2: SMA 400 (5% Prox, 27% Stop) ---
def run_strategy_2(df):
    data = df.copy()
    closes = data['close'].values
    smas = data['SMA_400'].values
    dists = data['Dist_Pct_400'].values
    asset_rets = data['Asset_Ret'].values
    n = len(data)
    
    position_arr = np.zeros(n)
    strat_daily_ret = np.zeros(n)
    
    current_trend = 0
    trade_equity = 1.0
    max_trade_equity = 1.0
    is_stopped_out = False
    
    stop_threshold = 1.0 - S2_STOP_PCT
    
    for i in range(n):
        # 1. Trend & Inputs
        trend_now = 1 if closes[i] > smas[i] else -1
        is_proximal = dists[i] < S2_PROX_PCT
        target_weight = 0.5 if is_proximal else 1.0
        
        # 2. Hard Trend Flip (Reset)
        if trend_now != current_trend:
            current_trend = trend_now
            trade_equity = 1.0
            max_trade_equity = 1.0
            is_stopped_out = False
            # Enter immediately at close
            position_arr[i] = current_trend * target_weight
            
        else:
            # Same Trend
            
            # Update PnL/Stops based on YESTERDAY'S position
            prev_pos = position_arr[i-1] if i > 0 else 0
            todays_pnl = prev_pos * asset_rets[i]
            strat_daily_ret[i] = todays_pnl
            
            # Track Trade Equity
            if prev_pos != 0 and np.sign(prev_pos) == current_trend:
                trade_equity *= (1 + todays_pnl)
                if trade_equity > max_trade_equity:
                    max_trade_equity = trade_equity
                
                # Check Stop
                if not is_stopped_out and trade_equity < (max_trade_equity * stop_threshold):
                    is_stopped_out = True
            
            # Position Logic
            if is_stopped_out:
                # Re-entry check
                if is_proximal:
                    is_stopped_out = False
                    trade_equity = 1.0
                    max_trade_equity = 1.0
                    position_arr[i] = current_trend * target_weight # Re-enter
                else:
                    position_arr[i] = 0.0 # Stay flat
            else:
                position_arr[i] = current_trend * target_weight

    data['Pos_2'] = position_arr
    data['Active_Pos_2'] = data['Pos_2'].shift(1).fillna(0.0) 
    data['Strat_2_Ret'] = strat_daily_ret
    data['Equity_2'] = (1 + data['Strat_2_Ret']).cumprod()
    return data

def calc_stats(equity_series):
    total_ret = (equity_series.iloc[-1] - 1) * 100
    
    daily_rets = equity_series.pct_change().dropna()
    mean = daily_rets.mean()
    std = daily_rets.std()
    
    sharpe = (mean / std) * np.sqrt(365) if std != 0 else 0
    
    # Max Drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return total_ret, sharpe, max_dd

@app.route('/')
def dashboard():
    df_raw = fetch_data()
    df_s1 = run_strategy_1(df_raw)
    df_final = run_strategy_2(df_s1)
    
    # Stats
    s1_tot, s1_shp, s1_dd = calc_stats(df_final['Equity_1'])
    s2_tot, s2_shp, s2_dd = calc_stats(df_final['Equity_2'])
    
    # Plotting
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    
    # 1. Main Price & SMAs
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title(f'Market Context: {SYMBOL} (Log Scale)', fontsize=14, fontweight='bold')
    ax1.plot(df_final.index, df_final['close'], color='black', alpha=0.4, lw=1, label='Price')
    ax1.plot(df_final.index, df_final['SMA_120'], color='orange', linestyle='--', lw=1.5, label='SMA 120')
    ax1.plot(df_final.index, df_final['SMA_400'], color='blue', linestyle='-', lw=1.5, label='SMA 400')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # 2. Head-to-Head Equity Curves
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title('Head-to-Head Equity Growth (Daily Compounding)', fontsize=14, fontweight='bold')
    ax2.plot(df_final.index, df_final['Equity_1'], color='orange', lw=2, label=f'S1: SMA 120 (Total: {s1_tot:,.0f}%)')
    ax2.plot(df_final.index, df_final['Equity_2'], color='blue', lw=2.5, label=f'S2: SMA 400 (Total: {s2_tot:,.0f}%)')
    ax2.fill_between(df_final.index, df_final['Equity_2'], df_final['Equity_1'], where=(df_final['Equity_2']>df_final['Equity_1']), color='blue', alpha=0.05, interpolate=True)
    ax2.fill_between(df_final.index, df_final['Equity_2'], df_final['Equity_1'], where=(df_final['Equity_2']<df_final['Equity_1']), color='orange', alpha=0.05, interpolate=True)
    
    ax2.axhline(1.0, color='black', linestyle='--')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Equity Multiple')
    
    # 3. Side-by-Side Annual Returns
    ax3 = fig.add_subplot(gs[2, :])
    ann_ret_1 = df_final['Strat_1_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
    ann_ret_2 = df_final['Strat_2_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
    
    years = ann_ret_1.index.year
    x = np.arange(len(years))
    width = 0.35
    
    rects1 = ax3.bar(x - width/2, ann_ret_1, width, label='SMA 120 (S1)', color='orange', alpha=0.85)
    rects2 = ax3.bar(x + width/2, ann_ret_2, width, label='SMA 400 (S2)', color='blue', alpha=0.85)
    
    ax3.set_title('Annual Performance Comparison (%)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.axhline(0, color='black', lw=1)
    ax3.legend()
    # Fixed: Changed 'Axis' to 'axis'
    ax3.grid(axis='y', alpha=0.2)
    
    # Add labels to bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax3.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Strategy Showdown</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6f9; padding: 20px; text-align: center; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
            .header {{ margin-bottom: 30px; }}
            .comparison-box {{ display: flex; justify-content: space-around; margin-bottom: 30px; gap: 20px; }}
            .stat-panel {{ flex: 1; padding: 20px; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .panel-s1 {{ background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); }}
            .panel-s2 {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); transform: scale(1.05); border: 2px solid gold; }}
            .stat-row {{ display: flex; justify-content: space-between; margin: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px; }}
            .stat-val {{ font-weight: bold; font-size: 1.2em; }}
            h2 {{ margin-top: 0; border-bottom: 2px solid rgba(255,255,255,0.5); padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏆 Strategy Showdown 🏆</h1>
                <p>Comparing the Standard Model vs. The Optimized Challenger</p>
            </div>
            
            <div class="comparison-box">
                <div class="stat-panel panel-s1">
                    <h2>Strategy 1 (SMA 120)</h2>
                    <div class="stat-row"><span>Total Return:</span> <span class="stat-val">{s1_tot:,.0f}%</span></div>
                    <div class="stat-row"><span>Sharpe Ratio:</span> <span class="stat-val">{s1_shp:.2f}</span></div>
                    <div class="stat-row"><span>Max Drawdown:</span> <span class="stat-val">{s1_dd:.1f}%</span></div>
                    <p style="font-size:0.9em; margin-top:15px;">Logic: 40-Day Decay</p>
                </div>
                
                <div class="stat-panel panel-s2">
                    <h2>Strategy 2 (SMA 400) 👑</h2>
                    <div class="stat-row"><span>Total Return:</span> <span class="stat-val">{s2_tot:,.0f}%</span></div>
                    <div class="stat-row"><span>Sharpe Ratio:</span> <span class="stat-val">{s2_shp:.2f}</span></div>
                    <div class="stat-row"><span>Max Drawdown:</span> <span class="stat-val">{s2_dd:.1f}%</span></div>
                    <p style="font-size:0.9em; margin-top:15px;">Logic: 5% Prox, 27% Stop, Re-entry</p>
                </div>
            </div>
            
            <img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto; border-radius: 8px; border: 1px solid #ddd;" />
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting showdown server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
