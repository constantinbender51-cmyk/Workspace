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
EVENT_STUDY_WINDOW = 120 

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
def run_strategy_1(df):
    data = df.copy()
    data['SMA_120'] = data['close'].rolling(window=SMA_PERIOD_1).mean()
    
    # Logic
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
        
        # Weighted Decay Days 1-40
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

# --- Strategy 2: SMA 400 Continuous ---
def run_strategy_2(df):
    data = df.copy() # Works on top of Strat 1 df
    data['SMA_400'] = data['close'].rolling(window=SMA_PERIOD_2).mean()
    
    # Logic: Always Long/Short based on SMA 400
    # 1 if Close > SMA, -1 if Close < SMA
    data['Raw_Trend_2'] = np.where(data['close'] > data['SMA_400'], 1, -1)
    
    # We set position to 0 if SMA is NaN (start of data)
    data['Raw_Trend_2'] = np.where(data['SMA_400'].isna(), 0, data['Raw_Trend_2'])
    
    # Enter next day
    data['Active_Pos_2'] = data['Raw_Trend_2'].shift(1).fillna(0.0)
    
    # Calc Returns
    data['Strat_2_Daily_Ret'] = data['Active_Pos_2'] * data['Asset_Ret']
    data['Equity_2'] = (1 + data['Strat_2_Daily_Ret']).cumprod()
    
    return data

def analyze_stats(df):
    # --- Stats Strat 1 ---
    # Trade defined as the 40-day weighted block
    df['S1_Start'] = (df['Active_Pos_1'].abs() > 0) & (df['Active_Pos_1'].shift(1).fillna(0).abs() == 0)
    s1_starts = df[df['S1_Start']].index
    
    s1_trades = []
    for s in s1_starts:
        e_idx = df.index.get_loc(s) + 39
        if e_idx >= len(df): continue
        e = df.index[e_idx]
        subset = df.loc[s:e]
        ret = (1 + subset['Strat_1_Daily_Ret']).prod() - 1
        s1_trades.append(ret)
    
    s1_win_rate = (sum(x > 0 for x in s1_trades) / len(s1_trades) * 100) if s1_trades else 0
    s1_total_ret = (df['Equity_1'].iloc[-1] - 1) * 100
    
    # --- Stats Strat 2 ---
    # Trade defined as continuous holding period between crossovers
    df['S2_Change'] = (df['Raw_Trend_2'] != df['Raw_Trend_2'].shift(1)) & (df['Raw_Trend_2'] != 0)
    s2_starts = df[df['S2_Change']].index
    
    s2_trades = []
    # Simplified trade loop for continuous
    current_idx = df.index.get_loc(s2_starts[0]) if len(s2_starts) > 0 else 0
    
    # Calculate returns for blocks
    # This is an approximation for stats display
    s2_total_ret = (df['Equity_2'].iloc[-1] - 1) * 100
    
    return {
        'S1_Win': s1_win_rate,
        'S1_Total': s1_total_ret,
        'S1_Count': len(s1_trades),
        'S2_Total': s2_total_ret
    }

# --- Server & Plotting ---
@app.route('/')
def dashboard():
    df_raw = fetch_data()
    df_s1 = run_strategy_1(df_raw)
    df_final = run_strategy_2(df_s1)
    
    stats = analyze_stats(df_final)
    
    # Plotting
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    
    # 1. Price + SMAs
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.set_title(f'{SYMBOL} Analysis: SMA 120 (Timed) vs SMA 400 (Continuous)', fontsize=14, fontweight='bold')
    ax1.plot(df_final.index, df_final['close'], color='black', alpha=0.5, label='Price', linewidth=0.8)
    ax1.plot(df_final.index, df_final['SMA_120'], color='orange', linestyle='--', label='SMA 120', linewidth=1)
    ax1.plot(df_final.index, df_final['SMA_400'], color='blue', linestyle='-', label='SMA 400', linewidth=1.5)
    
    # Highlight Strat 1 Trades (Red/Blue zones)
    ax1.fill_between(df_final.index, df_final['close'].min(), df_final['close'].max(), 
                     where=(df_final['Active_Pos_1'] > 0), color='orange', alpha=0.15, label='Strat 1 Active (Long)')
    ax1.fill_between(df_final.index, df_final['close'].min(), df_final['close'].max(), 
                     where=(df_final['Active_Pos_1'] < 0), color='purple', alpha=0.15, label='Strat 1 Active (Short)')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylabel('Price (Log Scale)')
    ax1.set_yscale('log')
    
    # 2. Equity Curves Comparison
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title(f"Equity Curves (Daily Compounded)", fontsize=12, fontweight='bold')
    
    # Strat 1
    ax2.plot(df_final.index, df_final['Equity_1'], color='orange', linewidth=2, label=f'Strat 1: SMA 120 Weighted (Total: {stats["S1_Total"]:.0f}%)')
    # Strat 2
    ax2.plot(df_final.index, df_final['Equity_2'], color='blue', linewidth=2, label=f'Strat 2: SMA 400 Continuous (Total: {stats["S2_Total"]:.0f}%)')
    
    ax2.axhline(1.0, color='black', linestyle='--')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Equity Multiple')
    
    # 3. Strat 1 Weight Profile (Visualizing the decay)
    ax3 = fig.add_subplot(gs[3, 0])
    days = np.arange(0, 50)
    weights = [1 - (d/40)**2 if 1 <= d <= 40 else 0 for d in days]
    ax3.plot(days, weights, color='orange', marker='o', markersize=3)
    ax3.set_title('Strategy 1: Position Weight Profile')
    ax3.set_xlabel('Days Since Signal')
    ax3.set_ylabel('Weight')
    ax3.grid(True, alpha=0.3)
    
    # 4. Annual Returns Comparison
    ax4 = fig.add_subplot(gs[3, 1])
    # Resample
    ann_ret_1 = df_final['Strat_1_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
    ann_ret_2 = df_final['Strat_2_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    # Plot grouped bar chart
    width = 0.35
    x = np.arange(len(ann_ret_1))
    years = ann_ret_1.index.year
    
    ax4.bar(x - width/2, ann_ret_1.values, width, label='Strat 1 (120)', color='orange', alpha=0.8)
    ax4.bar(x + width/2, ann_ret_2.values, width, label='Strat 2 (400)', color='blue', alpha=0.8)
    
    ax4.set_title('Annual Returns Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(years, rotation=45)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.2)

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
            .title-s1 {{ color: #e65100; font-weight:bold; }}
            .title-s2 {{ color: #0d47a1; font-weight:bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Comparison: SMA 120 Timed vs SMA 400 Continuous</h1>
            
            <div class="stats">
                <div class="card s1">
                    <div class="title-s1">Strategy 1 (SMA 120 Weighted)</div>
                    <hr>
                    <div class="val">{stats['S1_Total']:.0f}%</div>
                    <div class="lbl">Total Equity Return</div>
                    <div class="val" style="font-size: 1.2em; margin-top:10px;">{stats['S1_Win']:.1f}%</div>
                    <div class="lbl">Win Rate (per 40d signal)</div>
                </div>
                <div class="card s2">
                    <div class="title-s2">Strategy 2 (SMA 400 Continuous)</div>
                    <hr>
                    <div class="val">{stats['S2_Total']:.0f}%</div>
                    <div class="lbl">Total Equity Return</div>
                    <div class="lbl" style="margin-top:10px;">Trend Following (Always In)</div>
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
