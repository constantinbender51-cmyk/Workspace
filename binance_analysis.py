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
EVENT_STUDY_WINDOW = 365 # Extended to view the full year of Strat 2

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

# --- Strategy 1: SMA 120 Weighted Decay (1 - (d/40)^2) ---
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

# --- Strategy 2: SMA 400 Weighted Growth ((d/365)^2) ---
def run_strategy_2(df):
    data = df.copy() # Works on top of Strat 1 df
    data['SMA_400'] = data['close'].rolling(window=SMA_PERIOD_2).mean()
    
    closes = data['close'].values
    smas = data['SMA_400'].values
    n = len(data)
    position_arr = np.zeros(n)
    
    current_trend = 0
    last_signal_idx = -9999
    
    for i in range(SMA_PERIOD_2, n):
        # Determine trend
        trend_now = 1 if closes[i] > smas[i] else -1
        
        # If trend flips, we reset the timer (scale in starts from 0 again)
        if trend_now != current_trend:
            current_trend = trend_now
            last_signal_idx = i 
            
        d_trade = i - last_signal_idx
        
        # Weighted Growth Days 1-365
        if 1 <= d_trade <= 365:
            # Formula: (d/365)^2
            # This starts small and grows parabolically
            weight = (d_trade / 365.0)**2
            position_arr[i] = current_trend * weight
        else:
            # Go flat after 1 year (365 days)
            position_arr[i] = 0.0
    
    data['Pos_2'] = position_arr
    # Enter next day
    data['Active_Pos_2'] = data['Pos_2'].shift(1).fillna(0.0)
    
    # Calc Returns
    data['Strat_2_Daily_Ret'] = data['Active_Pos_2'] * data['Asset_Ret']
    data['Equity_2'] = (1 + data['Strat_2_Daily_Ret']).cumprod()
    
    return data

def analyze_stats(df):
    # --- Stats Strat 1 ---
    s1_total_ret = (df['Equity_1'].iloc[-1] - 1) * 100
    
    # --- Stats Strat 2 ---
    s2_total_ret = (df['Equity_2'].iloc[-1] - 1) * 100
    
    # Calculate simple annual returns for both
    ann_1 = df['Strat_1_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1).mean() * 100
    ann_2 = df['Strat_2_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1).mean() * 100
    
    return {
        'S1_Total': s1_total_ret,
        'S2_Total': s2_total_ret,
        'S1_Ann': ann_1,
        'S2_Ann': ann_2
    }

# --- Server & Plotting ---
@app.route('/')
def dashboard():
    df_raw = fetch_data()
    df_s1 = run_strategy_1(df_raw)
    df_final = run_strategy_2(df_s1)
    
    stats = analyze_stats(df_final)
    
    # Plotting
    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    gs = fig.add_gridspec(5, 2)
    
    # 1. Price + SMAs + Active Zones
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.set_title(f'{SYMBOL} Analysis: SMA 120 (Decay) vs SMA 400 (Growth)', fontsize=14, fontweight='bold')
    ax1.plot(df_final.index, df_final['close'], color='black', alpha=0.5, label='Price', linewidth=0.8)
    ax1.plot(df_final.index, df_final['SMA_120'], color='orange', linestyle='--', label='SMA 120', linewidth=1)
    ax1.plot(df_final.index, df_final['SMA_400'], color='blue', linestyle='-', label='SMA 400', linewidth=1.5)
    
    # Highlight Active Zones (Simple fills for visibility)
    # Strat 1 (Decay)
    ax1.fill_between(df_final.index, df_final['close'].min(), df_final['close'].max(), 
                     where=(df_final['Active_Pos_1'].abs() > 0.1), color='orange', alpha=0.1, label='S1 Active (Decay)')
    # Strat 2 (Growth)
    ax1.fill_between(df_final.index, df_final['close'].min(), df_final['close'].max(), 
                     where=(df_final['Active_Pos_2'].abs() > 0.1), color='blue', alpha=0.1, label='S2 Active (Growth)')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylabel('Price (Log Scale)')
    ax1.set_yscale('log')
    
    # 2. Equity Curves Comparison
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_title(f"Equity Curves (Daily Compounded)", fontsize=12, fontweight='bold')
    
    ax2.plot(df_final.index, df_final['Equity_1'], color='orange', linewidth=2, label=f'S1: SMA 120 (40d Decay) | Total: {stats["S1_Total"]:.0f}%')
    ax2.plot(df_final.index, df_final['Equity_2'], color='blue', linewidth=2, label=f'S2: SMA 400 (365d Growth) | Total: {stats["S2_Total"]:.0f}%')
    
    ax2.axhline(1.0, color='black', linestyle='--')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Equity Multiple')
    
    # 3. Weight Profile S1 (Decay)
    ax3 = fig.add_subplot(gs[3, 0])
    days_1 = np.arange(0, 50)
    weights_1 = [1 - (d/40)**2 if 1 <= d <= 40 else 0 for d in days_1]
    ax3.plot(days_1, weights_1, color='orange', marker='.', markersize=5)
    ax3.set_title('Strategy 1 Weight: 1 - (d/40)^2')
    ax3.set_xlabel('Days Since Signal')
    ax3.set_ylabel('Exposure')
    ax3.grid(True, alpha=0.3)

    # 4. Weight Profile S2 (Growth)
    ax4 = fig.add_subplot(gs[3, 1])
    days_2 = np.arange(0, 400)
    weights_2 = [(d/365)**2 if 1 <= d <= 365 else 0 for d in days_2]
    ax4.plot(days_2, weights_2, color='blue', linewidth=2)
    ax4.set_title('Strategy 2 Weight: (d/365)^2')
    ax4.set_xlabel('Days Since Signal')
    ax4.set_ylabel('Exposure')
    ax4.grid(True, alpha=0.3)
    
    # 5. Annual Returns Comparison
    ax5 = fig.add_subplot(gs[4, :])
    ann_ret_1 = df_final['Strat_1_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
    ann_ret_2 = df_final['Strat_2_Daily_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    width = 0.35
    x = np.arange(len(ann_ret_1))
    years = ann_ret_1.index.year
    
    ax5.bar(x - width/2, ann_ret_1.values, width, label='Strat 1 (Decay)', color='orange', alpha=0.8)
    ax5.bar(x + width/2, ann_ret_2.values, width, label='Strat 2 (Growth)', color='blue', alpha=0.8)
    
    ax5.set_title('Annual Returns Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(years, rotation=45)
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.legend()
    ax5.grid(True, alpha=0.2)

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
            <h1>Strategy Comparison: SMA 120 Decay vs SMA 400 Growth</h1>
            
            <div class="stats">
                <div class="card s1">
                    <div class="title-s1">Strategy 1 (SMA 120)</div>
                    <hr>
                    <div class="val">{stats['S1_Total']:.0f}%</div>
                    <div class="lbl">Total Equity Return</div>
                    <div class="lbl" style="margin-top:5px">Weight: Starts 100%, Decays to 0% (40 days)</div>
                </div>
                <div class="card s2">
                    <div class="title-s2">Strategy 2 (SMA 400)</div>
                    <hr>
                    <div class="val">{stats['S2_Total']:.0f}%</div>
                    <div class="lbl">Total Equity Return</div>
                    <div class="lbl" style="margin-top:5px">Weight: Starts ~0%, Grows to 100% (365 days)</div>
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
