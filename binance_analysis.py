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
PORT = 8080
EVENT_STUDY_WINDOW = 120 # Tracking 120 days post-signal for observation

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

# --- Time-Based Strategy Logic (40d ON with Weighted Decay) ---
def run_strategy(df):
    data = df.copy()
    data['SMA'] = data['close'].rolling(window=SMA_PERIOD).mean()
    
    # Initialize columns
    data['Raw_Trend'] = np.where(data['close'] > data['SMA'], 1, -1)
    data['Position'] = 0
    
    closes = data['close'].values
    smas = data['SMA'].values
    n = len(data)
    
    position_arr = np.zeros(n)
    
    # State variables
    current_trend = 0
    last_signal_idx = -9999
    
    for i in range(SMA_PERIOD, n):
        # Determine trend at this step
        trend_now = 1 if closes[i] > smas[i] else -1
        
        # Did the trend FLIP? This resets the entire sequence.
        if trend_now != current_trend:
            current_trend = trend_now
            last_signal_idx = i # Reset the clock
            
        # days_since_signal is the number of days passed (0 = signal day)
        days_since_signal = i - last_signal_idx
        
        # d_trade is the trade day number (1 to 40) for the position taken TODAY
        d_trade = days_since_signal
        
        # --- Weighted Decay Logic (Trade from Day 1 to Day 40) ---
        
        if 1 <= d_trade <= 40:
            # Weight formula: W_d = 1 - (d/40)^2
            weight = 1 - (d_trade / 40)**2
            # Position is Trend direction (+1 or -1) * Weight
            position_arr[i] = current_trend * weight
            
        # All other days (Day 0, or days > 40) are FLAT
        else:
            position_arr[i] = 0
            
    data['Position'] = position_arr
    
    # Shift position by 1 to prevent lookahead bias (signal on close -> enter next open)
    data['Active_Position'] = data['Position'].shift(1).fillna(0)
    
    # Calculate returns
    data['Log_Ret'] = np.log(data['close'] / data['close'].shift(1))
    # Strat_Ret is now Position Size * Log Return
    data['Strat_Ret'] = data['Active_Position'] * data['Log_Ret']
    
    return data.dropna()

def analyze_trades(df):
    # Determine distinct trade segments (blocks of non-zero position)
    # Since position is now fractional, we filter where position is not zero
    active_df = df[df['Active_Position'] != 0].copy()
    
    if active_df.empty:
        return pd.DataFrame(), pd.Series()

    # The entire 40-day block is now one "trade" per signal
    # We group by the start of the 40-day window
    
    # Identify signal days that actually resulted in a trade
    df['Trade_Start'] = (df['Active_Position'].abs() > 0) & (df['Active_Position'].shift(1).fillna(0).abs() == 0)
    signal_dates = df[df['Trade_Start']].index
    
    trades = []
    
    for start_date in signal_dates:
        # Find the end of the 40-day trade window
        end_date_index = df.index.get_loc(start_date) + 39
        if end_date_index >= len(df): continue
        end_date = df.index[end_date_index]
        
        trade_data = df.loc[start_date:end_date]
        
        if len(trade_data) < 1: continue

        # Calculate weighted return for this segment
        weighted_log_returns = trade_data['Strat_Ret'].sum()
        
        # Approximate unweighted return for win/loss (simplifying)
        position_direction = trade_data['Raw_Trend'].iloc[0]
        start_price = trade_data['close'].iloc[0]
        end_price = trade_data['close'].iloc[-1]
        
        if position_direction == 1:
            raw_ret = (end_price - start_price) / start_price
        else:
            raw_ret = (start_price - end_price) / start_price
            
        # The true return is the sum of weighted returns
        trades.append({
            'Trade_ID': start_date,
            'Position': 'Long' if position_direction == 1 else 'Short',
            'Start': start_date,
            'End': end_date,
            'Duration': len(trade_data),
            'Return': np.exp(weighted_log_returns) - 1, # Convert log returns to arithmetic
            'Win': weighted_log_returns > 0 # Win if total weighted log return is positive
        })
        
    trades_df = pd.DataFrame(trades)
    
    # Event Study: Average Price Curve after the *Initial* Signal (120 Days)
    df['Trend_Flip'] = (df['Raw_Trend'] != df['Raw_Trend'].shift(1))
    signal_dates_raw = df[df['Trend_Flip']].index
    
    curves = []
    look_forward = EVENT_STUDY_WINDOW 
    
    for date in signal_dates_raw:
        loc = df.index.get_loc(date)
        if loc + look_forward >= len(df): continue
        
        subset = df.iloc[loc : loc+look_forward]
        start_price = subset['close'].iloc[0]
        direction = subset['Raw_Trend'].iloc[0]
        
        norm_curve = subset['close'] / start_price
        
        # Invert Short trades for comparison
        if direction == -1:
            norm_curve = 2 - norm_curve
            
        norm_curve = norm_curve.reset_index(drop=True)
        curves.append(norm_curve)
        
    if curves:
        avg_curve = pd.concat(curves, axis=1).mean(axis=1)
    else:
        avg_curve = pd.Series()
        
    return trades_df, avg_curve

# --- Server & Plotting ---
@app.route('/')
def dashboard():
    df_raw = fetch_data()
    df_strat = run_strategy(df_raw)
    trades_df, avg_curve = analyze_trades(df_strat)
    
    # Calculate statistics based on trade segments
    if not trades_df.empty:
        total_trades = len(trades_df)
        win_rate = trades_df['Win'].mean() * 100
        # Total log return from the strategy
        cum_log_ret = df_strat['Strat_Ret'].cumsum().iloc[-1] 
        cum_ret = (np.exp(cum_log_ret) - 1) * 100 # Convert to arithmetic return percentage
        expectancy = trades_df['Return'].mean() * 100 # Average arithmetic return per trade
    else:
        total_trades = 0
        win_rate = 0
        cum_ret = 0
        expectancy = 0
    
    # Plotting
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    
    # 1. Main Price Chart with Strategy Zones
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.set_title(f'{SYMBOL} - SMA 120 Weighted Decay Strategy (40-Day Trade)', fontsize=14, fontweight='bold')
    ax1.plot(df_strat.index, df_strat['close'], color='black', alpha=0.6, label='Price', linewidth=1)
    ax1.plot(df_strat.index, df_strat['SMA'], color='orange', linestyle='--', label='SMA 120', alpha=0.8)
    
    # Highlight Active Zones (using fractional position for color intensity visual aid)
    ax1.fill_between(df_strat.index, df_strat['close'].min(), df_strat['close'].max(), 
                     where=(df_strat['Active_Position'] > 0), color='red', alpha=df_strat['Active_Position'].abs().clip(upper=0.3), label='Active Long (Weighted)')
    ax1.fill_between(df_strat.index, df_strat['close'].min(), df_strat['close'].max(), 
                     where=(df_strat['Active_Position'] < 0), color='blue', alpha=df_strat['Active_Position'].abs().clip(upper=0.3), label='Active Short (Weighted)')
    
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylabel('Price')
    
    # 2. Cumulative Returns
    ax2 = fig.add_subplot(gs[2, 0])
    df_strat['Cum_Ret'] = df_strat['Strat_Ret'].cumsum()
    ax2.plot(df_strat.index, df_strat['Cum_Ret'], color='green')
    ax2.fill_between(df_strat.index, df_strat['Cum_Ret'], 0, color='green', alpha=0.1)
    ax2.set_title(f'Cumulative Strategy Log Return (Total: {cum_ret:.0f}%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Event Study (The Curve - 120 Days)
    ax3 = fig.add_subplot(gs[2, 1])
    if not avg_curve.empty:
        # Plot the 120-day average curve
        ax3.plot(avg_curve.index, avg_curve.values, color='purple', linewidth=2)
        ax3.axhline(1.0, color='black', linestyle='-')
        
        # Highlight the active trade period (Day 1 to 40)
        ax3.axvspan(1, 40, color='green', alpha=0.15, label='Weighted Trade Period (40d)')
        
        ax3.set_title(f'Avg Price Move After SMA Signal ({EVENT_STUDY_WINDOW} Days)')
        ax3.set_xlabel('Days After Signal (Day 0 = Signal)')
        ax3.set_xlim(0, EVENT_STUDY_WINDOW)
        ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Annual Returns Bar Chart
    ax4 = fig.add_subplot(gs[3, :])
    annual_rets = df_strat['Strat_Ret'].resample('Y').apply(lambda x: np.exp(x.sum()) - 1)
    colors = ['green' if x > 0 else 'red' for x in annual_rets]
    ax4.bar(annual_rets.index.year, annual_rets.values, color=colors, alpha=0.7)
    ax4.set_title('Annual Strategy Returns (Arithmetic)')
    ax4.axhline(0, color='black', linewidth=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    # --- HTML Output ---
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Weighted Strategy Analysis</title>
        <style>
            body {{ font-family: sans-serif; background: #f0f2f5; padding: 20px; text-align: center; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 20px; }}
            .card {{ 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                border-left: 5px solid #007bff; 
                transition: transform 0.2s;
            }}
            .card:hover {{ transform: translateY(-5px); box-shadow: 0 6px 10px rgba(0,0,0,0.15); }}
            .val {{ font-size: 24px; font-weight: bold; color: #333; }}
            .lbl {{ font-size: 14px; color: #666; }}
            h1 {{ color: #2c3e50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SMA 120 Weighted Decay Strategy Analysis</h1>
            <p>Strategy: Trade Days 1-40 with position weight $W_d = 1 - (d/40)^2$.</p>
            <div class="stats">
                <div class="card"><div class="val">{win_rate:.2f}%</div><div class="lbl">Win Rate (Per 40-Day Signal)</div></div>
                <div class="card"><div class="val">{total_trades}</div><div class="lbl">Total 40-Day Signals</div></div>
                <div class="card"><div class="val">{expectancy:.2f}%</div><div class="lbl">Avg Return Per 40-Day Signal</div></div>
                <div class="card"><div class="val">{cum_ret:.0f}%</div><div class="lbl">Total Cumulative Arithmetic Return</div></div>
            </div>
            <img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;" alt="Trading Strategy Performance Plots" />
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting weighted analysis server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
