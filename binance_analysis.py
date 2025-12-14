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

# --- Test Parameters for Strategy 2 (SMA 400) ---
TEST_PARAMS = [
    (0.05, 0.27, "S2-A (5% Prox / 27% Stop)"),  # Proximity 5%, Stop 27%
    (0.05, 0.16, "S2-B (5% Prox / 16% Stop)"),  # Proximity 5%, Stop 16%
    (0.22, 0.16, "S2-C (22% Prox / 16% Stop)") # Proximity 22%, Stop 16%
]

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
    
    # Pre-calculate base returns and SMAs
    df['Asset_Ret'] = df['close'].pct_change().fillna(0.0)
    df['SMA_120'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['SMA_400'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    df['Trend_Base_400'] = np.where(df['close'] > df['SMA_400'], 1, -1)
    df['Dist_Pct_400'] = (df['close'] - df['SMA_400']).abs() / df['SMA_400']

    df.dropna(inplace=True)
    print(f"Data ready: {len(df)} rows.")
    return df

# --- Strategy 1: SMA 120 Weighted Decay (Returns DataFrame) ---
def run_strategy_1(df):
    data = df.copy()
    
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
            
    data['Active_Pos'] = pd.Series(position_arr, index=data.index).shift(1).fillna(0.0)
    data['Strat_Daily_Ret'] = data['Active_Pos'] * data['Asset_Ret']
    data['Equity'] = (1 + data['Strat_Daily_Ret']).cumprod()
    
    return data['Equity']

# --- Strategy 2: SMA 400 Proximity + Trailing Stop (Returns Series) ---
def run_strategy_2_test(df, prox_threshold, stop_threshold_pct):
    """Calculates Equity Curve for S2 with given parameters."""
    
    trends = df['Trend_Base_400'].values
    dists = df['Dist_Pct_400'].values
    asset_rets = df['Asset_Ret'].values
    n = len(df)
    
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
                
    # Calculate final equity curve
    equity_curve = (1 + pd.Series(strat_daily_rets, index=df.index)).cumprod()
    return equity_curve

# --- Helper for Sharpe Ratio Calculation ---
def calculate_sharpe(equity_series):
    # Daily returns = (Equity today / Equity yesterday) - 1
    daily_returns = equity_series.pct_change().dropna()
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    
    if std_ret == 0:
        return 0.0
        
    return (mean_ret / std_ret) * np.sqrt(365)


# --- Dashboard ---
@app.route('/')
def dashboard():
    df = fetch_data()
    
    equity_curves = {}
    
    # 1. Run Strategy 1 (SMA 120 Decay)
    equity_s1 = run_strategy_1(df)
    equity_curves['S1 (SMA 120 Decay)'] = equity_s1
    
    # 2. Run Strategy 2 parameter tests
    for prox_pct, stop_pct, label in TEST_PARAMS:
        equity_s2 = run_strategy_2_test(df, prox_pct, stop_pct)
        equity_curves[label] = equity_s2

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    
    ax.set_title(f'{SYMBOL} Equity Curve Comparison (Daily Compounding)', fontsize=14, fontweight='bold')
    ax.axhline(1.0, color='gray', linestyle='--')
    
    summary_data = []

    # Plot and calculate metrics
    for label, equity_series in equity_curves.items():
        total_ret = (equity_series.iloc[-1] - 1) * 100
        sharpe = calculate_sharpe(equity_series)
        
        summary_data.append({
            'Strategy': label,
            'Total Return': f"{total_ret:,.0f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Equity': f"{equity_series.iloc[-1]:.2f}x"
        })
        
        color = 'orange' if 'S1' in label else ('blue' if '5%' in label else 'red')
        style = '--' if 'S1' in label else '-'
        
        ax.plot(equity_series.index, equity_series.values, 
                label=f'{label} (Sharpe: {sharpe:.2f} / Ret: {total_ret:.0f}%)', 
                color=color, linestyle=style, linewidth=2 if 'S1' not in label else 1.5)
        
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Equity Multiple (Starting at 1.0)')
    
    # Save Plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    # Convert summary data to HTML table
    table_html = "<table><thead><tr><th>Strategy</th><th>Total Return</th><th>Sharpe Ratio</th><th>Max Equity</th></tr></thead><tbody>"
    for row in summary_data:
        table_html += f"<tr><td>{row['Strategy']}</td><td>{row['Total Return']}</td><td>{row['Sharpe Ratio']}</td><td>{row['Max Equity']}</td></tr>"
    table_html += "</tbody></table>"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Strategy Comparison Test</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; padding: 20px; text-align: center; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 1em; text-align: left; }}
            th, td {{ padding: 12px 15px; border: 1px solid #ddd; }}
            th {{ background-color: #007bff; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Equity Curve Comparison: Parameter Sensitivity</h1>
            <p>Comparing SMA 120 Decay vs. SMA 400 (Trailing Stop/Proximity Re-entry) under specific parameters.</p>
            
            {table_html}
            
            <img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;" />
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting comparison server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
