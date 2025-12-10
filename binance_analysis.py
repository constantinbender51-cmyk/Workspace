import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from flask import Flask, render_template_string
import io
import base64

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' 
START_DATE = '2018-01-01 00:00:00'
PORT = 8080

def fetch_data():
    """Fetches historical OHLCV data from Binance."""
    exchange = ccxt.binance()
    since = exchange.parse8601(START_DATE)
    all_candles = []
    
    print(f"Fetching {SYMBOL} data from {START_DATE}...")
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not candles:
                break
            
            last_timestamp = candles[-1][0]
            since = last_timestamp + 1
            all_candles += candles
            
            if len(candles) < 1000:
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_sharpe(returns):
    """Calculates annualized Sharpe Ratio (Crypto 365 days)."""
    if returns.std() == 0:
        return 0
    return np.sqrt(365) * (returns.mean() / returns.std())

def calculate_efficiency_ratio(df, period=30):
    """
    Calculates the Efficiency Ratio (III):
    Numerator: abs(sum(log_returns)) over period
    Denominator: sum(abs(log_returns)) over period
    """
    # Calculate Log Returns
    log_ret = np.log(df['close'] / df['close'].shift(1))
    
    # Numerator: Absolute value of the net directional move
    numerator = log_ret.rolling(window=period).sum().abs()
    
    # Denominator: Sum of the absolute individual moves (volatility/path length)
    denominator = log_ret.abs().rolling(window=period).sum()
    
    # Efficiency Ratio
    er = numerator / denominator
    
    return er

def grid_search(df):
    """
    Runs a 2D grid search on the first 50% of data.
    Optimizes:
    1. SMA Period (10-400)
    2. III Threshold (0.1 - 0.55) for leverage reduction
    """
    split_idx = int(len(df) * 0.5)
    train_df = df.iloc[:split_idx].copy()
    cutoff_date = train_df.index[-1]
    
    # 1. Calculate III ONCE (invariant of SMA)
    train_df['iii'] = calculate_efficiency_ratio(train_df, period=30)
    market_return = np.log(train_df['close'] / train_df['close'].shift(1))
    
    best_sharpe = -float('inf')
    best_period = 360
    best_threshold = 0.3
    
    print("Running 2D Grid Search (SMA x III Threshold)...")
    
    sma_range = range(10, 410, 10)
    # Search thresholds from 0.10 to 0.55 in steps of 0.05
    threshold_range = np.arange(0.10, 0.60, 0.05)
    
    for period in sma_range:
        # Calculate SMA Signal for this period
        sma = train_df['close'].rolling(window=period).mean()
        
        # Raw Direction: 1 (Long), -1 (Short)
        direction = np.where(train_df['close'] > sma, 1, -1)
        direction = pd.Series(direction, index=train_df.index).shift(1)
        
        # Inner Loop: Optimize III Threshold
        for thresh in threshold_range:
            # Leverage Logic: 0.5 if iii < thresh, else 1.0
            # Note: We shift leverage 1 step to match execution (decision made on close)
            leverage_mult = np.where(train_df['iii'] < thresh, 0.5, 1.0)
            leverage_series = pd.Series(leverage_multiplier_calc(train_df['iii'], thresh), index=train_df.index).shift(1)
            
            # Combine
            strategy_return = direction * leverage_series * market_return
            sharpe = calculate_sharpe(strategy_return)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_period = period
                best_threshold = thresh
            
    return best_period, best_threshold, best_sharpe, cutoff_date

def leverage_multiplier_calc(iii_series, threshold):
    """Helper for broadcasting numpy comparison"""
    return np.where(iii_series < threshold, 0.5, 1.0)

def run_strategy(df, best_period, best_threshold):
    """Calculates final strategy with optimized SMA and Threshold."""
    data = df.copy()
    
    # 1. Calculate Efficiency Ratio (III)
    data['efficiency_ratio'] = calculate_efficiency_ratio(data, period=30)
    
    # 2. Pre-calculate ALL SMAs for visualization
    sma_cols = []
    for period in range(10, 410, 10):
        col_name = f'SMA_{period}'
        data[col_name] = data['close'].rolling(window=period).mean()
        sma_cols.append(col_name)

    # 3. Strategy Logic
    signal_col = f'SMA_{best_period}'
    
    # A. Directional Signal
    direction_conditions = [
        (data['close'] > data[signal_col]),
        (data['close'] < data[signal_col])
    ]
    direction_choices = [1, -1]
    data['raw_direction'] = np.select(direction_conditions, direction_choices, default=0)
    
    # B. Leverage Signal (Using Optimized Threshold)
    data['leverage_mult'] = np.where(data['efficiency_ratio'] < best_threshold, 0.5, 1.0)
    
    # C. Final Position (Shifted 1 for lookahead prevention)
    data['position'] = (data['raw_direction'] * data['leverage_mult']).shift(1)
    
    # Returns
    data['market_return'] = np.log(data['close'] / data['close'].shift(1))
    data['strategy_return'] = data['position'] * data['market_return']
    
    # Equity
    data['equity'] = data['strategy_return'].cumsum().apply(np.exp)
    data['buy_hold_equity'] = data['market_return'].cumsum().apply(np.exp)
    
    return data, sma_cols

def generate_plot(df, best_period, best_threshold, training_cutoff):
    """Generates the 3-panel plot."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # --- PLOT 1: Price & SMAs ---
    cmap = matplotlib.colormaps['jet'] 
    norm = mcolors.Normalize(vmin=10, vmax=400)
    
    sma_range = range(10, 410, 10)
    for period in sma_range:
        col_name = f'SMA_{period}'
        color = cmap(norm(period))
        if period == best_period:
            continue 
        else:
            ax1.plot(df.index, df[col_name], color=color, alpha=0.5, linewidth=1)

    winner_col = f'SMA_{best_period}'
    ax1.plot(df.index, df[winner_col], label=f'Best SMA: {best_period}', color='gold', linewidth=4, zorder=10)
    ax1.plot(df.index, df['close'], label='Price', color='white', linewidth=1.5, alpha=0.9, zorder=5) 
    ax1.axvline(training_cutoff, color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    
    ax1.set_title(f'{SYMBOL} Price & 40 SMAs | Optimal: SMA {best_period}', fontsize=14)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log') 

    # --- PLOT 2: Equity Curve ---
    ax2.plot(df.index, df['equity'], label='Strategy Equity', color='lime', linewidth=2)
    ax2.plot(df.index, df['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.7, linestyle='--')
    ax2.axvline(training_cutoff, color='red', linestyle='--', linewidth=2)
    
    ax2.set_title('Equity Curve (Log Returns Accumulated)', fontsize=14)
    ax2.set_ylabel('Normalized Equity')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # --- PLOT 3: Efficiency Ratio & Leverage Zones ---
    ax3.plot(df.index, df['efficiency_ratio'], color='cyan', linewidth=1.5, label='Efficiency Ratio (III)')
    
    # Plot Optimized Threshold
    ax3.axhline(best_threshold, color='orange', linestyle='-', linewidth=2, label=f'Optimal Threshold ({best_threshold:.2f})')
    
    # Fill background
    ax3.fill_between(df.index, 0, best_threshold, color='red', alpha=0.1, label='Zone: 0.5x Leverage')
    ax3.fill_between(df.index, best_threshold, 1.0, color='green', alpha=0.1, label='Zone: 1.0x Leverage')
    
    ax3.axvline(training_cutoff, color='red', linestyle='--', linewidth=2)

    ax3.set_title(r'Efficiency Ratio (III) & Leverage Regime', fontsize=14)
    ax3.set_ylabel('Efficiency (0=Chop, 1=Trend)')
    ax3.set_xlabel('Date')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig)
    return data

@app.route('/')
def index():
    try:
        # 1. Fetch
        df = fetch_data()
        
        # 2. 2D Grid Search
        best_period, best_threshold, best_sharpe, split_date = grid_search(df)
        
        # 3. Run Strategy (Full Data with Optimal Params)
        result_df, _ = run_strategy(df, best_period, best_threshold)
        
        # 4. Plot
        plot_data = generate_plot(result_df, best_period, best_threshold, split_date)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Grid Search Backtest: {SYMBOL}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; text-align: center; background: #1e1e1e; color: #eee; padding: 20px; }}
                .container {{ background: #2d2d2d; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); display: inline-block; max-width: 95%; }}
                h1 {{ color: #fff; margin-bottom: 5px; }}
                .stats {{ background: #333; padding: 15px; border-radius: 8px; margin: 20px 0; border: 1px solid #444; }}
                .stats span {{ margin: 0 15px; font-size: 1.1em; }}
                .highlight {{ color: #FFD700; font-weight: bold; }} 
                img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                button {{ background: #2196F3; color: white; border: none; padding: 10px 20px; font-size: 16px; cursor: pointer; border-radius: 4px; margin-top: 20px; }}
                button:hover {{ background: #1976D2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Trading Lab: 2D Parameter Optimization</h1>
                <p>Optimization run on first 50% (In-Sample). Strategy scales leverage based on Efficiency Ratio.</p>
                
                <div class="stats">
                    <span>Best SMA: <span class="highlight">{best_period}</span></span>
                    <span>Best III Cutoff: <span class="highlight">{best_threshold:.2f}</span></span>
                    <span>Training Sharpe: <span class="highlight">{best_sharpe:.2f}</span></span>
                    <span>Split Date: {split_date.date()}</span>
                </div>
                
                <div style="margin-bottom: 20px; color: #aaa;">
                    Rule: If <b>III < {best_threshold:.2f}</b>, Leverage = <b>0.5x</b>. Else <b>1.0x</b>.
                </div>

                <img src="data:image/png;base64,{plot_data}" alt="Backtest Plot">
                <br>
                <button onclick="location.reload()">Run Again</button>
            </div>
        </body>
        </html>
        """
        return render_template_string(html)
        
    except Exception as e:
        import traceback
        return f"<h1>Error</h1><pre>{traceback.format_exc()}</pre>"

if __name__ == '__main__':
    print(f"Starting Server on {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=True)
