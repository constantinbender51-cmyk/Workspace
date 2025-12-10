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
    # Assuming risk-free rate is 0 for simplicity in crypto context
    return np.sqrt(365) * (returns.mean() / returns.std())

def grid_search(df):
    """
    Runs a grid search on the first 50% of data.
    Returns: (best_period, best_sharpe, training_cutoff_date)
    """
    split_idx = int(len(df) * 0.5)
    train_df = df.iloc[:split_idx].copy()
    cutoff_date = train_df.index[-1]
    
    best_sharpe = -float('inf')
    best_period = 360 # Default fallback
    
    print("Running Grid Search on Training Set (First 50%)...")
    
    sma_range = range(10, 410, 10)
    
    for period in sma_range:
        # Calculate temp SMA
        sma = train_df['close'].rolling(window=period).mean()
        
        # Logic: Long if Close > SMA, Short if Close < SMA
        # 1 = Long, -1 = Short
        position = np.where(train_df['close'] > sma, 1, -1)
        
        # Shift position to avoid look-ahead
        position = pd.Series(position, index=train_df.index).shift(1)
        
        # Calculate Returns
        market_return = np.log(train_df['close'] / train_df['close'].shift(1))
        strategy_return = position * market_return
        
        sharpe = calculate_sharpe(strategy_return)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_period = period
            
    return best_period, best_sharpe, cutoff_date

def run_strategy(df, best_period):
    """Calculates all SMAs for viz, but runs logic only on best_period."""
    data = df.copy()
    
    # 1. Pre-calculate ALL SMAs for the visualization
    sma_cols = []
    for period in range(10, 410, 10):
        col_name = f'SMA_{period}'
        data[col_name] = data['close'].rolling(window=period).mean()
        sma_cols.append(col_name)

    # 2. Run Strategy Logic using the OPTIMIZED period
    signal_col = f'SMA_{best_period}'
    
    conditions = [
        (data['close'] > data[signal_col]),
        (data['close'] < data[signal_col])
    ]
    choices = [1, -1] # Long, Short
    
    data['position'] = np.select(conditions, choices, default=0)
    data['position'] = data['position'].shift(1) 
    
    # Returns
    data['market_return'] = np.log(data['close'] / data['close'].shift(1))
    data['strategy_return'] = data['position'] * data['market_return']
    
    # Equity
    data['equity'] = data['strategy_return'].cumsum().apply(np.exp)
    data['buy_hold_equity'] = data['market_return'].cumsum().apply(np.exp)
    
    return data, sma_cols

def generate_plot(df, best_period, training_cutoff):
    """Generates the plot with colormap for SMAs and split line."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- PLOT 1: Price & Rainbow SMAs ---
    
    # Create a color map instance
    cmap = matplotlib.colormaps['jet'] # 'jet' or 'turbo' gives high contrast for spectral data
    norm = mcolors.Normalize(vmin=10, vmax=400)
    
    # Plot all SMAs using the colormap
    sma_range = range(10, 410, 10)
    for period in sma_range:
        col_name = f'SMA_{period}'
        # Color based on period length
        color = cmap(norm(period))
        
        if period == best_period:
            continue # Skip the winner, we plot it last on top
        else:
            ax1.plot(df.index, df[col_name], color=color, alpha=0.5, linewidth=1)

    # Plot the Winner on top
    winner_col = f'SMA_{best_period}'
    ax1.plot(df.index, df[winner_col], label=f'Best SMA: {best_period}', color='white', linewidth=3, path_effects=[matplotlib.patheffects.withStroke(linewidth=5, foreground='black')])
    
    # Plot Price
    ax1.plot(df.index, df['close'], label='Price', color='black', linewidth=1.5, alpha=0.8)
    
    # Training Split Line
    ax1.axvline(training_cutoff, color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    
    ax1.set_title(f'{SYMBOL} Price & 40 SMAs (Color Mapped) | Optimal: SMA {best_period}', fontsize=14)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log') # Log scale often looks better for long-term crypto

    # --- PLOT 2: Equity Curve ---
    ax2.plot(df.index, df['equity'], label='Strategy Equity', color='green', linewidth=2)
    ax2.plot(df.index, df['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.7, linestyle='--')
    
    # Training Split Line on Equity
    ax2.axvline(training_cutoff, color='red', linestyle='--', linewidth=2)
    ax2.text(training_cutoff, df['equity'].mean(), '  <-- Training (In-Sample) | Test (Out-of-Sample) -->', color='red', fontsize=10, verticalalignment='center')

    ax2.set_title('Equity Curve (Log Returns Accumulated)', fontsize=14)
    ax2.set_ylabel('Normalized Equity')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig)
    return data

@app.route('/')
def index():
    try:
        # 1. Fetch
        df = fetch_data()
        
        # 2. Grid Search (Train on 50%)
        best_period, best_sharpe, split_date = grid_search(df)
        
        # 3. Run Strategy (Full Data)
        result_df, _ = run_strategy(df, best_period)
        
        # 4. Plot
        plot_data = generate_plot(result_df, best_period, split_date)
        
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
                .highlight {{ color: #4caf50; font-weight: bold; }}
                img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                button {{ background: #2196F3; color: white; border: none; padding: 10px 20px; font-size: 16px; cursor: pointer; border-radius: 4px; margin-top: 20px; }}
                button:hover {{ background: #1976D2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Trading Lab: SMA Grid Search</h1>
                <p>Optimization run on first 50% of data (In-Sample)</p>
                
                <div class="stats">
                    <span>Best SMA Found: <span class="highlight">{best_period}</span></span>
                    <span>Training Sharpe: <span class="highlight">{best_sharpe:.2f}</span></span>
                    <span>Split Date: {split_date.date()}</span>
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
    print(f"Starting Grid Search Server on {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=True)
