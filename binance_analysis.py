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
WINDOW_SIZE = 800 # Rolling optimization window

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

def calculate_efficiency_ratio(df, period=30):
    log_ret = np.log(df['close'] / df['close'].shift(1))
    numerator = log_ret.rolling(window=period).sum().abs()
    denominator = log_ret.abs().rolling(window=period).sum()
    return numerator / denominator

def walk_forward_optimization(df):
    """
    Performs Vectorized Walk-Forward Optimization.
    1. Pre-calculates returns for ALL parameter combos (40 SMAs * 10 Thresholds).
    2. Calculates rolling Sharpe for all combos.
    3. Selects best combo daily.
    """
    print("Preparing Vectorized Strategy Matrix...")
    data = df.copy()
    data['market_return'] = np.log(data['close'] / data['close'].shift(1))
    data['iii'] = calculate_efficiency_ratio(data, period=30)
    
    # 1. Generate Parameter Ranges
    sma_periods = list(range(10, 410, 10))
    thresholds = np.arange(0.10, 0.60, 0.05)
    
    # 2. Build Strategy Returns Matrix (N_days x N_Strategies)
    # We will use a dictionary to store Series, then concat
    strategy_returns = {}
    
    # Pre-calculate SMA signals (Vectorized)
    # This might take a moment but is faster than loops for Sharpe calculation
    for period in sma_periods:
        sma = data['close'].rolling(window=period).mean()
        # Direction: 1 (Long), -1 (Short)
        # Shifted by 1 later, but we calculate raw signal first
        raw_direction = np.where(data['close'] > sma, 1, -1)
        
        for thresh in thresholds:
            col_name = f"{period}_{thresh:.2f}"
            
            # Leverage Logic: 0.5 if iii < thresh else 1.0
            leverage = np.where(data['iii'] < thresh, 0.5, 1.0)
            
            # Position = Direction * Leverage
            # Note: We do NOT shift here yet because we want to align Return(t) with Signal(t-1)
            # But to build the "Strategy Return at time t", we need Pos(t-1) * MktRet(t)
            
            # Let's construct the vector:
            # Pos(t) based on Close(t)
            pos_vector = raw_direction * leverage
            
            # Strategy Return(t) = Pos(t-1) * MarketRet(t)
            # We can calculate this using shift
            strat_ret = pd.Series(pos_vector, index=data.index).shift(1) * data['market_return']
            
            strategy_returns[col_name] = strat_ret

    # Convert to DataFrame (N_days x 400 columns)
    strat_df = pd.DataFrame(strategy_returns)
    
    # 3. Calculate Rolling Sharpe (Vectorized)
    print(f"Calculating Rolling Sharpe ({WINDOW_SIZE} days) for {len(strat_df.columns)} strategies...")
    
    # Mean / Std (We can ignore sqrt(365) for ranking purposes)
    rolling_mean = strat_df.rolling(window=WINDOW_SIZE).mean()
    rolling_std = strat_df.rolling(window=WINDOW_SIZE).std()
    
    # Avoid division by zero
    rolling_sharpe = rolling_mean / (rolling_std + 1e-9)
    
    # 4. Select Best Strategy Daily
    # idxmax returns the column name of the max value
    print("Selecting best strategies...")
    best_strategy_col = rolling_sharpe.idxmax(axis=1)
    
    # 5. Construct Walk-Forward Equity Curve
    # For day t, we use the strategy selected based on window ending at t-1
    # So we shift the selection signal by 1
    selected_strategy_lagged = best_strategy_col.shift(1)
    
    # We need to pick the specific return from strat_df based on the column name in selected_strategy_lagged
    # Pandas 'lookup' is deprecated, using specialized indexing
    
    # Create a Series for the WFO returns
    wfo_returns = pd.Series(0.0, index=data.index)
    
    # Optimization: Iterate only where selection changes or just loop (vector lookup is hard with changing cols)
    # Faster approach: Use numpy indexing
    
    # Get integer indices for columns
    col_map = {name: i for i, name in enumerate(strat_df.columns)}
    # Map column names to integers, fill NaNs with 0 (default col)
    col_indices = selected_strategy_lagged.map(col_map).fillna(0).astype(int)
    
    # Extract numpy array of returns
    returns_arr = strat_df.values
    # Row indices
    row_indices = np.arange(len(data))
    
    # Select returns: returns_arr[row, col_selected_for_that_row]
    # Note: col_indices is based on t-1, which is what we want (decision made yesterday applied today)
    selected_returns = returns_arr[row_indices, col_indices]
    
    # Handle the warm-up period (first WINDOW_SIZE + 1 days are invalid)
    # +1 because of the lag
    selected_returns[:WINDOW_SIZE+1] = 0
    
    data['strategy_return'] = selected_returns
    data['equity'] = data['strategy_return'].cumsum().apply(np.exp)
    data['buy_hold_equity'] = data['market_return'].cumsum().apply(np.exp)
    
    # Store dynamic params for plotting
    # Parse "period_thresh" string back to values
    def parse_params(val):
        if pd.isna(val): return None, None
        p, t = val.split('_')
        return int(p), float(t)
        
    params = selected_strategy_lagged.apply(parse_params)
    # Create DF from list of tuples
    params_df = pd.DataFrame(params.tolist(), index=params.index, columns=['best_sma', 'best_thresh'])
    data = pd.concat([data, params_df], axis=1)
    
    return data

def generate_plot(df):
    """Generates the plot showing Equity and Dynamic Parameter Evolution."""
    # Filter out warm-up period for cleaner plots
    plot_df = df.iloc[WINDOW_SIZE+1:].copy()
    
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # --- PLOT 1: Equity Curves ---
    ax1.plot(plot_df.index, plot_df['equity'], label='Walk-Forward Strategy', color='lime', linewidth=2)
    ax1.plot(plot_df.index, plot_df['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    
    ax1.set_title(f'Walk-Forward Optimization (Rolling {WINDOW_SIZE} Days)', fontsize=14)
    ax1.set_ylabel('Normalized Equity')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # --- PLOT 2: Best SMA Evolution ---
    ax2.scatter(plot_df.index, plot_df['best_sma'], c=plot_df['best_sma'], cmap='turbo', s=10, alpha=0.6)
    # Add a moving average of the selection to show trend
    ax2.plot(plot_df.index, plot_df['best_sma'].rolling(30).mean(), color='white', linewidth=1.5, alpha=0.8, label='30d Avg Selection')
    
    ax2.set_title('Dynamic SMA Selection (Optimized Daily)', fontsize=12)
    ax2.set_ylabel('Selected SMA Period')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- PLOT 3: Best Threshold Evolution ---
    ax3.plot(plot_df.index, plot_df['best_thresh'], color='orange', linewidth=1.5)
    ax3.fill_between(plot_df.index, 0, plot_df['best_thresh'], color='orange', alpha=0.2)
    
    ax3.set_title('Dynamic III Threshold Selection', fontsize=12)
    ax3.set_ylabel('Selected III Threshold')
    ax3.set_xlabel('Date')
    ax3.set_ylim(0, 0.6)
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
        
        # 2. Walk-Forward Optimization
        result_df = walk_forward_optimization(df)
        
        # 3. Stats
        # Calculate stats on the WFO period only
        valid_rets = result_df['strategy_return'].iloc[WINDOW_SIZE+1:]
        total_return = result_df['equity'].iloc[-1]
        sharpe = np.sqrt(365) * (valid_rets.mean() / valid_rets.std())
        
        # 4. Plot
        plot_data = generate_plot(result_df)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Walk-Forward Backtest: {SYMBOL}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; text-align: center; background: #1e1e1e; color: #eee; padding: 20px; }}
                .container {{ background: #2d2d2d; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); display: inline-block; max-width: 95%; }}
                h1 {{ color: #fff; margin-bottom: 5px; }}
                .stats {{ background: #333; padding: 15px; border-radius: 8px; margin: 20px 0; border: 1px solid #444; }}
                .stats span {{ margin: 0 15px; font-size: 1.1em; }}
                .highlight {{ color: #00e676; font-weight: bold; }} 
                img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                button {{ background: #2196F3; color: white; border: none; padding: 10px 20px; font-size: 16px; cursor: pointer; border-radius: 4px; margin-top: 20px; }}
                button:hover {{ background: #1976D2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Trading Lab: Walk-Forward Optimization</h1>
                <p>Daily Re-Optimization (Window: {WINDOW_SIZE} days). 400 Parameter Combos/Day.</p>
                
                <div class="stats">
                    <span>Total Return: <span class="highlight">{total_return:.2f}x</span></span>
                    <span>WFO Sharpe: <span class="highlight">{sharpe:.2f}</span></span>
                </div>
                
                <div style="margin-bottom: 20px; color: #aaa; font-size: 0.9em;">
                    The strategy recalibrates every single day based on the previous 100 days of data.
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
