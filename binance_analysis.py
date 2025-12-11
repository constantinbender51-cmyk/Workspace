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

# Optimization Parameters
TRAIN_WINDOW = 400   # Lookback window for finding best params
RETRAIN_DAYS = 60    # How often we re-optimize (and hold params)
FIXED_III_THRESHOLD = 0.1 # Fixed leverage threshold

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
    Performs Walk-Forward Optimization with Periodic Retraining.
    1. Pre-calculates returns for all 40 SMA strategies.
    2. Steps through time in 60-day chunks.
    3. Looks back 400 days to pick best SMA.
    4. Applies best SMA to next 60 days.
    """
    print(f"Preparing Strategy Matrix (40 SMAs, Fixed Threshold {FIXED_III_THRESHOLD})...")
    data = df.copy()
    data['market_return'] = np.log(data['close'] / data['close'].shift(1))
    
    # 1. Calculate III and Leverage Multiplier ONCE
    data['iii'] = calculate_efficiency_ratio(data, period=30)
    # Leverage: 0.5 if iii < 0.1, else 1.0 (Shifted 1 step later)
    leverage_mult = np.where(data['iii'] < FIXED_III_THRESHOLD, 0.5, 1.0)
    
    # 2. Generate Parameter Range (SMA only)
    sma_periods = list(range(10, 410, 10)) # 40 periods
    
    # 3. Build Strategy Returns Matrix (N_days x 40 Strategies)
    strategy_returns = {}
    
    for period in sma_periods:
        # Calculate SMA and Raw Direction
        sma = data['close'].rolling(window=period).mean()
        raw_direction = np.where(data['close'] > sma, 1, -1)
        
        # Position = Direction * Fixed Leverage
        pos_vector = raw_direction * leverage_mult
        
        # Strategy Return(t) = Pos(t-1) * MarketRet(t)
        strat_ret = pd.Series(pos_vector, index=data.index).shift(1) * data['market_return']
        strategy_returns[period] = strat_ret

    # DataFrame of all potential strategy returns
    strat_df = pd.DataFrame(strategy_returns)
    
    # 4. Step-wise Walk Forward Optimization
    print(f"Running Step-wise WFO (Train: {TRAIN_WINDOW}d, Hold: {RETRAIN_DAYS}d)...")
    
    # Result containers
    wfo_returns = np.zeros(len(data))
    best_sma_log = np.full(len(data), np.nan)
    
    # Start loop after the first training window is available
    # We iterate in steps of RETRAIN_DAYS
    for t in range(TRAIN_WINDOW, len(data), RETRAIN_DAYS):
        
        # A. Define Training Set (Lookback 400 days from t)
        train_start = t - TRAIN_WINDOW
        train_end = t
        train_slice = strat_df.iloc[train_start:train_end]
        
        # B. Find Best SMA in Training Set (Max Sharpe)
        # Avoid division by zero
        sharpes = train_slice.mean() / (train_slice.std() + 1e-9)
        best_sma = sharpes.idxmax()
        
        # C. Define Test Set (Next 60 days from t)
        test_start = t
        test_end = min(t + RETRAIN_DAYS, len(data))
        
        # D. Apply Best SMA to Test Set
        # We take the column corresponding to best_sma
        wfo_returns[test_start:test_end] = strat_df[best_sma].iloc[test_start:test_end]
        
        # Log the parameter for plotting
        best_sma_log[test_start:test_end] = best_sma
        
    # 5. Store Results
    data['strategy_return'] = wfo_returns
    data['equity'] = data['strategy_return'].cumsum().apply(np.exp)
    data['buy_hold_equity'] = data['market_return'].cumsum().apply(np.exp)
    data['best_sma'] = best_sma_log
    
    return data

def generate_plot(df):
    """Generates the plot showing Equity and Step-wise Parameter Evolution."""
    # Filter out warm-up period for cleaner plots
    plot_df = df.iloc[TRAIN_WINDOW:].copy()
    
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # --- PLOT 1: Equity Curves ---
    ax1.plot(plot_df.index, plot_df['equity'], label='WFO Strategy (Retrain 60d)', color='lime', linewidth=2)
    ax1.plot(plot_df.index, plot_df['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    
    ax1.set_title(f'Walk-Forward Optimization (Train: {TRAIN_WINDOW}d, Hold: {RETRAIN_DAYS}d)', fontsize=14)
    ax1.set_ylabel('Normalized Equity')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # --- PLOT 2: Best SMA Evolution (Step Function) ---
    # Plot the raw steps
    ax2.step(plot_df.index, plot_df['best_sma'], where='post', color='cyan', linewidth=2, label='Active SMA')
    ax2.scatter(plot_df.index, plot_df['best_sma'], color='cyan', s=10, alpha=0.5)
    
    ax2.set_title('Dynamic SMA Selection (Re-optimized every 60 days)', fontsize=12)
    ax2.set_ylabel('Selected SMA Period')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 420)

    # --- PLOT 3: Efficiency Ratio ---
    ax3.plot(plot_df.index, plot_df['iii'], color='magenta', linewidth=1, label='Efficiency Ratio (III)')
    ax3.axhline(FIXED_III_THRESHOLD, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({FIXED_III_THRESHOLD})')
    
    # Fill background
    ax3.fill_between(plot_df.index, 0, FIXED_III_THRESHOLD, color='red', alpha=0.1, label='Zone: 0.5x Leverage')
    
    ax3.set_title(f'Efficiency Ratio (III) < {FIXED_III_THRESHOLD} triggers De-leveraging', fontsize=12)
    ax3.set_ylabel('Efficiency')
    ax3.set_xlabel('Date')
    ax3.set_ylim(0, 0.6)
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
        
        # 2. Walk-Forward Optimization
        result_df = walk_forward_optimization(df)
        
        # 3. Stats
        valid_rets = result_df['strategy_return'].iloc[TRAIN_WINDOW:]
        total_return = result_df['equity'].iloc[-1] if not result_df['equity'].empty else 1.0
        
        sharpe_calc = valid_rets.mean() / (valid_rets.std() + 1e-9)
        sharpe = np.sqrt(365) * sharpe_calc
        
        # 4. Plot
        plot_data = generate_plot(result_df)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Walk-Forward WFO: {SYMBOL}</title>
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
                <h1>AI Trading Lab: Periodic WFO</h1>
                <p>Train Window: {TRAIN_WINDOW} Days | Retrain Frequency: {RETRAIN_DAYS} Days</p>
                
                <div class="stats">
                    <span>Total Return: <span class="highlight">{total_return:.2f}x</span></span>
                    <span>WFO Sharpe: <span class="highlight">{sharpe:.2f}</span></span>
                </div>
                
                <div style="margin-bottom: 20px; color: #aaa; font-size: 0.9em;">
                    The model looks back {TRAIN_WINDOW} days to find the best SMA, locks it in for {RETRAIN_DAYS} days, and then repeats.
                    <br>Leverage is halved if Efficiency Ratio < {FIXED_III_THRESHOLD}.
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
