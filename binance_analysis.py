import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive for server use
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, Response
import io
import base64
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' # Daily candles to keep data volume manageable
START_DATE = '2018-01-01 00:00:00'
PORT = 8080

def fetch_data():
    """Fetches historical OHLCV data from Binance with pagination."""
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
            
            # Break if we've reached the current time
            if len(candles) < 1000:
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_strategy(df):
    """Calculates SMAs, signals, and equity curve."""
    data = df.copy()
    
    # Calculate SMAs from 10 to 400
    sma_range = range(10, 410, 10) # 10, 20, ... 400
    for period in sma_range:
        data[f'SMA_{period}'] = data['close'].rolling(window=period).mean()

    # Strategy Logic based on SMA 360
    # Long (1) if Close > SMA 360
    # Short (-1) if Close < SMA 360
    # Flat (0) otherwise (rare equality)
    
    sma_signal = data['SMA_360']
    
    conditions = [
        (data['close'] > sma_signal),
        (data['close'] < sma_signal)
    ]
    choices = [1, -1] # Long, Short
    
    # Shift position by 1 to avoid look-ahead bias (trade occurs on next candle open)
    data['position'] = np.select(conditions, choices, default=0)
    data['position'] = data['position'].shift(1) 
    
    # Calculate Returns
    # Using log returns for simpler accumulation: ln(p1/p0)
    data['market_return'] = np.log(data['close'] / data['close'].shift(1))
    data['strategy_return'] = data['position'] * data['market_return']
    
    # Calculate Equity Curve (Cumulative Sum of log returns, exponentiated for normalized price)
    # Starting with capital of 1
    data['equity'] = data['strategy_return'].cumsum().apply(np.exp)
    
    # Buy & Hold Equity for comparison
    data['buy_hold_equity'] = data['market_return'].cumsum().apply(np.exp)
    
    return data

def generate_plot(df):
    """Generates a matplotlib figure and returns it as a base64 string."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Price and SMAs
    
    for col in df.columns:
        if 'SMA_' in col:
            # Highlight SMA 360 specifically
            if col == 'SMA_360':
                # INCREASED PROMINENCE
                ax1.plot(df.index, df[col], label='SMA 360 (Signal)', color='gold', linewidth=3)
            else:
                # Increased visibility for other SMAs
                ax1.plot(df.index, df[col], color='gray', alpha=0.2, linewidth=1.0)
                
    # Increased visibility for Price
    ax1.plot(df.index, df['close'], label='Price', color='black', linewidth=1.5)
    ax1.set_title(f'{SYMBOL} Price & SMAs (10-400) | Signal: SMA 360')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equity Curve
    ax2.plot(df.index, df['equity'], label='Strategy Equity', color='green', linewidth=2)
    ax2.plot(df.index, df['buy_hold_equity'], label='Buy & Hold', color='blue', alpha=0.5, linestyle='--')
    ax2.set_title('Equity Curve (Starting 1.0)')
    ax2.set_ylabel('Normalized Equity')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig)
    return data

@app.route('/')
def index():
    try:
        # 1. Fetch
        df = fetch_data()
        
        # 2. Process
        result_df = run_strategy(df)
        
        # 3. Plot
        plot_data = generate_plot(result_df)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Binance Backtest: {SYMBOL}</title>
            <style>
                body {{ font-family: sans-serif; text-align: center; background: #f4f4f4; padding: 20px; }}
                .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }}
                h1 {{ color: #333; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Strategy Backtest Result</h1>
                <p><b>Symbol:</b> {SYMBOL} | <b>Timeframe:</b> {TIMEFRAME} | <b>Start:</b> {START_DATE}</p>
                <p><b>Logic:</b> Long if > SMA 360, Short if < SMA 360</p>
                <img src="data:image/png;base64,{plot_data}" alt="Backtest Plot">
                <br><br>
                <button onclick="location.reload()">Refresh Data</button>
            </div>
        </body>
        </html>
        """
        return render_template_string(html)
        
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p>"

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=True)
