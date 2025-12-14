import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
import io
import base64
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
RSI_PERIOD = 14
RSI_EMA_PERIOD = 21  # Standard smoothing for the signal line

def fetch_data(symbol, timeframe, start_date_str):
    """
    Fetches historical OHLCV data from Binance via CCXT.
    Handles pagination to get data from 2018 to present.
    """
    print(f"Fetching data for {symbol} since {start_date_str}...")
    exchange = ccxt.binance()
    
    # Convert start date to timestamp ms
    since = exchange.parse8601(start_date_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            last_timestamp = ohlcv[-1][0]
            if since == last_timestamp: # Prevent infinite loop if no new data
                break
                
            since = last_timestamp + 1
            all_ohlcv += ohlcv
            
            # Rate limit protection (rudimentary)
            # time.sleep(exchange.rateLimit / 1000)
            
            # Break if we reached current time (roughly)
            if last_timestamp >= exchange.milliseconds() - 60000:
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Fetched {len(df)} candles.")
    return df

def calculate_indicators(df):
    """
    Calculates RSI and the EMA of the RSI manually using Pandas.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()

    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate EMA of the RSI
    df['rsi_ema'] = df['rsi'].ewm(span=RSI_EMA_PERIOD, adjust=False).mean()
    
    return df

def run_backtest(df):
    """
    Applies the Long/Short logic.
    Long (1) when RSI > EMA
    Short (-1) when RSI < EMA
    """
    df = df.copy()
    
    # 1 for Long, -1 for Short
    # We use shift(1) because we trade at the Open of the next candle based on Close of previous
    df['signal'] = np.where(df['rsi'] > df['rsi_ema'], 1, -1)
    df['position'] = df['signal'].shift(1)
    
    # Calculate returns
    # Market return: (Close - Open) / Open or pct_change
    df['market_returns'] = df['close'].pct_change()
    
    # Strategy return: Position * Market Return
    # If we are short (-1) and price drops (negative return), we make profit.
    df['strategy_returns'] = df['position'] * df['market_returns']
    
    # Cumulative returns for plotting
    df['cumulative_market'] = (1 + df['market_returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    
    return df

def create_plot(df):
    """
    Generates a Matplotlib figure and returns it as a base64 string.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Cumulative Returns
    ax1.plot(df.index, df['cumulative_market'], label='Buy & Hold (BTC)', color='gray', alpha=0.5)
    ax1.plot(df.index, df['cumulative_strategy'], label='RSI/EMA Strategy', color='blue')
    ax1.set_title(f'Backtest Results: {SYMBOL} ({START_DATE} - Present)')
    ax1.set_ylabel('Cumulative Return (Multiplier)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: RSI and EMA
    # Slice the last 365 days for clearer view of the indicators, or plot all
    recent_df = df.tail(365) # Just showing last year for clarity on indicators
    ax2.plot(recent_df.index, recent_df['rsi'], label='RSI', color='purple', linewidth=1)
    ax2.plot(recent_df.index, recent_df['rsi_ema'], label='EMA of RSI', color='orange', linewidth=1)
    ax2.axhline(70, linestyle='--', color='red', alpha=0.3)
    ax2.axhline(30, linestyle='--', color='green', alpha=0.3)
    ax2.set_title('Indicator View (Last 365 Days)')
    ax2.set_ylabel('RSI Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64

# --- Flask Routes ---

@app.route('/')
def dashboard():
    # 1. Fetch
    df = fetch_data(SYMBOL, TIMEFRAME, START_DATE)
    
    # 2. Calculate
    df = calculate_indicators(df)
    
    # 3. Backtest
    df = run_backtest(df)
    
    # 4. Plot
    plot_url = create_plot(df)
    
    # 5. Stats
    total_return = df['cumulative_strategy'].iloc[-1]
    market_return = df['cumulative_market'].iloc[-1]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Trading Bot</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; background: #f4f4f4; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .stat-box {{ background: #eef; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Backtest: RSI Cross EMA</h1>
            <div class="stats">
                <div class="stat-box"><strong>Strategy Return:</strong> {total_return:.2f}x</div>
                <div class="stat-box"><strong>Buy & Hold Return:</strong> {market_return:.2f}x</div>
                <div class="stat-box"><strong>Current Position:</strong> {'LONG' if df['position'].iloc[-1] == 1 else 'SHORT'}</div>
            </div>
            <img src="data:image/png;base64,{plot_url}" style="width:100%; height:auto;" />
            <br><br>
            <p><em>Data fetched from Binance. Strategy: Long when RSI > EMA(RSI), Short when RSI < EMA(RSI).</em></p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print("Starting server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=True)
