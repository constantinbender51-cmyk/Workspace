import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from flask import Flask, Response
import io
import base64

app = Flask(__name__)

# Function to fetch OHLCV data from Binance

def fetch_ohlcv(symbol='BTCUSDT', interval='1d', start_date='2018-01-01'):
    """
    Fetch OHLCV data from Binance public API.
    Returns a pandas DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    base_url = 'https://api.binance.com/api/v3/klines'
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    all_data = []
    
    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'limit': 1000
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_ts = data[-1][0] + 1  # Move to next candle
        
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                          'close_time', 'quote_asset_volume', 'number_of_trades', 
                                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

# Function to calculate strategy returns

def calculate_strategy_returns(df):
    """
    Calculate daily returns based on the strategy:
    - Long when open > 365 SMA and open > 120 SMA of open.
    - Short when open < 365 SMA and open < 120 SMA of open.
    - Flat otherwise.
    - Apply 2% stop loss: if long and low <= open * 0.98, return -2%; if short and high >= open * 1.02, return -2%.
    - Apply 1x leverage to daily returns.
    Returns a DataFrame with strategy returns.
    """
    df = df.copy()
    # Calculate SMAs
    df['sma_365'] = df['open'].rolling(window=365).mean()
    df['sma_120'] = df['open'].rolling(window=120).mean()
    
    # Determine position: 1 for long, -1 for short, 0 for flat
    df['position'] = 0
    long_condition = (df['open'] > df['sma_365']) & (df['open'] > df['sma_120'])
    short_condition = (df['open'] < df['sma_365']) & (df['open'] < df['sma_120'])
    df.loc[long_condition, 'position'] = 1
    df.loc[short_condition, 'position'] = -1
    
    # Calculate daily returns based on position
    df['daily_return'] = 0.0
    for i in range(len(df)):
        if i == 0:
            continue
        open_price = df.iloc[i]['open']
        high_price = df.iloc[i]['high']
        low_price = df.iloc[i]['low']
        close_price = df.iloc[i]['close']
        position = df.iloc[i]['position']
        
        if position == 1:  # Long
            # Check stop loss
            if low_price <= open_price * 0.98:
                daily_return = -0.02  # -2%
            else:
                daily_return = (close_price - open_price) / open_price
        elif position == -1:  # Short
            # Check stop loss
            if high_price >= open_price * 1.02:
                daily_return = -0.02  # -2%
            else:
                daily_return = (open_price - close_price) / open_price
        else:  # Flat
            daily_return = 0.0
        
        df.iloc[i, df.columns.get_loc('daily_return')] = daily_return
    
    # Apply leverage
    df['leveraged_return'] = df['daily_return'] * 1
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['leveraged_return']).cumprod() - 1
    
    return df

# Function to generate plot

def generate_plot(df):
    """
    Generate a Matplotlib plot showing cumulative returns with background colors:
    - Green for long days (position == 1)
    - Red for short days (position == -1)
    - Purple for stop loss days (daily_return == -0.02)
    - Price plotted on secondary y-axis.
    Returns a base64 encoded image string.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Add background colors for long and short days
    dates = df.index
    positions = df['position'].values
    daily_returns = df['daily_return'].values
    
    # Find intervals for long days (position == 1)
    long_start = None
    for i in range(len(dates)):
        if positions[i] == 1 and long_start is None:
            long_start = dates[i]
        elif positions[i] != 1 and long_start is not None:
            ax1.axvspan(long_start, dates[i-1], alpha=0.3, color='green', label='Long Days' if i == 1 else '')
            long_start = None
    if long_start is not None:
        ax1.axvspan(long_start, dates[-1], alpha=0.3, color='green', label='Long Days' if len(dates) == 1 else '')
    
    # Find intervals for short days (position == -1)
    short_start = None
    for i in range(len(dates)):
        if positions[i] == -1 and short_start is None:
            short_start = dates[i]
        elif positions[i] != -1 and short_start is not None:
            ax1.axvspan(short_start, dates[i-1], alpha=0.3, color='red', label='Short Days' if i == 1 else '')
            short_start = None
    if short_start is not None:
        ax1.axvspan(short_start, dates[-1], alpha=0.3, color='red', label='Short Days' if len(dates) == 1 else '')
    
    # Add background colors for stop loss days (daily_return == -0.02)
    stop_loss_start = None
    for i in range(len(dates)):
        if daily_returns[i] == -0.02 and stop_loss_start is None:
            stop_loss_start = dates[i]
        elif daily_returns[i] != -0.02 and stop_loss_start is not None:
            ax1.axvspan(stop_loss_start, dates[i-1], alpha=0.2, color='purple', label='Stop Loss Days' if i == 1 else '')
            stop_loss_start = None
    if stop_loss_start is not None:
        ax1.axvspan(stop_loss_start, dates[-1], alpha=0.2, color='purple', label='Stop Loss Days' if len(dates) == 1 else '')
    
    # Plot cumulative returns on primary y-axis with log scale
    cumulative_returns_percent = df['cumulative_return'] * 100
    ax1.plot(df.index, cumulative_returns_percent, label='Cumulative Return (%)', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('linear')
    ax1.grid(True)
    
    # Create secondary y-axis for price and SMAs
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['close'], label='Price (USD)', color='orange', alpha=0.7)
    ax2.plot(df.index, df['sma_365'], label='365 SMA', color='green', linestyle='--', alpha=0.7)
    ax2.plot(df.index, df['sma_120'], label='120 SMA', color='red', linestyle=':', alpha=0.7)
    ax2.set_ylabel('Price (USD)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Title and legends
    plt.title('Strategy Cumulative Returns and Price with Leverage (1x) and Stop Loss (2%)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    """
    Main route to fetch data, calculate strategy, and display plot.
    """
    try:
        # Fetch data
        df = fetch_ohlcv(symbol='BTCUSDT', interval='1d', start_date='2018-01-01')
        if df.empty:
            return "Error: No data fetched from Binance."
        
        # Calculate strategy returns
        df = calculate_strategy_returns(df)
        
        # Generate plot
        plot_img = generate_plot(df)
        
        # Create HTML response
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .info {{ margin-top: 20px; padding: 10px; background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Strategy Results: BTC/USDT from 2018</h1>
            <p>Strategy: Long when open > 365 SMA and 120 SMA of open, short when open < both SMAs, flat otherwise.</p>
            <p>Stop loss: 2%, Leverage: 1x.</p>
            <img src="data:image/png;base64,{plot_img}" alt="Cumulative Returns Plot">
            <div class="info">
                <p>Data fetched from Binance. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total days: {len(df)}</p>
                <p>Final cumulative return: {df['cumulative_return'].iloc[-1]*100:.2f}%</p>
            </div>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
