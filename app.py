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
    - Apply 5% stop loss: if long and low <= open * 0.95, return -5%; if short and high >= open * 1.05, return -5%.
    - Apply 1x leverage to daily returns.
    Returns a DataFrame with strategy returns.
    """
    df = df.copy()
    # Calculate SMAs
    df['sma_365'] = df['open'].rolling(window=365).mean()
    df['sma_120'] = df['open'].rolling(window=120).mean()
    # Calculate ATR 29 and range
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                    abs(df['low'] - df['close'].shift(1))))
    df['atr_29'] = df['tr'].rolling(window=14).mean()
    df['range'] = df['high'] - df['low']
    df['sma_7_range'] = df['range'].rolling(window=7).mean()
    df['sma_14_range'] = df['range'].rolling(window=14).mean()
    
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
            if low_price <= open_price * 0.95:
                daily_return = -0.05  # -5%
            else:
                daily_return = (close_price - open_price) / open_price
        elif position == -1:  # Short
            # Check stop loss
            if high_price >= open_price * 1.05:
                daily_return = -0.05  # -5%
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
    - Green if yesterday's range/ATR 14 < 1
    - Red if yesterday's range/ATR 14 >= 1
    - Price plotted on secondary y-axis.
    Returns a base64 encoded image string.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Add background colors based on yesterday's ATR 29
    dates = df.index
    sma_14_range_values = df['sma_14_range'].values
    
    # Iterate through dates to apply shading
    for i in range(len(dates)):
        # Get yesterday's SMA 14 of range, handle NaNs and first day
        sma_14_range_yesterday = sma_14_range_values[i-1] if i > 0 and not pd.isna(sma_14_range_values[i-1]) else np.nan
        
        color = 'white' # Default background color
        
        # Determine color based on the condition: yesterday's SMA 14 of range > 2000
        if not pd.isna(sma_14_range_yesterday):
            if sma_14_range_yesterday > 3000:
                color = 'grey'
        
        # Apply shading for the current day
        if i < len(dates) - 1:
            ax1.axvspan(dates[i], dates[i+1], alpha=0.3, color=color, edgecolor='none')
        else:
            # For the last day, shade to the end of the plot
            ax1.axvspan(dates[i], dates[i], alpha=0.3, color=color, edgecolor='none')
    
    # Plot cumulative returns on primary y-axis with log scale
    cumulative_returns_percent = df['cumulative_return'] * 100
    ax1.plot(df.index, cumulative_returns_percent, label='Cumulative Return (%)', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('linear')
    ax1.grid(True)
    
    # Create secondary y-axis for price, SMAs, and ATR 29
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['close'], label='Price (USD)', color='orange', alpha=0.7)
    ax2.plot(df.index, df['sma_365'], label='365 SMA', color='green', linestyle='--', alpha=0.7)
    ax2.plot(df.index, df['sma_120'], label='120 SMA', color='red', linestyle=':', alpha=0.7)
    ax2.plot(df.index, df['atr_29'], label='ATR 29', color='purple', linestyle='-', alpha=0.7)
    ax2.plot(df.index, df['sma_7_range'], label='SMA 7 Range', color='brown', linestyle='--', alpha=0.7)
    ax2.plot(df.index, df['sma_14_range'], label='SMA 14 Range', color='black', linestyle='-.', alpha=0.7)
    ax2.set_ylabel('Price (USD) / ATR / Range SMA', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Title and legends
    plt.title('Strategy Cumulative Returns and Price with Leverage (1x), Stop Loss (5%), and Background (Grey if Yesterday\'s SMA 14 of Range > 3000, White otherwise)')
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
            <p>Stop loss: 5%, Leverage: 1x. ATR 29, SMA 7 of daily range, and SMA 14 of daily range calculated and plotted. Background: grey if yesterday's SMA 14 of Range > 3000, white otherwise.</p>
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
