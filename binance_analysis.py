import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1M'  # 1 Month
PORT = 8080

def fetch_binance_data(symbol, interval):
    """
    Fetches historical kline (candlestick) data from Binance.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Binance limit is 1000 candles. Since Bitcoin's monthly history 
    # fits well within 1000 months (~83 years), one call is sufficient.
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000 
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # DataFrame columns based on Binance API documentation
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_plot(df):
    """
    Generates a matplotlib plot and returns it as a base64 encoded string.
    """
    if df.empty:
        return None

    # Scientific Layout Setup
    plt.figure(figsize=(12, 6))
    plt.style.use('bmh')  # 'bmh' is a clean, scientific style
    
    # Plot Close Price
    plt.plot(df['open_time'], df['close'], label='Close Price', color='#2c3e50', linewidth=2)
    
    # --- Custom Events Logic ---
    # Format: (Date String, Label, Color)
    events = [
        ('2012-11-01', 'Buy 1 Year', 'green'),
        ('2013-11-01', 'Sell', 'red'),
        ('2016-12-01', 'Buy 1 Year', 'green'),
        ('2017-12-01', 'Sell 1 Year', 'red'),
        ('2021-10-01', 'Buy 1 Year', 'green'),
        ('2022-10-01', 'Sell 1 Year', 'red'),
        ('2024-08-01', 'Buy 1 Year', 'green'),
        ('2025-07-01', 'Sell 1 Year', 'red'),
    ]

    for date_str, label, color in events:
        target_date = pd.to_datetime(date_str)
        
        # Find the closest available data point to this date
        # We look for a date within a short tolerance (e.g., matching month)
        # Calculate time difference
        time_diff = (df['open_time'] - target_date).abs()
        
        # Get the index of the minimum difference
        if time_diff.min() < pd.Timedelta(days=32):  # Ensure we are within ~1 month
            idx = time_diff.idxmin()
            row = df.loc[idx]
            
            plot_date = row['open_time']
            plot_price = row['close']
            
            # Plot Marker
            marker_style = '^' if 'Buy' in label else 'v'
            plt.scatter(plot_date, plot_price, color=color, s=120, marker=marker_style, zorder=5)
            
            # Add Label Annotation
            plt.annotate(
                label, 
                xy=(plot_date, plot_price),
                xytext=(0, 15 if 'Buy' in label else -25),
                textcoords='offset points',
                ha='center', 
                color=color, 
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8)
            )

    # Configuration for Scientific Look
    plt.title(f'Historical Monthly Price Action: {SYMBOL}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT) - Log Scale', fontsize=12)
    plt.yscale('log')  # Log scale is standard for long-term crypto history
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()

    # Save to IO buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def home():
    # 1. Fetch Data
    df = fetch_binance_data(SYMBOL, INTERVAL)
    
    # 2. Create Plot
    plot_url = create_plot(df)
    
    # 3. Stats for display
    if not df.empty:
        current_price = f"${df['close'].iloc[-1]:,.2f}"
        all_time_high = f"${df['high'].max():,.2f}"
        start_date = df['open_time'].iloc[0].strftime('%Y-%m-%d')
    else:
        current_price = "N/A"
        all_time_high = "N/A"
        start_date = "N/A"

    # 4. Render HTML
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Binance Market Data</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; color: #333; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #444; }
            .stats { display: flex; justify-content: space-around; margin-bottom: 30px; background: #eef2f5; padding: 15px; border-radius: 5px; }
            .stat-box { text-align: center; }
            .stat-label { font-size: 0.9em; color: #666; text-transform: uppercase; letter-spacing: 1px; }
            .stat-value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
            .plot-container { text-align: center; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            .footer { margin-top: 20px; text-align: center; font-size: 0.8em; color: #888; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Market Analysis: {{ symbol }}</h1>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Current Price</div>
                    <div class="stat-value">{{ current }}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">All Time High</div>
                    <div class="stat-value">{{ ath }}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Data Since</div>
                    <div class="stat-value">{{ start }}</div>
                </div>
            </div>

            <div class="plot-container">
                {% if plot_url %}
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Price Chart">
                {% else %}
                    <p>Error loading data from Binance.</p>
                {% endif %}
            </div>
            
            <div class="footer">
                Data fetches live from Binance API | Interval: Monthly
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                  plot_url=plot_url, 
                                  symbol=SYMBOL, 
                                  current=current_price, 
                                  ath=all_time_high, 
                                  start=start_date)

if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices on the network
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)