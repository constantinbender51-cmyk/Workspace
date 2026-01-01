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
    plt.style.use('bmh')
    
    # Plot Close Price
    plt.plot(df['open_time'], df['close'], label='Close Price', color='#2c3e50', linewidth=2)
    
    # --- Local Close Maxima Logic (2 Year Radius) ---
    
    # We want to find months where Close is the highest within a 2-year radius.
    # Radius of 2 years = 24 months before + 24 months after.
    # Window size = 24 + 1 (current) + 24 = 49 months.
    
    # min_periods=25 allows the calculation to start immediately at the beginning of the data
    # (checking the current month vs the next 24 months). 
    # This effectively satisfies the condition "assume price before our data was always 0".
    df['rolling_close_max'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    
    # IMPORTANT: min_periods will also calculate a max for the END of the dataframe 
    # (ignoring the missing future). We must strictly invalidate the last 24 months 
    # because we cannot know if those prices are local maxima without seeing the future.
    if len(df) > 24:
        df.loc[df.index[-24:], 'rolling_close_max'] = -1.0
    
    # Filter: It is a local peak if the Close equals the rolling max.
    local_peaks = df[df['close'] == df['rolling_close_max']]

    # Plot markers for these peaks
    if not local_peaks.empty:
        # Plot the Peak itself (Gold Star)
        plt.scatter(
            local_peaks['open_time'], 
            local_peaks['close'], 
            color='#FFD700',  # Gold
            s=200, 
            marker='*', 
            edgecolors='black', 
            linewidth=0.5, 
            zorder=6, 
            label='2-Year Radius Max'
        )

        # Iterate to add Offset Markers (-1 Year and +1 Year)
        first_green = True
        first_grey = True

        for _, row in local_peaks.iterrows():
            peak_date = row['open_time']
            
            # --- 1 Year Before (Green) ---
            target_date_before = peak_date - pd.DateOffset(years=1)
            # Find closest existing data point in the dataframe
            dt_diff_before = (df['open_time'] - target_date_before).abs()
            
            # Only plot if we find a date within ~45 days (to account for varying month lengths/missing data)
            if dt_diff_before.min() < pd.Timedelta(days=45):
                idx_before = dt_diff_before.idxmin()
                row_before = df.loc[idx_before]
                
                plt.scatter(
                    row_before['open_time'], 
                    row_before['close'], 
                    color='mediumseagreen', 
                    s=100, 
                    marker='^', 
                    zorder=5, 
                    edgecolors='black', 
                    linewidth=0.5,
                    label='1 Year Before' if first_green else ""
                )
                first_green = False

            # --- 1 Year After (Grey) ---
            target_date_after = peak_date + pd.DateOffset(years=1)
            dt_diff_after = (df['open_time'] - target_date_after).abs()
            
            if dt_diff_after.min() < pd.Timedelta(days=45):
                idx_after = dt_diff_after.idxmin()
                row_after = df.loc[idx_after]
                
                plt.scatter(
                    row_after['open_time'], 
                    row_after['close'], 
                    color='darkgrey', 
                    s=100, 
                    marker='v', 
                    zorder=5, 
                    edgecolors='black', 
                    linewidth=0.5,
                    label='1 Year After' if first_grey else ""
                )
                first_grey = False

    # Configuration for Scientific Look
    plt.title(f'Historical Monthly Price Action: {SYMBOL}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT) - Log Scale', fontsize=12)
    plt.yscale('log')
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