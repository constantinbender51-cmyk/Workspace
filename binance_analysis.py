import os
import threading
import time
import io
import requests
import pandas as pd
import matplotlib
# Force non-interactive backend for Railway/Headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, send_file, render_template_string
from datetime import datetime, timedelta

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1M'  # Monthly
START_YEAR = 2017 # Approx start of reliable binance data
END_YEAR = 2050
CYCLE_MONTHS = 12 + 22 + 12 # 46 Months total

# Global storage for the latest plot buffer
plot_cache = None
last_update = 0

def fetch_binance_data():
    """Downloads all available monthly OHLCV data from Binance."""
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    
    # Get start time (Binance launched ~July 2017)
    # We'll just fetch the max allowed which covers all history for Monthly
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'limit': limit
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        # DataFrame columns: Open Time, Open, High, Low, Close, Volume, ...
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def apply_strategy(df):
    """
    Applies the Peak Strategy:
    1. Find ATH
    2. Cycle: Short (12m) -> Flat (22m) -> Long (12m)
    3. Project backward and forward to 2050
    """
    if df.empty:
        return None, None

    # 1. Find the Anchor (Global ATH)
    # We assume the ATH marks the BEGINNING of the Short period
    ath_date = df['high'].idxmax()
    ath_price = df['high'].max()
    
    print(f"Anchor ATH Found: {ath_date.date()} at ${ath_price:,.2f}")

    # 2. Create the full timeline (Start of Data -> 2050)
    start_date = df.index[0]
    end_date = pd.Timestamp(f"{END_YEAR}-12-31")
    
    # Generate a monthly range for the strategy projection
    # usage of MS (Month Start) to align cleanly
    strategy_index = pd.date_range(start=start_date, end=end_date, freq='MS')
    strategy_df = pd.DataFrame(index=strategy_index)
    strategy_df['action'] = 'FLAT' # Default
    strategy_df['cycle_phase'] = 0

    # 3. Propagate the cycle
    # The cycle is 46 months long. 
    # Logic: Calculate months difference from ATH. 
    # Use modulo 46 to determine position in cycle.
    
    # Helper to calculate month difference
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    for date in strategy_index:
        # Calculate offset from ATH
        # If date is before ATH, offset is negative, modulo still works correctly in Python
        offset = diff_month(date, ath_date)
        
        # Normalize offset to [0, 45]
        cycle_pos = offset % CYCLE_MONTHS
        
        if 0 <= cycle_pos < 12:
            action = 'SHORT'
        elif 12 <= cycle_pos < 34: # 12 + 22 = 34
            action = 'FLAT'
        else: # 34 to 46
            action = 'LONG'
            
        strategy_df.at[date, 'action'] = action
        strategy_df.at[date, 'cycle_phase'] = cycle_pos

    return df, strategy_df

def generate_plot():
    """Generates the Matplotlib chart and saves it to a buffer."""
    global plot_cache, last_update
    
    # Simple caching: don't regenerate if requested frequently (e.g. < 1 hr)
    # Since monthly data changes slowly, we can cache aggressively or just run once at startup
    if plot_cache is not None and (time.time() - last_update) < 3600:
        return plot_cache

    print("Generating plot...")
    df, strategy = fetch_binance_data(), None
    
    if not df.empty:
        df, strategy = apply_strategy(df)

    if strategy is None:
        return None

    # Setup Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.style.use('dark_background')
    
    # 1. Plot BTC Price (Log Scale for long term)
    ax.semilogy(df.index, df['close'], color='white', linewidth=1.5, label='BTC Price (Log)')
    
    # 2. Visualize Strategy Zones
    # We iterate through the strategy dataframe and paint the background
    # This can be heavy if we iterate row by row, so we group by consecutive actions
    
    # Resample strategy to daily for smoother filling or just use the monthly blocks
    # Using 'step' plot style for the background colors is easier visually
    
    # Define colors
    colors = {'SHORT': '#ff4d4d', 'FLAT': '#404040', 'LONG': '#00cc66'}
    alphas = {'SHORT': 0.3, 'FLAT': 0.2, 'LONG': 0.3}

    # Identify chunks of continuous actions to create span blocks
    strategy['group'] = (strategy['action'] != strategy['action'].shift()).cumsum()
    
    for _, group in strategy.groupby('group'):
        start = group.index[0]
        # For the end date, we add 1 month to cover the full bar width visually
        end = group.index[-1] + pd.DateOffset(months=1)
        action = group['action'].iloc[0]
        
        ax.axvspan(start, end, color=colors[action], alpha=alphas[action], ec=None)

        # Add text labels for future cycles (optional, helps readability)
        if start > df.index[-1] and action != 'FLAT':
            midpoint = start + (end - start) / 2
            ax.text(midpoint, df['close'].min(), action[0], color=colors[action], 
                    ha='center', va='bottom', fontsize=8, alpha=0.7)

    # 3. Aesthetics
    ax.set_title(f'BTC Peak Strategy (ATH Anchor) - Forward Projection to {END_YEAR}', fontsize=16, color='white')
    ax.set_ylabel('Price (USDT) - Log Scale')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Set X limits
    ax.set_xlim(df.index[0], pd.Timestamp(f'{END_YEAR}-01-01'))
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color='white', lw=2, label='BTC Price'),
        Patch(facecolor=colors['SHORT'], alpha=alphas['SHORT'], label='Short (1 Yr)'),
        Patch(facecolor=colors['FLAT'], alpha=alphas['FLAT'], label='Flat (22 Mo)'),
        Patch(facecolor=colors['LONG'], alpha=alphas['LONG'], label='Long (1 Yr)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    plot_cache = buf
    last_update = time.time()
    return buf

# --- Web Server Routes ---

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Peak Strategy</title>
        <style>
            body { background-color: #121212; color: #e0e0e0; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
            img { max-width: 100%; height: auto; border: 1px solid #333; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { margin-bottom: 10px; }
            p { color: #888; margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BTC Strategy Visualization</h1>
            <p>Strategy: Peak ATH → Short (1y) → Flat (22m) → Buy (1y) | Projected to 2050</p>
            <img src="/plot.png" alt="Strategy Chart" />
            <p><small>Data source: Binance Monthly OHLCV</small></p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/plot.png')
def plot_img():
    buf = generate_plot()
    if buf:
        # Create a new buffer from the cached bytes to avoid seeking issues on concurrent requests
        return send_file(io.BytesIO(buf.getvalue()), mimetype='image/png')
    return "Error generating plot", 500

def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    # Host 0.0.0.0 is required for Docker/Railway
    app.run(host='0.0.0.0', port=port)

# --- Main Application Entry Point ---
if __name__ == "__main__":
    print("--- Starting BTC Strategy Application ---")
    
    # 1. Initial Data Load (Optional, but good for logs)
    # We let the first web request trigger the plot generation or do it here
    try:
        generate_plot()
        print("Initial plot generated successfully.")
    except Exception as e:
        print(f"Initial plot generation failed (will retry on web request): {e}")

    # 2. Start Web Server in a daemon thread
    # In a real production wsgi env this might be different, but for a 
    # "script running like an application" this works well.
    server_thread = threading.Thread(target=run_web_server)
    server_thread.daemon = True
    server_thread.start()

    # 3. Main Loop
    # Keep the main thread alive to allow the server thread to run
    # This is also where you would put other trading logic/scheduled tasks
    try:
        while True:
            time.sleep(60) # Sleep to save CPU, keep container alive
    except KeyboardInterrupt:
        print("Shutting down...")