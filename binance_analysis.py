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
START_YEAR = 2017 
END_YEAR = 2050

# Strategy Definition
SHORT_DURATION = 12
FLAT_DURATION = 22
LONG_DURATION = 12
CYCLE_MONTHS = SHORT_DURATION + FLAT_DURATION + LONG_DURATION # 46 Months total

# --- ANCHOR SETTINGS ---
# Set this to 'YYYY-MM-DD' to force the strategy start date.
# Set to None to let the script find the highest price automatically.
FORCE_ANCHOR_DATE = '2025-07-01' 

# Global storage for the latest plot buffer
plot_cache = None
last_update = 0

def fetch_binance_data():
    """Downloads all available monthly OHLCV data from Binance."""
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'limit': limit
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        
        # DataFrame columns: Open Time, Open, High, Low, Close, Volume...
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['high'] = df['high'].astype(float)
        df['close'] = df['close'].astype(float)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def apply_strategy(df):
    if df.empty:
        return None, None, None, None

    # 1. Determine Anchor
    # Debugging: Print top 5 highs to see what the data actually contains
    print("\n--- DATA DEBUG: TOP 5 MONTHLY HIGHS ---")
    top_5 = df.sort_values(by='high', ascending=False).head(5)
    for date, row in top_5.iterrows():
        print(f"{date.date()}: ${row['high']:,.2f}")
    print("---------------------------------------\n")

    if FORCE_ANCHOR_DATE:
        ath_date = pd.Timestamp(FORCE_ANCHOR_DATE)
        # Try to find the exact price in data, otherwise estimate or max
        if ath_date in df.index:
            ath_price = df.loc[ath_date]['high']
        else:
            # If the forced date is outside our current data (future), use the max found
            ath_price = df['high'].max()
        print(f"Strategy: Using FORCED Anchor: {ath_date.date()}")
    else:
        ath_date = df['high'].idxmax()
        ath_price = df['high'].max()
        print(f"Strategy: Auto-detected Anchor: {ath_date.date()} at ${ath_price:,.2f}")

    # 2. Create the full timeline
    start_date = df.index[0]
    end_date = pd.Timestamp(f"{END_YEAR}-12-31")
    
    strategy_index = pd.date_range(start=start_date, end=end_date, freq='MS')
    strategy_df = pd.DataFrame(index=strategy_index)
    strategy_df['action'] = 'FLAT' 
    strategy_df['cycle_phase'] = 0

    # 3. Propagate the cycle
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    for date in strategy_index:
        # Calculate offset from ATH
        offset = diff_month(date, ath_date)
        
        # Normalize offset to [0, 45]
        cycle_pos = offset % CYCLE_MONTHS
        
        # 0-11: Short (12m)
        # 12-33: Flat (22m)
        # 34-45: Long (12m)
        if 0 <= cycle_pos < SHORT_DURATION:
            action = 'SHORT'
        elif SHORT_DURATION <= cycle_pos < (SHORT_DURATION + FLAT_DURATION):
            action = 'FLAT'
        else: 
            action = 'LONG'
            
        strategy_df.at[date, 'action'] = action
        strategy_df.at[date, 'cycle_phase'] = cycle_pos

    return df, strategy_df, ath_date, ath_price

def generate_plot():
    global plot_cache, last_update
    
    # Simple cache (optional, helpful if many requests hit at once)
    if plot_cache is not None and (time.time() - last_update) < 60:
        return plot_cache

    print("Generating plot...")
    df_raw = fetch_binance_data()
    
    if df_raw.empty:
        return None
        
    df, strategy, ath_date, ath_price = apply_strategy(df_raw)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.style.use('dark_background')
    
    # 1. Plot BTC Price (Log Scale)
    ax.semilogy(df.index, df['close'], color='white', linewidth=1.5, label='BTC Price (Log)')
    
    # 2. Visualize Strategy Zones
    colors = {'SHORT': '#ff4d4d', 'FLAT': '#404040', 'LONG': '#00cc66'}
    alphas = {'SHORT': 0.25, 'FLAT': 0.15, 'LONG': 0.25}

    # Group continuous actions to draw spans
    strategy['group'] = (strategy['action'] != strategy['action'].shift()).cumsum()
    
    for _, group in strategy.groupby('group'):
        start = group.index[0]
        # We want the block to cover the entire month. 
        # Adding 1 month exactly fills the gap to the next candle.
        end = group.index[-1] + pd.DateOffset(months=1)
        action = group['action'].iloc[0]
        
        ax.axvspan(start, end, color=colors[action], alpha=alphas[action], ec=None)

        # Label future cycles
        if start > df.index[-1]:
             midpoint = start + (end - start) / 2
             # Place text near recent price or a fixed level
             y_pos = df['close'].iloc[-1] if not df.empty else 50000
             ax.text(midpoint, y_pos, action[0], color=colors[action], 
                     ha='center', va='bottom', fontsize=8, alpha=0.7, fontweight='bold')

    # 3. Mark the Anchor
    if ath_date in df.index:
        ax.scatter([ath_date], [ath_price], color='yellow', s=150, zorder=10, label='Anchor', marker='*')
        ax.text(ath_date, ath_price * 1.2, f'ANCHOR\n{ath_date.strftime("%Y-%m")}', color='yellow', ha='center', fontsize=10, fontweight='bold')
    else:
        # If anchor is forced to a date not in data (e.g. data hasn't loaded fully or future)
        # We manually plot it if it's within X-axis range
        pass

    # 4. Aesthetics
    ax.set_title(f'BTC Peak Strategy | Anchor: {ath_date.strftime("%Y-%m")} | Data to {END_YEAR}', fontsize=14, color='white')
    ax.set_ylabel('Price (USDT) - Log Scale')
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    
    # Set limits
    ax.set_xlim(df.index[0], pd.Timestamp(f'{END_YEAR}-01-01'))
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color='white', lw=2, label='BTC Price'),
        matplotlib.lines.Line2D([0], [0], color='yellow', marker='*', lw=0, markersize=10, label='Anchor'),
        Patch(facecolor=colors['SHORT'], alpha=alphas['SHORT'], label=f'Short ({SHORT_DURATION}m)'),
        Patch(facecolor=colors['FLAT'], alpha=alphas['FLAT'], label=f'Flat ({FLAT_DURATION}m)'),
        Patch(facecolor=colors['LONG'], alpha=alphas['LONG'], label=f'Long ({LONG_DURATION}m)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

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
            img { max-width: 95%; height: auto; border: 1px solid #333; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { margin-bottom: 5px; }
            .note { color: #f39c12; font-size: 0.9em; margin-bottom: 20px; max-width: 800px; margin-left: auto; margin-right: auto;}
            .meta { color: #666; font-size: 0.8em; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BTC Strategy Visualization</h1>
            <div class="note">
                <strong>Strategy Configuration:</strong><br>
                Anchor Date: <strong>{{ anchor }}</strong><br>
                Sequence: Short (12m) → Flat (22m) → Buy (12m)
            </div>
            <img src="/plot.png" alt="Strategy Chart" />
            <div class="meta">Data source: Binance Monthly OHLCV</div>
        </div>
    </body>
    </html>
    """
    anchor_display = FORCE_ANCHOR_DATE if FORCE_ANCHOR_DATE else "Auto-Detected Highest High"
    return render_template_string(html.replace("{{ anchor }}", str(anchor_display)))

@app.route('/plot.png')
def plot_img():
    buf = generate_plot()
    if buf:
        return send_file(io.BytesIO(buf.getvalue()), mimetype='image/png')
    return "Error generating plot", 500

def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    print("--- Starting BTC Strategy Application ---")
    server_thread = threading.Thread(target=run_web_server)
    server_thread.daemon = True
    server_thread.start()

    try:
        # Generate initial plot on startup
        generate_plot()
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down...")