import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
PAIR = 'XBTUSD'
# Kraken Interval: 10080 minutes = 1 week. 
# This allows fetching ~14 years of history in one call (720 count limit * 1 week).
INTERVAL = 10080 
PORT = 8080

def fetch_kraken_data(pair, interval):
    """
    Fetches historical OHLC data from Kraken public API.
    Uses Weekly interval to get maximum history, then resamples to Monthly.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        'pair': pair,
        'interval': interval
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('error'):
            print(f"Kraken API Error: {data['error']}")
            return pd.DataFrame()

        # Kraken returns a dict with pair name as key. We need to find that key.
        # The result keys are usually [PairName, 'last'].
        result_data = data['result']
        # Extract the list of candles (find the key that isn't 'last')
        candles_key = [k for k in result_data.keys() if k != 'last'][0]
        candles = result_data[candles_key]
        
        # Kraken Columns: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'count'])
        
        # Convert types
        df['time'] = pd.to_datetime(df['time'], unit='s')
        numeric_cols = ['open', 'high', 'low', 'close', 'vol']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # Resample to Monthly ('MS' = Month Start) to standardize
        df.set_index('time', inplace=True)
        
        # Logic: 
        # Open = first open of the month
        # High = max high of the month
        # Low = min low of the month
        # Close = last close of the month
        df_monthly = df.resample('MS').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum'
        })
        
        # Drop incomplete months (if any NaN) or just the very last one if it's currently active
        df_monthly.dropna(inplace=True)
        
        # Reset index to make 'time' a column again for easy plotting
        df_monthly.reset_index(inplace=True)
        
        return df_monthly

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_plot(df):
    if df.empty:
        return None

    # Scientific Layout
    plt.figure(figsize=(14, 7))
    plt.style.use('bmh')
    
    # 1. Base Plot
    plt.plot(df['time'], df['close'], label='Close Price', color='#2c3e50', linewidth=1.5, zorder=2)
    
    # --- Pattern 2: Consolidation (30% Range for >= 1 Year) ---
    # Logic: Look at rolling 12-month windows.
    # If (WindowMax - WindowMin) / WindowMin <= 0.30, mark this window.
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    df['rolling_max'] = df['high'].rolling(window=indexer).max()
    df['rolling_min'] = df['low'].rolling(window=indexer).min()
    
    # Calculate Range Percentage
    df['range_pct'] = (df['rolling_max'] - df['rolling_min']) / df['rolling_min']
    
    # Identify Start of Consolidation Periods
    consolidation_starts = df[df['range_pct'] <= 0.30]
    
    # Create a mask for "In Consolidation" for filtering later
    # We map the 12-month forward window back to the actual dates
    consolidation_mask = pd.Series(False, index=df.index)
    
    # Shade the regions and build the mask
    # We use a set to avoid re-plotting overlapping spans excessively
    shaded_indices = set()
    
    for idx in consolidation_starts.index:
        # The window starts at idx and covers 12 months (rows idx to idx+11)
        start_date = df.loc[idx, 'time']
        # Be careful with index out of bounds if window is near the end
        end_idx = min(idx + 11, len(df) - 1)
        end_date = df.loc[end_idx, 'time']
        
        # Mark these indices as consolidated
        consolidation_mask.loc[idx:end_idx] = True
        
        # Visualize (Shading)
        # We perform a rough visual union by just plotting rectangles
        plt.axvspan(start_date, end_date, color='slategrey', alpha=0.1, zorder=1, edgecolor=None)

    # Label for legend
    plt.plot([], [], color='slategrey', alpha=0.3, linewidth=10, label='Consolidation (<30% range >1yr)')

    # --- Pattern 1: Local High (Close, 2yr radius) ---
    # Close has not been surpassed 2 years before and 2 years after.
    # Radius = 24 months. Window = 49.
    df['rolling_max_close'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    
    # We must exclude the last 24 months from being highs because we don't know the future
    valid_highs = df.copy()
    if len(valid_highs) > 24:
        valid_highs.loc[valid_highs.index[-24:], 'rolling_max_close'] = np.inf # invalidate future

    local_highs = valid_highs[valid_highs['close'] == valid_highs['rolling_max_close']]
    
    plt.scatter(
        local_highs['time'], 
        local_highs['close'], 
        color='#d63031', # Red
        s=150, 
        marker='v', 
        zorder=5, 
        edgecolors='white',
        linewidth=0.8,
        label='Local High (2yr radius)'
    )

    # --- Pattern 3: Local Low (Low, 2yr radius, ignoring consolidation) ---
    # "The period in which price has not moved removes the prices from analysis"
    # We effectively treat 'Low' prices inside consolidation zones as Infinity
    # so they cannot be the minimum of the window.
    
    df_lows_clean = df.copy()
    # Apply mask: Set Low to Inf where consolidation exists
    df_lows_clean.loc[consolidation_mask, 'low'] = np.inf
    
    # Find rolling min on this "cleaned" data
    df_lows_clean['rolling_min_low'] = df_lows_clean['low'].rolling(window=49, center=True, min_periods=25).min()
    
    # Invalidate last 24 months for safety
    if len(df_lows_clean) > 24:
        df_lows_clean.loc[df_lows_clean.index[-24:], 'rolling_min_low'] = -1.0

    # Match condition: The original Low (if not consolidated) must match the rolling min
    # Note: If df['low'] was set to Inf, it won't match rolling min (unless everything is Inf)
    local_lows = df_lows_clean[df_lows_clean['low'] == df_lows_clean['rolling_min_low']]
    
    # We plot the Low price
    plt.scatter(
        local_lows['time'], 
        local_lows['low'], 
        color='#00b894', # Green
        s=150, 
        marker='^', 
        zorder=5, 
        edgecolors='white',
        linewidth=0.8,
        label='Local Low (2yr radius, clean)'
    )

    # --- Layout & Styling ---
    plt.title(f'Market Structure Analysis: {PAIR} (Kraken)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price (USD) - Log Scale', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="major", color='#dcdde1', linestyle='-')
    plt.grid(True, which="minor", color='#f5f6fa', linestyle=':', alpha=0.5)
    plt.legend(frameon=True, loc='upper left', facecolor='white', framealpha=0.9)
    plt.tight_layout()

    # Save
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def home():
    df = fetch_kraken_data(PAIR, INTERVAL)
    plot_url = create_plot(df)
    
    if not df.empty:
        current_price = f"${df['close'].iloc[-1]:,.2f}"
        all_time_high = f"${df['high'].max():,.2f}"
        start_date = df['time'].iloc[0].strftime('%Y-%m-%d')
    else:
        current_price = "N/A"
        all_time_high = "N/A"
        start_date = "N/A"

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kraken Market Structure</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f1f2f6; color: #2f3542; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 10px 20px rgba(0,0,0,0.05); }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 10px; }
            .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 0.9em; }
            .stats { display: flex; justify-content: center; gap: 40px; margin-bottom: 30px; flex-wrap: wrap; }
            .stat-box { background: #f8f9fa; padding: 15px 25px; border-radius: 8px; border-left: 4px solid #3498db; min-width: 150px; text-align: center; }
            .stat-label { font-size: 0.8em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
            .stat-value { font-size: 1.4em; font-weight: 700; color: #2c3e50; }
            .plot-container { text-align: center; margin-bottom: 20px; }
            img { max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #dfe6e9; }
            .legend-box { background: #fff3cd; color: #856404; padding: 15px; border-radius: 6px; font-size: 0.9em; margin-top: 20px; border: 1px solid #ffeeba; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Market Structure Analysis</h1>
            <p class="subtitle">Data Source: Kraken Public API ({{ symbol }})</p>
            
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
                    <div class="stat-label">History Since</div>
                    <div class="stat-value">{{ start }}</div>
                </div>
            </div>

            <div class="plot-container">
                {% if plot_url %}
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Price Chart">
                {% else %}
                    <p style="color:red">Error: Could not retrieve data from Kraken.</p>
                {% endif %}
            </div>
            
            <div class="legend-box">
                <strong>Analysis Legend:</strong><br>
                <span style="color:#d63031">▼ Red Marker:</span> Local High (Close not surpassed within ±2 years)<br>
                <span style="color:#00b894">▲ Green Marker:</span> Local Low (Low not undercut within ±2 years, ignoring consolidations)<br>
                <span style="color:slategrey">■ Grey Zones:</span> Consolidation (Price range < 30% for > 1 year) - Lows here are ignored.
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                  plot_url=plot_url, 
                                  symbol=PAIR, 
                                  current=current_price, 
                                  ath=all_time_high, 
                                  start=start_date)

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)