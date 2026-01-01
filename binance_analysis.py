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

        result_data = data['result']
        candles_key = [k for k in result_data.keys() if k != 'last'][0]
        candles = result_data[candles_key]
        
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        numeric_cols = ['open', 'high', 'low', 'close', 'vol']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        df.set_index('time', inplace=True)
        df_monthly = df.resample('MS').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum'
        })
        
        df_monthly.dropna(inplace=True)
        df_monthly.reset_index(inplace=True)
        
        return df_monthly
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_plot(df):
    if df.empty:
        return None

    plt.figure(figsize=(14, 7))
    plt.style.use('bmh')
    
    ax = plt.gca()
    # Set a subtle light gray background for the plotting area 
    # This makes the "White" spans visible
    ax.set_facecolor('#f0f2f6')
    
    # Base Plot - Price line
    plt.plot(df['time'], df['close'], label='Close Price', color='#2c3e50', linewidth=1.8, zorder=3)
    
    # --- Pattern 2: Consolidation (30% Range for >= 1 Year) ---
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    df['rolling_max'] = df['high'].rolling(window=indexer).max()
    df['rolling_min'] = df['low'].rolling(window=indexer).min()
    df['range_pct'] = (df['rolling_max'] - df['rolling_min']) / df['rolling_min']
    
    consolidation_starts = df[df['range_pct'] <= 0.30]
    consolidation_mask = pd.Series(False, index=df.index)
    
    for idx in consolidation_starts.index:
        start_date = df.loc[idx, 'time']
        end_idx = min(idx + 11, len(df) - 1)
        end_date = df.loc[end_idx, 'time']
        consolidation_mask.loc[idx:end_idx] = True
        
        # Plotting the consolidation span in pure white
        # zorder=0 ensures it is the bottom-most layer (behind the grid)
        plt.axvspan(start_date, end_date, color='white', alpha=1.0, zorder=0)

    # Proxy for Legend
    plt.axvspan(None, None, color='white', label='Consolidation (White Area)', ec='#ccc')

    # --- Pattern 1: Local High (Close, 2yr radius) ---
    df['rolling_max_close'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    
    valid_highs = df.copy()
    if len(valid_highs) > 24:
        valid_highs.loc[valid_highs.index[-24:], 'rolling_max_close'] = np.inf

    local_highs = valid_highs[valid_highs['close'] == valid_highs['rolling_max_close']]
    plt.scatter(local_highs['time'], local_highs['close'], color='#d63031', s=140, marker='v', zorder=5, edgecolors='white', label='Local High (2yr)')

    # --- Pattern 3: Local Low (Low, 1yr radius, ignoring consolidation) ---
    df_lows_clean = df.copy()
    df_lows_clean.loc[consolidation_mask, 'low'] = np.inf
    
    df_lows_clean['rolling_min_low'] = df_lows_clean['low'].rolling(window=25, center=True, min_periods=13).min()
    
    if len(df_lows_clean) > 12:
        df_lows_clean.loc[df_lows_clean.index[-12:], 'rolling_min_low'] = -1.0

    local_lows = df_lows_clean[df_lows_clean['low'] == df_lows_clean['rolling_min_low']]
    plt.scatter(local_lows['time'], local_lows['low'], color='#00b894', s=140, marker='^', zorder=5, edgecolors='white', label='Local Low (1yr)')

    # Styling
    plt.title(f'Scientific Market Analysis: {PAIR} (Kraken)', fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price (USD) - Logarithmic Scale', fontsize=12)
    plt.yscale('log')
    
    # Grid customization - keeping it subtle but visible
    plt.grid(True, which="major", color='#e1e5ea', linestyle='-', alpha=0.8, zorder=1)
    
    plt.legend(frameon=True, loc='upper left', facecolor='white', framealpha=0.95, edgecolor='#ddd')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=110)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def home():
    df = fetch_kraken_data(PAIR, INTERVAL)
    plot_url = create_plot(df)
    
    current_price = f"${df['close'].iloc[-1]:,.2f}" if not df.empty else "N/A"
    ath = f"${df['high'].max():,.2f}" if not df.empty else "N/A"
    start_date = df['time'].iloc[0].strftime('%Y-%m-%d') if not df.empty else "N/A"

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kraken Market Analysis</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Inter', -apple-system, sans-serif; margin: 0; padding: 20px; background-color: #f1f2f6; color: #2f3542; }
            .container { max-width: 1300px; margin: 0 auto; background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            .header-info { display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }
            .stat-box { background: #fff; padding: 15px 30px; border-radius: 10px; text-align: center; border: 1px solid #eef; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
            .stat-label { font-size: 0.75rem; color: #747d8c; text-transform: uppercase; margin-bottom: 5px; font-weight: 600; }
            .stat-value { font-size: 1.25rem; font-weight: 700; color: #2f3542; }
            .chart-frame { text-align: center; background: #fafafa; border-radius: 8px; padding: 10px; border: 1px solid #eee; }
            img { max-width: 100%; height: auto; border-radius: 4px; }
            .legend-panel { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #3498db; }
            .legend-item { margin-bottom: 8px; font-size: 0.95rem; display: flex; align-items: center; }
            .color-box { width: 14px; height: 14px; display: inline-block; margin-right: 10px; border-radius: 3px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align:center; margin-bottom: 10px;">Market Structure & Pattern Analysis</h1>
            <p style="text-align:center; color: #747d8c; margin-bottom: 30px;">Instrument: Kraken {{ symbol }} (Monthly Resolution)</p>
            
            <div class="header-info">
                <div class="stat-box"><div class="stat-label">Price</div><div class="stat-value">{{ current }}</div></div>
                <div class="stat-box"><div class="stat-label">ATH</div><div class="stat-value">{{ ath }}</div></div>
                <div class="stat-box"><div class="stat-label">Active Since</div><div class="stat-value">{{ start }}</div></div>
            </div>

            <div class="chart-frame">
                {% if plot_url %}
                    <img src="data:image/png;base64,{{ plot_url }}" alt="BTC Market Chart">
                {% else %}
                    <p style="padding: 50px;">Connection Error: Unable to fetch market data from Kraken.</p>
                {% endif %}
            </div>

            <div class="legend-panel">
                <h3 style="margin-top:0">Analysis Details</h3>
                <div class="legend-item"><span class="color-box" style="background:#d63031"></span> <strong>Local High:</strong> Highest close within a 2-year forward/backward radius.</div>
                <div class="legend-item"><span class="color-box" style="background:#00b894"></span> <strong>Local Low:</strong> Lowest low within a 1-year forward/backward radius (excluding consolidations).</div>
                <div class="legend-item"><span class="color-box" style="background:white"></span> <strong>Consolidation:</strong> Periods where the 1-year price range was &lt; 30% (Lows ignored here).</div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, plot_url=plot_url, symbol=PAIR, current=current_price, ath=ath, start=start_date)

if __name__ == '__main__':
    print("Server launching on port 8080...")
    app.run(host='0.0.0.0', port=PORT)