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
    
    # Base Plot
    plt.plot(df['time'], df['close'], label='Close Price', color='#2c3e50', linewidth=1.5, zorder=2)
    
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
        plt.axvspan(start_date, end_date, color='slategrey', alpha=0.1, zorder=1, edgecolor=None)

    plt.plot([], [], color='slategrey', alpha=0.3, linewidth=10, label='Consolidation Area')

    # --- Pattern 1: Local High (Close, 2yr radius) ---
    # Radius = 24 months. Window = 49 (24 pre + 1 curr + 24 post)
    df['rolling_max_close'] = df['close'].rolling(window=49, center=True, min_periods=25).max()
    
    valid_highs = df.copy()
    if len(valid_highs) > 24:
        valid_highs.loc[valid_highs.index[-24:], 'rolling_max_close'] = np.inf

    local_highs = valid_highs[valid_highs['close'] == valid_highs['rolling_max_close']]
    plt.scatter(local_highs['time'], local_highs['close'], color='#d63031', s=120, marker='v', zorder=5, edgecolors='white', label='Local High (2yr)')

    # --- Pattern 3: Local Low (Low, 1yr radius, ignoring consolidation) ---
    # Radius = 12 months. Window = 25 (12 pre + 1 curr + 12 post)
    df_lows_clean = df.copy()
    df_lows_clean.loc[consolidation_mask, 'low'] = np.inf
    
    df_lows_clean['rolling_min_low'] = df_lows_clean['low'].rolling(window=25, center=True, min_periods=13).min()
    
    if len(df_lows_clean) > 12:
        df_lows_clean.loc[df_lows_clean.index[-12:], 'rolling_min_low'] = -1.0

    local_lows = df_lows_clean[df_lows_clean['low'] == df_lows_clean['rolling_min_low']]
    plt.scatter(local_lows['time'], local_lows['low'], color='#00b894', s=120, marker='^', zorder=5, edgecolors='white', label='Local Low (1yr)')

    # Styling
    plt.title(f'Market Structure Analysis: {PAIR} (Kraken)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price (USD) - Log Scale', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="major", color='#dcdde1', linestyle='-')
    plt.legend(frameon=True, loc='upper left', facecolor='white', framealpha=0.9)
    plt.tight_layout()

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
            body { font-family: system-ui; margin: 0; padding: 20px; background-color: #f1f2f6; color: #2f3542; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .stats { display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; }
            .stat-box { background: #f8f9fa; padding: 10px 20px; border-radius: 8px; text-align: center; border-bottom: 3px solid #3498db; }
            img { max-width: 100%; height: auto; border-radius: 4px; }
            .legend { font-size: 0.85em; margin-top: 20px; padding: 15px; background: #fffbe6; border-radius: 6px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align:center">Kraken Market Structure</h1>
            <div class="stats">
                <div class="stat-box"><strong>Price:</strong> {{ current }}</div>
                <div class="stat-box"><strong>ATH:</strong> {{ ath }}</div>
                <div class="stat-box"><strong>Since:</strong> {{ start }}</div>
            </div>
            <div style="text-align:center">
                {% if plot_url %}<img src="data:image/png;base64,{{ plot_url }}">{% endif %}
            </div>
            <div class="legend">
                <strong>Markers:</strong><br>
                - <span style="color:#d63031">▼ Red:</span> Local Close High (2-year radius)<br>
                - <span style="color:#00b894">▲ Green:</span> Local Low (1-year radius, excludes consolidation zones)<br>
                - <span style="color:slategrey">■ Shaded:</span> 30% Price Range Consolidation (min 1-year duration)
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, plot_url=plot_url, current=current_price, ath=ath, start=start_date)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)