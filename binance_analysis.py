import io
import os
import requests
import pandas as pd
import matplotlib
# Set backend to Agg (Anti-Grain Geometry) for non-interactive server environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from flask import Flask, send_file
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
BASE_URL = "https://api.blockchain.info/charts/{slug}"

METRICS_TO_FETCH = [
    {"slug": "market-price", "title": "Market Price (USD)", "color": "#f7931a", "key": "price"},
    {"slug": "hash-rate", "title": "Hash Rate (TH/s)", "color": "#007bff", "key": "hash"},
    {"slug": "n-transactions", "title": "Daily Transactions", "color": "#28a745", "key": "tx_count"},
    {"slug": "miners-revenue", "title": "Miners Revenue (USD)", "color": "#dc3545", "key": "revenue"},
    # CHANGED: 'estimated-transaction-volume-usd' (On-Chain) -> 'trade-volume' (Exchange Trading Volume)
    {"slug": "trade-volume", "title": "Exchange Volume (USD)", "color": "#6f42c1", "key": "volume"}
]

def fetch_metric_data(slug, timespan="1year"):
    """
    Fetches chart data from Blockchain.com API.
    Returns a pandas Series or None.
    """
    url = BASE_URL.format(slug=slug)
    params = {"timespan": timespan, "format": "json", "sampled": "true"}
    headers = {"User-Agent": "Mozilla/5.0 (Cloud Deployment)"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['date'] = pd.to_datetime(df['x'], unit='s')
            df.set_index('date', inplace=True)
            return df['y']
    except Exception as e:
        print(f"Error fetching {slug}: {e}")
        return None

def format_currency(x, pos):
    """Format large numbers with K, M, B suffixes."""
    if x >= 1e9:
        return f'${x*1e-9:.1f}B'
    elif x >= 1e6:
        return f'${x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'${x*1e-3:.0f}K'
    return f'${x:.0f}'

def format_number(x, pos):
    """Format large numbers with K, M, B suffixes (No dollar sign)."""
    if x >= 1e9:
        return f'{x*1e-9:.1f}B'
    elif x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'
    return f'{x:.0f}'

@app.route('/')
def home():
    """
    Main route: Fetches data, calculates derived metrics, and generates a 3x2 grid.
    """
    # Dictionary to store fetched series for calculation
    data_store = {}
    
    # 1. Fetch all base metrics
    for item in METRICS_TO_FETCH:
        data_store[item['key']] = {
            "series": fetch_metric_data(item['slug']),
            "info": item
        }

    # 2. Calculate "Volume / Transactions" 
    # Logic: Exchange Trading Volume / Daily Confirmed Transactions
    # This represents the dollar amount traded on exchanges per single on-chain transaction.
    avg_tx_val_series = None
    if data_store['volume']['series'] is not None and data_store['tx_count']['series'] is not None:
        # Pandas aligns indices (dates) automatically during division
        avg_tx_val_series = data_store['volume']['series'] / data_store['tx_count']['series']

    # 3. Define the Plotting Order (6 Plots)
    plots_config = [
        data_store['price'],
        data_store['hash'],
        data_store['tx_count'],
        data_store['revenue'],
        data_store['volume'],
        {
            "series": avg_tx_val_series,
            "info": {"title": "Exchange Vol / Tx (Ratio)", "color": "#20c997"} 
        }
    ]

    # 4. Create Figure (3 Rows x 2 Columns)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 14))
    axes = axes.flatten()
    plt.style.use('ggplot')

    for i, plot_obj in enumerate(plots_config):
        ax = axes[i]
        series = plot_obj.get("series")
        info = plot_obj.get("info")
        
        if series is not None and not series.empty:
            ax.plot(series.index, series.values, color=info["color"], linewidth=2)
            ax.set_title(info["title"], fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Smart Y-Axis Formatting
            if "USD" in info["title"] or "Vol" in info["title"]:
                ax.yaxis.set_major_formatter(FuncFormatter(format_currency))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(format_number))
                
            # Rotate dates
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Data Unavailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(info["title"])

    fig.suptitle(f"Bitcoin Metrics (Exchange & On-Chain) - {datetime.now().strftime('%Y-%m-%d')}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust for suptitle

    # Save to memory buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    img_buffer.seek(0)
    plt.close(fig)

    return send_file(img_buffer, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)