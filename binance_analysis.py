import io
import os
import requests
import pandas as pd
import matplotlib
# Set backend to Agg (Anti-Grain Geometry) for non-interactive server environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, send_file, make_response
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
METRICS_TO_FETCH = [
    {"slug": "market-price", "title": "Bitcoin Market Price (USD)", "color": "#f7931a"},
    {"slug": "hash-rate", "title": "Hash Rate (TH/s)", "color": "#007bff"},
    {"slug": "n-transactions", "title": "Daily Transactions", "color": "#28a745"},
    {"slug": "miners-revenue", "title": "Miners Revenue (USD)", "color": "#dc3545"}
]

BASE_URL = "https://api.blockchain.info/charts/{slug}"

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

@app.route('/')
def home():
    """
    Main route: Fetches data, generates plot, and returns it as an image.
    """
    # Create the plot figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axes = axes.flatten()
    plt.style.use('ggplot')

    # Fetch and plot each metric
    for i, metric in enumerate(METRICS_TO_FETCH):
        ax = axes[i]
        slug = metric["slug"]
        
        # Fetch data
        series = fetch_metric_data(slug)
        
        if series is not None and not series.empty:
            ax.plot(series.index, series.values, color=metric["color"], linewidth=2)
            ax.set_title(metric["title"], fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Format Y-axis with commas
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Rotate dates
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        else:
            ax.text(0.5, 0.5, 'Data Unavailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric["title"])

    fig.suptitle(f"Bitcoin On-Chain Metrics (Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')})", fontsize=16)
    plt.tight_layout()

    # Save plot to an in-memory buffer (no disk file needed)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    img_buffer.seek(0)
    plt.close(fig)

    # Return the image directly to the browser
    return send_file(img_buffer, mimetype='image/png')

if __name__ == "__main__":
    # Get PORT from environment for Railway compatibility (default to 5000)
    port = int(os.environ.get("PORT", 5000))
    # Host must be 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port)