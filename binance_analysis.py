import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, send_file
from datetime import timedelta
import io
import os

# --- 1. Data Download & Loading ---
def get_data():
    file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = 'ohlcv_data.csv'

    if not os.path.exists(output_file):
        print("Downloading data from Google Drive...")
        try:
            gdown.download(url, output_file, quiet=False)
        except Exception as e:
            print(f"Download failed: {e}")
            return pd.DataFrame(), None
    else:
        print("File already exists. Using local copy.")

    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame(), None

    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next((col for col in df.columns if 'date' in col or 'time' in col), None)
    if not date_col:
        raise ValueError("Could not identify a date/timestamp column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    return df, date_col

# --- 2. Mathematical Logic ---
def get_regression_for_days(df, date_col, days):
    """Calculates y = mx + b for a specific lookback window."""
    max_date = df[date_col].max()
    cutoff_date = max_date - timedelta(days=days)
    subset = df.loc[df[date_col] >= cutoff_date].copy()

    if subset.empty:
        return None, None
    
    # Calculate X in minutes (Unix epoch seconds // 60)
    subset['x_minutes'] = subset[date_col].astype('int64') // 10**9 // 60
    price_col = 'close' if 'close' in subset.columns else subset.columns[1]
    
    X = subset['x_minutes'].values
    Y = subset[price_col].values

    # Formula per request: 
    # m = sum((x - avg_x)(y - avg_y)) / sum(x - avg_x)^2
    # b = avg_y - m(avg_x)
    avg_x, avg_y = np.mean(X), np.mean(Y)
    num = np.sum((X - avg_x) * (Y - avg_y))
    den = np.sum((X - avg_x) ** 2)
    
    m = num / den if den != 0 else 0
    b = avg_y - (m * avg_x)
    
    y_pred = (m * X) + b
    
    return subset, (m, b, price_col, y_pred)

# --- 3. Flask Web Server with Matplotlib ---
app = Flask(__name__)
full_df = None
date_column_name = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OHLCV Regression Analysis</title>
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: white; text-align: center; padding: 20px; }
        .controls { margin-bottom: 30px; background: #2a2a2a; padding: 20px; border-radius: 10px; display: inline-block; }
        img { border: 2px solid #444; border-radius: 5px; max-width: 95%; height: auto; }
        .btn-group { display: flex; flex-wrap: wrap; justify-content: center; gap: 5px; max-width: 800px; margin: 0 auto; }
        a { text-decoration: none; padding: 8px 12px; background: #444; color: white; border-radius: 4px; font-size: 12px; }
        a:hover { background: #666; }
        a.active { background: #ffcc00; color: black; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Market Regression Analysis</h1>
    <div class="controls">
        <p>Select Lookback Window (Days):</p>
        <div class="btn-group">
            {% for i in range(1, 31) %}
            <a href="/{{ i }}" class="{{ 'active' if i == current_day else '' }}">{{ i }}d</a>
            {% endfor %}
        </div>
    </div>
    <div id="chart">
        <img src="/plot/{{ current_day }}" alt="Regression Plot">
    </div>
</body>
</html>
"""

@app.route('/')
@app.route('/<int:days>')
def index(days=30):
    return render_template_string(HTML_TEMPLATE, current_day=days)

@app.route('/plot/<int:days>')
def plot_png(days):
    subset, stats = get_regression_for_days(full_df, date_column_name, days)
    if subset is None:
        return "No data", 404
    
    m, b, price_col, y_pred = stats

    # Create Matplotlib Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Plot price data
    ax.plot(subset[date_column_name], subset[price_col], color='#00d2ff', alpha=0.6, label=f'Price ({price_col})')
    
    # Plot regression line
    ax.plot(subset[date_column_name], y_pred, color='#ffcc00', linewidth=2, label=f'Fit: y = {m:.6f}x + {b:.2f}')
    
    ax.set_title(f"Linear Regression: Last {days} Day(s)", fontsize=14, color='#ffcc00')
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Rotation for dates
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor='#1a1a1a')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    full_df, date_column_name = get_data()
    if not full_df.empty:
        print("\nStarting Web Server on http://0.0.0.0:8050")
        # Set host to 0.0.0.0 for public network access
        app.run(host='0.0.0.0', port=8050, debug=False)