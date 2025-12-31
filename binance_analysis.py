import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, send_file
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

    # Clean headers
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next((col for col in df.columns if 'date' in col or 'time' in col), None)
    if not date_col:
        raise ValueError("Could not identify a date/timestamp column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    return df, date_col

# --- 2. Calculation Logic ---
def calculate_all_lines(df, date_col):
    """Calculates regression params for lookbacks 1 through 30."""
    max_date = df[date_col].max()
    price_col = 'close' if 'close' in df.columns else df.columns[1]
    results = []

    for days in range(1, 31):
        cutoff = max_date - timedelta(days=days)
        subset = df.loc[df[date_col] >= cutoff].copy()
        
        if len(subset) < 2:
            continue
            
        # X in minutes (Unix epoch seconds // 60)
        subset['x_minutes'] = subset[date_col].astype('int64') // 10**9 // 60
        X = subset['x_minutes'].values
        Y = subset[price_col].values

        # Formula: m = sum((x - avg_x)(y - avg_y)) / sum(x - avg_x)^2
        avg_x, avg_y = np.mean(X), np.mean(Y)
        num = np.sum((X - avg_x) * (Y - avg_y))
        den = np.sum((X - avg_x) ** 2)
        
        m = num / den if den != 0 else 0
        b = avg_y - (m * avg_x)
        
        # We only need the start and end points of the line to draw it
        x_ends = np.array([X[0], X[-1]])
        y_ends = (m * x_ends) + b
        t_ends = [subset[date_col].iloc[0], subset[date_col].iloc[-1]]
        
        results.append({
            'days': days,
            'm': m,
            'b': b,
            'times': t_ends,
            'prices': y_ends
        })
        
    return results, price_col

# --- 3. Flask Server ---
app = Flask(__name__)
full_df = None
date_column_name = None

@app.route('/')
def plot_all():
    lines, price_col = calculate_all_lines(full_df, date_column_name)
    
    # Create the Matplotlib Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    
    # Plot original price data (the last 30 days of it)
    max_date = full_df[date_column_name].max()
    plot_subset = full_df[full_df[date_column_name] >= (max_date - timedelta(days=30))]
    ax.plot(plot_subset[date_column_name], plot_subset[price_col], 
            color='white', alpha=0.3, linewidth=1, label='Actual Price')

    # Use a colormap to distinguish between days (1 to 30)
    cmap = plt.cm.get_cmap('plasma', 30)

    for i, line in enumerate(lines):
        # Color goes from yellow (1d) to purple (30d)
        color = cmap(i/30)
        alpha = 0.8 if line['days'] in [1, 7, 14, 30] else 0.3
        linewidth = 2 if line['days'] in [1, 7, 14, 30] else 0.8
        
        ax.plot(line['times'], line['prices'], color=color, 
                alpha=alpha, linewidth=linewidth, 
                label=f"{line['days']}d Fit" if line['days'] in [1, 7, 14, 30] else "")

    ax.set_title("30 Concurrent Regression Lines (1 to 30 Day Lookbacks)", fontsize=16, pad=20)
    ax.set_xlabel("Date/Time", fontsize=12)
    ax.set_ylabel(f"Price ({price_col.upper()})", fontsize=12)
    ax.legend(loc='upper left', fontsize='small', framealpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.2)
    
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Stream the image back to the browser
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor='#000000')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    full_df, date_column_name = get_data()
    if not full_df.empty:
        print("\nServer running. Access via http://<your-ip>:8050")
        app.run(host='0.0.0.0', port=8050, debug=False)