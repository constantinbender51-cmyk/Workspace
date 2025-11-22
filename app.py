from flask import Flask, render_template
import pandas as pd
import os
from fetch_price_data import fetch_price_data

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch top 10 assets by volume
    from fetch_top_assets import fetch_top_assets_by_volume
    top_assets = fetch_top_assets_by_volume(10)
    if not top_assets:
        return "Error: Could not fetch top assets"
    # Return simple list of top assets without market cap data
    return f"Top 10 assets by volume: {', '.join(top_assets)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)