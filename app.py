from flask import Flask, render_template
import pandas as pd
import os
from fetch_price_data import fetch_price_data

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch top 10 liquid assets and their price changes relative to BTC
    from fetch_top_assets import fetch_top_assets_by_volume
    from fetch_price_data import fetch_price_data
    
    # Fetch top 10 assets by volume
    top_assets = fetch_top_assets_by_volume(10)
    if not top_assets:
        return "Error: Could not fetch top assets"
    
    # Fetch BTC price data for comparison
    btc_data = fetch_price_data('BTCUSDT')
    if btc_data is None:
        return "Error: Could not fetch BTC price data"
    
    # Calculate price change percentages for each asset relative to BTC
    asset_changes = []
    for symbol in top_assets:
        asset_data = fetch_price_data(symbol)
        if asset_data is not None and not asset_data.empty:
            # Calculate percentage change: ((asset_price - btc_price) / btc_price) * 100
            asset_price = asset_data['Close'].iloc[-1]  # Latest price
            btc_price = btc_data['Close'].iloc[-1]  # Latest BTC price
            change_percent = ((asset_price - btc_price) / btc_price) * 100
            asset_changes.append({'symbol': symbol, 'change_percent': change_percent})
    
    if not asset_changes:
        return "Error: No valid price data for top assets"
    return render_template('index.html', asset_changes=asset_changes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)