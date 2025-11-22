from flask import Flask, render_template
import pandas as pd
import os
from fetch_price_data import fetch_price_data

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch top 10 liquid assets and calculate total market cap
    from fetch_top_assets import fetch_top_assets_by_volume
    from fetch_price_data import fetch_price_data
    
    # Fetch top 10 assets by volume
    top_assets = fetch_top_assets_by_volume(10)
    if not top_assets:
        return "Error: Could not fetch top assets"
    
    # Calculate total market cap for all top assets
    total_market_cap = 0
    market_caps = []
    
    for symbol in top_assets:
        asset_data = fetch_price_data(symbol)
        if asset_data is not None and not asset_data.empty:
            # Get latest price
            latest_price = asset_data['Close'].iloc[-1]
            # Get 24h volume for circulating supply estimation
            from binance.client import Client
            client = Client()
            ticker = client.get_ticker(symbol=symbol)
            # Estimate market cap using price * 24h volume (simplified approach)
            market_cap = latest_price * float(ticker['quoteVolume'])
            total_market_cap += market_cap
            market_caps.append({'symbol': symbol, 'market_cap': market_cap})
    
    if total_market_cap == 0:
        return "Error: Could not calculate market cap data"
    
    return render_template('index.html', total_market_cap=total_market_cap, market_caps=market_caps)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)