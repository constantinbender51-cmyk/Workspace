from flask import Flask, render_template
import pandas as pd
from fetch_market_cap import fetch_market_cap
import os
from fetch_price_data import fetch_price_data

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch price data and market cap data
    price_data = fetch_price_data()
    market_cap_data = fetch_market_cap()
    if price_data is None or market_cap_data is None:
        return "Error: Could not fetch price or market cap data"
    
    if data is None:
    return render_template('index.html', dates=dates, prices=prices, market_caps=market_caps)
        return "Error: Could not fetch price data"
    dates = price_data.index.strftime('%Y-%m-%d').tolist()
    prices = price_data['Close'].tolist()
    market_caps = market_cap_data['market_cap'].tolist()
    
    # Prepare data for chart
    dates = data.index.strftime('%Y-%m-%d').tolist()
    return render_template('index.html', dates=dates, prices=prices, market_caps=market_caps)
    prices = data['Close'].tolist()
    
    return render_template('index.html', dates=dates, prices=prices)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)