from flask import Flask, render_template
import pandas as pd
import os
from fetch_price_data import fetch_price_data

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch price data and market cap data
    price_data = fetch_price_data()
    
    if price_data is None:
        return "Error: Could not fetch price data"
    
    # Prepare data for chart
    dates = price_data.index.strftime('%Y-%m-%d').tolist()
    prices = price_data['Close'].tolist()
    return render_template('index.html', dates=dates, prices=prices)
    market_caps = market_cap_data['market_cap'].tolist()
    
    return render_template('index.html', dates=dates, prices=prices, market_caps=market_caps)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)