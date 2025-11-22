from flask import Flask, render_template
import pandas as pd
import os
from fetch_price_data import fetch_price_data

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch price data
    data = fetch_price_data()
    
    if data is None:
        return "Error: Could not fetch price data"
    
    # Prepare data for chart
    dates = data.index.strftime('%Y-%m-%d').tolist()
    prices = data['Close'].tolist()
    
    return render_template('index.html', dates=dates, prices=prices)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)