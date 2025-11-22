from flask import Flask, render_template
import plotly.express as px
import pandas as pd
import os
from fetch_price_data import fetch_price_data

app = Flask(__name__)

@app.route('/btc-price')
def btc_price():
    try:
        # Fetch historical BTC price data
        btc_data = fetch_price_data(symbol='BTCUSDT', start_date='2022-01-01', end_date='2023-09-30')
        if btc_data is None:
            return "Error: Could not fetch BTC price data. Please check the API connection."
        # Generate a plotly graph
        fig = px.line(btc_data, x=btc_data.index, y='Close', title='BTC Price Over Time')
        graph_html = fig.to_html(full_html=False)
        return render_template('btc_price.html', graph_html=graph_html)
    except Exception as e:
        return f"Error generating BTC price graph: {str(e)}"

@app.route('/')
def index():
    # Fetch top 10 assets by volume
    from fetch_top_assets import fetch_top_assets_by_volume
    top_assets = fetch_top_assets_by_volume(10)
    if not top_assets:
        return "Error: Could not fetch top assets. Please check the API connection."
    # Render template with top assets list
    return render_template('index.html', top_assets=top_assets)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)