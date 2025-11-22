from flask import Flask, render_template
import plotly.express as px
import pandas as pd
import os
import logging
from fetch_price_data import fetch_price_data

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/')
def index():
    try:
        logging.debug("Starting index route - fetching BTC price data")
        # Fetch historical BTC price data
        btc_data = fetch_price_data(symbol='BTCUSDT', start_date='2022-01-01', end_date='2023-09-30')
        if btc_data is None:
            logging.error("Failed to fetch BTC price data - returned None")
            return "Error: Could not fetch BTC price data. Please check the API connection."
        logging.debug(f"Successfully fetched BTC data with {len(btc_data)} records")
        # Generate a plotly graph
        logging.debug("Generating Plotly graph from BTC data")
        fig = px.line(btc_data, x=btc_data.index, y='Close', title='BTC Price Over Time')
        graph_html = fig.to_html(full_html=False)
        logging.debug("Successfully generated graph HTML, rendering template")
        return render_template('index.html', graph_html=graph_html)
    except Exception as e:
        logging.error(f"Error generating BTC price graph: {str(e)}")
        return "An error occurred while generating the graph. Please try again later."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)