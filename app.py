from flask import Flask, render_template
import plotly.express as px
import pandas as pd
import os
import logging
import time
from fetch_price_data import fetch_multiple_cryptos

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/')
def index():
    try:
        logging.debug("Starting index route - fetching multiple cryptocurrency data")
        # Fetch historical data for multiple cryptocurrencies
        crypto_data = fetch_multiple_cryptos()
        if crypto_data is None:
            logging.error("Failed to fetch cryptocurrency data - returned None")
            return "Error: Could not fetch cryptocurrency data. Please check the API connection."
        logging.debug(f"Successfully fetched data for {len(crypto_data.columns)} cryptocurrencies")
        # Generate a plotly graph
        logging.debug("Generating Plotly graph from cryptocurrency data")
        fig = px.line(crypto_data, x=crypto_data.index, y=crypto_data.columns, 
                     title='Cryptocurrency Prices Over Time',
                     labels={'value': 'Price (USDT)', 'variable': 'Cryptocurrency'})
        graph_html = fig.to_html(full_html=False)
        logging.debug("Successfully generated graph HTML, rendering template")
        return render_template('index.html', graph_html=graph_html)
    except Exception as e:
        logging.error(f"Error generating cryptocurrency price graph: {str(e)}")
        return "An error occurred while generating the graph. Please try again later."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)