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

# Static circulating supply values (in millions of coins)
CIRCULATING_SUPPLY = {
    'Bitcoin': 19.5,  # ~19.5 million BTC
    'Ethereum': 120.2,  # ~120.2 million ETH
    'Ripple': 54375.0,  # ~54.375 billion XRP
    'Cardano': 35500.0,  # ~35.5 billion ADA
    'Binance Coin': 153.4  # ~153.4 million BNB
}

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
        
        # Calculate market caps
        logging.debug("Calculating market capitalization data")
        btc_market_cap = crypto_data['Bitcoin'] * CIRCULATING_SUPPLY['Bitcoin']
        
        # Calculate total market cap for all cryptocurrencies
        total_market_cap = pd.Series(0, index=crypto_data.index)
        for crypto_name in crypto_data.columns:
            if crypto_name in CIRCULATING_SUPPLY:
                total_market_cap += crypto_data[crypto_name] * CIRCULATING_SUPPLY[crypto_name]
        
        # Create market cap DataFrame
        market_cap_data = pd.DataFrame({
            'Bitcoin Market Cap (Billion USD)': btc_market_cap,
            'Total Market Cap (Billion USD)': total_market_cap
        })
        
        # Generate price graph
        logging.debug("Generating Plotly graph from cryptocurrency data")
        fig_prices = px.line(crypto_data, x=crypto_data.index, y=crypto_data.columns, 
                     title='Cryptocurrency Prices Over Time',
                     labels={'value': 'Price (USDT)', 'variable': 'Cryptocurrency'})
        prices_graph_html = fig_prices.to_html(full_html=False)
        
        # Generate market cap graph
        logging.debug("Generating Plotly graph for market cap data")
        fig_market_cap = px.line(market_cap_data, x=market_cap_data.index, y=market_cap_data.columns,
                     title='Cryptocurrency Market Capitalization Over Time',
                     labels={'value': 'Market Cap (Billion USD)', 'variable': 'Market Cap'})
        market_cap_graph_html = fig_market_cap.to_html(full_html=False)
        
        logging.debug("Successfully generated graphs, rendering template")
        return render_template('index.html', 
                             prices_graph_html=prices_graph_html, 
                             market_cap_graph_html=market_cap_graph_html)
    except Exception as e:
        logging.error(f"Error generating cryptocurrency price graph: {str(e)}")
        return "An error occurred while generating the graph. Please try again later."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)