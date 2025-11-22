from flask import Flask, render_template
import plotly.express as px
import pandas as pd
import os
import logging
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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

def prepare_data_for_prediction():
    """
    Prepare data for linear regression model to predict 7-day BTC price movement.
    Uses BTC price, volume, and total crypto market cap as features.
    """
    try:
        logging.debug("Preparing data for linear regression prediction")
        crypto_data = fetch_multiple_cryptos()
        if crypto_data is None:
            logging.error("Failed to fetch data for prediction")
            return None, None, None, None
        
        # Calculate features: BTC price, volume (not available, using placeholder), and total market cap
        btc_price = crypto_data['Bitcoin']
        # Note: Volume data is not fetched in current implementation; using BTC price as placeholder for volume
        volume = btc_price  # Placeholder; in practice, fetch actual volume from Binance
        total_market_cap = pd.Series(0, index=crypto_data.index)
        for crypto_name in crypto_data.columns:
            if crypto_name in CIRCULATING_SUPPLY:
                total_market_cap += crypto_data[crypto_name] * CIRCULATING_SUPPLY[crypto_name]
        
        # Create feature DataFrame and normalize
        features = pd.DataFrame({
            'BTC_Price': btc_price,
            'Volume': volume,
            'Total_Market_Cap': total_market_cap
        })
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
        
        # Calculate target: 7-day price movement (percentage change in BTC price)
        target = btc_price.pct_change(periods=7).shift(-7)  # Shift to align with current features
        
        # Drop NaN values from target and align features
        valid_indices = target.notna()
        features_clean = features_scaled[valid_indices]
        target_clean = target[valid_indices]
        
        logging.debug(f"Prepared data with {len(features_clean)} samples")
        return features_clean, target_clean, scaler, crypto_data.index[valid_indices]
    except Exception as e:
        logging.error(f"Error preparing data for prediction: {str(e)}")
        return None, None, None, None


@app.route('/predict')
def predict():
    """
    Route to train linear regression model and display predictions vs actual.
    """
    try:
        logging.debug("Starting prediction route")
        
        # First fetch the regular cryptocurrency data for the other graphs
        crypto_data = fetch_multiple_cryptos()
        if crypto_data is None:
            return "Error: Could not fetch cryptocurrency data."
        
        # Calculate market caps for the regular display
        btc_market_cap = crypto_data['Bitcoin'] * CIRCULATING_SUPPLY['Bitcoin']
        total_market_cap = pd.Series(0, index=crypto_data.index)
        for crypto_name in crypto_data.columns:
            if crypto_name in CIRCULATING_SUPPLY:
                total_market_cap += crypto_data[crypto_name] * CIRCULATING_SUPPLY[crypto_name]
        
        market_cap_data = pd.DataFrame({
            'Bitcoin Market Cap (Billion USD)': btc_market_cap,
            'Total Market Cap (Billion USD)': total_market_cap
        })
        
        # Generate regular price and market cap graphs
        fig_prices = px.line(crypto_data, x=crypto_data.index, y=crypto_data.columns, 
                     title='Cryptocurrency Prices Over Time',
                     labels={'value': 'Price (USDT)', 'variable': 'Cryptocurrency'})
        prices_graph_html = fig_prices.to_html(full_html=False)
        
        fig_market_cap = px.line(market_cap_data, x=market_cap_data.index, y=market_cap_data.columns,
                     title='Cryptocurrency Market Capitalization Over Time',
                     labels={'value': 'Market Cap (Billion USD)', 'variable': 'Market Cap'})
        market_cap_graph_html = fig_market_cap.to_html(full_html=False)
        
        # Now prepare data for linear regression prediction
        features, target, scaler, dates = prepare_data_for_prediction()
        if features is None or target is None:
            return "Error: Could not prepare data for prediction."
        
        # Split data into train and test sets (40% test)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42, shuffle=False)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Create DataFrame for plotting - use the actual test indices
        test_indices = X_test.index
        plot_data = pd.DataFrame({
            'Date': dates[test_indices],
            'Actual': y_test.values,
            'Predicted': y_pred
        }).set_index('Date')
        
        # Generate prediction graph
        fig = px.line(plot_data, x=plot_data.index, y=['Actual', 'Predicted'],
                     title='Linear Regression: Predicted vs Actual 7-Day BTC Price Movement',
                     labels={'value': 'Price Movement (Fraction)', 'variable': 'Type'})
        prediction_graph_html = fig.to_html(full_html=False)
        
        logging.debug("All graphs generated successfully")
        logging.debug(f"Prediction graph HTML length: {len(prediction_graph_html) if prediction_graph_html else 0}")
        return render_template('index.html', 
                             prediction_graph_html=prediction_graph_html,
                             prices_graph_html=prices_graph_html, 
                             market_cap_graph_html=market_cap_graph_html)
    except Exception as e:
        logging.error(f"Error in prediction route: {str(e)}")
        return "An error occurred during prediction. Please try again later."

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