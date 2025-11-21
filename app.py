#!/usr/bin/env python3

import pandas as pd
import numpy as np
from binance.client import Client
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from flask import Flask, render_template, jsonify
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store current state
current_btc_price = 0
    print("Starting BTC Price Prediction App...")
current_volume = 0
predictions = []

@app.route('/')
def index():
    """Main page with BTC price and predictions"""
        print(f"Fetching current BTC price from Binance...")
    global current_btc_price, current_volume, predictions
    
    # Get current BTC price
    try:
        client = Client()
        ticker = client.get_symbol_ticker(symbol='BTCUSDT')
        current_btc_price = float(ticker['price'])
        
        # Get 24h stats for volume
        stats = client.get_24hr_ticker_statistics(symbol='BTCUSDT')
        current_volume = float(stats['volume'])
        
        # Generate some sample predictions
        predictions = generate_sample_predictions(current_btc_price)
        
        # Generate plot
        print(f"Using fallback data due to error: {e}")
        plot_url = generate_plot(current_btc_price, predictions)
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        current_btc_price = 45000.0
        current_volume = 25000
        predictions = generate_sample_predictions(current_btc_price)
        plot_url = generate_plot(current_btc_price, predictions)
    
    return render_template('index.html',
                         current_price=current_btc_price,
                         price_change=2.5,  # Sample data
                         volume=current_volume,
                         data_count=2000,   # Sample data
                         predictions=predictions,
                         plot_url=plot_url)

@app.route('/covariance')
def covariance():
    """Calculate and display BTC-S&P500 covariance"""
    try:
        # Define date range (last 30 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Fetch BTC data
        btc_data = fetch_btc_data_daily(start_date, end_date)
        
        if len(btc_data) == 0:
            return jsonify({'error': 'No BTC data available'})
        
        # Fetch S&P 500 data
        sp500_data = fetch_sp500_data(start_date, end_date)
        
        if sp500_data is None or len(sp500_data) == 0:
            return jsonify({'error': 'No S&P 500 data available'})
        
        # Calculate covariance
        results = calculate_covariance(btc_data, sp500_data)
        
        if results:
            return jsonify(results)
        else:
            return jsonify({'error': 'Could not calculate covariance'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/retrain')
def retrain():
    """Endpoint to retrain the model"""
    # In a real implementation, this would retrain the ML model
    # For now, just return success
    return jsonify({'status': 'success', 'message': 'Model retrained (simulated)'})

def fetch_btc_data_daily(start_date, end_date):
    """Fetch daily BTC candles from Binance for specified date range"""
    print(f"Fetching BTC data from {start_date} to {end_date}...")
    
    client = Client()
    
    # Fetch daily candles
    klines = client.get_historical_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start_date,
        end_str=end_date
    )
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['timestamp'].dt.date
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Calculate daily returns
    df['btc_return'] = df['close'].pct_change()
    
    # Keep only relevant columns
    df = df[['date', 'close', 'btc_return']].dropna()
    
    return df

def fetch_sp500_data(start_date, end_date):
    """Fetch real S&P 500 data from Yahoo Finance"""
    print(f"Fetching S&P 500 data from {start_date} to {end_date}...")
    
    # Use SPY ETF as proxy for S&P 500
    ticker = "SPY"
    
    try:
        # Download historical data
        sp500 = yf.download(ticker, start=start_date, end=end_date)
        
        if sp500.empty:
            print("No S&P 500 data available for the specified date range")
            return None
        
        # Reset index and prepare data
        sp500 = sp500.reset_index()
        sp500['date'] = sp500['Date'].dt.date
        sp500['sp500_return'] = sp500['Close'].pct_change()
        
        # Keep only relevant columns
        sp500 = sp500[['date', 'Close', 'sp500_return']].rename(columns={'Close': 'close'})
        sp500 = sp500.dropna()
        
        print(f"Fetched {len(sp500)} days of S&P 500 data")
        return sp500
        
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        return None

def calculate_covariance(btc_data, sp500_data):
    """Calculate covariance between BTC and S&P 500 returns"""
    
    # Merge data on date
    merged_data = pd.merge(btc_data, sp500_data, on='date', suffixes=('_btc', '_sp500'))
    
    if len(merged_data) < 2:
        print("Not enough overlapping data to calculate covariance")
        return None
    
    # Calculate covariance
    covariance = np.cov(merged_data['btc_return'], merged_data['sp500_return'])[0, 1]
    
    # Calculate correlation for additional insight
    correlation = np.corrcoef(merged_data['btc_return'], merged_data['sp500_return'])[0, 1]
    
    # Calculate descriptive statistics
    btc_mean_return = merged_data['btc_return'].mean()
    sp500_mean_return = merged_data['sp500_return'].mean()
    btc_volatility = merged_data['btc_return'].std()
    sp500_volatility = merged_data['sp500_return'].std()
    
    # Calculate additional metrics
    btc_total_return = (merged_data['close_btc'].iloc[-1] / merged_data['close_btc'].iloc[0] - 1)
    sp500_total_return = (merged_data['close_sp500'].iloc[-1] / merged_data['close_sp500'].iloc[0] - 1)
    
    return {
        'covariance': covariance,
        'correlation': correlation,
        'btc_mean_return': btc_mean_return,
        'sp500_mean_return': sp500_mean_return,
        'btc_volatility': btc_volatility,
        'sp500_volatility': sp500_volatility,
        'btc_total_return': btc_total_return,
        'sp500_total_return': sp500_total_return,
        'data_points': len(merged_data),
        'period': f"{merged_data['date'].min()} to {merged_data['date'].max()}",
        'merged_data': merged_data.to_dict('records')
    }

def generate_sample_predictions(current_price):
    """Generate sample predictions based on current price"""
    # Simple linear growth with some randomness
    predictions = []
    price = current_price
    for i in range(10):
        # Small random change between -1% and +2%
        change = np.random.uniform(-0.01, 0.02)
        price = price * (1 + change)
        predictions.append(round(price, 2))
    return predictions

def generate_plot(current_price, predictions):
    """Generate a plot showing current price and predictions"""
    plt.figure(figsize=(10, 6))
    
    # Historical data (simulated)
    historical_prices = [current_price * (1 - 0.05 * i) for i in range(5, 0, -1)]
    historical_prices.append(current_price)
    
    # Plot historical data
    plt.plot(range(len(historical_prices)), historical_prices, 'b-', label='Historical', linewidth=2)
    
    # Plot predictions
    prediction_indices = range(len(historical_prices) - 1, len(historical_prices) + len(predictions))
    all_prices = [historical_prices[-1]] + predictions
    plt.plot(prediction_indices, all_prices, 'r--', label='Predictions', linewidth=2)
    
    plt.axvline(x=len(historical_prices) - 1, color='gray', linestyle=':', alpha=0.7)
    plt.text(len(historical_prices) - 1, min(all_prices) * 0.95, 'Now', ha='center')
    
    plt.title('BTC Price Prediction')
    plt.xlabel('Time Period')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Convert plot to base64 for HTML embedding
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)