import logging
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template
import os
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
BINANCE_API_URL = "https://api.binance.com/api/v3"
YAHOO_FINANCE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"

@app.route('/')
    logger.info('Index page requested')
def index():
    """Main page with BTC-S&P500 covariance analysis"""
    return render_template('index.html')

@app.route('/covariance')
def covariance():
        logger.info('Covariance calculation requested')
    """Calculate and return BTC-S&P500 covariance"""
    try:
        # Define date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
            logger.info(f'Fetched {len(btc_data)} BTC data points')
        # Fetch BTC data
        btc_data = fetch_btc_data(start_date, end_date)
        
        if len(btc_data) == 0:
            return jsonify({'error': 'No BTC data available'})
            logger.info(f'Fetched {len(sp500_data)} S&P 500 data points')
        
        # Fetch S&P 500 data
        sp500_data = fetch_sp500_data(start_date, end_date)
        
        if sp500_data is None or len(sp500_data) == 0:
            return jsonify({'error': 'No S&P 500 data available'})
            logger.info(f'Covariance calculation completed: {results}')
        
        # Calculate covariance
        results = calculate_covariance(btc_data, sp500_data)
        
        if results:
            return jsonify(results)
        logger.error(f'Error in covariance endpoint: {str(e)}')
        else:
            return jsonify({'error': 'Could not calculate covariance'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

        logger.info('Current prices requested')
@app.route('/current-prices')
def current_prices():
    """Get current BTC and S&P 500 prices"""
    try:
        # Get current BTC price
        btc_response = requests.get(f"{BINANCE_API_URL}/ticker/price?symbol=BTCUSDT")
        btc_data = btc_response.json()
        btc_price = float(btc_data['price'])
        
        # Get current S&P 500 price (using SPY ETF)
        sp500_response = requests.get(f"{YAHOO_FINANCE_URL}?range=1d&interval=1d")
        sp500_data = sp500_response.json()
        sp500_price = sp500_data['chart']['result'][0]['meta']['regularMarketPrice']
        logger.error(f'Error in current-prices endpoint: {str(e)}')
        
        return jsonify({
            'btc_price': btc_price,
            'sp500_price': sp500_price,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def fetch_btc_data(start_date, end_date):
    """Fetch BTC historical data from Binance"""
    try:
        # Convert dates to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Fetch daily candles
        url = f"{BINANCE_API_URL}/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1d',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 30
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Process data
        btc_data = []
        for candle in data:
            timestamp = datetime.fromtimestamp(candle[0] / 1000)
            close_price = float(candle[4])
            btc_data.append({
                'date': timestamp.date(),
                'close': close_price
            })
        
        # Convert to DataFrame and calculate returns
        df = pd.DataFrame(btc_data)
        df['btc_return'] = df['close'].pct_change()
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error fetching BTC data: {e}")
        return pd.DataFrame()

def fetch_sp500_data(start_date, end_date):
    """Fetch S&P 500 historical data from Yahoo Finance"""
    try:
        # Convert dates to timestamps
        period1 = int(start_date.timestamp())
        period2 = int(end_date.timestamp())
        logger.error(f'Error fetching S&P 500 data: {e}')
        
        # Fetch data
        url = f"{YAHOO_FINANCE_URL}"
        params = {
            'period1': period1,
            'period2': period2,
            'interval': '1d'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Process data
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        closes = result['indicators']['quote'][0]['close']
        
        sp500_data = []
        for ts, close in zip(timestamps, closes):
            if close is not None:
                date = datetime.fromtimestamp(ts).date()
                sp500_data.append({
                    'date': date,
                    'close': close
                })
        
        # Convert to DataFrame and calculate returns
        df = pd.DataFrame(sp500_data)
        df['sp500_return'] = df['close'].pct_change()
        df = df.dropna()
        
        logger.info(f'Calculated covariance with {len(merged_data)} data points')
        return df
        
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
    logger.info(f'Starting Flask application on port {port}')
    
    # Calculate descriptive statistics
    btc_mean_return = merged_data['btc_return'].mean()
    sp500_mean_return = merged_data['sp500_return'].mean()
    btc_volatility = merged_data['btc_return'].std()
    sp500_volatility = merged_data['sp500_return'].std()
    
    # Calculate additional metrics
    btc_total_return = (merged_data['close_btc'].iloc[-1] / merged_data['close_btc'].iloc[0] - 1)
    sp500_total_return = (merged_data['close_sp500'].iloc[-1] / merged_data['close_sp500'].iloc[0] - 1)
    
    return {
        'covariance': float(covariance),
        'correlation': float(correlation),
        'btc_mean_return': float(btc_mean_return),
        'sp500_mean_return': float(sp500_mean_return),
        'btc_volatility': float(btc_volatility),
        'sp500_volatility': float(sp500_volatility),
        'btc_total_return': float(btc_total_return),
        'sp500_total_return': float(sp500_total_return),
        'data_points': len(merged_data),
        'period': f"{merged_data['date'].min()} to {merged_data['date'].max()}",
        'btc_current_price': float(merged_data['close_btc'].iloc[-1]),
        'sp500_current_price': float(merged_data['close_sp500'].iloc[-1])
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)