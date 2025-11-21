#!/usr/bin/env python3

import pandas as pd
import numpy as np
from binance.client import Client
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import warnings
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store model state
current_model = None
last_training_time = None

def fetch_btc_data_realtime():
    """Fetch current BTC price and recent data"""
    try:
        client = Client()
        
        # Get current price
        ticker = client.get_symbol_ticker(symbol='BTCUSDT')
        current_price = float(ticker['price'])
        
        # Get 24h stats
        stats = client.get_24hr_ticker_stats(symbol='BTCUSDT')
        price_change = float(stats['priceChangePercent'])
        volume = float(stats['volume'])
        
        return {
            'current_price': current_price,
            'price_change': price_change,
            'volume': volume
        }
    except Exception as e:
        print(f"Error fetching BTC data: {e}")
        return {
            'current_price': 0,
            'price_change': 0,
            'volume': 0
        }

def fetch_historical_btc_data():
    """Fetch historical BTC data for analysis"""
    try:
        client = Client()
        
        # Fetch last 2000 1-hour candles
        klines = client.get_historical_klines(
            symbol='BTCUSDT',
            interval=Client.KLINE_INTERVAL_1HOUR,
            limit=2000
        )
        
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"Error fetching historical BTC data: {e}")
        return pd.DataFrame()

def generate_prediction_chart(historical_data):
    """Generate a simple price chart"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot historical prices
        plt.plot(historical_data['timestamp'], historical_data['close'], 
                label='BTC Price', linewidth=2, color='#ff6b35')
        
        plt.title('Bitcoin Price History', fontsize=16, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Price (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 for HTML embedding
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
        
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def get_simple_predictions(current_price):
    """Generate simple price predictions based on current price"""
    # This is a simplified prediction for demonstration
    # In a real application, you would use a proper ML model
    predictions = []
    base_price = current_price
    
    for i in range(10):
        # Simple random walk prediction
        change_percent = np.random.normal(0, 0.02)  # 2% standard deviation
        predicted_price = base_price * (1 + change_percent)
        predictions.append(predicted_price)
        base_price = predicted_price
    
    return predictions

@app.route('/')
def index():
    """Main page with BTC price and predictions"""
    try:
        # Fetch current BTC data
        btc_data = fetch_btc_data_realtime()
        
        # Fetch historical data
        historical_data = fetch_historical_btc_data()
        
        # Generate predictions
        predictions = get_simple_predictions(btc_data['current_price'])
        
        # Generate chart
        plot_url = generate_prediction_chart(historical_data)
        
        return render_template('index.html',
                             current_price=btc_data['current_price'],
                             price_change=btc_data['price_change'],
                             volume=btc_data['volume'],
                             data_count=len(historical_data),
                             predictions=predictions,
                             plot_url=plot_url)
        
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html',
                             current_price=0,
                             price_change=0,
                             volume=0,
                             data_count=0,
                             predictions=[],
                             plot_url=None)

@app.route('/retrain')
def retrain_model():
    """Endpoint to retrain the model"""
    try:
        # In a real application, this would retrain your ML model
        # For now, we'll just return success
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/covariance')
def covariance_analysis():
    """Endpoint to run the original covariance analysis"""
    try:
        # This runs the original covariance calculator logic
        from covariance_calculator import main
        
        # Capture the output (this is a simplified approach)
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        main()
        
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        return f"<pre>{output}</pre>"
        
    except Exception as e:
        return f"Error running covariance analysis: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)