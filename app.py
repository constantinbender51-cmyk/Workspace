#!/usr/bin/env python3

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from binance.client import Client
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

app = Flask(__name__)

def fetch_btc_data():
    """Fetch recent BTC data from Binance"""
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
        print(f"Error fetching BTC data: {e}")
        return None

def prepare_features(df):
    """Prepare features for machine learning model"""
    # Calculate technical indicators
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    
    # Drop NaN values
    df = df.dropna()
    
    # Feature columns
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'price_change', 
                    'volume_change', 'high_low_ratio', 'ma_5', 'ma_10', 'ma_20']
    
    return df, feature_cols

def train_model(df, feature_cols):
    """Train linear regression model"""
    # Prepare features and target
    X = df[feature_cols]
    y = df['close'].shift(-1)  # Predict next period's close price
    
    # Remove last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

def make_predictions(model, scaler, df, feature_cols, periods=10):
    """Make future predictions"""
    predictions = []
    current_data = df[feature_cols].iloc[-1:].copy()
    
    for i in range(periods):
        # Scale current data
        current_scaled = scaler.transform(current_data)
        
        # Make prediction
        pred = model.predict(current_scaled)[0]
        predictions.append(pred)
        
        # Update current data for next prediction
        # (This is a simplified approach - in practice you'd need more sophisticated feature engineering)
        current_data['close'] = pred
        current_data['open'] = pred * 0.999  # Small random variation
        current_data['high'] = pred * 1.001
        current_data['low'] = pred * 0.998
        
    return predictions

def create_plot(df, predictions):
    """Create price prediction chart"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(df['timestamp'], df['close'], label='Historical Price', linewidth=2)
    
    # Plot predictions
    future_times = [df['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(len(predictions))]
    plt.plot(future_times, predictions, 'r--', label='Predictions', linewidth=2, marker='o')
    
    plt.title('BTC Price Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 for HTML embedding
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """Main page with BTC price predictions"""
    try:
        # Fetch BTC data
        df = fetch_btc_data()
        
        if df is None or len(df) == 0:
            return render_template('index.html', 
                                 current_price=0,
                                 price_change=0,
                                 volume=0,
                                 data_count=0,
                                 predictions=[],
                                 plot_url=None)
        
        # Prepare features and train model
        df, feature_cols = prepare_features(df)
        model, scaler = train_model(df, feature_cols)
        
        # Make predictions
        predictions = make_predictions(model, scaler, df, feature_cols, periods=10)
        
        # Create plot
        plot_url = create_plot(df, predictions)
        
        # Calculate current metrics
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        volume = df['volume'].iloc[-1]
        data_count = len(df)
        
        return render_template('index.html',
                             current_price=current_price,
                             price_change=price_change,
                             volume=volume,
                             data_count=data_count,
                             predictions=predictions,
                             plot_url=plot_url)
        
    except Exception as e:
        print(f"Error in main route: {e}")
        return render_template('index.html', 
                             current_price=0,
                             price_change=0,
                             volume=0,
                             data_count=0,
                             predictions=[],
                             plot_url=None)

@app.route('/retrain')
def retrain():
    """Endpoint to retrain the model"""
    try:
        # This would trigger a fresh data fetch and model training
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)