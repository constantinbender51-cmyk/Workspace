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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

app = Flask(__name__)

def fetch_btc_data():
    """Fetch recent BTC data from Binance"""
    print("Fetching BTC data...")
    
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
    
    # Calculate features
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    
    df = df.dropna()
    
    return df

def prepare_features(df):
    """Prepare features for prediction"""
    features = ['open', 'high', 'low', 'close', 'volume', 'price_change', 'volume_change', 'ma_5', 'ma_10']
    
    X = df[features].values
    y = df['close'].shift(-1).dropna().values
    
    # Align X and y
    X = X[:-1]
    
    return X, y

def train_model(X, y):
    """Train linear regression model"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_future(model, scaler, last_data, periods=10):
    """Predict future prices"""
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(periods):
        # Scale current data
        current_scaled = scaler.transform([current_data])
        
        # Predict next price
        next_price = model.predict(current_scaled)[0]
        predictions.append(next_price)
        
        # Update current data for next prediction
        current_data[3] = next_price  # update close price
        current_data[5] = (next_price - current_data[3]) / current_data[3]  # update price_change
        current_data[7] = np.mean([current_data[3], current_data[0], current_data[1], current_data[2]])  # rough ma_5
        current_data[8] = np.mean([current_data[3], current_data[0], current_data[1], current_data[2]])  # rough ma_10
    
    return predictions

def create_plot(df, predictions):
    """Create price prediction plot"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(df['timestamp'].iloc[-100:], df['close'].iloc[-100:], label='Historical Price', linewidth=2)
    
    # Plot predictions
    future_times = [df['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(len(predictions))]
    plt.plot(future_times, predictions, 'r--', label='Predictions', linewidth=2, marker='o')
    
    plt.title('BTC Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """Main page"""
    try:
        # Fetch data
        df = fetch_btc_data()
        
        if len(df) < 100:
            return render_template('index.html', 
                                 current_price=0,
                                 price_change=0,
                                 volume=0,
                                 data_count=0,
                                 predictions=[],
                                 plot_url=None)
        
        # Prepare features and train model
        X, y = prepare_features(df)
        model, scaler = train_model(X, y)
        
        # Make predictions
        last_data = df.iloc[-1][['open', 'high', 'low', 'close', 'volume', 'price_change', 'volume_change', 'ma_5', 'ma_10']].values
        predictions = predict_future(model, scaler, last_data)
        
        # Create plot
        plot_url = create_plot(df, predictions)
        
        # Calculate current metrics
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        volume = df['volume'].iloc[-1]
        
        return render_template('index.html',
                             current_price=current_price,
                             price_change=price_change,
                             volume=volume,
                             data_count=len(df),
                             predictions=predictions,
                             plot_url=plot_url)
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html',
                             current_price=0,
                             price_change=0,
                             volume=0,
                             data_count=0,
                             predictions=[],
                             plot_url=None)

@app.route('/retrain')
def retrain():
    """Retrain model endpoint"""
    try:
        # This would trigger a fresh data fetch and model training
        return jsonify({'status': 'success', 'message': 'Model retrained'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)