#!/usr/bin/env python3

import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, render_template, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize Binance client (no API key needed for public data)
client = Client()

# Global variables to store model and data
model = None
X_scaler = None
y_scaler = None
latest_data = None
predictions = None

def fetch_btc_data():
    """Fetch 2,000 BTC candles from Binance"""
    print("Fetching BTC data from Binance...")
    
    # Fetch 1-hour candles (2000 candles = ~83 days of data)
    klines = client.get_klines(
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
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Calculate additional features
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['price_ma_5'] = df['close'].rolling(window=5).mean()
    df['price_ma_20'] = df['close'].rolling(window=20).mean()
    df['volatility'] = df['high'] - df['low']
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    # Features for prediction
    features = ['open', 'high', 'low', 'volume', 'price_change', 'volume_change', 
               'high_low_ratio', 'price_ma_5', 'price_ma_20', 'volatility']
    
    # Target: next candle's close price
    df['target'] = df['close'].shift(-1)
    df = df.dropna()
    
    X = df[features]
    y = df['target']
    
    return X, y, df

def train_model():
    """Train linear regression model"""
    global model, latest_data
    
    # Fetch and prepare data
    df = fetch_btc_data()
    X, y, df = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Store latest data for display
    latest_data = df.iloc[-100:]  # Last 100 data points for display
    
    return model, X_test, y_test, y_pred, mse, r2

def make_future_predictions():
    """Make predictions for future prices"""
    global model, predictions
    
    if model is None:
        return None
    
    # Get latest data
    df = fetch_btc_data()
    X, y, df = prepare_features(df)
    
    # Use the last data point for prediction
    latest_features = X.iloc[-1:].values
    
    # Make predictions for next 10 periods
    future_predictions = []
    current_features = latest_features.copy()
    
    for i in range(10):
        pred_price = model.predict(current_features)[0]
        future_predictions.append(pred_price)
        
        # Update features for next prediction (simplified approach)
        # In a real scenario, you'd need to update all features properly
        current_features[0][0] = pred_price  # Update open price
        current_features[0][1] = pred_price * 1.001  # Estimate high
        current_features[0][2] = pred_price * 0.999  # Estimate low
    
    predictions = future_predictions
    return future_predictions

def create_plot():
    """Create matplotlib plot for web display"""
    if latest_data is None or predictions is None:
        return None
    
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(latest_data['timestamp'], latest_data['close'], 
             label='Historical Prices', linewidth=2, color='blue')
    
    # Plot predictions
    future_times = [latest_data['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(len(predictions))]
    plt.plot(future_times, predictions, 
             label='Predictions', linewidth=2, color='red', linestyle='--', marker='o')
    
    plt.title('BTC/USDT Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 for web display
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """Main page with data and predictions"""
    global model, latest_data, predictions
    
    if model is None:
        # Train model on first access
        train_model()
        make_future_predictions()
    
    plot_url = create_plot()
    
    if latest_data is not None:
        current_price = latest_data['close'].iloc[-1]
        price_change = latest_data['price_change'].iloc[-1] * 100
        volume = latest_data['volume'].iloc[-1]
    else:
        current_price = 0
        price_change = 0
        volume = 0
    
    return render_template('index.html',
                         plot_url=plot_url,
                         current_price=current_price,
                         price_change=price_change,
                         volume=volume,
                         predictions=predictions,
                         data_count=len(latest_data) if latest_data else 0)

@app.route('/api/data')
def api_data():
    """API endpoint for JSON data"""
    global latest_data, predictions
    
    if latest_data is None:
        return jsonify({'error': 'No data available'})
    
    data = {
        'current_price': float(latest_data['close'].iloc[-1]),
        'predictions': [float(p) for p in predictions] if predictions else [],
        'last_updated': datetime.now().isoformat(),
        'data_points': len(latest_data)
    }
    
    return jsonify(data)

@app.route('/retrain')
def retrain():
    """Retrain the model with fresh data"""
    train_model()
    make_future_predictions()
    return jsonify({'status': 'Model retrained successfully'})

if __name__ == '__main__':
    print("Starting BTC Price Prediction Service...")
    print("Training initial model...")
    train_model()
    make_future_predictions()
    print("Starting web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)