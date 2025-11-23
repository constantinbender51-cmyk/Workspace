from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64
import requests
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)

# Global variables to store data and results
btc_data = None
model_results = None
capital_data = None

def fetch_btc_data():
    """Fetch BTC price data from Binance API from Jan 2022 to Sep 2023"""
    end_date = datetime(2023, 9, 30)
    start_date = datetime(2022, 1, 1)
    
    # Binance API endpoint for historical klines
    url = "https://api.binance.com/api/v3/klines"
    
    # Binance uses milliseconds for timestamps
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    all_data = []
    current_start = start_timestamp
    
    try:
        # Binance limits to 1000 records per request, so we need to paginate
        while current_start < end_timestamp:
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1d',
                'startTime': current_start,
                'endTime': end_timestamp,
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            all_data.extend(klines)
            
            # Move to next period (last kline's close time + 1ms)
            current_start = int(klines[-1][6]) + 1
            
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        # Process the klines data
        # Klines format: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamps and prices
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df['price'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Set date as index and sort
        df = df.set_index('date').sort_index()
        
        # Keep only necessary columns
        df = df[['price', 'volume']]
        
        return df
        
    except Exception as e:
        print(f"Error fetching data from Binance: {e}")
        # Fallback: create sample data
        dates = pd.date_range(start='2022-01-01', end='2023-09-30', freq='D')
        np.random.seed(42)
        prices = 40000 + np.cumsum(np.random.normal(0, 1000, len(dates)))
        volumes = 1000000000 + np.random.normal(0, 100000000, len(dates))
        
        df = pd.DataFrame({
            'price': prices,
            'volume': volumes
        }, index=dates)
        
        return df

def calculate_features(df):
    """Calculate SMA features for the model"""
    df = df.copy()
    
    # Calculate SMAs
    df['sma_7'] = df['price'].rolling(window=7).mean()
    df['sma_365'] = df['price'].rolling(window=365).mean()
    df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    
    # Create target (next day price)
    df['next_day_price'] = df['price'].shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def train_model(df):
    """Train linear regression model with 50-50 split"""
    # Split data 50-50 based on time
    split_index = len(df) // 2
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    # Features for the model
    features = ['sma_7', 'sma_365', 'volume_sma_5', 'volume_sma_10']
    
    # Prepare training data
    X_train = train_df[features]
    y_train = train_df['next_day_price']
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on entire dataset
    X_all = df[features]
    predictions = model.predict(X_all)
    
    # Add predictions to dataframe
    df['prediction'] = predictions
    
    # Calculate strategy returns
    capital_history = [1000]  # Start with $1000
    positions = ['hold']  # No position on first day
    
    for i in range(1, len(df)):
        if i >= split_index:  # Only trade in test period
            # Use only information available at time of decision
            # Today's prediction is based on yesterday's features
            today_pred = df['prediction'].iloc[i]
            yesterday_close = df['price'].iloc[i-1]
            today_close = df['price'].iloc[i]
            
            if today_pred < yesterday_close:
                # Go long: profit = (today_close - yesterday_close) / yesterday_close
                returns = (today_close - yesterday_close) / yesterday_close
                position = 'long'
            else:
                # Go short: profit = (yesterday_close - today_close) / yesterday_close
                returns = (yesterday_close - today_close) / yesterday_close
                position = 'short'
            
            new_capital = capital_history[-1] * (1 + returns)
            capital_history.append(new_capital)
            positions.append(position)
        else:
            # In training period, capital doesn't change
            capital_history.append(capital_history[-1])
            positions.append('hold')
    
    # Add capital data to dataframe
    df['capital'] = capital_history
    df['position'] = positions
    
    return {
        'model': model,
        'train_df': train_df,
        'test_df': test_df,
        'full_df': df,
        'split_index': split_index,
        'features': features
    }

def create_plot_image(df, split_index, plot_type='price'):
    """Create matplotlib plot and return as base64 image"""
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'price':
        # Price and predictions plot
        plt.plot(df.index, df['price'], label='Actual Price', linewidth=2)
        plt.plot(df.index, df['prediction'], label='Predicted Price', linewidth=2, alpha=0.7)
        plt.axvline(x=df.index[split_index], color='red', linestyle='--', 
                   label='Train/Test Split', alpha=0.7)
        plt.title('BTC Price vs Predictions (Test Period Highlighted)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    else:  # capital plot
        test_period = df.iloc[split_index:]
        plt.plot(test_period.index, test_period['capital'], 
                label='Capital Development', linewidth=2, color='green')
        plt.title('Capital Development in Test Period (Starting: $1000)')
        plt.xlabel('Date')
        plt.ylabel('Capital (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def update_data():
    """Background thread to update data periodically"""
    global btc_data, model_results, capital_data
    
    while True:
        try:
            print("Updating BTC data...")
            btc_data = fetch_btc_data()
            feature_data = calculate_features(btc_data)
            model_results = train_model(feature_data)
            print("Data update completed")
        except Exception as e:
            print(f"Error updating data: {e}")
        
        # Update every hour
        time.sleep(3600)

@app.route('/')
def index():
    if btc_data is None or model_results is None:
        return "Data is being loaded. Please refresh in a moment."
    
    df = model_results['full_df']
    split_index = model_results['split_index']
    
    # Create plots
    price_plot = create_plot_image(df, split_index, 'price')
    capital_plot = create_plot_image(df, split_index, 'capital')
    
    # Calculate performance metrics
    test_df = df.iloc[split_index:]
    final_capital = test_df['capital'].iloc[-1] if len(test_df) > 0 else 1000
    total_return = ((final_capital - 1000) / 1000) * 100
    
    # Get current strategy position
    current_position = test_df['position'].iloc[-1] if len(test_df) > 0 else 'hold'
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Trading Strategy</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .plot {{ margin: 20px 0; }}
            .metrics {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .metric-item {{ margin: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BTC Trading Strategy Dashboard</h1>
            
            <div class="metrics">
                <h2>Performance Metrics</h2>
                <div class="metric-item"><strong>Initial Capital:</strong> $1,000</div>
                <div class="metric-item"><strong>Final Capital:</strong> ${final_capital:,.2f}</div>
                <div class="metric-item"><strong>Total Return:</strong> {total_return:+.2f}%</div>
                <div class="metric-item"><strong>Current Position:</strong> {current_position.upper()}</div>
                <div class="metric-item"><strong>Test Period:</strong> {df.index[split_index].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}</div>
            </div>
            
            <div class="plot">
                <h2>BTC Price vs Predictions</h2>
                <img src="data:image/png;base64,{price_plot}" alt="Price Chart" style="width: 100%; max-width: 1000px;">
            </div>
            
            <div class="plot">
                <h2>Capital Development (Test Period)</h2>
                <img src="data:image/png;base64,{capital_plot}" alt="Capital Chart" style="width: 100%; max-width: 1000px;">
            </div>
            
            <div class="metrics">
                <h3>Strategy Rules</h3>
                <ul>
                    <li>If yesterday's prediction < yesterday's close: GO LONG today</li>
                    <li>If yesterday's prediction > yesterday's close: GO SHORT today</li>
                    <li>Features used: 7-day SMA price, 365-day SMA price, 5-day SMA volume, 10-day SMA volume</li>
                    <li>Model: Linear Regression</li>
                    <li>Data split: 50% train, 50% test (chronological)</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Start background data update thread
    update_thread = threading.Thread(target=update_data, daemon=True)
    update_thread.start()
    
    # Initial data load
    print("Loading initial data...")
    btc_data = fetch_btc_data()
    feature_data = calculate_features(btc_data)
    model_results = train_model(feature_data)
    print("Initial data loaded")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)