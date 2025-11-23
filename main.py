import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template
from datetime import datetime, timedelta

app = Flask(__name__)

def fetch_btc_data():
    # Fetch BTC daily price data from Binance starting from 2025-01-01
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int(datetime(2025, 1, 1).timestamp() * 1000)
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start_time}&endTime={end_time}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['date', 'close', 'volume']]
    return df

def calculate_features(df):
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    df['sma_volume_5'] = df['volume'].rolling(window=5).mean()
    df['sma_volume_10'] = df['volume'].rolling(window=10).mean()
    return df

def prepare_data(df):
    df = calculate_features(df)
    df = df.dropna()  # Remove rows with NaN values from rolling averages
    
    # Create features with 5-day lookback
    features = []
    targets = []
    for i in range(5, len(df)):
        row = df.iloc[i]
        lookback = df.iloc[i-5:i]
        feature_row = [
            lookback['close'].iloc[-1],  # Latest close in lookback
            row['sma_7'],
            row['sma_365'],
            row['sma_volume_5'],
            row['sma_volume_10']
        ]
        features.append(feature_row)
        targets.append(df.iloc[i]['close'])  # Predict next day's close (i.e., current row's close)
    
    return np.array(features), np.array(targets), df

def train_model(features, targets):
    # Split data 50% for training, 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.5, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def trading_strategy(df, model, features, start_capital=1000, transaction_cost=0.001):
    capital = start_capital
    capital_history = [capital]
    positions = []  # Track positions for plotting
    
    # Use test set indices for trading
    test_start_idx = len(features) // 2  # 50% split
    for i in range(test_start_idx, len(features)):
        prediction = model.predict([features[i]])[0]
        actual_close = df.iloc[i + 5]['close']  # Adjust index since features start from index 5
        yesterday_close = df.iloc[i + 4]['close'] if i + 4 < len(df) else df.iloc[i + 3]['close']  # Simplified for edge cases
        
        # Decision: buy if yesterday's close > prediction, sell if <
        if yesterday_close > prediction:
            # Buy: long position
            investment = capital
            capital_after_trade = investment * (1 - transaction_cost)
            capital = capital_after_trade * (actual_close / yesterday_close)  # Profit/loss
            positions.append('buy')
        else:
            # Sell: short position
            investment = capital
            capital_after_trade = investment * (1 - transaction_cost)
            capital = capital_after_trade * (yesterday_close / actual_close)  # Profit/loss for short
            positions.append('sell')
        
        capital_history.append(capital)
    
    return capital_history, positions

def create_plot(capital_history, df, predictions, test_start_idx):
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot capital development
    ax1.plot(range(len(capital_history)), capital_history)
    ax1.set_title('Capital Development Over Time')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True)
    
    # Plot price over time
    test_dates = df['date'].iloc[test_start_idx + 5: test_start_idx + 5 + len(capital_history)]  # Adjust for lookback
    ax2.plot(test_dates, df['close'].iloc[test_start_idx + 5: test_start_idx + 5 + len(capital_history)])
    ax2.set_title('BTC Price Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USDT)')
    ax2.grid(True)
    
    # Plot predictions vs actual
    actual_prices = df['close'].iloc[test_start_idx + 5: test_start_idx + 5 + len(predictions)]
    ax3.plot(test_dates[:len(predictions)], predictions, label='Predicted Price', color='red')
    ax3.plot(test_dates[:len(actual_prices)], actual_prices, label='Actual Price', color='blue')
    ax3.set_title('Predicted vs Actual Price')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price (USDT)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Convert plot to base64 string for HTML embedding
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    # Fetch and prepare data
    df = fetch_btc_data()
    features, targets, df = prepare_data(df)
    
    # Train model once
    model, X_test, y_test = train_model(features, targets)
    
    # Generate predictions for test set
    predictions = model.predict(X_test)
    
    # Apply trading strategy
    start_capital = 1000
    transaction_cost = 0.001
    test_start_idx = len(features) // 2
    capital_history, positions = trading_strategy(df, model, X_test, start_capital, transaction_cost)
    
    # Create plot
    plot_url = create_plot(capital_history, df, predictions, test_start_idx)
    
    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)