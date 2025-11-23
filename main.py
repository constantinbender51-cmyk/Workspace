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
    # Fetch BTC daily price data from Binance starting from 2022-01-01 to ensure enough data for proper train/test split
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int(datetime(2022, 1, 1).timestamp() * 1000)
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start_time}&endTime={end_time}"
    response = requests.get(url)
    data = response.json()
    if not data:
        raise ValueError("No data fetched from Binance. Check the date range or API availability.")
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['date', 'close', 'volume']]
    return df

def calculate_features(df):
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()  # Use full 365-day SMA as specified
    df['sma_volume_5'] = df['volume'].rolling(window=5).mean()
    df['sma_volume_10'] = df['volume'].rolling(window=10).mean()
    return df

def prepare_data(df):
    df = calculate_features(df)
    df = df.dropna()  # Remove rows with NaN values from rolling averages
    
    # Check if we have enough data after dropping NaN; minimum 6 days for 5-day lookback + target
    if len(df) < 6:
        raise ValueError(f"Insufficient data after processing. Have {len(df)} days, need at least 6. Try fetching more data.")
    
    # Create features with 5-day lookback for predicting the next day's close
    features = []
    targets = []
    for i in range(5, len(df) - 1):  # Start from index 5 to have 5 days of lookback, stop at len(df)-1 to have a target
        # Use technical indicators from the past 5 days (i-5 to i-1) to predict close on day i
        feature_row = [
            df.iloc[i-5]['sma_7'],       # SMA 7 from 5 days ago
            df.iloc[i-4]['sma_7'],       # SMA 7 from 4 days ago
            df.iloc[i-3]['sma_7'],       # SMA 7 from 3 days ago
            df.iloc[i-2]['sma_7'],       # SMA 7 from 2 days ago
            df.iloc[i-1]['sma_7'],       # SMA 7 from 1 day ago
            df.iloc[i-5]['sma_365'],     # SMA 365 from 5 days ago
            df.iloc[i-4]['sma_365'],     # SMA 365 from 4 days ago
            df.iloc[i-3]['sma_365'],     # SMA 365 from 3 days ago
            df.iloc[i-2]['sma_365'],     # SMA 365 from 2 days ago
            df.iloc[i-1]['sma_365'],     # SMA 365 from 1 day ago
            df.iloc[i-5]['sma_volume_5'], # SMA volume 5 from 5 days ago
            df.iloc[i-4]['sma_volume_5'], # SMA volume 5 from 4 days ago
            df.iloc[i-3]['sma_volume_5'], # SMA volume 5 from 3 days ago
            df.iloc[i-2]['sma_volume_5'], # SMA volume 5 from 2 days ago
            df.iloc[i-1]['sma_volume_5'], # SMA volume 5 from 1 day ago
            df.iloc[i-5]['sma_volume_10'], # SMA volume 10 from 5 days ago
            df.iloc[i-4]['sma_volume_10'], # SMA volume 10 from 4 days ago
            df.iloc[i-3]['sma_volume_10'], # SMA volume 10 from 3 days ago
            df.iloc[i-2]['sma_volume_10'], # SMA volume 10 from 2 days ago
            df.iloc[i-1]['sma_volume_10']  # SMA volume 10 from 1 day ago
        ]
        features.append(feature_row)
        targets.append(df.iloc[i]['close'])  # Target is close on day i (next day after the 5-day lookback)
    
    return np.array(features), np.array(targets), df

def train_model(features, targets):
    # Check if features and targets are not empty
    if len(features) == 0 or len(targets) == 0:
        raise ValueError("No features or targets available for training. Ensure sufficient data.")
    # Split data 70% for training, 30% for testing to get longer testing period
    split_index = int(len(features) * 0.7)
    if split_index == 0:
        raise ValueError("Insufficient data for 70/30 train-test split. Need at least 2 samples.")
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def trading_strategy(df, model, X_test, start_capital=1000, transaction_cost=0.001):
    capital = start_capital
    capital_history = [capital]
    positions = []  # Track positions for plotting
    
    # X_test corresponds to features for the test set; indices in df need adjustment
    # Features were built for indices 5 to len(df)-2, so test set starts after train split
    # With 70/30 split, test_start_idx in features array is at 70% of features
    total_features_start = 5  # Features start from index 5 in df
    for i in range(len(X_test)):
        prediction = model.predict([X_test[i]])[0]
        # Map test index back to df index: features index i corresponds to df index i + total_features_start
        df_idx = i + total_features_start
        if df_idx >= len(df) - 1:
            break  # Avoid index out of bounds
        yesterday_close = df.iloc[df_idx - 1]['close']  # Close on the day before prediction
        actual_close = df.iloc[df_idx]['close']  # Actual close on the predicted day
        
        # Decision: long if actual_close > prediction, short if <
        if actual_close > prediction:
            # Long position: expect price to increase
            investment = capital
            capital_after_trade = investment * (1 - transaction_cost)  # Apply transaction cost on entry
            # Profit/loss: (actual_close / yesterday_close) for long
            capital = capital_after_trade * (actual_close / yesterday_close)
            positions.append('long')
        else:
            # Short position: expect price to decrease
            investment = capital
            capital_after_trade = investment * (1 - transaction_cost)  # Apply transaction cost on entry
            # Profit/loss: (yesterday_close / actual_close) for short
            capital = capital_after_trade * (yesterday_close / actual_close)
            positions.append('short')
        
        capital_history.append(capital)
    
    return capital_history, positions

def create_plot(capital_history, df, predictions, test_start_idx):
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot capital development
    ax1.plot(range(len(capital_history)), capital_history)
    ax1.set_title('Capital Development Over Time')
    ax1.set_xlabel('Trading Day in Test Set')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True)
    
    # Plot price over time for the test period
    # test_start_idx in features corresponds to df index: test_start_idx + 5 (since features start from index 5)
    start_df_idx = test_start_idx + 5
    end_df_idx = start_df_idx + len(capital_history) - 1  # -1 because capital_history includes start
    if end_df_idx > len(df):
        end_df_idx = len(df)
    test_dates = df['date'].iloc[start_df_idx:end_df_idx]
    test_prices = df['close'].iloc[start_df_idx:end_df_idx]
    ax2.plot(test_dates, test_prices)
    ax2.set_title('BTC Price Over Time (Test Period)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USDT)')
    ax2.grid(True)
    
    # Plot predictions vs actual for the test set
    actual_prices = df['close'].iloc[start_df_idx:start_df_idx + len(predictions)]
    pred_dates = test_dates[:len(predictions)]
    ax3.plot(pred_dates, predictions, label='Predicted Price', color='red')
    ax3.plot(pred_dates, actual_prices, label='Actual Price', color='blue')
    ax3.set_title('Predicted vs Actual Price (Test Set)')
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
    try:
        # Fetch and prepare data
        df = fetch_btc_data()
        features, targets, df = prepare_data(df)
        
        # Train model once with 50% split
        model, X_test, y_test = train_model(features, targets)
        
        # Generate predictions for test set
        predictions = model.predict(X_test)
        
        # Apply trading strategy on test set
        start_capital = 1000
        transaction_cost = 0.001
        capital_history, positions = trading_strategy(df, model, X_test, start_capital, transaction_cost)
        
        # Create plot; adjust test_start_idx for plotting
        test_start_idx = int(len(features) * 0.7)  # Start of test set in features array (70% split)
        plot_url = create_plot(capital_history, df, predictions, test_start_idx)
        
        return render_template('index.html', plot_url=plot_url)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)