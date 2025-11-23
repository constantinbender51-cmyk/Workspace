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
    # Fetch BTC daily price data from Binance starting from 2022-01-01 to ensure enough data for proper train/test split    print("DEBUG: Starting to fetch BTC data from Binance")
    end_time = int(datetime(2025, 11, 30).timestamp() * 1000)
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
    # Calculate features without lookahead bias - use shift to ensure no future data    print("DEBUG: Calculating features for the dataset")
    df['sma_7'] = df['close'].rolling(window=7).mean().shift(1)
    df['sma_365'] = df['close'].rolling(window=365).mean().shift(1)
    df['sma_volume_5'] = df['volume'].rolling(window=5).mean().shift(1)
    df['sma_volume_10'] = df['volume'].rolling(window=10).mean().shift(1)
    return df

def prepare_data(df):
    df = calculate_features(df)
    df = df.dropna()  # Remove rows with NaN values from rolling averages
    print(f"DEBUG: Data shape after calculate_features: {df.shape}")
    
    # Check if we have enough data after dropping NaN; minimum 4 days for 3-day lookback + target
    print(f"DEBUG: Data after dropping NaN has {len(df)} rows")
    if len(df) < 4:
        raise ValueError(f"Insufficient data after processing. Have {len(df)} days, need at least 4. Try fetching more data.")
    
    # Create features with 3-day lookback for predicting the next day's close
    features = []
    targets = []
    print("DEBUG: Preparing features and targets with 3-day lookback")
    for i in range(3, len(df) - 1):  # Start from index 3 to have 3 days of lookback, stop at len(df)-1 to have a target
        # Use technical indicators from the past 3 days (i-3 to i-1) to predict close on day i
        feature_row = [
            df.iloc[i-3]['sma_7'],       # SMA 7 from 3 days ago
            df.iloc[i-2]['sma_7'],       # SMA 7 from 2 days ago
            df.iloc[i-1]['sma_7'],       # SMA 7 from 1 day ago
            df.iloc[i-3]['sma_365'],     # SMA 365 from 3 days ago
            df.iloc[i-2]['sma_365'],     # SMA 365 from 2 days ago
            df.iloc[i-1]['sma_365'],     # SMA 365 from 1 day ago
            df.iloc[i-3]['sma_volume_5'], # SMA volume 5 from 3 days ago
            df.iloc[i-2]['sma_volume_5'], # SMA volume 5 from 2 days ago
            df.iloc[i-1]['sma_volume_5'], # SMA volume 5 from 1 day ago
            df.iloc[i-3]['sma_volume_10'], # SMA volume 10 from 3 days ago
            df.iloc[i-2]['sma_volume_10'], # SMA volume 10 from 2 days ago
            df.iloc[i-1]['sma_volume_10']  # SMA volume 10 from 1 day ago
        ]
        features.append(feature_row)
        targets.append(df.iloc[i+1]['close'])  # Target is close on day i+1 (true next day prediction)
    
    print(f"DEBUG: Prepared {len(features)} feature samples and {len(targets)} target samples")
    print(f"DEBUG: Features shape: {np.array(features).shape}, Targets shape: {np.array(targets).shape}")
    return np.array(features), np.array(targets), df

def train_model(features, targets):
    # Check if features and targets are not empty    print("DEBUG: Starting model training with 50/50 train-test split")
    if len(features) == 0 or len(targets) == 0:
        raise ValueError("No features or targets available for training. Ensure sufficient data.")
    # Split data 50% for training, 50% for testing
    split_index = int(len(features) * 0.5)
    if split_index == 0:
        raise ValueError("Insufficient data for 50/50 train-test split. Need at least 2 samples.")
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def trading_strategy(df, model, X_test, start_capital=1000, transaction_cost=0.001):
    capital = start_capital
    capital_history = [capital]
    positions = []  # Track positions for plotting    print("DEBUG: Executing trading strategy on test set")
    
    # X_test corresponds to features for the test set; indices in df need adjustment
    # Features were built for indices 3 to len(df)-2, so test set starts after train split
    total_features_start = 3  # Features start from index 3 in df
    for i in range(len(X_test)):
        prediction = model.predict([X_test[i]])[0]
        # Map test index back to df index: features index i corresponds to df index i + total_features_start
        df_idx = i + total_features_start
        if df_idx >= len(df) - 1:
            break  # Avoid index out of bounds
        actual_close_next = df.iloc[df_idx + 1]['close']  # Actual close on the next day (target of prediction)
        
        # Always take a long position
        # Simple long return: capital grows based on current price divided by previous day's price
        current_price = df.iloc[df_idx]['close']
        previous_price = df.iloc[df_idx - 1]['close']
        capital = capital * (current_price / previous_price)
        positions.append('long')
        
        capital_history.append(capital)
    
    return capital_history, positions

def create_plot(capital_history, df, predictions, test_start_idx, positions):
    # Create a figure with subplots    print("DEBUG: Generating plot for visualization")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot capital development with colors for long (green) and short (red) periods
    ax1.plot(range(len(capital_history)), capital_history, color='black', linewidth=1)
    for i in range(len(positions)):
        if positions[i] == 'long':
            ax1.axvspan(i, i+1, color='green', alpha=0.3)
        elif positions[i] == 'short':
            ax1.axvspan(i, i+1, color='red', alpha=0.3)
    ax1.set_title('Capital Development Over Time (Green: Long, Red: Short)')
    ax1.set_xlabel('Trading Day in Test Set')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True)
    
    # Plot price over time for the test period
    # test_start_idx in features corresponds to df index: test_start_idx + 3 (since features start from index 3)
    start_df_idx = test_start_idx + 3
    end_df_idx = start_df_idx + len(X_test)  # Use full test set length
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
        # Fetch and prepare data        print("DEBUG: Starting index route execution")
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
        test_start_idx = int(len(features) * 0.5)  # Start of test set in features array (50% split)
        plot_url = create_plot(capital_history, df, predictions, test_start_idx, positions)
        
        return render_template('index.html', plot_url=plot_url)
    except Exception as e:        print("DEBUG: Successfully generated and rendered plot")
        print(f"DEBUG: Rendered template with plot URL of length: {len(plot_url)}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)