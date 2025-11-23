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
    print("DEBUG: Starting to fetch BTC data from Binance")
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
    # Calculate features without lookahead bias - use shift to ensure no future data
    print("DEBUG: Calculating features for the dataset")
    df['sma_5'] = df['close'].rolling(window=5).mean().shift(1)
    df['sma_365'] = df['close'].rolling(window=365).mean().shift(1)
    df['sma_volume_5'] = df['volume'].rolling(window=5).mean().shift(1)
    df['sma_volume_10'] = df['volume'].rolling(window=10).mean().shift(1)
    return df

def prepare_data(df):
    df = calculate_features(df)
    df = df.dropna()  # Remove rows with NaN values from rolling averages
    print(f"DEBUG: Data shape after calculate_features: {df.shape}")
    
    # Check if we have enough data after dropping NaN; minimum 8 days for 5-day lookback + 3-day future target
    print(f"DEBUG: Data after dropping NaN has {len(df)} rows")
    if len(df) < 8:
        raise ValueError(f"Insufficient data after processing. Have {len(df)} days, need at least 8. Try fetching more data.")
    
    # Create features with 5-day lookback for predicting the close 3 days in the future
    features = []
    targets = []
    
    print("DEBUG: Preparing features and targets with 5-day lookback for 3-day future target")
    for i in range(5, len(df) - 3):  # Start from index 5 to have 5 days of lookback, stop at len(df)-3 to have a 3-day future target
        # Use technical indicators from the past 5 days (i-5 to i-1) to predict close on day i+3
        feature_row = [
            df.iloc[i-5]['sma_5'],       # SMA 5 from 5 days ago
            df.iloc[i-4]['sma_5'],       # SMA 5 from 4 days ago
            df.iloc[i-3]['sma_5'],       # SMA 5 from 3 days ago
            df.iloc[i-2]['sma_5'],       # SMA 5 from 2 days ago
            df.iloc[i-1]['sma_5'],       # SMA 5 from 1 day ago
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
        targets.append(df.iloc[i+3]['close'])  # Target is close on day i+3 (3 days in the future)
    
    print(f"DEBUG: Prepared {len(features)} feature samples and {len(targets)} target samples")
    print(f"DEBUG: Features shape: {np.array(features).shape}, Targets shape: {np.array(targets).shape}")
    return np.array(features), np.array(targets), df

def train_model(features, targets):
    # Check if features and targets are not empty
    print("DEBUG: Starting model training with 50/50 train-test split")
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

def trading_strategy(df, model, X_test, test_start_idx, start_capital=1000, transaction_cost=0.001):
    capital = start_capital
    capital_history = [capital]
    positions = []  # Track positions for plotting
    
    print("DEBUG: Executing trading strategy on test set")
    
    # test_start_idx is the index in the features array where test set begins
    # Map this to df index: features start at index 3 in df, so add 3
    for i in range(len(X_test)):
        prediction = model.predict([X_test[i]])[0]
        
        # Current df index for this test sample
        df_idx = test_start_idx + i + 3
        
        if df_idx >= len(df) - 3:
            break
            
        current_price = df.iloc[df_idx]['close']
        future_price = df.iloc[df_idx + 3]['close']
        
        # Trading logic: 
        # If current price > prediction: go LONG (expecting mean reversion down)
        # If current price < prediction: go SHORT (expecting mean reversion up)
        if current_price > prediction:
            # Long position: profit if price goes up
            position_return = (future_price / current_price)
            positions.append('long')
        else:
            # Short position: profit if price goes down
            position_return = (current_price / future_price)
            positions.append('short')
        
        # Apply transaction cost and update capital
        capital = capital * position_return * (1 - transaction_cost)
        capital_history.append(capital)
    
    return capital_history, positions

def create_plot(capital_history, df, predictions, test_start_idx, positions, X_test):
    # Create a figure with subplots
    print("DEBUG: Generating plot for visualization")
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
        # Fetch and prepare data
        print("DEBUG: Starting index route execution")
        df = fetch_btc_data()
        features, targets, df = prepare_data(df)
        
        # Train model once with 50% split
        model, X_test, y_test = train_model(features, targets)
        
        # Generate predictions for test set
        predictions = model.predict(X_test)
        
        # Apply trading strategy on test set
        start_capital = 1000
        transaction_cost = 0.001
        test_start_idx = int(len(features) * 0.5)  # Start of test set in features array (50% split)
        capital_history, positions = trading_strategy(df, model, X_test, test_start_idx, start_capital, transaction_cost)
        
        # Create plot
        plot_url = create_plot(capital_history, df, predictions, test_start_idx, positions, X_test)
        
        print("DEBUG: Successfully generated and rendered plot")
        print(f"DEBUG: Rendered template with plot URL of length: {len(plot_url)}")
        
        return render_template('index.html', plot_url=plot_url)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
