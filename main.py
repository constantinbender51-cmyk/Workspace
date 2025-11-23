import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

def fetch_btc_data():
    """Fetch Bitcoin price data from Binance API"""
    print("DEBUG: Fetching BTC data from Binance")
    end_time = int(datetime(2025, 11, 30).timestamp() * 1000)
    start_time = int(datetime(2022, 1, 1).timestamp() * 1000)
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start_time}&endTime={end_time}"
    
    response = requests.get(url)
    data = response.json()
    
    if not data:
        raise ValueError("No data fetched from Binance")
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['date', 'close', 'volume']].reset_index(drop=True)
    
    print(f"DEBUG: Fetched {len(df)} days of data")
    return df

def create_features_and_targets(df):
    """
    Create features and targets for prediction.
    
    For each day, we use:
    - Last 3 days of SMA_7 (price moving average)
    - Last 3 days of SMA_365 (price moving average)
    - Last 3 days of SMA_volume_5 (volume moving average)
    - Last 3 days of SMA_volume_10 (volume moving average)
    
    To predict: Next day's close price
    """
    print("DEBUG: Creating features and targets")
    
    # Calculate moving averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    df['sma_volume_5'] = df['volume'].rolling(window=5).mean()
    df['sma_volume_10'] = df['volume'].rolling(window=10).mean()
    
    # Drop rows where we don't have all the moving averages yet
    df = df.dropna().reset_index(drop=True)
    print(f"DEBUG: After calculating moving averages, have {len(df)} days")
    
    # Now create our feature matrix and target vector
    X = []  # Features
    y = []  # Targets (next day close price)
    dates = []  # Keep track of dates for plotting
    
    # We need 3 days of history for features, plus 1 day ahead for target
    # So we start at index 3 and go until len(df)-1
    for i in range(3, len(df)):
        # Features: last 3 days of each indicator
        features = [
            # Day i-3 indicators
            df.loc[i-3, 'sma_7'],
            df.loc[i-3, 'sma_365'],
            df.loc[i-3, 'sma_volume_5'],
            df.loc[i-3, 'sma_volume_10'],
            # Day i-2 indicators
            df.loc[i-2, 'sma_7'],
            df.loc[i-2, 'sma_365'],
            df.loc[i-2, 'sma_volume_5'],
            df.loc[i-2, 'sma_volume_10'],
            # Day i-1 indicators
            df.loc[i-1, 'sma_7'],
            df.loc[i-1, 'sma_365'],
            df.loc[i-1, 'sma_volume_5'],
            df.loc[i-1, 'sma_volume_10'],
        ]
        
        # Target: next day's close price (day i)
        target = df.loc[i, 'close']
        
        X.append(features)
        y.append(target)
        dates.append(df.loc[i, 'date'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"DEBUG: Created {len(X)} samples")
    print(f"DEBUG: Feature shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y, dates, df

def train_test_split_data(X, y, dates, train_ratio=0.5):
    """Split data into train and test sets"""
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    dates_test = dates[split_idx:]
    
    print(f"DEBUG: Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test, dates_test

def train_model(X_train, y_train):
    """Train linear regression model"""
    print("DEBUG: Training model")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("DEBUG: Model trained")
    return model

def backtest_strategy(model, X_test, y_test, start_capital=1000, transaction_cost=0.001):
    """
    Backtest trading strategy:
    - If current price > predicted price: GO LONG (buy)
    - If current price < predicted price: GO SHORT (sell)
    """
    print("DEBUG: Running backtest")
    
    capital = start_capital
    capital_history = [capital]
    positions = []
    
    predictions = model.predict(X_test)
    
    for i in range(len(predictions)):
        predicted_price = predictions[i]
        actual_current_price = y_test[i]
        
        # Get next day's price for calculating return
        if i < len(y_test) - 1:
            next_day_price = y_test[i + 1]
        else:
            # Last day, use same price (no return)
            next_day_price = actual_current_price
        
        # Trading decision
        if actual_current_price > predicted_price:
            # GO LONG: we think price will rise
            return_pct = (next_day_price / actual_current_price) - 1
            positions.append('long')
        else:
            # GO SHORT: we think price will fall
            return_pct = (actual_current_price / next_day_price) - 1
            positions.append('short')
        
        # Apply return and transaction cost
        capital = capital * (1 + return_pct - transaction_cost)
        capital_history.append(capital)
    
    print(f"DEBUG: Final capital: ${capital:.2f}")
    print(f"DEBUG: Total return: {((capital/start_capital - 1) * 100):.2f}%")
    
    return capital_history, positions, predictions

def create_plots(capital_history, positions, dates_test, y_test, predictions):
    """Create visualization plots"""
    print("DEBUG: Creating plots")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot 1: Capital over time
    ax1.plot(range(len(capital_history)), capital_history, color='black', linewidth=2)
    
    # Color background based on position
    for i in range(len(positions)):
        if positions[i] == 'long':
            ax1.axvspan(i, i+1, color='green', alpha=0.2)
        else:
            ax1.axvspan(i, i+1, color='red', alpha=0.2)
    
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Portfolio Value', 'Long Position', 'Short Position'])
    
    # Plot 2: Bitcoin price over time
    ax2.plot(dates_test, y_test, color='blue', linewidth=1.5)
    ax2.set_title('Bitcoin Price (Test Period)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USDT)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Predicted vs Actual
    ax3.plot(dates_test, y_test, label='Actual Price', color='blue', linewidth=1.5, alpha=0.7)
    ax3.plot(dates_test, predictions, label='Predicted Price', color='red', linewidth=1.5, alpha=0.7)
    ax3.set_title('Model Predictions vs Actual Price', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price (USDT)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    try:
        # Step 1: Fetch data
        df = fetch_btc_data()
        
        # Step 2: Create features and targets
        X, y, dates, df_with_features = create_features_and_targets(df)
        
        # Step 3: Split into train and test
        X_train, y_train, X_test, y_test, dates_test = train_test_split_data(X, y, dates, train_ratio=0.5)
        
        # Step 4: Train model
        model = train_model(X_train, y_train)
        
        # Step 5: Backtest strategy
        capital_history, positions, predictions = backtest_strategy(
            model, X_test, y_test, start_capital=1000, transaction_cost=0.001
        )
        
        # Step 6: Create plots
        plot_url = create_plots(capital_history, positions, dates_test, y_test, predictions)
        
        print("DEBUG: Successfully completed all steps")
        
        return render_template('index.html', plot_url=plot_url)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
