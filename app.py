from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import subprocess

app = Flask(__name__)

# Load and preprocess data
def load_data():
    if not os.path.exists('btc_data.csv'):
        # Run the data fetching script if file doesn't exist
        subprocess.run(['python', 'fetch_price_data.py'], check=True)
    df = pd.read_csv('btc_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Prepare features and target
def prepare_data(df):
    # Calculate specified SMA
    df['sma_7'] = df['close'].rolling(window=7).mean()
    
    # Remove rows with NaN values from SMA calculation
    df_clean = df.dropna()
    
    features = []
    targets = []
    for i in range(len(df_clean)):
        # Features: 7-day price SMA
        feature = [
            df_clean['sma_7'].iloc[i]
        ]
        features.append(feature)
        # Target: next day's closing price
        if i < len(df_clean) - 3:  # Reduced lookback to 3 days
            target = df_clean['close'].iloc[i + 3]
            targets.append(target)
    
    # Remove the last 3 features since they have no corresponding target
    features = features[:-3]
    return np.array(features), np.array(targets)

# Train model
def train_model(features, targets):
    # Use time series split: first 50% for training, last 50% for testing
    split_idx = int(len(features) * 0.5)
    X_train = features[:split_idx]
    X_test = features[split_idx:]
    y_train = targets[:split_idx]
    y_test = targets[split_idx:]
    # Test indices correspond to the indices in the cleaned DataFrame for the test set
    test_indices = list(range(split_idx, split_idx + len(y_test)))
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, X_test, y_test, predictions, mse, test_indices

# Generate plot
def create_plot(df, y_test, predictions, test_indices):
    plt.figure(figsize=(10, 8))
    dates = df.index[test_indices]
    # Sort by date to ensure chronological plotting
    sorted_indices = np.argsort(dates)
    sorted_dates = dates[sorted_indices]
    sorted_y_test = y_test[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    
    # Calculate capital with daily accumulation based on prediction from 4 days ago vs actual price yesterday
    capital = [1000]  # Start with $1000
    positions = []  # Store position type for coloring
    
    for i in range(len(sorted_y_test)):
        # Calculate return using the previous two days' prices: yesterday and the day before
        if i >= 2:  # Ensure there are at least two previous days
            price_yesterday = sorted_y_test[i - 1]
            price_day_before = sorted_y_test[i - 2]
            return_calc = (sorted_y_test[i] - price_yesterday) / price_yesterday
        else:
            # For the first two days in test set, use available data; skip if not enough history
            if i == 0 and test_indices[sorted_indices[i]] >= 2:
                price_yesterday = df['close'].iloc[test_indices[sorted_indices[i]] - 1]
                price_day_before = df['close'].iloc[test_indices[sorted_indices[i]] - 2]
                return_calc = (sorted_y_test[i] - price_yesterday) / price_yesterday
            elif i == 1 and test_indices[sorted_indices[i]] >= 2:
                price_yesterday = sorted_y_test[i - 1]
                price_day_before = df['close'].iloc[test_indices[sorted_indices[i]] - 2]
                return_calc = (sorted_y_test[i] - price_yesterday) / price_yesterday
            else:
                return_calc = 0  # Default to no return if insufficient data
        
        # ML Strategy: If predicted price from 4 days ago is lower than actual price yesterday, apply positive return, else negative
        if i >= 1:  # Ensure prediction from 4 days ago is available
            pred_index = test_indices[sorted_indices[i]] - 4  # Prediction from 4 days ago
            if pred_index >= 0 and pred_index < len(df):
                # Find if pred_index is in test_indices; use prediction if available, else actual price
                pred_in_test = np.where(test_indices == pred_index)[0]
                if pred_in_test.size > 0:
                    pred_price_4_days_ago = sorted_predictions[pred_in_test[0]]
                else:
                    pred_price_4_days_ago = df['close'].iloc[pred_index]
                actual_price_yesterday = sorted_y_test[i - 1]
                if pred_price_4_days_ago < actual_price_yesterday:
                    ret = return_calc  # Positive signal: long position
                    positions.append('long')  # Mark as long
                else:
                    ret = -return_calc  # Negative signal: short position
                    positions.append('short')  # Mark as short
            else:
                ret = 0  # Default if prediction not available
                positions.append('neutral')  # Mark as neutral
        else:
            ret = 0  # Default for first day
            positions.append('neutral')  # Mark as neutral
        
        capital.append(capital[-1] * (1 + ret))
    
    capital = capital[1:]  # Remove the initial 1000 to match the number of dates
    
    # Plot price and predictions with position markers
    plt.subplot(2, 1, 1)
    plt.plot(sorted_dates, sorted_y_test, label='Actual Price', color='blue')
    plt.plot(sorted_dates, sorted_predictions, label='Predicted Price', color='red')
    # Add markers for long and short positions on the prediction line
    for j in range(len(sorted_dates)):
        if positions[j] == 'long':
            plt.scatter(sorted_dates[j], sorted_predictions[j], color='green', marker='^', s=50, zorder=5)
        elif positions[j] == 'short':
            plt.scatter(sorted_dates[j], sorted_predictions[j], color='red', marker='v', s=50, zorder=5)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('BTC Price Prediction vs Actual')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot capital with color based on position
    plt.subplot(2, 1, 2)
    # Create segments for coloring based on positions
    prev_idx = 0
    for k in range(1, len(capital)):
        if positions[k-1] == 'long':
            color = 'green'
        elif positions[k-1] == 'short':
            color = 'red'
        else:
            color = 'gray'  # Neutral in gray
        plt.plot(sorted_dates[prev_idx:k+1], capital[prev_idx:k+1], color=color, linewidth=2)
        prev_idx = k
    plt.xlabel('Date')
    plt.ylabel('Capital (USD)')
    plt.title('Trading Strategy Capital (Green: Long, Red: Short)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    df = load_data()
    features, targets = prepare_data(df)
    model, X_test, y_test, predictions, mse, test_indices = train_model(features, targets)
    plot_url = create_plot(df, y_test, predictions, test_indices)
    return render_template('index.html', plot_url=plot_url, mse=mse)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
