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
    # Calculate price SMAs
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate volume SMAs
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_sma_50'] = df['volume'].rolling(window=50).mean()
    
    # Remove rows with NaN values from SMA calculations
    df_clean = df.dropna()
    
    features = []
    targets = []
    for i in range(len(df_clean)):
        # Features: 20, 50, 100, and 200-day price SMAs + volume and 10, 20, 50-day volume SMAs
        feature = [
            df_clean['sma_20'].iloc[i],
            df_clean['sma_50'].iloc[i],
            df_clean['sma_100'].iloc[i],
            df_clean['sma_200'].iloc[i],
            df_clean['volume'].iloc[i],
            df_clean['volume_sma_10'].iloc[i],
            df_clean['volume_sma_20'].iloc[i],
            df_clean['volume_sma_50'].iloc[i]
        ]
        features.append(feature)
        # Target: next day's closing price
        if i < len(df_clean) - 1:
            target = df_clean['close'].iloc[i + 1]
            targets.append(target)
    
    # Remove the last feature since it has no corresponding target
    features = features[:-1]
    return np.array(features), np.array(targets)

# Train model
def train_model(features, targets):
    # Use time series split: first 80% for training, last 20% for testing
    split_idx = int(len(features) * 0.8)
    X_train = features[:split_idx]
    X_test = features[split_idx:]
    y_train = targets[:split_idx]
    y_test = targets[split_idx:]
    # Test indices start from split_idx + 200 (since we lost first 200 rows to SMA calculation)
    test_indices = list(range(split_idx + 200, split_idx + 200 + len(y_test)))
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
    
    # Calculate trading returns
    positions = np.where(sorted_y_test > sorted_predictions, 1, -1)  # Long if actual > predicted, short otherwise
    # Calculate returns: for long positions, profit when price increases; for short positions, profit when price decreases
    returns = positions[:-1] * (sorted_y_test[1:] - sorted_y_test[:-1]) / sorted_y_test[:-1]
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    # Plot price and predictions
    plt.subplot(2, 1, 1)
    plt.plot(sorted_dates, sorted_y_test, label='Actual Price', color='blue')
    plt.plot(sorted_dates, sorted_predictions, label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('BTC Price Prediction vs Actual')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    plt.plot(sorted_dates[1:], cumulative_returns, label='Cumulative Returns', color='green')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Trading Strategy Cumulative Returns')
    plt.legend()
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