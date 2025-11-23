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
    df['sma_1'] = df['close'].rolling(window=0).mean()
    
    # Remove rows with NaN values from SMA calculation
    df_clean = df.dropna()
    
    features = []
    targets = []
    for i in range(len(df_clean)):
        # Features: 1-day price SMA
        feature = [
            df_clean['sma_1'].iloc[i]
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
    
    # Calculate capital with daily accumulation based on prediction vs actual and actual Bitcoin returns
    capital = [1000]  # Start with $1000
    
    for i in range(len(sorted_y_test)):
        if i == 0:
            # For the first day, use the price from the day before the test period starts
            prev_index = test_indices[sorted_indices[i]] - 1
            prev_price = df['close'].iloc[prev_index] if prev_index >= 0 else sorted_y_test[i]
        else:
            prev_price = sorted_y_test[i - 1]
        
        actual_return = (sorted_y_test[i] - prev_price) / prev_price
        
        # ML Strategy
        if sorted_predictions[i] > sorted_y_test[i]:  # Prediction above actual: negative actual return
            ret = -actual_return
        else:  # Prediction below actual: positive actual return
            ret = actual_return
        capital.append(capital[-1] * (1 + ret))
    
    capital = capital[1:]  # Remove the initial 1000 to match the number of dates
    
    # Plot price and predictions
    plt.subplot(2, 1, 1)
    plt.plot(sorted_dates, sorted_y_test, label='Actual Price', color='blue')
    plt.plot(sorted_dates, sorted_predictions, label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('BTC Price Prediction vs Actual')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot capital
    plt.subplot(2, 1, 2)
    plt.plot(sorted_dates, capital, label='ML Strategy Capital', color='green')
    plt.xlabel('Date')
    plt.ylabel('Capital (USD)')
    plt.title('Trading Strategy Capital')
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
