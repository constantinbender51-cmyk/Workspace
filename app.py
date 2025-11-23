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
    df_price = pd.read_csv('btc_data.csv')
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price.set_index('date', inplace=True)
    
    # Fetch on-chain metrics
    import requests
    import time
    BASE_URL = "https://api.blockchain.info/charts/"
    METRICS = {
        'Active_Addresses': 'n-unique-addresses',
        'Net_Transaction_Count': 'n-transactions',
        'Transaction_Volume_USD': 'estimated-transaction-volume-usd',
    }
    START_DATE = '2022-01-01'
    END_DATE = '2023-09-30'
    
    def fetch_chart_data(chart_name, start_date):
        params = {
            'format': 'json',
            'start': start_date,
            'timespan': '1year',
            'rollingAverage': '1d'
        }
        url = f"{BASE_URL}{chart_name}"
        print(f"DEBUG: Fetching {chart_name} from {url}")
        try:
            response = requests.get(url, params=params, timeout=30)
            print(f"DEBUG: Response status for {chart_name}: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            print(f"DEBUG: Data keys for {chart_name}: {list(data.keys()) if data else 'No data'}")
            if 'values' not in data or not data['values']:
                print(f"DEBUG: No values found for {chart_name}")
                return pd.DataFrame()
            df = pd.DataFrame(data['values'])
            print(f"DEBUG: Fetched {len(df)} rows for {chart_name}")
            df['Date'] = pd.to_datetime(df['x'], unit='s', utc=True).dt.tz_localize(None)
            df = df.set_index('Date')['y'].rename(chart_name)
            print(f"DEBUG: Successfully processed {chart_name}, date range: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            print(f"ERROR: Failed to fetch {chart_name}: {str(e)}")
            return pd.DataFrame()
    
    all_data = [df_price]
    for metric_name, chart_endpoint in METRICS.items():
        print(f"\nDEBUG: Processing {metric_name} ({chart_endpoint})")
        time.sleep(2)  # Rate limiting
        df_metric = fetch_chart_data(chart_endpoint, START_DATE)
        if not df_metric.empty:
            df_metric = df_metric.rename(metric_name)
            all_data.append(df_metric)
            print(f"DEBUG: Successfully added {metric_name} to dataset")
        else:
            print(f"WARNING: No data for {metric_name}")
    
    print(f"DEBUG: Combining {len(all_data)} datasets")
    df_combined = pd.concat(all_data, axis=1, join='inner')
    print(f"DEBUG: Combined dataset shape: {df_combined.shape}")
    print(f"DEBUG: Combined dataset columns: {list(df_combined.columns)}")
    df_final = df_combined.loc[START_DATE:END_DATE].ffill()
    df_final = df_final[~df_final.index.duplicated(keep='first')]
    print(f"DEBUG: Final dataset shape: {df_final.shape}")
    print(f"DEBUG: Final dataset columns: {list(df_final.columns)}")
    
    return df_final

# Prepare features and target
def prepare_data(df):
    # Calculate specified SMAs
    df['sma_14'] = df['close'].rolling(window=14).mean()
    df['sma_14_squared'] = df['sma_14'] ** 2
    
    # Remove rows with NaN values from SMA and on-chain metric calculations
    df_clean = df.dropna()
    
    features = []
    targets = []
    for i in range(len(df_clean)):
        # Features: 14-day SMA, squared 14-day SMA, and on-chain metrics (if available)
        feature = [
            df_clean['sma_14'].iloc[i],
            df_clean['sma_14_squared'].iloc[i]
        ]
        # Add on-chain metrics only if they exist in the DataFrame
        if 'Net_Transaction_Count' in df_clean.columns:
            feature.append(df_clean['Net_Transaction_Count'].iloc[i])
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
    # Training indices correspond to the indices in the cleaned DataFrame for the training set
    train_indices = list(range(len(y_train)))
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate predictions and MSE for both training and test sets
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    return model, X_train, y_train, train_predictions, train_mse, test_mse, train_indices

# Generate plot
def create_plot(df, y_train, predictions, train_indices):
    plt.figure(figsize=(10, 8))
    dates = df.index[train_indices]
    # Sort by date to ensure chronological plotting
    sorted_indices = np.argsort(dates)
    sorted_dates = dates[sorted_indices]
    sorted_y_train = y_train[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    
    # Calculate capital with daily accumulation based on yesterday's prediction vs actual price
    capital = [1000]  # Start with $1000
    positions = []  # Store position type for coloring
    
    for i in range(len(sorted_y_train)):
        # Calculate return using today's price vs yesterday's price
        if i >= 1:  # Ensure there is at least one previous day
            price_yesterday = sorted_y_train[i - 1]
            return_calc = (sorted_y_train[i] - price_yesterday) / price_yesterday
        else:
            # For the first day in training set, use available data; skip if not enough history
            if i == 0 and train_indices[sorted_indices[i]] >= 1:
                price_yesterday = df['close'].iloc[train_indices[sorted_indices[i]] - 1]
                return_calc = (sorted_y_train[i] - price_yesterday) / price_yesterday
            else:
                return_calc = 0  # Default to no return if insufficient data
        
        # ML Strategy: If yesterday's predicted price is lower than yesterday's actual price, apply positive return, else negative
        if i >= 1:  # Ensure yesterday's prediction is available
            pred_price_yesterday = sorted_predictions[i - 1]
            actual_price_yesterday = sorted_y_train[i - 1]
            if pred_price_yesterday < actual_price_yesterday:
                ret = return_calc  # Positive signal: long position
                positions.append('long')  # Mark as long
            else:
                ret = -return_calc  # Negative signal: short position
                positions.append('short')  # Mark as short
        else:
            ret = 0  # Default for first day
            positions.append('neutral')  # Mark as neutral
        
        capital.append(capital[-1] * (1 + ret))
    
    capital = capital[1:]  # Remove the initial 1000 to match the number of dates
    
    # Plot price and predictions with colored line segments for positions
    plt.subplot(2, 1, 1)
    plt.plot(sorted_dates, sorted_y_train, label='Actual Price', color='blue')
    # Plot prediction line with color based on positions
    prev_idx = 0
    for j in range(1, len(sorted_dates)):
        if positions[j-1] == 'long':
            color = 'green'
        elif positions[j-1] == 'short':
            color = 'red'
        else:
            color = 'gray'  # Neutral in gray
        plt.plot(sorted_dates[prev_idx:j+1], sorted_predictions[prev_idx:j+1], color=color, linewidth=2)
        prev_idx = j
    plt.plot([], [], color='green', label='Predicted Price (Long)')
    plt.plot([], [], color='red', label='Predicted Price (Short)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('BTC Price Prediction vs Actual (Training Period)')
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
    model, X_train, y_train, predictions, train_mse, test_mse, train_indices = train_model(features, targets)
    plot_url = create_plot(df, y_train, predictions, train_indices)
    return render_template('index.html', plot_url=plot_url, train_mse=train_mse)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
