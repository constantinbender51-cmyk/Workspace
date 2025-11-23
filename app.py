from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
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
    START_DATE = '2018-01-01'
    END_DATE = '2024-11-25'
    
    def fetch_chart_data(chart_name, start_date):
        params = {
            'format': 'json',
            'start': start_date,
            'timespan': '1year',
            'rollingAverage': '1d'
        }
        url = f"{BASE_URL}{chart_name}"
        print(f"DEBUG: Fetching {chart_name} from {url}")
        
        # Add headers for rate limiting awareness
        headers = {
            'User-Agent': 'BTC-Prediction-App/1.0'
        }
        
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                print(f"DEBUG: Response status for {chart_name}: {response.status_code}")
                
                # Handle rate limiting (429 status code)
                if response.status_code == 429:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"WARNING: Rate limit hit for {chart_name}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                    
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
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Request failed for {chart_name} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
                else:
                    print(f"ERROR: All retries failed for {chart_name}")
                    return pd.DataFrame()
            except Exception as e:
                print(f"ERROR: Failed to fetch {chart_name}: {str(e)}")
                return pd.DataFrame()
    
    all_data = [df_price]
    for metric_name, chart_endpoint in METRICS.items():
        print(f"\nDEBUG: Processing {metric_name} ({chart_endpoint})")
        time.sleep(3)  # Rate limiting - increased from 2 to 3 seconds
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
    # Calculate technical indicators
    # 3-day SMA for close price
    df['sma_3_close'] = df['close'].rolling(window=3).mean()
    # 9-day SMA for close price
    df['sma_9_close'] = df['close'].rolling(window=9).mean()
    # 3-day EMA for volume
    df['ema_3_volume'] = df['volume'].ewm(span=3).mean()
    
    # MACD (12,26,9)
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd_line'] = ema_12 - ema_26
    df['signal_line'] = df['macd_line'].ewm(span=9).mean()
    
    # Stochastic RSI (14,3,3)
    rsi_period = 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic RSI calculation
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    df['stoch_rsi'] = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    
    # Day of week (1-7)
    df['day_of_week'] = df.index.dayofweek + 1
    
    # Remove rows with NaN values from indicator calculations
    df_clean = df.dropna()
    
    features = []
    targets = []
    for i in range(len(df_clean)):
        # Features: technical indicators and on-chain metrics for previous 20 days
        if i >= 40:  # Ensure enough history for MACD and RSI calculations and 20-day lookback
            feature = []
            # Add features from the last 20 days (t-20 to t-1)
            for lookback in range(1, 21):
                if i - lookback >= 0:
                    # Technical indicators for each day in lookback period
                    feature.append(df_clean['sma_3_close'].iloc[i - lookback])
                    feature.append(df_clean['sma_9_close'].iloc[i - lookback])
                    feature.append(df_clean['ema_3_volume'].iloc[i - lookback])
                    feature.append(df_clean['macd_line'].iloc[i - lookback])
                    feature.append(df_clean['signal_line'].iloc[i - lookback])
                    feature.append(df_clean['stoch_rsi'].iloc[i - lookback])
                    feature.append(df_clean['day_of_week'].iloc[i - lookback])
                    
                    # On-chain metrics for each day in lookback period
                    if 'Net_Transaction_Count' in df_clean.columns:
                        feature.append(df_clean['Net_Transaction_Count'].iloc[i - lookback])
                    else:
                        feature.append(0)
                        
                    if 'Transaction_Volume_USD' in df_clean.columns:
                        feature.append(df_clean['Transaction_Volume_USD'].iloc[i - lookback])
                    else:
                        feature.append(0)
                        
                    if 'Active_Addresses' in df_clean.columns:
                        feature.append(df_clean['Active_Addresses'].iloc[i - lookback])
                    else:
                        feature.append(0)
                else:
                    # Pad with zeros if not enough history
                    feature.extend([0] * 10)  # 7 technical + 3 on-chain = 10 features per day
            
            features.append(feature)
            # Target: today's closing price
            target = df_clean['close'].iloc[i]
            targets.append(target)
    
    # Convert to array and remove rows with NaN in features (due to window padding)
    features = np.array(features)
    # Remove rows where any feature is NaN (from window padding)
    valid_indices = ~np.isnan(features).any(axis=1)
    features = features[valid_indices]
    # Adjust targets to match valid features
    targets = np.array(targets)
    targets = targets[valid_indices[:len(targets)]]
    # Ensure features and targets have the same length
    min_len = min(len(features), len(targets))
    features = features[:min_len]
    targets = targets[:min_len]
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized, np.array(targets), scaler

# Train model
def train_model(features, targets):
    # Use time series split: first 80% for training, last 20% for testing
    split_idx = int(len(features) * 0.8)
    X_train = features[:split_idx]
    X_test = features[split_idx:]
    y_train = targets[:split_idx]
    y_test = targets[split_idx:]
    # Training indices correspond to the indices in the cleaned DataFrame for the training set
    # Start from index 40 (due to MACD and RSI calculations and 20-day lookback requirement) and use the first split_idx rows after that
    train_indices = list(range(40, 40 + split_idx))
    
    # Reshape features for LSTM input: (samples, time_steps, features)
    # Each sample has 20 time steps (lookback days) and 10 features per day
    X_train_reshaped = X_train.reshape(X_train.shape[0], 20, 10)
    X_test_reshaped = X_test.reshape(X_test.shape[0], 20, 10)
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(40, activation='relu', return_sequences=True, input_shape=(20, 10)))
    model.add(LSTM(40, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Calculate predictions and MSE for both training and test sets
    train_predictions = model.predict(X_train_reshaped, verbose=0).flatten()
    test_predictions = model.predict(X_test_reshaped, verbose=0).flatten()
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    # Test indices correspond to the indices in the cleaned DataFrame for the test set
    test_indices = list(range(40 + split_idx, 40 + len(features)))
    
    return model, X_train, y_train, train_predictions, train_mse, test_mse, train_indices, X_test, y_test, test_predictions, test_indices

# Generate plot
def create_plot(df, y_train, train_predictions, train_indices, y_test, test_predictions, test_indices):
    plt.figure(figsize=(10, 8))
    
    # Combine training and test data for plotting
    all_dates = df.index[train_indices + test_indices]
    all_y_actual = np.concatenate([y_train, y_test])
    all_predictions = np.concatenate([train_predictions, test_predictions])
    
    # Sort by date to ensure chronological plotting
    sorted_indices = np.argsort(all_dates)
    sorted_dates = all_dates[sorted_indices]
    sorted_y_actual = all_y_actual[sorted_indices]
    sorted_predictions = all_predictions[sorted_indices]
    
    # Calculate capital with daily accumulation based on yesterday's prediction vs actual price for entire period
    capital = [1000]  # Start with $1000
    positions = []  # Store position type for coloring
    
    for i in range(len(sorted_y_actual)):
        # Calculate return using today's price vs yesterday's price
        if i >= 1:  # Ensure there is at least one previous day
            price_yesterday = sorted_y_actual[i - 1]
            return_calc = (sorted_y_actual[i] - price_yesterday) / price_yesterday
        else:
            # For the first day, use available data; skip if not enough history
            if i == 0 and (train_indices + test_indices)[sorted_indices[i]] >= 1:
                price_yesterday = df['close'].iloc[(train_indices + test_indices)[sorted_indices[i]] - 1]
                return_calc = (sorted_y_actual[i] - price_yesterday) / price_yesterday
            else:
                return_calc = 0  # Default to no return if insufficient data
        
        # ML Strategy: If yesterday's predicted price is above yesterday's actual price, apply positive return, else negative
        if i >= 1:  # Ensure yesterday's prediction is available
            pred_price_yesterday = sorted_predictions[i - 1]
            actual_price_yesterday = sorted_y_actual[i - 1]
            if pred_price_yesterday > actual_price_yesterday:
                ret = return_calc * 1  # Positive signal: long position with 1x leverage
                positions.append('long')  # Mark as long
            else:
                ret = -return_calc * 1  # Negative signal: short position with 1x leverage
                positions.append('short')  # Mark as short
        else:
            ret = 0  # Default for first day
            positions.append('neutral')  # Mark as neutral
        
        capital.append(capital[-1] * (1 + ret))
    
    capital = capital[1:]  # Remove the initial 1000 to match the number of dates
    
    # Plot actual and predicted prices with daily granularity and colored line segments for positions
    plt.subplot(2, 1, 1)
    plt.plot(sorted_dates, sorted_y_actual, label='Actual Price', color='blue')
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
    plt.title('BTC Price Prediction vs Actual (Training and Test Periods, Daily)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot capital with color based on position for entire period
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
    features, targets, scaler = prepare_data(df)
    model, X_train, y_train, train_predictions, train_mse, test_mse, train_indices, X_test, y_test, test_predictions, test_indices = train_model(features, targets)
    plot_url = create_plot(df, y_train, train_predictions, train_indices, y_test, test_predictions, test_indices)
    return render_template('index.html', plot_url=plot_url, train_mse=train_mse)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
