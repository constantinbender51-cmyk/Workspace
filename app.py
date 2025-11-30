import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# Fetch OHLCV data from Binance public API
def fetch_ohlcv_data():
    symbol = 'BTCUSDT'  # Example symbol, can be parameterized
    interval = '1d'
    start_time = int(datetime(2018, 1, 1).timestamp() * 1000)  # Binance uses milliseconds
    end_time = int(datetime.now().timestamp() * 1000)
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit=1000'
    
    all_data = []
    while start_time < end_time:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1  # Move to next time after last entry
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit=1000'
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df.set_index('date', inplace=True)
    return df

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def prepare_features_target(data, feature_window=90, target_window=365):
    data['sma_365'] = calculate_sma(data['close'], target_window)
    
    # Create features: past 90 days of OHLCV
    features = []
    targets = []
    for i in range(feature_window, len(data) - 1):
        if not pd.isna(data['sma_365'].iloc[i + 1]):
            feature = data[['open', 'high', 'low', 'close', 'volume']].iloc[i - feature_window + 1: i + 1].values.flatten()
            target = data['sma_365'].iloc[i + 1]
            features.append(feature)
            targets.append(target)
    
    return np.array(features), np.array(targets), data

def train_lstm_model(features, targets):
    # Reshape features for LSTM input: (samples, timesteps, features)
    # Assuming features are flattened from 90 days * 5 features (OHLCV), reshape to (samples, 90, 5)
    n_samples = features.shape[0]
    features_reshaped = features.reshape(n_samples, 90, 5)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(90, 5)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    print("Training LSTM model with TensorFlow/Keras")
    model.fit(features_reshaped, targets, epochs=50, validation_split=0.2, verbose=0)
    return model

@app.route('/')
def index():
    # Fetch and prepare data
    data = fetch_ohlcv_data()
    features, targets, data_with_sma = prepare_features_target(data)
    
    # Train model
    model = train_lstm_model(features, targets)
    
    # Generate predictions
    features_reshaped = features.reshape(features.shape[0], 90, 5)
    predictions = model.predict(features_reshaped).flatten()
    
    # Prepare data for chart
    valid_indices = data_with_sma.index[90:-1]  # Align with features
    actual_sma = data_with_sma['sma_365'].loc[valid_indices].values
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(valid_indices, actual_sma, label='Actual 365 SMA', color='blue')
    plt.plot(valid_indices, predictions, label='Model Predictions', color='red', linestyle='--')
    plt.title('LSTM Model Predictions vs Actual 365-Day SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Convert plot to base64 for HTML embedding
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # HTML template
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Model vs 365 SMA</title>
    </head>
    <body>
        <h1>LSTM Model Predictions vs 365-Day Simple Moving Average</h1>
        <img src="data:image/png;base64,{{ plot_url }}" alt="Chart">
        <p>Note: Using TensorFlow/Keras for LSTM model training.</p>
    </body>
    </html>
    '''
    return render_template_string(html_template, plot_url=plot_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)