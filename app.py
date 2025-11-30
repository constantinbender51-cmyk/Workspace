import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import requests

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
    # Mock model training for demonstration
    # In a real scenario, implement LSTM using TensorFlow/Keras or PyTorch
    print("Training LSTM model (mock implementation)")
    # Placeholder for model training logic
    # For example: model.fit(features, targets, epochs=10, batch_size=32)
    return lambda x: np.mean(x[:, -5:], axis=1)  # Mock predictor: average of last 5 closes

@app.route('/')
def index():
    # Fetch and prepare data
    data = fetch_ohlcv_data()
    features, targets, data_with_sma = prepare_features_target(data)
    
    # Train model
    model = train_lstm_model(features, targets)
    
    # Generate predictions (mock)
    predictions = model(features)
    
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
        <p>Note: This is a demonstration with mock data. Replace with real Binance API and LSTM implementation for production use.</p>
    </body>
    </html>
    '''
    return render_template_string(html_template, plot_url=plot_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)