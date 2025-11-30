import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

# Mock data generation for demonstration
# In a real scenario, replace this with actual Binance API calls
def fetch_ohlcv_data():
    # Generate mock daily OHLCV data from 2018-01-01 to current date
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate price data with some noise and trend
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    high = price + np.abs(np.random.randn(len(dates)) * 2)
    low = price - np.abs(np.random.randn(len(dates)) * 2)
    open_price = price + np.random.randn(len(dates)) * 1
    volume = np.random.uniform(1000, 10000, len(dates))
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume
    })
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