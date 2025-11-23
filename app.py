from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import subprocess
import threading
import time
import json

app = Flask(__name__)

# --- Global State for Training Progress ---
# In a production app with multiple workers, use Redis or a database.
training_state = {
    'status': 'idle', # idle, training, completed, error
    'progress': 0,
    'epoch': 0,
    'total_epochs': 0,
    'train_loss': [],
    'val_loss': [],
    'current_train_loss': 0,
    'current_val_loss': 0,
    'plot_url': None,
    'train_mse': 0,
    'error': None
}

class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        training_state['epoch'] = epoch + 1
        training_state['progress'] = int(((epoch + 1) / self.total_epochs) * 100)
        training_state['current_train_loss'] = logs.get('loss')
        training_state['current_val_loss'] = logs.get('val_loss')
        training_state['train_loss'].append(logs.get('loss'))
        training_state['val_loss'].append(logs.get('val_loss'))

# Load and preprocess data (Same logic as before)
def load_data():
    if not os.path.exists('btc_data.csv'):
        subprocess.run(['python', 'fetch_price_data.py'], check=True)
    df_price = pd.read_csv('btc_data.csv')
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price.set_index('date', inplace=True)
    
    # Simple On-chain fetch (Simplified for stability in threading)
    # Using existing logic but handling potential API failures gracefully
    try:
        import requests
        BASE_URL = "https://api.blockchain.info/charts/"
        METRICS = {
            'Active_Addresses': 'n-unique-addresses',
            'Net_Transaction_Count': 'n-transactions',
            'Transaction_Volume_USD': 'estimated-transaction-volume-usd',
        }
        START_DATE = '2022-08-10'
        END_DATE = '2023-09-30'
        
        all_data = [df_price]
        for metric_name, chart_endpoint in METRICS.items():
            time.sleep(1)
            try:
                params = {'format': 'json', 'start': START_DATE, 'timespan': '1year', 'rollingAverage': '1d'}
                response = requests.get(f"{BASE_URL}{chart_endpoint}", params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'values' in data:
                        df = pd.DataFrame(data['values'])
                        df['Date'] = pd.to_datetime(df['x'], unit='s', utc=True).dt.tz_localize(None)
                        df = df.set_index('Date')['y'].rename(metric_name)
                        all_data.append(df)
            except Exception as e:
                print(f"Skipping {metric_name}: {e}")
                
        df_combined = pd.concat(all_data, axis=1, join='inner')
        df_final = df_combined.loc[START_DATE:END_DATE].ffill()
        df_final = df_final[~df_final.index.duplicated(keep='first')]
        return df_final
    except Exception as e:
        print(f"Data fetch error: {e}")
        return df_price # Fallback to just price data

def prepare_data(df):
    df['sma_3_close'] = df['close'].rolling(window=3).mean()
    df['sma_9_close'] = df['close'].rolling(window=9).mean()
    df['ema_3_volume'] = df['volume'].ewm(span=3).mean()
    
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd_line'] = ema_12 - ema_26
    df['signal_line'] = df['macd_line'].ewm(span=9).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    df['stoch_rsi'] = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    df['day_of_week'] = df.index.dayofweek + 1
    
    df_clean = df.dropna()
    
    features = []
    targets = []
    for i in range(len(df_clean)):
        if i >= 40:
            feature = []
            for lookback in range(1, 21):
                if i - lookback >= 0:
                    feature.append(df_clean['sma_3_close'].iloc[i - lookback])
                    feature.append(df_clean['sma_9_close'].iloc[i - lookback])
                    feature.append(df_clean['ema_3_volume'].iloc[i - lookback])
                    feature.append(df_clean['macd_line'].iloc[i - lookback])
                    feature.append(df_clean['signal_line'].iloc[i - lookback])
                    feature.append(df_clean['stoch_rsi'].iloc[i - lookback])
                    feature.append(df_clean['day_of_week'].iloc[i - lookback])
                    
                    for col in ['Net_Transaction_Count', 'Transaction_Volume_USD', 'Active_Addresses']:
                        feature.append(df_clean[col].iloc[i - lookback] if col in df_clean.columns else 0)
                else:
                    feature.extend([0] * 10)
            features.append(feature)
            targets.append(df_clean['close'].iloc[i])
    
    features = np.array(features)
    valid_indices = ~np.isnan(features).any(axis=1)
    features = features[valid_indices]
    targets = np.array(targets)
    targets = targets[valid_indices[:len(targets)]]
    min_len = min(len(features), len(targets))
    return features[:min_len], targets[:min_len], StandardScaler().fit_transform(features[:min_len])

def create_plot(df, y_train, predictions, train_indices, history_loss, history_val_loss):
    plt.figure(figsize=(10, 12))
    
    # 1. Price Plot
    dates = df.index[train_indices]
    sorted_indices = np.argsort(dates)
    sorted_dates = dates[sorted_indices]
    sorted_y_train = y_train[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    
    plt.subplot(3, 1, 1)
    plt.plot(sorted_dates, sorted_y_train, label='Actual Price', color='blue')
    plt.plot(sorted_dates, sorted_predictions, label='Predicted', color='green', alpha=0.7)
    plt.title('BTC Price Prediction (Fit)')
    plt.legend()
    plt.xticks(rotation=45)

    # 2. Capital Plot (Simplified for brevity)
    capital = [1000]
    for i in range(1, len(sorted_y_train)):
        ret = (sorted_y_train[i] - sorted_y_train[i-1]) / sorted_y_train[i-1]
        pos = 1 if sorted_predictions[i-1] > sorted_y_train[i-1] else -1
        capital.append(capital[-1] * (1 + (ret * pos * 5))) # 5x leverage
    
    plt.subplot(3, 1, 2)
    plt.plot(sorted_dates, capital, color='purple')
    plt.title('Strategy Capital (5x Leverage)')
    plt.xticks(rotation=45)

    # 3. Double Descent Loss Curve
    plt.subplot(3, 1, 3)
    plt.semilogy(history_loss, label='Train Loss', color='blue')
    plt.semilogy(history_val_loss, label='Test Loss (Val)', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title('Double Descent Visualization')
    plt.legend()

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def run_training_task():
    global training_state
    try:
        training_state['status'] = 'training'
        training_state['train_loss'] = []
        training_state['val_loss'] = []
        
        df = load_data()
        features, targets, _ = prepare_data(df)
        
        # Split
        split_idx = int(len(features) * 0.8)
        X_train = features[:split_idx]
        X_test = features[split_idx:]
        y_train = targets[:split_idx]
        y_test = targets[split_idx:]
        train_indices = list(range(40, 40 + split_idx))
        
        X_train_reshaped = X_train.reshape(X_train.shape[0], 20, 10)
        X_test_reshaped = X_test.reshape(X_test.shape[0], 20, 10)
        
        # --- DOUBLE DESCENT CONFIGURATION ---
        EPOCHS = 1500  # Increased from 50
        UNITS = 512    # Increased from 100
        
        training_state['total_epochs'] = EPOCHS
        
        model = Sequential()
        model.add(LSTM(UNITS, activation='relu', return_sequences=True, input_shape=(20, 10)))
        model.add(LSTM(UNITS, activation='relu', return_sequences=True))
        model.add(LSTM(UNITS, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
        
        # Train with callback
        history = model.fit(
            X_train_reshaped, y_train,
            epochs=EPOCHS,
            batch_size=32,
            verbose=0,
            validation_data=(X_test_reshaped, y_test),
            callbacks=[ProgressCallback(EPOCHS)]
        )
        
        # Generate Results
        train_predictions = model.predict(X_train_reshaped, verbose=0).flatten()
        
        plot_url = create_plot(
            df, y_train, train_predictions, train_indices, 
            history.history['loss'], history.history['val_loss']
        )
        
        training_state['plot_url'] = plot_url
        training_state['train_mse'] = history.history['loss'][-1]
        training_state['status'] = 'completed'
        
    except Exception as e:
        print(f"Training failed: {e}")
        training_state['status'] = 'error'
        training_state['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    if training_state['status'] == 'training':
        return jsonify({'status': 'already_running'})
    
    # Reset state
    training_state['status'] = 'starting'
    training_state['progress'] = 0
    training_state['epoch'] = 0
    training_state['plot_url'] = None
    
    # Start background thread
    thread = threading.Thread(target=run_training_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/status')
def status():
    return jsonify(training_state)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
