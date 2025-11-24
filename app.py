from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout # Import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2 # Import L2 Regularizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import subprocess
import threading
import time
import logging
import traceback
import math

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Global State & Lock ---
state_lock = threading.Lock()
training_state = {
    'status': 'idle',
    'progress': 0,
    'epoch': 0,
    'total_epochs': 0,
    'batch': 0,
    'total_batches': 0,
    'train_loss': [],
    'val_loss': [],
    'current_train_loss': 0,
    'current_val_loss': 0,
    'plot_url': None,
    'train_mse': 0,
    'error': None,
    'last_update': 0
}

# --- Auto-start training on app startup ---
def start_training_on_startup():
    """Start training automatically when the app starts"""
    time.sleep(2)  # Give the app time to fully initialize
    with state_lock:
        if training_state['status'] == 'idle':
            training_state['status'] = 'starting'
            training_state['progress'] = 0
            training_state['epoch'] = 0
            training_state['plot_url'] = None
            training_state['error'] = None
    
    thread = threading.Thread(target=run_training_task)
    thread.daemon = True
    thread.start()
    logger.info("Auto-started training on app startup")

# --- Helper to prevent JSON crashes ---
def sanitize_float(val):
    try:
        if val is None: return None
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return None
        return float(val)
    except:
        return None

class ThrottledProgressCallback(Callback):
    def __init__(self, total_epochs, update_frequency=5):
        self.total_epochs = total_epochs
        self.update_frequency = update_frequency

    def on_epoch_begin(self, epoch, logs=None):
        with state_lock:
            training_state['epoch'] = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.update_frequency == 0:
            logs = logs or {}
            loss = sanitize_float(logs.get('loss'))
            
            with state_lock:
                training_state['batch'] = batch + 1
                training_state['current_train_loss'] = loss
                training_state['last_update'] = time.time()
                
                if self.params and 'steps' in self.params:
                    total_steps = self.params['steps']
                    training_state['total_batches'] = total_steps
                    current_epoch_progress = (training_state['epoch'] - 1) / self.total_epochs
                    current_batch_progress = (batch + 1) / total_steps
                    total_progress = current_epoch_progress + (current_batch_progress / self.total_epochs)
                    training_state['progress'] = int(total_progress * 100)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = sanitize_float(logs.get('val_loss'))
        loss = sanitize_float(logs.get('loss'))
        
        with state_lock:
            training_state['current_val_loss'] = val_loss
            training_state['train_loss'].append(loss)
            training_state['val_loss'].append(val_loss)
        
        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{self.total_epochs} - Loss: {loss}")

def load_data():
    logger.info("Step 1: Loading Data...")
    if not os.path.exists('btc_data.csv'):
        logger.info("btc_data.csv not found. Running fetch script...")
        try:
            subprocess.run(['python', 'fetch_price_data.py'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Fetch script failed: {e}")
            raise e

    df_price = pd.read_csv('btc_data.csv')
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price.set_index('date', inplace=True)
    
    try:
        import requests
        BASE_URL = "https://api.blockchain.info/charts/"
        METRICS = {
            'Active_Addresses': 'n-unique-addresses',
            'Net_Transaction_Count': 'n-transactions',
            'Transaction_Volume_USD': 'estimated-transaction-volume-usd',
        }
        START_DATE = '2022-01-01'
        END_DATE = '2023-12-31'
        
        all_data = [df_price]
        for metric_name, chart_endpoint in METRICS.items():
            yearly_data = []
            # Make 8 separate API calls for each year from 2018 to 2025 to cover the full range
            for year in range(2018, 2026):
                time.sleep(1)  # Sleep for 1 second between API calls to respect rate limits
                try:
                    year_start = f"{year}-01-01"
                    year_end = f"{year}-12-31"
                    # Adjust end date for 2025 to match END_DATE
                    if year == 2025:
                        year_end = END_DATE
                    params = {'format': 'json', 'start': year_start, 'end': year_end, 'timespan': '1year'}
                    response = requests.get(f"{BASE_URL}{chart_endpoint}", params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'values' in data:
                            df = pd.DataFrame(data['values'])
                            df['Date'] = pd.to_datetime(df['x'], unit='s', utc=True).dt.tz_localize(None)
                            df = df.set_index('Date')['y']
                            yearly_data.append(df)
                except Exception as e:
                    logger.warning(f"Skipping {metric_name} for year {year}: {e}")
            
            # Combine all yearly data for this metric
            if yearly_data:
                combined_df = pd.concat(yearly_data, axis=0)
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df = combined_df.sort_index()
                combined_df = combined_df.rename(metric_name)
                all_data.append(combined_df)
                
        df_combined = pd.concat(all_data, axis=1, join='inner')
        df_final = df_combined.loc[START_DATE:END_DATE].ffill()
        df_final = df_final[~df_final.index.duplicated(keep='first')]
        

        
        return df_final
    except Exception as e:
        logger.error(f"Data merge error: {e}")
        return df_price 

def prepare_data(df):
    logger.info("Step 2: Preparing Features...")
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
            for lookback in range(1, 13):
                if i - lookback >= 0:
                    # Use actual values for each feature
                    feature.append(df_clean['sma_3_close'].iloc[i - lookback])
                    feature.append(df_clean['sma_9_close'].iloc[i - lookback])
                    feature.append(df_clean['ema_3_volume'].iloc[i - lookback])
                    feature.append(df_clean['macd_line'].iloc[i - lookback])
                    feature.append(df_clean['signal_line'].iloc[i - lookback])
                    feature.append(df_clean['stoch_rsi'].iloc[i - lookback])
                    feature.append(df_clean['day_of_week'].iloc[i - lookback])
                    
                    # On-chain metrics actual values
                    for col in ['Net_Transaction_Count', 'Transaction_Volume_USD', 'Active_Addresses']:
                        if col in df_clean.columns:
                            feature.append(df_clean[col].iloc[i - lookback])
                        else:
                            feature.append(0)
                else:
                    # For days before the start of the sequence, use zeros
                    feature.extend([0] * 10)
            features.append(feature)
            targets.append(df_clean['close'].iloc[i])
    
    features = np.array(features)
    valid_indices = ~np.isnan(features).any(axis=1)
    features = features[valid_indices]
    targets = np.array(targets)
    targets = targets[valid_indices[:len(targets)]]
    min_len = min(len(features), len(targets))
    
    # Scale Features
    scaler_features = StandardScaler()
    features_scaled = scaler_features.fit_transform(features[:min_len])
    
    # Scale Targets
    scaler_target = StandardScaler()
    targets_reshaped = targets[:min_len].reshape(-1, 1)
    targets_scaled = scaler_target.fit_transform(targets_reshaped)
    
    return features_scaled, targets_scaled, scaler_features, scaler_target

def create_plot(df, y_train, predictions, train_indices, history_loss, history_val_loss, y_test=None, test_predictions=None, test_indices=None):
    logger.info("Step 4: Generating Plots...")
    history_loss = [x if x is not None else 0 for x in history_loss]
    history_val_loss = [x if x is not None else 0 for x in history_val_loss]

    plt.figure(figsize=(10, 12))
    
    # Combine training and test data for continuous timeline
    all_dates = []
    all_y_actual = []
    all_y_predicted = []
    
    # Add training data
    train_dates = df.index[train_indices]
    sorted_train_indices = np.argsort(train_dates)
    sorted_train_dates = train_dates[sorted_train_indices]
    sorted_y_train = y_train[sorted_train_indices]
    sorted_predictions = predictions[sorted_train_indices]
    all_dates.extend(sorted_train_dates)
    all_y_actual.extend(sorted_y_train)
    all_y_predicted.extend(sorted_predictions)
    
    # Add test data if available
    if y_test is not None and test_predictions is not None and test_indices is not None:
        test_dates = df.index[test_indices]
        sorted_test_indices = np.argsort(test_dates)
        sorted_test_dates = test_dates[sorted_test_indices]
        sorted_y_test = y_test[sorted_test_indices]
        sorted_test_predictions = test_predictions[sorted_test_indices]
        all_dates.extend(sorted_test_dates)
        all_y_actual.extend(sorted_y_test)
        all_y_predicted.extend(sorted_test_predictions)
    
    # Plot 1: Combined actual vs predicted prices
    plt.subplot(3, 1, 1)
    plt.plot(all_dates, all_y_actual, label='Actual Price', color='blue')
    plt.plot(all_dates, all_y_predicted, label='Predicted', color='green', alpha=0.7)
    plt.title('BTC Price Prediction (Training and Test Sets)')
    plt.legend()
    plt.xticks(rotation=45)

    # Strategy capital calculation for combined data
    capital = [1000]
    for i in range(1, len(all_y_actual)):
        ret = (all_y_actual[i] - all_y_actual[i-1]) / all_y_actual[i-1]
        pos = 1 if all_y_predicted[i] > all_y_actual[i-1] else -1 
        capital.append(capital[-1] * (1 + (ret * pos * 1)))
    
    plt.subplot(3, 1, 2)
    plt.plot(all_dates, capital, color='purple')
    plt.title('Strategy Capital (Long/Short based on Predicted vs Yesterday Price)')
    plt.xticks(rotation=45)

    # Plot 3: Loss curves
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
    plt.close() # Important to close plot after saving
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def run_training_task():
    global training_state
    logger.info("Background thread started.")
    K.clear_session()
    
    try:
        with state_lock:
            training_state['status'] = 'training'
            training_state['train_loss'] = []
            training_state['val_loss'] = []
        
        df = load_data()
        features, targets_scaled, _, scaler_target = prepare_data(df)
        
        split_idx = int(len(features) * 0.7)
        X_train = features[:split_idx]
        X_test = features[split_idx:]
        y_train = targets_scaled[:split_idx]
        y_test = targets_scaled[split_idx:]
        train_indices = list(range(40, 40 + split_idx))
        
        X_train_reshaped = X_train.reshape(X_train.shape[0], 12, 10)
        X_test_reshaped = X_test.reshape(X_test.shape[0], 12, 10)
        
        # INCREASED EPOCHS AND ADDED REGULARIZATION
        EPOCHS = 1000
        UNITS = 512
        REG_RATE = 1e-4 # L2 Regularization rate
        
        with state_lock:
            training_state['total_epochs'] = EPOCHS
        
        logger.info(f"Building Model (Units: {UNITS}, Epochs: {EPOCHS}, L2: {REG_RATE})...")
        model = Sequential()
        
        # LSTM 1: L2 regularization added to the kernel weights
        model.add(LSTM(UNITS, activation='relu', return_sequences=True, 
                       input_shape=(12, 10), kernel_regularizer=l2(REG_RATE)))
        model.add(Dropout(0.5)) # Dropout to force redundancy
        
        # LSTM 2: L2 regularization added
        model.add(LSTM(UNITS, activation='relu', return_sequences=True, 
                       kernel_regularizer=l2(REG_RATE)))
        model.add(Dropout(0.5)) # Dropout to force redundancy
        
        # LSTM 3: L2 regularization added
        model.add(LSTM(UNITS, activation='relu', kernel_regularizer=l2(REG_RATE)))
        model.add(Dropout(0.5)) # Dropout to force redundancy
        
        model.add(Dense(1))
        
        # Use low learning rate and clipnorm for stability
        optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')
        
        logger.info("Starting model.fit() ...")
        from tensorflow.keras.callbacks import EarlyStopping
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=500,
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train_reshaped, y_train,
            epochs=EPOCHS,
            batch_size=128,
            verbose=0,
            validation_data=(X_test_reshaped, y_test),
            callbacks=[ThrottledProgressCallback(EPOCHS, update_frequency=5), early_stopping]
        )
        logger.info("Training completed.")
        
        # Get predictions in Scaled format
        train_predictions_scaled = model.predict(X_train_reshaped, verbose=0)
        test_predictions_scaled = model.predict(X_test_reshaped, verbose=0)
        
        # INVERSE TRANSFORM: Convert scaled outputs back to real dollar prices
        train_predictions_real = scaler_target.inverse_transform(train_predictions_scaled).flatten()
        y_train_real = scaler_target.inverse_transform(y_train).flatten()
        test_predictions_real = scaler_target.inverse_transform(test_predictions_scaled).flatten()
        y_test_real = scaler_target.inverse_transform(y_test).flatten()
        test_indices = list(range(40 + split_idx, 40 + split_idx + len(y_test_real)))
        
        plot_url = create_plot(
            df, y_train_real, train_predictions_real, train_indices, 
            history.history['loss'], history.history['val_loss'],
            y_test_real, test_predictions_real, test_indices
        )
        
        with state_lock:
            training_state['plot_url'] = plot_url
            # Note: The 'loss' here now includes the L2 penalty, which is why it might not approach 0 as closely as before.
            training_state['train_mse'] = sanitize_float(history.history['loss'][-1])
            training_state['status'] = 'completed'
        
    except Exception as e:
        logger.error(f"CRITICAL TRAINING FAILURE: {e}")
        logger.error(traceback.format_exc())
        with state_lock:
            training_state['status'] = 'error'
            training_state['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    with state_lock:
        if training_state['status'] == 'training':
            return jsonify({'status': 'already_running'})
        
        training_state['status'] = 'starting'
        training_state['progress'] = 0
        training_state['epoch'] = 0
        training_state['plot_url'] = None
        training_state['error'] = None
    
    thread = threading.Thread(target=run_training_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/status')
def status():
    with state_lock:
        return jsonify(training_state)

# Start training automatically when app starts
startup_thread = threading.Thread(target=start_training_on_startup)
startup_thread.daemon = True
startup_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
