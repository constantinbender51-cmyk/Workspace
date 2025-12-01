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
from sklearn.preprocessing import MinMaxScaler
import threading
import time
import json

app = Flask(__name__)

# Global variables for model and training status
trained_model = None
training_history = None
training_in_progress = False
training_complete = False
training_start_time = None
loss_history = []
val_loss_history = []

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

def prepare_features_target(data, feature_window=30, target_window=365):
    data['sma_365'] = calculate_sma(data['close'], target_window)
    
    # Calculate SMA 28 as feature
    data['sma_28'] = calculate_sma(data['close'], 28)
    
    # Create features: past 30 days of SMA 28 values
    features = []
    targets = []
    valid_indices = []
    for i in range(feature_window, len(data) - 1):
        if not pd.isna(data['sma_365'].iloc[i + 1]):
            feature = data['sma_28'].iloc[i - feature_window + 1: i + 1].values.flatten()
            target = data['sma_365'].iloc[i + 1]
            features.append(feature)
            targets.append(target)
            valid_indices.append(data.index[i + 1])
    
    features_array = np.array(features)
    targets_array = np.array(targets)
    
    # Normalize features and targets using MinMaxScaler (0 to 1) to prevent large values
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_normalized = feature_scaler.fit_transform(features_array)
    targets_normalized = target_scaler.fit_transform(targets_array.reshape(-1, 1)).flatten()
    
    # Check for NaN values
    if np.any(np.isnan(features_normalized)) or np.any(np.isnan(targets_normalized)):
        raise ValueError("NaN values detected in normalized data")
    
    return features_normalized, targets_normalized, data, valid_indices, feature_scaler, target_scaler

def train_lstm_model(features, targets):
    # Reshape features for LSTM input: (samples, timesteps, features)
    # Features are flattened from 30 days * 1 feature (SMA 28 / volume ratio), reshape to (samples, 30, 1)
    n_samples = features.shape[0]
    features_reshaped = features.reshape(n_samples, 30, 1)
    
    # Build LSTM model with gradient clipping to prevent exploding gradients
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(30, 1), kernel_constraint=tf.keras.constraints.MaxNorm(3), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-3, l2=5e-3)),
        tf.keras.layers.Dropout(0.164025),
        Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-3, l2=5e-3))
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Add early stopping to prevent overfitting and improve stability
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    # Custom callback to capture loss and val_loss during training
    class LossHistoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                loss_history.append(logs.get('loss'))
                val_loss_history.append(logs.get('val_loss'))
                # Print for debugging
                print(f"Epoch {epoch+1}: loss={logs.get('loss')}, val_loss={logs.get('val_loss')}")
    
    # Train the model
    print("Training LSTM model with TensorFlow/Keras")
    history = model.fit(features_reshaped, targets, epochs=20, validation_split=0.2, batch_size=64, verbose=1, callbacks=[early_stopping, LossHistoryCallback()])
    return model, history

# Function to train model in background
def train_model_background():
    global trained_model, training_history, training_in_progress, training_complete, training_start_time, loss_history, val_loss_history
    training_in_progress = True
    training_complete = False
    training_start_time = time.time()
    loss_history.clear()
    val_loss_history.clear()
    
    try:
        # Fetch and prepare data
        data = fetch_ohlcv_data()
        features, targets, data_with_sma, valid_indices, feature_scaler, target_scaler = prepare_features_target(data)
        
        # Train model
        trained_model, training_history = train_lstm_model(features, targets)
        
        training_complete = True
        print(f"Training completed in {time.time() - training_start_time:.2f} seconds")
    except Exception as e:
        print(f"Training failed: {e}")
        training_complete = False
    finally:
        training_in_progress = False

# Start training when the script runs
print("Starting model training in background...")
training_thread = threading.Thread(target=train_model_background, daemon=True)
training_thread.start()

@app.route('/')
def index():
    global trained_model, training_history, training_in_progress, training_complete, training_start_time
    
    # HTML template with dynamic content
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Model Training Progress</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .training { background-color: #fff3cd; border: 1px solid #ffeaa7; }
            .complete { background-color: #d4edda; border: 1px solid #c3e6cb; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            .plot { margin: 20px 0; }
            #loss-data { margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>LSTM Model Training Progress</h1>
        
        <div id="status-container">
            <!-- Status will be updated dynamically -->
        </div>
        
        <div id="loss-data">
            <h2>Loss and Validation Loss</h2>
            <table id="loss-table">
                <thead>
                    <tr>
                        <th>Epoch</th>
                        <th>Loss</th>
                        <th>Validation Loss</th>
                    </tr>
                </thead>
                <tbody id="loss-table-body">
                    <!-- Rows will be added dynamically -->
                </tbody>
            </table>
        </div>
        
        <div id="plots-container">
            <!-- Plots will be inserted here after training completes -->
        </div>
        
        <script>
            const eventSource = new EventSource('/progress');
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Update status container
                const statusContainer = document.getElementById('status-container');
                let statusHTML = '';
                
                if (data.training_in_progress) {
                    statusHTML = `
                        <div class="status training">
                            <h2>Training in Progress...</h2>
                            <p>Model training started at ${data.start_time}.</p>
                            <p>Elapsed time: ${data.elapsed_time} seconds</p>
                            <p>Please wait while the model trains with 20 epochs.</p>
                            <p>Epochs completed: ${data.epochs_completed}</p>
                            <p>Check the console for detailed progress.</p>
                        </div>
                    `;
                } else if (data.training_complete && data.trained_model_exists) {
                    statusHTML = `
                        <div class="status complete">
                            <h2>Training Complete!</h2>
                            <p>Model trained successfully in ${data.elapsed_time} seconds.</p>
                            <p>Total epochs: ${data.total_epochs}</p>
                        </div>
                    `;
                    // Insert plots after training completes
                    const plotsContainer = document.getElementById('plots-container');
                    plotsContainer.innerHTML = `
                        <div class="plot">
                            <h2>LSTM Model Predictions vs 365-Day Simple Moving Average</h2>
                            <img src="data:image/png;base64,${data.plot_url1}" alt="Predictions Chart">
                        </div>
                        <div class="plot">
                            <h2>Training Loss vs Validation Loss</h2>
                            <p>Epochs reduced from 40 to 20 (2x).</p>
                            <img src="data:image/png;base64,${data.plot_url2}" alt="Loss Chart">
                        </div>
                        <p>Note: Using TensorFlow/Keras for LSTM model training.</p>
                    `;
                    eventSource.close(); // Close SSE connection after training completes
                } else {
                    statusHTML = `
                        <div class="status error">
                            <h2>Training Status Unknown</h2>
                            <p>Model training has not started or encountered an error.</p>
                            <p>Please check the application logs.</p>
                        </div>
                    `;
                }
                
                statusContainer.innerHTML = statusHTML;
                
                // Update loss table
                const lossTableBody = document.getElementById('loss-table-body');
                if (data.loss_history && data.val_loss_history) {
                    lossTableBody.innerHTML = '';
                    for (let i = 0; i < data.loss_history.length; i++) {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${i + 1}</td>
                            <td>${data.loss_history[i].toFixed(6)}</td>
                            <td>${data.val_loss_history[i].toFixed(6)}</td>
                        `;
                        lossTableBody.appendChild(row);
                    }
                }
            };
            
            eventSource.onerror = function(error) {
                console.error('EventSource failed:', error);
                const statusContainer = document.getElementById('status-container');
                statusContainer.innerHTML = `
                    <div class="status error">
                        <h2>Connection Error</h2>
                        <p>Failed to connect to progress stream. Please refresh the page.</p>
                    </div>
                `;
            };
        </script>
    </body>
    </html>
    '''
    
    # Calculate elapsed time
    elapsed = 0
    if training_start_time:
        elapsed = int(time.time() - training_start_time)
    
    # Format start time
    start_time_str = "N/A"
    if training_start_time:
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))
    
    return render_template_string(html_template)

@app.route('/progress')
def progress():
    global trained_model, training_history, training_in_progress, training_complete, training_start_time, loss_history, val_loss_history
    
    def generate():
        while True:
            # Calculate elapsed time
            elapsed = 0
            if training_start_time:
                elapsed = int(time.time() - training_start_time)
            
            # Format start time
            start_time_str = "N/A"
            if training_start_time:
                start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))
            
            # Prepare data to send
            data = {
                'training_in_progress': training_in_progress,
                'training_complete': training_complete,
                'start_time': start_time_str,
                'elapsed_time': elapsed,
                'loss_history': loss_history.copy(),
                'val_loss_history': val_loss_history.copy(),
                'epochs_completed': len(loss_history),
                'total_epochs': 20,
                'trained_model_exists': trained_model is not None
            }
            
            # If training is complete, add plot URLs
            if training_complete and trained_model is not None and training_history is not None:
                # Fetch and prepare data for predictions
                data_fetched = fetch_ohlcv_data()
                features, targets, data_with_sma, valid_indices, feature_scaler, target_scaler = prepare_features_target(data_fetched)
                features_reshaped = features.reshape(features.shape[0], 30, 1)
                predictions_normalized = trained_model.predict(features_reshaped).flatten()
                predictions = target_scaler.inverse_transform(predictions_normalized.reshape(-1, 1)).flatten()
                actual_sma = target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
                
                # Create first plot
                plt.figure(figsize=(12, 6))
                plt.plot(valid_indices, actual_sma, label='Actual 365 SMA', color='blue')
                plt.plot(valid_indices, predictions, label='Model Predictions', color='red', linestyle='--')
                plt.title('LSTM Model Predictions vs Actual 365-Day SMA')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                
                img1 = io.BytesIO()
                plt.savefig(img1, format='png', bbox_inches='tight')
                img1.seek(0)
                data['plot_url1'] = base64.b64encode(img1.getvalue()).decode()
                plt.close()
                
                # Create second plot with log scale
                plt.figure(figsize=(12, 6))
                plt.plot(training_history.history['loss'], label='Training Loss', color='blue')
                plt.plot(training_history.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
                plt.title('Training Loss vs Validation Loss (20 Epochs) - Log Scale')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE) - Log Scale')
                plt.yscale('log')
                plt.legend()
                plt.grid(True)
                
                img2 = io.BytesIO()
                plt.savefig(img2, format='png', bbox_inches='tight')
                img2.seek(0)
                data['plot_url2'] = base64.b64encode(img2.getvalue()).decode()
                plt.close()
            
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)  # Send updates every second
    
    return app.response_class(generate(), mimetype='text/event-stream')
    global trained_model, training_history, training_in_progress, training_complete, training_start_time
    
    # HTML template with dynamic content
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Model Training Progress</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .training { background-color: #fff3cd; border: 1px solid #ffeaa7; }
            .complete { background-color: #d4edda; border: 1px solid #c3e6cb; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            .plot { margin: 20px 0; }
            #loss-data { margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>LSTM Model Training Progress</h1>
        
        <div id="status-container">
            <!-- Status will be updated dynamically -->
        </div>
        
        <div id="loss-data">
            <h2>Loss and Validation Loss</h2>
            <table id="loss-table">
                <thead>
                    <tr>
                        <th>Epoch</th>
                        <th>Loss</th>
                        <th>Validation Loss</th>
                    </tr>
                </thead>
                <tbody id="loss-table-body">
                    <!-- Rows will be added dynamically -->
                </tbody>
            </table>
        </div>
        
        <div id="plots-container">
            <!-- Plots will be inserted here after training completes -->
        </div>
        
        <script>
            const eventSource = new EventSource('/progress');
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Update status container
                const statusContainer = document.getElementById('status-container');
                let statusHTML = '';
                
                if (data.training_in_progress) {
                    statusHTML = `
                        <div class="status training">
                            <h2>Training in Progress...</h2>
                            <p>Model training started at ${data.start_time}.</p>
                            <p>Elapsed time: ${data.elapsed_time} seconds</p>
                            <p>Please wait while the model trains with 40 epochs.</p>
                            <p>Epochs completed: ${data.epochs_completed}</p>
                            <p>Check the console for detailed progress.</p>
                        </div>
                    `;
                } else if (data.training_complete && data.trained_model_exists) {
                    statusHTML = `
                        <div class="status complete">
                            <h2>Training Complete!</h2>
                            <p>Model trained successfully in ${data.elapsed_time} seconds.</p>
                            <p>Total epochs: ${data.total_epochs}</p>
                        </div>
                    `;
                    // Insert plots after training completes
                    const plotsContainer = document.getElementById('plots-container');
                    plotsContainer.innerHTML = `
                        <div class="plot">
                            <h2>LSTM Model Predictions vs 365-Day Simple Moving Average</h2>
                            <img src="data:image/png;base64,${data.plot_url1}" alt="Predictions Chart">
                        </div>
                        <div class="plot">
                            <h2>Training Loss vs Validation Loss</h2>
                            <p>Epochs increased from 20 to 40 (2x).</p>
                            <img src="data:image/png;base64,${data.plot_url2}" alt="Loss Chart">
                        </div>
                        <p>Note: Using TensorFlow/Keras for LSTM model training.</p>
                    `;
                    eventSource.close(); // Close SSE connection after training completes
                } else {
                    statusHTML = `
                        <div class="status error">
                            <h2>Training Status Unknown</h2>
                            <p>Model training has not started or encountered an error.</p>
                            <p>Please check the application logs.</p>
                        </div>
                    `;
                }
                
                statusContainer.innerHTML = statusHTML;
                
                // Update loss table
                const lossTableBody = document.getElementById('loss-table-body');
                if (data.loss_history && data.val_loss_history) {
                    lossTableBody.innerHTML = '';
                    for (let i = 0; i < data.loss_history.length; i++) {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${i + 1}</td>
                            <td>${data.loss_history[i].toFixed(6)}</td>
                            <td>${data.val_loss_history[i].toFixed(6)}</td>
                        `;
                        lossTableBody.appendChild(row);
                    }
                }
            };
            
            eventSource.onerror = function(error) {
                console.error('EventSource failed:', error);
                const statusContainer = document.getElementById('status-container');
                statusContainer.innerHTML = `
                    <div class="status error">
                        <h2>Connection Error</h2>
                        <p>Failed to connect to progress stream. Please refresh the page.</p>
                    </div>
                `;
            };
        </script>
    </body>
    </html>
    '''
    
    # Calculate elapsed time
    elapsed = 0
    if training_start_time:
        elapsed = int(time.time() - training_start_time)
    
    # Format start time
    start_time_str = "N/A"
    if training_start_time:
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))
    
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)