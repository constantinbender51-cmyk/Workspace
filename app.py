from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os, subprocess, threading, time, logging, traceback
from statsmodels.tsa.stattools import acf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
    'last_update': 0,
    'mae': None,
    'smape': None,
    'resid_std': None,
    'resid_auto': None
}

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

def load_data():
    if not os.path.exists('btc_data.csv'):
        subprocess.run(['python', 'fetch_price_data.py'], check=True)
    df = pd.read_csv('btc_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def prepare_data(df):
    df_clean = df.dropna()
    features, targets = [], []
    for i in range(len(df_clean)):
        if i >= 40:
            feature = []
            for lookback in range(1, 13):
                if i - lookback >= 0:
                    feature.extend([
                        df_clean['sma_3_close'].iloc[i - lookback] if 'sma_3_close' in df_clean else 0,
                        df_clean['sma_9_close'].iloc[i - lookback] if 'sma_9_close' in df_clean else 0,
                        df_clean['ema_3_volume'].iloc[i - lookback] if 'ema_3_volume' in df_clean else 0,
                        df_clean['macd_line'].iloc[i - lookback] if 'macd_line' in df_clean else 0,
                        df_clean['signal_line'].iloc[i - lookback] if 'signal_line' in df_clean else 0,
                        df_clean['day_of_week'].iloc[i - lookback] if 'day_of_week' in df_clean else 0,
                        df_clean['volatility_7d'].iloc[i - lookback] if 'volatility_7d' in df_clean else 0,
                        df_clean['return_lag1'].iloc[i - lookback] if 'return_lag1' in df_clean else 0,
                        df_clean['return_lag3'].iloc[i - lookback] if 'return_lag3' in df_clean else 0,
                        df_clean['return_lag7'].iloc[i - lookback] if 'return_lag7' in df_clean else 0,
                        df_clean['bollinger_bandwidth'].iloc[i - lookback] if 'bollinger_bandwidth' in df_clean else 0
                    ])
                else:
                    feature.extend([0]*11)
            features.append(feature)
            targets.append(df_clean['log_return'].iloc[i])
    features = np.array(features)
    targets = np.array(targets)
    valid = ~np.isnan(features).any(axis=1) & ~np.isnan(targets)
    features, targets = features[valid], targets[valid]
    split_idx = int(len(features)*0.7)
    scaler_features = StandardScaler().fit(features[:split_idx])
    features_scaled = scaler_features.transform(features)
    scaler_target = StandardScaler().fit(targets[:split_idx].reshape(-1,1))
    targets_scaled = scaler_target.transform(targets.reshape(-1,1))
    return features_scaled, targets_scaled, scaler_features, scaler_target

# ... previous imports and code ...

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def create_plot(df, y_train, predictions, train_indices, history_loss, history_val_loss, y_test=None, test_predictions=None, test_indices=None):
    plt.figure(figsize=(10, 12))
    all_dates, all_y_actual, all_y_predicted = [], [], []

    train_dates = df.index[train_indices]
    sorted_train_indices = np.argsort(train_dates)
    sorted_train_dates = train_dates[sorted_train_indices]
    sorted_y_train = y_train[sorted_train_indices]
    sorted_predictions = predictions[sorted_train_indices]
    all_dates.extend(sorted_train_dates)
    all_y_actual.extend(sorted_y_train)
    all_y_predicted.extend(sorted_predictions)

    if y_test is not None and test_predictions is not None and test_indices is not None:
        test_dates = df.index[test_indices]
        sorted_test_indices = np.argsort(test_dates)
        sorted_test_dates = test_dates[sorted_test_indices]
        sorted_y_test = y_test[sorted_test_indices]
        sorted_test_predictions = test_predictions[sorted_test_indices]
        all_dates.extend(sorted_test_dates)
        all_y_actual.extend(sorted_y_test)
        all_y_predicted.extend(sorted_test_predictions)

    plt.subplot(3, 1, 1)
    plt.plot(all_dates, all_y_actual, label='Actual', color='blue')
    plt.plot(all_dates, all_y_predicted, label='Predicted', color='green', alpha=0.7)
    plt.title('BTC Log Return Prediction')
    plt.legend()
    plt.xticks(rotation=45)

    capital = [1000]
    for i in range(1, len(all_y_actual)):
        ret = (all_y_actual[i] - all_y_actual[i-1])
        pos = 1 if all_y_predicted[i] > 0 else -1
        capital.append(capital[-1] * (1 + (ret * pos)))
    plt.subplot(3, 1, 2)
    plt.plot(all_dates, capital, color='purple')
    plt.title('Strategy Capital (Long/Short)')
    plt.xticks(rotation=45)

    plt.subplot(3, 1, 3)
    plt.semilogy(history_loss, label='Train Loss', color='blue')
    plt.semilogy(history_val_loss, label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Huber Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def run_training_task():
    global training_state
    K.clear_session()
    try:
        with state_lock:
            training_state['status'] = 'training'
            training_state['train_loss'] = []
            training_state['val_loss'] = []

        df = load_data()
        features, targets_scaled, _, scaler_target = prepare_data(df)

        split_idx = int(len(features)*0.7)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = targets_scaled[:split_idx], targets_scaled[split_idx:]
        train_indices = list(range(40, 40+split_idx))

        X_train = X_train.reshape(X_train.shape[0], 12, -1)
        X_test = X_test.reshape(X_test.shape[0], 12, -1)

        EPOCHS = 500
        UNITS = 256
        REG_RATE = 1e-4

        with state_lock:
            training_state['total_epochs'] = EPOCHS

        model = Sequential()
        model.add(LSTM(UNITS, activation='relu', return_sequences=True,
                       input_shape=(12, X_train.shape[2]), kernel_regularizer=l2(REG_RATE)))
        model.add(Dropout(0.5))
        model.add(LSTM(UNITS, activation='relu', return_sequences=True, kernel_regularizer=l2(REG_RATE)))
        model.add(Dropout(0.5))
        model.add(LSTM(UNITS, activation='relu', kernel_regularizer=l2(REG_RATE)))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=Huber())

        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=128,
            verbose=0,
            validation_data=(X_test, y_test),
            callbacks=[ThrottledProgressCallback(EPOCHS, update_frequency=5), early_stopping]
        )

        train_predictions_scaled = model.predict(X_train, verbose=0)
        test_predictions_scaled = model.predict(X_test, verbose=0)

        train_predictions_real = scaler_target.inverse_transform(train_predictions_scaled).flatten()
        y_train_real = scaler_target.inverse_transform(y_train).flatten()
        test_predictions_real = scaler_target.inverse_transform(test_predictions_scaled).flatten()
        y_test_real = scaler_target.inverse_transform(y_test).flatten()
        test_indices = list(range(40+split_idx, 40+split_idx+len(y_test_real)))

        plot_url = create_plot(df, y_train_real, train_predictions_real, train_indices,
                               history.history['loss'], history.history['val_loss'],
                               y_test_real, test_predictions_real, test_indices)

        mae_val = mean_absolute_error(y_test_real, test_predictions_real)
        smape_val = smape(y_test_real, test_predictions_real)
        resid = y_test_real - test_predictions_real
        resid_std = np.std(resid)
        resid_auto = acf(resid, fft=True, nlags=1)[1]

        with state_lock:
            training_state['plot_url'] = plot_url
            training_state['train_mse'] = sanitize_float(history.history['loss'][-1])
            training_state['mae'] = sanitize_float(mae_val)
            training_state['smape'] = sanitize_float(smape_val)
            training_state['resid_std'] = sanitize_float(resid_std)
            training_state['resid_auto'] = sanitize_float(resid_auto)
            training_state['status'] = 'completed'

    except Exception as e:
        logger.error(f"Training failed: {e}")
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
