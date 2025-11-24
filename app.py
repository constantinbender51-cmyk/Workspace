import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import sys
import http.server
import socketserver
import os
import time

# -----------------------------------------------------------------------------
# 0. UTILS
# -----------------------------------------------------------------------------

def log(message):
    """Prints a message with a timestamp and forces flush."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

# -----------------------------------------------------------------------------
# 1. DATA ACQUISITION
# -----------------------------------------------------------------------------

def get_binance_data(symbol="BTCUSDT", interval="1d", limit=365):
    """
    Fetches historical kline data from Binance public API.
    """
    log(f"Fetching {symbol} data from Binance (Limit: {limit} days)...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=limit + 30)
    
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": 1000
    }
    
    try:
        log(f"Requesting URL: {base_url}")
        # Added timeout to prevent hanging
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        log(f"Successfully fetched {len(data)} records from Binance.")
    except Exception as e:
        log(f"Error fetching Binance data: {e}")
        sys.exit(1)
        
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["date"] = df["timestamp"].dt.normalize()
    return df[["date", "close"]]

def get_blockchain_data(chart_name, timespan="1year"):
    """
    Fetches chart data from Blockchain.com public API.
    """
    log(f"Fetching {chart_name} from Blockchain.com...")
    url = f"https://api.blockchain.info/charts/{chart_name}?timespan={timespan}&format=json"
    
    try:
        # Added timeout to prevent hanging
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        log(f"Successfully fetched {chart_name}. Data points: {len(data['values'])}")
    except Exception as e:
        log(f"Error fetching Blockchain.com data: {e}")
        sys.exit(1)
        
    values = data["values"]
    df = pd.DataFrame(values)
    df.rename(columns={"x": "timestamp", "y": chart_name}, inplace=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["date"] = df["timestamp"].dt.normalize()
    
    df = df.groupby("date")[chart_name].mean().reset_index()
    return df

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def prepare_dataset():
    log("--- Starting Data Pipeline ---")
    btc_df = get_binance_data(limit=365)
    hash_df = get_blockchain_data("hash-rate")
    diff_df = get_blockchain_data("difficulty")
    
    log("Merging datasets...")
    df = btc_df.merge(hash_df, on="date", how="inner")
    df = df.merge(diff_df, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["target_sma7"] = df["log_ret"].rolling(window=7).mean()
    
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped_len = initial_len - len(df)
    log(f"Dropped {dropped_len} NaN rows. Final Shape: {df.shape}")
    return df

# -----------------------------------------------------------------------------
# 3. MODEL DEFINITION
# -----------------------------------------------------------------------------

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(features, targets, seq_length=10):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:(i + seq_length)]
        y = targets[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# -----------------------------------------------------------------------------
# 4. MAIN EXPERIMENT LOOP
# -----------------------------------------------------------------------------

def run_experiment():
    log("Initializing experiment...")
    
    # --- Config ---
    SEQ_LENGTH = 14
    EPOCHS = 2000
    LEARNING_RATE = 0.001
    HIDDEN_SIZES = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 45, 64]
    
    df = prepare_dataset()
    
    feature_cols = ["log_ret", "hash-rate", "difficulty"]
    target_col = ["target_sma7"]
    
    log("Normalizing features...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_x.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[target_col])
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
    
    noise_level = 0.05
    log(f"Adding Gaussian noise (std={noise_level}) to training labels...")
    
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    y_train_noisy = y_train + np.random.normal(0, noise_level, y_train.shape)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train_noisy)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    train_losses = []
    test_losses = []
    
    log("\n=== Starting Double Descent Training Loop ===")
    print(f"{'Hidden Size':<15} | {'Train MSE':<15} | {'Test MSE':<15} | {'Time (s)':<10}", flush=True)
    print("-" * 65, flush=True)
    
    for h_size in HIDDEN_SIZES:
        log(f"Training Hidden Size: {h_size}") # Log start of specific model
        model_start_time = time.time()
        
        model = SimpleLSTM(input_size=len(feature_cols), hidden_size=h_size)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # More frequent logging (every 100 epochs)
            if (epoch + 1) % 100 == 0:
                 # Print a dot or small status update without newline to show liveness if needed, 
                 # or just log regularly. We'll stick to log() but maybe less verbose.
                 if (epoch + 1) % 1000 == 0:
                    log(f"  [Size {h_size}] Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t)
            test_pred = model(X_test_t)
            train_mse = criterion(train_pred, torch.FloatTensor(y_train)).item() 
            test_mse = criterion(test_pred, y_test_t).item()
            
        train_losses.append(train_mse)
        test_losses.append(test_mse)
        
        elapsed = time.time() - model_start_time
        print(f"{h_size:<15} | {train_mse:.6f}          | {test_mse:.6f}          | {elapsed:.2f}s", flush=True)

    # -----------------------------------------------------------------------------
    # 5. VISUALIZATION
    # -----------------------------------------------------------------------------
    
    log("Generating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(HIDDEN_SIZES, train_losses, 'o--', label='Train MSE (Generalization)', color='blue')
    plt.plot(HIDDEN_SIZES, test_losses, 'o-', label='Test MSE', color='red', linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Model Complexity (Hidden Size) - Log Scale')
    plt.ylabel('Mean Squared Error')
    plt.title('Double Descent in LSTM (BTC Log Return SMA7)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Annotate the "Peak"
    if test_losses:
        max_test_loss = max(test_losses)
        max_idx = test_losses.index(max_test_loss)
        peak_x = HIDDEN_SIZES[max_idx]
        
        plt.annotate('Interpolation Threshold\n(Peak Overfitting)', 
                    xy=(peak_x, max_test_loss), 
                    xytext=(peak_x, max_test_loss * 1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center')
                 
    plt.tight_layout()
    
    plot_filename = "double_descent_plot.png"
    plt.savefig(plot_filename)
    log(f"Plot saved to {plot_filename}")
    
    html_content = f"""
    <html>
        <head><title>Double Descent Experiment</title></head>
        <body style="font-family: sans-serif; text-align: center; padding: 20px;">
            <h1>Double Descent Phenomenon in LSTM</h1>
            <p>Training vs Test MSE across model complexity</p>
            <img src="{plot_filename}" style="max-width: 100%; height: auto; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </body>
    </html>
    """
    with open("index.html", "w") as f:
        f.write(html_content)

    PORT = 8080
    Handler = http.server.SimpleHTTPRequestHandler
    
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    print(f"\nStarting web server on 0.0.0.0:{PORT}...", flush=True)
    print(f"View the result at http://localhost:{PORT}", flush=True)
    
    with ReusableTCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        log("Please install torch: pip install torch")
        sys.exit(1)
        
    run_experiment()
