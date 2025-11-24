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

# -----------------------------------------------------------------------------
# 1. DATA ACQUISITION
# -----------------------------------------------------------------------------

def get_binance_data(symbol="BTCUSDT", interval="1d", limit=365):
    """
    Fetches historical kline data from Binance public API.
    """
    print(f"Fetching {symbol} data from Binance...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Calculate timestamps for the last 'limit' days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=limit + 30) # Buffer for SMA calculation
    
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
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        sys.exit(1)
        
    # Columns: Open Time, Open, High, Low, Close, Volume, ...
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = df["close"].astype(float)
    
    # Set index to date (remove time component for merging)
    df["date"] = df["timestamp"].dt.normalize()
    return df[["date", "close"]]

def get_blockchain_data(chart_name, timespan="1year"):
    """
    Fetches chart data from Blockchain.com public API.
    """
    print(f"Fetching {chart_name} from Blockchain.com...")
    url = f"https://api.blockchain.info/charts/{chart_name}?timespan={timespan}&format=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching Blockchain.com data: {e}")
        sys.exit(1)
        
    values = data["values"]
    df = pd.DataFrame(values)
    df.rename(columns={"x": "timestamp", "y": chart_name}, inplace=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["date"] = df["timestamp"].dt.normalize()
    
    # Handle duplicates if multiple points per day, take mean
    df = df.groupby("date")[chart_name].mean().reset_index()
    return df

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def prepare_dataset():
    # 1. Fetch
    btc_df = get_binance_data(limit=365)
    hash_df = get_blockchain_data("hash-rate")
    diff_df = get_blockchain_data("difficulty")
    
    # 2. Merge
    df = btc_df.merge(hash_df, on="date", how="inner")
    df = df.merge(diff_df, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    
    # 3. Calculate Log Returns
    # R_t = ln(P_t / P_{t-1})
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    
    # 4. Calculate Target: SMA 7 of Log Returns
    df["target_sma7"] = df["log_ret"].rolling(window=7).mean()
    
    # 5. Drop NaNs created by shifting/rolling
    df.dropna(inplace=True)
    
    print(f"Dataset prepared. Shape: {df.shape}")
    return df

# -----------------------------------------------------------------------------
# 3. MODEL DEFINITION
# -----------------------------------------------------------------------------

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # No dropout to keep the model 'pure' for double descent observation
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last time step output
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
    # --- Config ---
    SEQ_LENGTH = 14
    EPOCHS = 2000  # High epochs to ensure we hit the interpolation regime
    LEARNING_RATE = 0.001
    # We vary hidden size to sweep through under-parameterized -> peak -> over-parameterized
    HIDDEN_SIZES = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 45, 64]
    
    # --- Data Prep ---
    df = prepare_dataset()
    
    # Features: Log Return, Hash Rate, Difficulty
    feature_cols = ["log_ret", "hash-rate", "difficulty"]
    target_col = ["target_sma7"]
    
    # Normalize
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_x.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[target_col])
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
    
    # Add noise to training labels to accentuate the double descent peak
    # (Common technique in Double Descent literature to make the effect visible on small data)
    noise_level = 0.05
    
    # Split
    # We use a random split here rather than time-series split to strictly demonstrate 
    # the capacity phenomenon (bias-variance trade-off) without distribution shift interference.
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    # Add noise ONLY to training targets
    y_train_noisy = y_train + np.random.normal(0, noise_level, y_train.shape)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train_noisy)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    train_losses = []
    test_losses = []
    
    print("\nStarting Double Descent Experiment...")
    print(f"{'Hidden Size':<15} | {'Train MSE':<15} | {'Test MSE':<15}")
    print("-" * 50)
    
    for h_size in HIDDEN_SIZES:
        model = SimpleLSTM(input_size=len(feature_cols), hidden_size=h_size)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # Training Loop
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t)
            test_pred = model(X_test_t)
            
            # Calculate final MSE (using clean y_train for metric logging if preferred, 
            # but standard is to measure against what it saw. We'll measure against clean to see generalization)
            train_mse = criterion(train_pred, torch.FloatTensor(y_train)).item() 
            test_mse = criterion(test_pred, y_test_t).item()
            
        train_losses.append(train_mse)
        test_losses.append(test_mse)
        
        print(f"{h_size:<15} | {train_mse:.6f}          | {test_mse:.6f}")

    # -----------------------------------------------------------------------------
    # 5. VISUALIZATION
    # -----------------------------------------------------------------------------
    
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
    max_test_loss = max(test_losses)
    max_idx = test_losses.index(max_test_loss)
    peak_x = HIDDEN_SIZES[max_idx]
    
    plt.annotate('Interpolation Threshold\n(Peak Overfitting)', 
                 xy=(peak_x, max_test_loss), 
                 xytext=(peak_x, max_test_loss * 1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center')
                 
    plt.tight_layout()
    
    # Save plot and start server
    plot_filename = "double_descent_plot.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    
    # Create a simple HTML viewer
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
    
    # Allow port reuse to prevent 'Address already in use' errors
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    print(f"\nStarting web server on 0.0.0.0:{PORT}...")
    print(f"View the result at http://localhost:{PORT}")
    
    with ReusableTCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    # Check for library availability
    try:
        import torch
    except ImportError:
        print("Please install torch: pip install torch")
        sys.exit(1)
        
    run_experiment()
