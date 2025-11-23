from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    df = pd.read_csv('btc_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Prepare features and target
def prepare_data(df):
    # Calculate specified SMAs
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
    
    # Remove rows with NaN values from SMA calculations
    df_clean = df.dropna()
    
    features = []
    targets = []
    for i in range(len(df_clean)):
        # Features: 7-day and 365-day price SMAs, 5-day and 10-day volume SMAs
        feature = [
            df_clean['sma_7'].iloc[i],
            df_clean['sma_365'].iloc[i],
            df_clean['volume_sma_5'].iloc[i],
            df_clean['volume_sma_10'].iloc[i]
        ]
        features.append(feature)
        # Target: next day's closing price
        if i < len(df_clean) - 3:  # Reduced lookback to 3 days
            target = df_clean['close'].iloc[i + 3]
            targets.append(target)
    
    # Remove the last 3 features since they have no corresponding target
    features = features[:-3]
    return np.array(features), np.array(targets)

# Train model
def train_model(features, targets, df):
    # Filter data for training set: 2022 to 2023
    train_start_date = pd.to_datetime('2022-01-01')
    train_end_date = pd.to_datetime('2023-12-31')
    train_mask = (df.index >= train_start_date) & (df.index <= train_end_date)
    train_df = df[train_mask]
    # Filter data for test set: 2024 to 2025
    test_start_date = pd.to_datetime('2024-01-01')
    test_end_date = pd.to_datetime('2025-12-31')
    test_mask = (df.index >= test_start_date) & (df.index <= test_end_date)
    test_df = df[test_mask]
    # Map filtered indices to original features and targets
    train_indices = df.index[train_mask].tolist()
    test_indices = df.index[test_mask].tolist()
    # Align features and targets with filtered indices
    # Since features and targets are derived from df_clean, we need to align with filtered DataFrames
    # This is a simplification; in practice, ensure indices match
    X_train = features[:len(train_indices)]
    X_test = features[len(train_indices):len(train_indices) + len(test_indices)]
    y_train = targets[:len(train_indices)]
    y_test = targets[len(train_indices):len(train_indices) + len(test_indices)]
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, X_test, y_test, predictions, mse, test_indices

# Generate plot
def create_plot(df, y_test, predictions, test_indices):
    plt.figure(figsize=(10, 8))
    dates = df.index[test_indices]
    # Sort by date to ensure chronological plotting
    sorted_indices = np.argsort(dates)
    sorted_dates = dates[sorted_indices]
    sorted_y_test = y_test[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    
    # Calculate capital with daily accumulation based on prediction vs actual and actual Bitcoin returns
    capital = [1000]  # Start with $1000
    sma_capital = [1000]  # Start with $1000 for SMA strategy
    
    for i in range(len(sorted_y_test)):
        if i == 0:
            # For the first day, use the price from the day before the test period starts
            prev_index = test_indices[sorted_indices[i]] - 1
            prev_price = df['close'].iloc[prev_index] if prev_index >= 0 else sorted_y_test[i]
        else:
            prev_price = sorted_y_test[i - 1]
        
        actual_return = (sorted_y_test[i] - prev_price) / prev_price
        
        # ML Strategy
        if sorted_predictions[i] > sorted_y_test[i]:  # Prediction above actual: negative actual return
            ret = -actual_return
        else:  # Prediction below actual: positive actual return
            ret = actual_return
        capital.append(capital[-1] * (1 + ret))
        
        # 365-day SMA Strategy
        current_date = sorted_dates[i]
        sma_365 = df.loc[current_date, 'sma_365']
        current_price = sorted_y_test[i]
        
        if current_price > sma_365:
            # Go long: positive actual return
            sma_ret = actual_return
        else:
            # Go short: negative actual return
            sma_ret = -actual_return
        sma_capital.append(sma_capital[-1] * (1 + sma_ret))
    
    capital = capital[1:]  # Remove the initial 1000 to match the number of dates
    sma_capital = sma_capital[1:]  # Remove the initial 1000 to match the number of dates
    
    # Plot price and predictions
    plt.subplot(2, 1, 1)
    plt.plot(sorted_dates, sorted_y_test, label='Actual Price', color='blue')
    plt.plot(sorted_dates, sorted_predictions, label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('BTC Price Prediction vs Actual')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot capital
    plt.subplot(2, 1, 2)
    plt.plot(sorted_dates, capital, label='ML Strategy Capital', color='green')
    plt.plot(sorted_dates, sma_capital, label='365-day SMA Strategy Capital', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Capital (USD)')
    plt.title('Trading Strategy Capital Comparison')
    plt.legend()
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
    features, targets = prepare_data(df)
    model, X_test, y_test, predictions, mse, test_indices = train_model(features, targets, df)
    plot_url = create_plot(df, y_test, predictions, test_indices)
    return render_template('index.html', plot_url=plot_url, mse=mse)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
