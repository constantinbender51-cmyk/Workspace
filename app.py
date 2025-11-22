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
def train_model(features, targets):
    df = load_data()
    df_clean = df.dropna()  # Cleaned data used for features/targets
    # Define date ranges
    start_date = pd.Timestamp('2022-01-01')
    end_date = pd.Timestamp('2023-09-23')
    # Mask for the range from January 2022 to September 23, 2023
    mask_range = (df_clean.index >= start_date) & (df_clean.index <= end_date)
    df_range = df_clean.loc[mask_range]
    total_in_range = len(df_range)
    split_idx = total_in_range // 2  # 50% split
    # Find start index in df_clean for the range
    start_idx = df_clean.index.get_loc(df_range.index[0])
    # Training set: first 50% of the range
    X_train = features[start_idx:start_idx + split_idx]
    y_train = targets[start_idx:start_idx + split_idx]
    # First test set: second 50% of the range
    X_test1 = features[start_idx + split_idx:start_idx + total_in_range]
    y_test1 = targets[start_idx + split_idx:start_idx + total_in_range]
    test1_indices = list(range(start_idx + split_idx, start_idx + total_in_range))
    # Second test set: from September 23, 2023 onward
    mask_after = df_clean.index > end_date
    df_after = df_clean.loc[mask_after]
    if len(df_after) > 0:
        after_start_idx = df_clean.index.get_loc(df_after.index[0])
        X_test2 = features[after_start_idx:after_start_idx + len(df_after)]
        y_test2 = targets[after_start_idx:after_start_idx + len(df_after)]
        test2_indices = list(range(after_start_idx, after_start_idx + len(y_test2)))
    else:
        X_test2 = np.array([])
        y_test2 = np.array([])
        test2_indices = []
    # Train model on training set
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predictions for first test set
    predictions1 = model.predict(X_test1)
    mse1 = mean_squared_error(y_test1, predictions1) if len(y_test1) > 0 else None
    # Predictions for second test set
    predictions2 = model.predict(X_test2) if len(X_test2) > 0 else np.array([])
    mse2 = mean_squared_error(y_test2, predictions2) if len(y_test2) > 0 else None
    # Return model, test sets, predictions, and MSEs
    return model, X_test1, y_test1, predictions1, mse1, test1_indices, X_test2, y_test2, predictions2, mse2, test2_indices

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
    model, X_test1, y_test1, predictions1, mse1, test1_indices, X_test2, y_test2, predictions2, mse2, test2_indices = train_model(features, targets)
    # For simplicity, plot only the first test set; adjust if needed for second test set
    plot_url = create_plot(df, y_test1, predictions1, test1_indices)
    return render_template('index.html', plot_url=plot_url, mse=mse1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
