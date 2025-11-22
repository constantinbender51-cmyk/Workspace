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
    lookback = 5
    features = []
    targets = []
    for i in range(lookback, len(df)):
        # Features: price and volume for the last 5 days
        price_window = df['close'].iloc[i-lookback:i].values
        volume_window = df['volume'].iloc[i-lookback:i].values
        feature = np.concatenate([price_window, volume_window])
        features.append(feature)
        # Target: next day's closing price
        target = df['close'].iloc[i]
        targets.append(target)
    return np.array(features), np.array(targets)

# Train model
def train_model(features, targets):
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        features, targets, range(len(targets)), test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, X_test, y_test, predictions, mse, test_indices

# Generate plot
def create_plot(df, y_test, predictions, test_indices):
    plt.figure(figsize=(10, 6))
    dates = df.index[test_indices]
    plt.plot(dates, y_test, label='Actual Price', color='blue')
    plt.plot(dates, predictions, label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('BTC Price Prediction vs Actual')
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
    model, X_test, y_test, predictions, mse, test_indices = train_model(features, targets)
    plot_url = create_plot(df, y_test, predictions, test_indices)
    return render_template('index.html', plot_url=plot_url, mse=mse)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)