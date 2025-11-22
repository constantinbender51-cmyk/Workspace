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

app = Flask(__name__)

# Load and preprocess data
def load_data():
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
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, X_test, y_test, predictions, mse

# Generate plot
def create_plot(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Price', color='blue')
    plt.plot(predictions, label='Predicted Price', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Price (USD)')
    plt.title('BTC Price Prediction vs Actual')
    plt.legend()
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
    model, X_test, y_test, predictions, mse = train_model(features, targets)
    plot_url = create_plot(y_test, predictions)
    return render_template('index.html', plot_url=plot_url, mse=mse)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)