import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_TIME = 1640995200000 # Jan 1 2022
END_TIME = 1696032000000   # Sep 30 2023
INITIAL_CAPITAL = 1000

def fetch_binance_data():
    """Fetches daily candles from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    data = []
    current_start = START_TIME
    
    while current_start < END_TIME:
        params = {
            'symbol': SYMBOL, 'interval': INTERVAL,
            'startTime': current_start, 'endTime': END_TIME, 'limit': 1000
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            candles = resp.json()
            if not candles: break
            data.extend(candles)
            current_start = candles[-1][6] + 1
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'QA Vol', 'Trades', 'Taker Base', 'Taker Quote', 'Ign'
    ])
    
    numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric] = df[numeric].apply(pd.to_numeric, axis=1)
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    return df[['Open Time', 'Close', 'Volume']]

def prepare_data(df):
    """Calculates SMAs and sets targets."""
    df = df.copy()
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_365'] = df['Close'].rolling(window=365).mean()
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    df.dropna(inplace=True)
    
    # Target: The Close price of the NEXT day
    df['Target_Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

def run_backtest():
    raw_df = fetch_binance_data()
    if raw_df.empty: return None, "No Data"
    
    df = prepare_data(raw_df)
    
    # Features (X) and Target (y)
    X = df[['Close', 'Volume', 'SMA_7', 'SMA_365', 'Volume_SMA_5', 'Volume_SMA_10']]
    y = df['Target_Next_Close']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Backtest Logic
    results = pd.DataFrame(index=X_test.index)
    results['Date'] = df['Open Time'].loc[X_test.index]
    results['Actual_Price'] = X_test['Close']        # Today's Price
    results['Next_Actual_Price'] = y_test            # Tomorrow's Price
    results['Predicted_Next_Price'] = predictions    # Prediction for Tomorrow
    
    # Calculate Market Return (Tomorrow - Today) / Today
    results['Market_Return'] = (results['Next_Actual_Price'] - results['Actual_Price']) / results['Actual_Price']
    
    strategy_returns = []
    positions = []
    
    for i, row in results.iterrows():
        current_price = row['Actual_Price']
        predicted_price = row['Predicted_Next_Price']
        market_ret = row['Market_Return']
        
        # --- USER SPECIFIED LOGIC ---
        if current_price < predicted_price:
            # "If price is below predicted value -> Negative return (Short)"
            positions.append('Short')
            strategy_returns.append(-1 * market_ret)
        elif current_price > predicted_price:
            # "If price is above predicted value -> Positive return (Long)"
            positions.append('Long')
            strategy_returns.append(1 * market_ret)
        else:
            positions.append('Neutral')
            strategy_returns.append(0)
            
    results['Strategy_Return'] = strategy_returns
    results['Position'] = positions
    
    # Equity Curve
    results['Equity'] = INITIAL_CAPITAL * (1 + results['Strategy_Return']).cumprod()
    results['Benchmark'] = INITIAL_CAPITAL * (1 + results['Market_Return']).cumprod()
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['Date'], results['Equity'], 'g-', label='Your Strategy')
    plt.plot(results['Date'], results['Benchmark'], 'k--', alpha=0.5, label='Buy & Hold')
    plt.title(f'Strategy Performance (Initial: ${INITIAL_CAPITAL})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(results['Date'], results['Next_Actual_Price'], label='Actual Price', color='blue', alpha=0.6)
    plt.plot(results['Date'], results['Predicted_Next_Price'], label='Predicted', color='orange', alpha=0.6, linestyle='--')
    plt.title('Price vs Prediction')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    final_cap = results['Equity'].iloc[-1]
    rmse = np.sqrt(mean_squared_error(y_test, predictions)) # Fixed syntax
    
    return plot_url, round(final_cap, 2), round(rmse, 2)

@app.route('/')
def index():
    try:
        plot, cap, error = run_backtest()
        return f"""
        <html>
            <body style="font-family: sans-serif; text-align: center; padding: 20px;">
                <h1>Trading Bot Dashboard</h1>
                <div style="display: flex; justify-content: center; gap: 40px; margin-bottom: 20px;">
                    <div><h2>${cap}</h2><p>Final Capital</p></div>
                    <div><h2>{error}</h2><p>RMSE Score</p></div>
                </div>
                <img src="data:image/png;base64,{plot}" style="max-width: 100%; border: 1px solid #ccc;" />
                <div style="margin-top: 20px; padding: 10px; background: #f0f0f0; display: inline-block; text-align: left;">
                    <strong>Logic Applied:</strong><br>
                    - If Current Price < Predicted: <b>SHORT</b> (Negative Return)<br>
                    - If Current Price > Predicted: <b>LONG</b> (Positive Return)
                </div>
            </body>
        </html>
        """
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
