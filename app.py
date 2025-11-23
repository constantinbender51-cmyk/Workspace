import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string

# Initialize Flask App
app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
# Timestamps for Jan 1, 2022 to Sep 30, 2023 (Milliseconds)
START_TIME = 1640995200000 
END_TIME = 1696032000000 
INITIAL_CAPITAL = 1000

def fetch_binance_data(symbol, interval, start, end):
    """
    Fetches historical kline data from Binance public API.
    Handles pagination as Binance limits to 1000 candles per call.
    """
    url = "https://api.binance.com/api/v3/klines"
    data = []
    
    current_start = start
    while current_start < end:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end,
            'limit': 1000
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data: {response.text}")
            break
        
        candles = response.json()
        if not candles:
            break
            
        data.extend(candles)
        # Update start time to the last candle's close time + 1ms
        current_start = candles[-1][6] + 1
        
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base', 'Taker Buy Quote', 'Ignore'
    ])
    
    # Type conversion
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    
    return df[['Open Time', 'Close', 'Volume']]

def prepare_data(df):
    """
    Calculates technical indicators and sets up features/targets.
    """
    df = df.copy()
    
    # 1. Technical Indicators
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_365'] = df['Close'].rolling(window=365).mean()
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    
    # Drop NaN values created by SMAs (especially SMA_365)
    df.dropna(inplace=True)
    
    # 2. Features and Target
    # Target: Next day's closing price (Shifted -1)
    # Note: We use Shift -1 to align "Today's Features" with "Tomorrow's Price"
    df['Target_Next_Close'] = df['Close'].shift(-1)
    
    # Drop the last row as it has no target
    df.dropna(inplace=True)
    
    return df

def run_strategy():
    """
    Main logic pipeline: Fetch -> Process -> Train -> Backtest -> Visualize
    """
    # 1. Fetch Data
    raw_df = fetch_binance_data(SYMBOL, INTERVAL, START_TIME, END_TIME)
    if raw_df.empty:
        return None, "Failed to fetch data from Binance."

    # 2. Process Data
    df = prepare_data(raw_df)
    
    if df.empty:
        return None, "Not enough data points after calculating SMAs (Check SMA_365)."

    # Define Features (X) and Target (y)
    feature_cols = ['Close', 'Volume', 'SMA_7', 'SMA_365', 'Volume_SMA_5', 'Volume_SMA_10']
    X = df[feature_cols]
    y = df['Target_Next_Close']

    # 3. Train/Test Split (80/20 Time-based)
    split_idx = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Date index for plotting
    test_dates = df['Open Time'].iloc[split_idx:]
    
    # 4. Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # 5. Backtesting Strategy
    # Create a results DataFrame for the test period
    results = pd.DataFrame(index=X_test.index)
    results['Date'] = test_dates
    results['Actual_Price'] = X_test['Close'] # Price at time t (Decision time)
    results['Next_Actual_Price'] = y_test     # Price at time t+1 (Result time)
    results['Predicted_Next_Price'] = predictions
    
    # Calculate Market Return (Percent change from t to t+1)
    results['Market_Return'] = (results['Next_Actual_Price'] - results['Actual_Price']) / results['Actual_Price']
    
    # Decision Rule (Per user prompt):
    # If prediction > actual price -> Short (Bet against market)
    # If prediction < actual price -> Long (Bet with market)
    # Note: "Actual Price" here refers to the price at the moment of decision (t)
    
    positions = []
    strategy_returns = []
    
    for i, row in results.iterrows():
        pred = row['Predicted_Next_Price']
        current = row['Actual_Price']
        mkt_ret = row['Market_Return']
        
        if pred > current:
            # Short Position
            # If market goes down (mkt_ret < 0), we make money. 
            # Short return approx = -1 * Market Return
            positions.append('Short')
            strategy_returns.append(-1 * mkt_ret)
        else:
            # Long Position
            # If market goes up (mkt_ret > 0), we make money.
            positions.append('Long')
            strategy_returns.append(mkt_ret)
            
    results['Position'] = positions
    results['Strategy_Return'] = strategy_returns
    
    # Capital Accumulation
    results['Equity_Curve'] = INITIAL_CAPITAL * (1 + results['Strategy_Return']).cumprod()
    results['Market_Curve'] = INITIAL_CAPITAL * (1 + results['Market_Return']).cumprod() # Benchmark

    # 6. Visualization
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Equity Curves
    plt.subplot(2, 1, 1)
    plt.plot(results['Date'], results['Equity_Curve'], label='Strategy Equity', color='green', linewidth=2)
    plt.plot(results['Date'], results['Market_Curve'], label='Buy & Hold (Benchmark)', color='gray', linestyle='--')
    plt.title(f'Algo Performance: BTC/USDT (Test Set)\nInitial Capital: ${INITIAL_CAPITAL}', fontsize=14)
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Price vs Prediction
    plt.subplot(2, 1, 2)
    plt.plot(results['Date'], results['Next_Actual_Price'], label='Actual Next Price', color='blue', alpha=0.7)
    plt.plot(results['Date'], results['Predicted_Next_Price'], label='Predicted Next Price', color='orange', alpha=0.7, linestyle=':')
    plt.title('Linear Regression Predictions vs Actual', fontsize=12)
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to Base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Metrics
    final_capital = results['Equity_Curve'].iloc[-1]
    total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    accuracy = mean_squared_error(y_test, predictions, squared=False)
    
    return plot_url, {
        "final_capital": round(final_capital, 2),
        "total_return": round(total_return, 2),
        "rmse": round(accuracy, 2),
        "trades": len(results)
    }

# --- Flask Routes ---

@app.route('/')
def dashboard():
    plot_url, stats = run_strategy()
    
    if plot_url is None:
        return f"<h3>Error: {stats}</h3>"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BTC Trading Algo Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: #eef2f7; padding: 15px; border-radius: 8px; text-align: center; }}
            .stat-value {{ font-size: 1.5em; font-weight: bold; color: #2980b9; }}
            .stat-label {{ color: #7f8c8d; font-size: 0.9em; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bitcoin Algo Strategy Dashboard</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${stats['final_capital']}</div>
                    <div class="stat-label">Final Capital</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['total_return']}%</div>
                    <div class="stat-label">Total Return</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['rmse']}</div>
                    <div class="stat-label">RMSE (Price Error)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['trades']}</div>
                    <div class="stat-label">Days Traded</div>
                </div>
            </div>

            <div style="text-align: center;">
                <img src="data:image/png;base64,{plot_url}" alt="Performance Chart">
            </div>
            
            <div style="margin-top: 20px; font-size: 0.9em; color: #666;">
                <p><strong>Configuration:</strong><br>
                Model: Linear Regression<br>
                Features: SMA_7, SMA_365, Vol_SMA_5, Vol_SMA_10<br>
                Strategy: Contrarian (Pred > Actual → Short, Pred < Actual → Long)<br>
                Source: Binance API (BTC/USDT 1d)</p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    # Listen on 0.0.0.0 for external access (required for Railway/Docker)
    app.run(host='0.0.0.0', port=8080)
