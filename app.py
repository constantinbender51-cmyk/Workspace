import requests
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
from flask import Flask

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_TIME = 1640995200000 
END_TIME = 1696032000000 
INITIAL_CAPITAL = 1000

def fetch_binance_data():
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
        except:
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
    df = df.copy()
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_365'] = df['Close'].rolling(window=365).mean()
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    df.dropna(inplace=True)
    
    # Target: The Close price of the NEXT day
    # When we are at index `t`, we want to predict Close at `t+1`
    df['Target_Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

def run_strategy():
    raw_df = fetch_binance_data()
    if raw_df.empty: return None, "No Data"
    
    df = prepare_data(raw_df)
    
    # X (Features) at time t
    X = df[['Close', 'Volume', 'SMA_7', 'SMA_365', 'Volume_SMA_5', 'Volume_SMA_10']]
    # y (Target) is Close at time t+1
    y = df['Target_Next_Close']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    # These are predictions generated using X_test (data at t)
    # So these predict the price at t+1
    predictions = model.predict(X_test)
    
    # --- ALIGNMENT ---
    # We construct the results dataframe to represent "The Next Day" (t+1)
    results = pd.DataFrame(index=X_test.index)
    results['Date'] = df['Open Time'].loc[X_test.index] # This is date t
    # To make it clear, let's shift the date for plotting so it shows the "Result Day"
    # But for calculation, we just need the prices aligned.
    
    results['Yesterday_Close'] = X_test['Close']   # Price at t (Basis for return)
    results['Today_Close'] = y_test                # Price at t+1 (The Target)
    results['Today_Predicted'] = predictions       # Prediction for t+1 (The Forecast)
    
    # Market Return for "Today" (t+1)
    results['Daily_Return'] = (results['Today_Close'] - results['Yesterday_Close']) / results['Yesterday_Close']
    
    positions = []
    strategy_returns = []
    
    for i, row in results.iterrows():
        actual = row['Today_Close']
        predicted = row['Today_Predicted']
        market_ret = row['Daily_Return']
        
        # --- YOUR SPECIFIED LOGIC ---
        # "If the price [Today_Close] is below the predicted value [Today_Predicted]"
        if actual < predicted:
            # "...calculate the return of today the negative return of Bitcoin" (Short)
            positions.append('Short')
            strategy_returns.append(-1 * market_ret)
            
        # "If the price [Today_Close] is above the predicted value [Today_Predicted]"
        elif actual > predicted:
            # "...calculate the return is the positive return of Bitcoin" (Long)
            positions.append('Long')
            strategy_returns.append(1 * market_ret)
        else:
            positions.append('Neutral')
            strategy_returns.append(0)
            
    results['Strategy_Return'] = strategy_returns
    
    # Capital Accumulation
    results['Equity'] = INITIAL_CAPITAL * (1 + results['Strategy_Return']).cumprod()
    results['Benchmark'] = INITIAL_CAPITAL * (1 + results['Daily_Return']).cumprod()
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(results['Date'], results['Equity'], 'g-', linewidth=2, label='Strategy Equity')
    plt.plot(results['Date'], results['Benchmark'], 'k--', alpha=0.5, label='Buy & Hold')
    plt.title(f'Strategy Performance (Initial: ${INITIAL_CAPITAL})')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Price vs Prediction
    plt.subplot(2, 1, 2)
    plt.plot(results['Date'], results['Today_Close'], label='Actual Price (Today)', color='blue')
    plt.plot(results['Date'], results['Today_Predicted'], label='Predicted Price (For Today)', color='orange', linestyle='--')
    plt.title('Actual Close vs Predicted Close')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    final_cap = results['Equity'].iloc[-1]
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return plot_url, round(final_cap, 2), round(rmse, 2)

@app.route('/')
def index():
    try:
        plot, cap, error = run_strategy()
        return f"""
        <html>
            <head><title>Algo Dashboard</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 20px; background-color: #f4f4f9;">
                <div style="background: white; max-width: 900px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h1 style="color: #333;">Strategy Results</h1>
                    
                    <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
                        <div style="background: #e0f7fa; padding: 15px; border-radius: 8px; width: 40%;">
                            <h2 style="margin: 0; color: #006064;">${cap}</h2>
                            <p style="margin: 5px 0 0;">Final Capital</p>
                        </div>
                        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; width: 40%;">
                            <h2 style="margin: 0; color: #e65100;">{error}</h2>
                            <p style="margin: 5px 0 0;">RMSE Score</p>
                        </div>
                    </div>

                    <img src="data:image/png;base64,{plot}" style="width: 100%; border: 1px solid #ddd; border-radius: 4px;" />
                    
                    <div style="margin-top: 20px; text-align: left; background: #f9f9f9; padding: 15px; border-radius: 5px;">
                        <strong>Logic Used:</strong>
                        <ul style="margin-top: 5px;">
                            <li>Compares <b>Today's Actual Close</b> vs <b>Today's Predicted Close</b> (made yesterday).</li>
                            <li>If Actual < Predicted: <b>SHORT</b> (Return = Negative of Daily Move)</li>
                            <li>If Actual > Predicted: <b>LONG</b> (Return = Positive of Daily Move)</li>
                        </ul>
                    </div>
                </div>
            </body>
        </html>
        """
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
