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
    # We train the model to predict t+1 based on t
    df['Target_Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

def run_strategy():
    raw_df = fetch_binance_data()
    if raw_df.empty: return None, "No Data"
    
    df = prepare_data(raw_df)
    
    # X (Features) at time t-1
    X = df[['Close', 'Volume', 'SMA_7', 'SMA_365', 'Volume_SMA_5', 'Volume_SMA_10']]
    # y (Target) is Close at time t
    y = df['Target_Next_Close']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train the model
    # The model learns: "Given yesterday's data, what is today's close?"
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    # X_test contains data from "Yesterday". 
    # The model uses X_test to predict "Today's Close".
    predictions_today = model.predict(X_test)
    
    # --- ALIGNMENT ---
    results = pd.DataFrame(index=X_test.index)
    results['Date'] = df['Open Time'].loc[X_test.index] # Date of X (Yesterday)
    # Shift date by 1 day so the row represents "Today"
    results['Date'] = results['Date'] + pd.Timedelta(days=1)
    
    results['Actual_Today'] = y_test            # This is the actual Close of Today
    results['Predicted_Today'] = predictions_today # This is the Model's Prediction for Today
    
    # Calculate Today's Market Return (Today Close vs Yesterday Close)
    # We need Yesterday's Close to calculate the percentage move
    results['Yesterday_Close'] = X_test['Close']
    results['Daily_Return'] = (results['Actual_Today'] - results['Yesterday_Close']) / results['Yesterday_Close']
    
    positions = []
    strategy_returns = []
    
    for i, row in results.iterrows():
        actual = row['Actual_Today']
        predicted = row['Predicted_Today']
        market_ret = row['Daily_Return']
        
        # --- YOUR SPECIFIED LOGIC ---
        # "Calculate the models prediction of today and compare it to the close of today"
        
        # 1. If Price (Today) < Prediction (Today)
        # "If the price is below the predicted value -> calculate negative return"
        if actual < predicted:
            positions.append('Short')
            strategy_returns.append(-1 * market_ret)
            
        # 2. If Price (Today) > Prediction (Today)
        # "If the price is above the predicted value -> calculate positive return"
        elif actual > predicted:
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
    plt.plot(results['Date'], results['Actual_Today'], label='Actual Price (Today)', color='blue')
    plt.plot(results['Date'], results['Predicted_Today'], label='Predicted Price (Today)', color='orange', linestyle='--')
    plt.title('Actual vs Predicted (Same Day Comparison)')
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
    rmse = np.sqrt(mean_squared_error(y_test, predictions_today))
    
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
                        <strong>Strict Comparison Logic:</strong>
                        <ul style="margin-top: 5px;">
                            <li><b>Prediction:</b> What the model thought Today's Close would be (based on yesterday).</li>
                            <li><b>Actual:</b> What Today's Close actually is.</li>
                            <li>If <b>Actual < Predicted</b>: Return = Negative Daily Return (Short).</li>
                            <li>If <b>Actual > Predicted</b>: Return = Positive Daily Return (Long).</li>
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
