import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

def fetch_btc_candles():
    base_url = 'https://api.binance.com/api/v3/klines'
    symbol = 'BTCUSDT'
    interval = '1d'
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    all_data = []
    current_start = start_date
    
    while current_start <= end_date:
        start_time = int(current_start.timestamp() * 1000)
        end_time = int((current_start + timedelta(days=1000)).timestamp() * 1000)
        if end_time > int(end_date.timestamp() * 1000):
            end_time = int(end_date.timestamp() * 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        headers = {'User-Agent': 'BTC-Prediction-App/1.0'}
        
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data:
                for candle in data:
                    timestamp = candle[0]
                    open_price = float(candle[1])
                    high = float(candle[2])
                    low = float(candle[3])
                    close = float(candle[4])
                    volume = float(candle[5])
                    date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                    all_data.append([date, open_price, high, low, close, volume])
                current_start = datetime.fromtimestamp(data[-1][0] / 1000) + timedelta(days=1)
            else:
                current_start += timedelta(days=1000)
        else:
            print(f"Error fetching data: {response.status_code}")
            break
        
        time.sleep(0.15)
    
    df = pd.DataFrame(all_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # --- Feature Engineering ---
    df['log_return'] = np.log(df['close']).diff()
    df['volatility_7d'] = df['log_return'].rolling(window=7).std()
    df['return_lag1'] = df['log_return'].shift(1)
    df['return_lag3'] = df['log_return'].shift(3)
    df['return_lag7'] = df['log_return'].shift(7)
    df['bollinger_bandwidth'] = (df['close'].rolling(20).std() / df['close'].rolling(20).mean())

    df.to_csv('btc_data.csv')
    print("Data fetched and saved to btc_data.csv with engineered features")

if __name__ == '__main__':
    fetch_btc_candles()
