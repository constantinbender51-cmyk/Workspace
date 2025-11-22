import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_btc_candles():
    base_url = 'https://api.binance.com/api/v3/klines'
    symbol = 'BTCUSDT'
    interval = '1d'
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 11, 25)
    
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        start_time = int(current_date.timestamp() * 1000)
        # Fetch up to 1000 days at once
        end_time = int((current_date + timedelta(days=1000)).timestamp() * 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        response = requests.get(base_url, params=params)
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
                # Move to the day after the last fetched date
                last_timestamp = data[-1][0]
                current_date = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(days=1)
            else:
                current_date += timedelta(days=1)
        else:
            print(f"Error fetching data: {response.status_code}")
            break
        
        time.sleep(0.1)  # Rate limiting
    
    df = pd.DataFrame(all_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df.to_csv('btc_data.csv', index=False)
    print("Data fetched and saved to btc_data.csv")

if __name__ == '__main__':
    fetch_btc_candles()
