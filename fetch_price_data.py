import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Configuration for on-chain metrics
BASE_URL = "https://api.blockchain.info/charts/"
METRICS = {
    'Active_Addresses': 'n-unique-addresses',
    'Net_Transaction_Count': 'n-transactions',
    'Transaction_Volume_USD': 'estimated-transaction-volume-usd',
}
START_DATE = '2022-01-01'
END_DATE = '2023-09-30'

def fetch_chart_data(chart_name, start_date):
    """
    Fetches historical data for a single chart from the Blockchain.com API.
    """
    params = {
        'format': 'json',
        'start': start_date,
        'timespan': 'all',
        'sampled': 'false'
    }
    url = f"{BASE_URL}{chart_name}"
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if 'values' not in data or not data['values']:
            return pd.DataFrame()
        df = pd.DataFrame(data['values'])
        df['Date'] = pd.to_datetime(df['x'], unit='s', utc=True).dt.tz_localize(None)
        df = df.set_index('Date')['y'].rename(chart_name)
        return df
    except Exception as e:
        print(f"Error fetching {chart_name}: {e}")
        return pd.DataFrame()

def fetch_btc_candles():
    base_url = 'https://api.binance.com/api/v3/klines'
    symbol = 'BTCUSDT'
    interval = '1d'
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 9, 30)
    
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
                current_start = datetime.fromtimestamp(data[-1][0] / 1000) + timedelta(days=1)
            else:
                current_start += timedelta(days=1000)
        else:
            print(f"Error fetching price data: {response.status_code}")
            break
        
        time.sleep(0.1)
    
    df_price = pd.DataFrame(all_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    # Fetch on-chain metrics
    all_metrics = []
    for metric_name, chart_endpoint in METRICS.items():
        time.sleep(1.5)
        df_metric = fetch_chart_data(chart_endpoint, START_DATE)
        if not df_metric.empty:
            df_metric = df_metric.rename(metric_name)
            all_metrics.append(df_metric)
    
    # Combine price and on-chain data
    if all_metrics:
        df_combined = pd.concat([df_price.set_index('date')] + all_metrics, axis=1)
        df_combined = df_combined.loc[START_DATE:END_DATE].ffill().dropna()
    else:
        df_combined = df_price.set_index('date')
        df_combined = df_combined.loc[START_DATE:END_DATE].dropna()
    df_combined.reset_index(inplace=True)
    df_combined.rename(columns={'index': 'date'}, inplace=True)
    df_combined.to_csv('btc_data.csv', index=False)
    print("Data fetched and saved to btc_data.csv")

if __name__ == '__main__':
    fetch_btc_candles()
