import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1d', limit=2000):
    """
    Fetch OHLCV data from Binance API
    
    Args:
        symbol (str): Trading pair symbol (default: BTCUSDT)
        interval (str): Kline interval (default: 1d)
        limit (int): Number of data points to fetch (default: 2000)
    
    Returns:
        pandas.DataFrame: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    base_url = 'https://api.binance.com/api/v3/klines'
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        response = requests.get(base_url, params=params, timeout=30)
        data = response.json()
        response.raise_for_status()
        
        
        # Convert to DataFrame
        # Check if we got valid data
        df = pd.DataFrame(data, columns=[
        if not data or not isinstance(data, list):
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            print(f"Error: Invalid response data from Binance API")
            return None
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime and numeric columns to float
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        
        # Select only the relevant columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance: {e}")
        return None

def save_to_csv(df, filename='binance_ohlcv_data.csv'):
    """Save DataFrame to CSV file"""
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        print(f"Total records fetched: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    # Fetch 2000 OHLCV data points for BTCUSDT with 1-day interval
    print("Fetching 2000 OHLCV data points from Binance...")
    ohlcv_data = fetch_binance_ohlcv(limit=2000)
    
    if ohlcv_data is not None:
        print("\nFirst 5 rows of data:")
        print(ohlcv_data.head())
        print("\nDataFrame info:")
        print(ohlcv_data.info())
        
        # Save to CSV
        save_to_csv(ohlcv_data)
        
        print("\nData fetching completed successfully!")
    else:
        print("Failed to fetch data from Binance")