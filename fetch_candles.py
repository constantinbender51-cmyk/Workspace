import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_binance_candles(symbol='BTCUSDT', interval='1m', limit=3000):
    """
    Fetch candlestick data from Binance API
    
    Args:
        symbol (str): Trading pair symbol (default: BTCUSDT)
        interval (str): Kline interval (1m, 5m, 15m, 1h, 1d, etc.)
        limit (int): Number of candles to fetch (max 1000 per request)
    
    Returns:
        pandas.DataFrame: DataFrame with candlestick data
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    all_candles = []
    remaining = limit
    end_time = None
    
    while remaining > 0:
        # Binance API limit is 1000 candles per request
        current_limit = min(remaining, 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': current_limit
        }
        
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            candles = response.json()
            
            if not candles:
                break
            
            all_candles.extend(candles)
            remaining -= len(candles)
            
            # Set end_time for next request to the open time of the oldest candle
            end_time = candles[0][0] - 1
            
            # Rate limiting
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(all_candles, columns=columns)
    
    # Convert timestamp to datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    return df

def display_candles(candle_data):
    """Display candle information in a readable format"""
    if candle_data is not None and len(candle_data) > 0:
        print(f"\n📊 Fetched {len(candle_data)} candles from Binance")
        print(f"   Time range: {candle_data['open_time'].iloc[-1]} to {candle_data['open_time'].iloc[0]}")
        print(f"   Latest price: {candle_data['close'].iloc[0]:.2f}")
        print(f"   High: {candle_data['high'].iloc[0]:.2f}")
        print(f"   Low: {candle_data['low'].iloc[0]:.2f}")
        print(f"   Volume: {candle_data['volume'].iloc[0]:.4f}")
        print(f"\nFirst 5 candles:")
        print(candle_data.head())
        
        # Save to CSV
        filename = f"binance_candles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        candle_data.to_csv(filename, index=False)
        print(f"\n💾 Data saved to: {filename}")
    else:
        print("No candle data available")

if __name__ == "__main__":
    print("Fetching 3000 candles from Binance...")
    
    # You can change these parameters
    symbol = "BTCUSDT"
    interval = "1m"  # 1 minute candles
    limit = 3000
    
    candles = fetch_binance_candles(symbol=symbol, interval=interval, limit=limit)
    
    if candles is not None and len(candles) > 0:
        display_candles(candles)
        print(f"\n✅ Successfully fetched {len(candles)} {symbol} {interval} candles!")
    else:
        print("❌ Failed to fetch candle data. Please check your internet connection.")