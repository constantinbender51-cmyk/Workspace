from binance.client import Client
import pandas as pd
def fetch_price_data(symbol='BTCUSDT', start_date='2022-01-01', end_date='2023-09-30'):

def fetch_price_data(symbol='AAPL', start_date='2022-01-01', end_date='2023-09-30'):
    """
    Fetch historical price data for a given symbol from start_date to end_date.
    
    Parameters:
    symbol (str): Stock symbol, default is 'AAPL'.
    start_date (str): Start date in 'YYYY-MM-DD' format, default is '2022-01-01'.
    try:
        # Initialize Binance client with API keys (set environment variables BINANCE_API_KEY and BINANCE_API_SECRET)
        api_key = 'your_api_key_here'  # Replace with your actual API key or use os.environ.get('BINANCE_API_KEY')
        api_secret = 'your_api_secret_here'  # Replace with your actual API secret or use os.environ.get('BINANCE_API_SECRET')
        client = Client(api_key, api_secret)
    end_date (str): End date in 'YYYY-MM-DD' format, default is '2023-09-30'.
    
    Returns:
    pandas.DataFrame: Historical price data.
    """
    try:
        # Download data using yfinance
        # Fetch historical klines data from Binance
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_date, end_date)
        
        # Convert to DataFrame
        data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data['Close time'] = pd.to_datetime(data['Close time'], unit='ms')
        data.set_index('Open time', inplace=True)
        # Convert price columns to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = data[col].astype(float)
        data = yf.download(symbol, start=start_date, end=end_date)
        # Check if data is empty
        if data.empty:
            print(f"No data found for symbol {symbol} in the specified date range.")
            return None
        # Save to CSV file
        filename = f"{symbol}_price_data_{start_date}_to_{end_date}.csv"
        data.to_csv(filename)
        print(f"Data saved to {filename}")
        # Check if data is empty
        if data.empty:
            print(f"No data found for symbol {symbol} in the specified date range.")
            return None
        
        # Save to CSV file
        filename = f"{symbol}_price_data_{start_date}_to_{end_date}.csv"
        data.to_csv(filename)
        print(f"Data saved to {filename}")
        
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Fetch data for Apple Inc. from Jan 2022 to Sep 2023
        return data
    fetch_price_data()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None