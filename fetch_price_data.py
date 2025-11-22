from binance.client import Client
import pandas as pd
import os
import logging

def fetch_price_data(symbol='BTCUSDT', start_date='2022-01-01', end_date='2023-09-30'):
    """
    Fetch historical price data for a given symbol from start_date to end_date using Binance API.
    
    Parameters:
    symbol (str): Cryptocurrency symbol, default is 'BTCUSDT'.
    start_date (str): Start date in 'YYYY-MM-DD' format, default is '2022-01-01'.
    end_date (str): End date in 'YYYY-MM-DD' format, default is '2023-09-30'.
    
    Returns:
    pandas.DataFrame: Historical price data.
    """
    try:
        # Initialize Binance client (no API key needed for public data)
        client = Client()
        
        # Fetch historical klines data
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str=start_date,
            end_str=end_date
        )
        
        if not klines:
            logging.warning(f"No data found for symbol {symbol} in the specified date range.")
            return None
        
        # Convert to DataFrame
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                  'Close time', 'Quote asset volume', 'Number of trades',
                  'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        
        data = pd.DataFrame(klines, columns=columns)
        
        # Convert timestamp to datetime and set as index
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data.set_index('Open time', inplace=True)
        
        # Convert price columns to float
        data['Close'] = data['Close'].astype(float)
        
        # Keep only the Close column for simplicity
        data = data[['Close']]
        
        # Save to CSV file
        filename = f"{symbol}_price_data_{start_date}_to_{end_date}.csv"
        data.to_csv(filename)
        logging.info(f"Data saved to {filename}")
        
        return data
    except Exception as e:
        logging.error(f"An error occurred fetching price data: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Fetch data for BTCUSDT from Jan 2022 to Sep 2023
    fetch_price_data()