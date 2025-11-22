from binance.client import Client
import pandas as pd
import os
import logging
import time

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
        logging.debug(f"Starting data fetch for {symbol} from {start_date} to {end_date}")
        # Initialize Binance client (no API key needed for public data)
        client = Client()
        logging.debug("Binance client initialized successfully")
        
        # Fetch historical klines data
        logging.debug(f"Fetching klines data for {symbol} with daily interval")
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str=start_date,
            end_str=end_date
        )
        
        if not klines:
            logging.warning(f"No data found for symbol {symbol} in the specified date range.")
            return None
        
        logging.debug(f"Retrieved {len(klines)} klines records")
        
        # Convert to DataFrame
        logging.debug("Converting klines data to DataFrame")
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                  'Close time', 'Quote asset volume', 'Number of trades',
                  'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        
        data = pd.DataFrame(klines, columns=columns)
        logging.debug(f"Created DataFrame with shape {data.shape}")
        
        # Convert timestamp to datetime and set as index
        logging.debug("Processing datetime and price data")
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data.set_index('Open time', inplace=True)
        
        # Convert price columns to float
        data['Close'] = data['Close'].astype(float)
        logging.debug(f"Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
        
        # Keep only the Close column for simplicity
        data = data[['Close']]
        logging.debug("Filtered DataFrame to keep only Close column")
        
        # Save to CSV file
        filename = f"{symbol}_price_data_{start_date}_to_{end_date}.csv"
        data.to_csv(filename)
        logging.info(f"Data saved to {filename}")
        logging.debug("Data fetch completed successfully")
        
        return data
    except Exception as e:
        logging.error(f"An error occurred fetching price data: {e}")
        return None

def fetch_multiple_cryptos():
    """
    Fetch historical price data for multiple cryptocurrencies with rate limiting.
    
    Returns:
    pandas.DataFrame: Combined historical price data for all cryptocurrencies.
    """
    try:
        logging.debug("Starting multi-cryptocurrency data fetch")
        
        # Define the cryptocurrencies to fetch
        cryptocurrencies = {
            'BTCUSDT': 'Bitcoin',
            'ETHUSDT': 'Ethereum', 
            'XRPUSDT': 'Ripple',
            'ADAUSDT': 'Cardano',
            'BNBUSDT': 'Binance Coin'
        }
        
        combined_data = pd.DataFrame()
        
        for symbol, name in cryptocurrencies.items():
            logging.debug(f"Fetching data for {symbol} ({name})")
            
            # Fetch data for current cryptocurrency
            data = fetch_price_data(symbol=symbol)
            
            if data is not None:
                # Rename the column to the cryptocurrency name
                data.rename(columns={'Close': name}, inplace=True)
                
                if combined_data.empty:
                    combined_data = data
                else:
                    combined_data = combined_data.join(data, how='outer')
                
                logging.debug(f"Successfully added {name} data to combined dataset")
            else:
                logging.warning(f"Failed to fetch data for {symbol}")
            
            # Add delay to avoid rate limiting (2 seconds between requests)
            logging.debug("Adding delay to avoid rate limiting")
            time.sleep(2)
        
        if combined_data.empty:
            logging.error("No cryptocurrency data was successfully fetched")
            return None
        
        logging.info(f"Successfully fetched data for {len(combined_data.columns)} cryptocurrencies")
        logging.debug(f"Combined dataset shape: {combined_data.shape}")
        
        return combined_data
        
    except Exception as e:
        logging.error(f"An error occurred in multi-cryptocurrency fetch: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Fetch data for BTCUSDT from Jan 2022 to Sep 2023
    fetch_price_data()