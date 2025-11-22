import requests
import pandas as pd
import os
import logging

def fetch_price_data(symbol='BTC', start_date='2022-01-01', end_date='2023-09-30'):
    """
    Fetch historical price data for a given symbol from start_date to end_date using Alpha Vantage API.
    
    Parameters:
    symbol (str): Cryptocurrency symbol, default is 'BTC'.
    start_date (str): Start date in 'YYYY-MM-DD' format, default is '2022-01-01'.
    end_date (str): End date in 'YYYY-MM-DD' format, default is '2023-09-30'.
    
    Returns:
    pandas.DataFrame: Historical price data.
    """
    try:
        api_key = 'OR1V5E82M3OOSMMD'
        # Alpha Vantage API endpoint for digital currency daily data
        url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={api_key}'
        response = requests.get(url)
        if response.status_code != 200:
            logging.error(f"Failed to fetch data from Alpha Vantage API: {response.status_code}")
            return None
        
        data_json = response.json()
        if 'Error Message' in data_json:
            logging.error(f"API Error: {data_json['Error Message']}")
            return None
        
        # Extract time series data
        time_series = data_json.get('Time Series (Digital Currency Daily)', {})
        if not time_series:
            logging.warning(f"No data found for symbol {symbol} in the specified date range.")
            return None
        
        # Convert to DataFrame
        data = pd.DataFrame.from_dict(time_series, orient='index')
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # Filter data based on start_date and end_date
        data = data.loc[start_date:end_date]
        if data.empty:
            logging.warning(f"No data found for symbol {symbol} in the specified date range.")
            return None
        
        # Select and rename relevant columns (e.g., '4a. close (USD)' for closing price)
        data = data[['4a. close (USD)']].rename(columns={'4a. close (USD)': 'Close'})
        data['Close'] = data['Close'].astype(float)
        
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