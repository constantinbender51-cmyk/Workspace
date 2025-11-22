import yfinance as yf
import pandas as pd

def fetch_price_data(symbol='AAPL', start_date='2022-01-01', end_date='2023-09-30'):
    """
    Fetch historical price data for a given symbol from start_date to end_date.
    
    Parameters:
    symbol (str): Stock symbol, default is 'AAPL'.
    start_date (str): Start date in 'YYYY-MM-DD' format, default is '2022-01-01'.
    end_date (str): End date in 'YYYY-MM-DD' format, default is '2023-09-30'.
    
    Returns:
    pandas.DataFrame: Historical price data.
    """
    try:
        # Download data using yfinance
        data = yf.download(symbol, start=start_date, end=end_date)
        
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
    fetch_price_data()