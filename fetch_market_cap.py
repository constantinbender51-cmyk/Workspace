import requests
import pandas as pd
import os

def fetch_market_cap(start_date='2022-01-01', end_date='2023-09-30'):
    """
    Fetch historical total cryptocurrency market cap data from CoinGecko API.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format, default is '2022-01-01'.
    end_date (str): End date in 'YYYY-MM-DD' format, default is '2023-09-30'.
    
    Returns:
    pandas.DataFrame: Historical market cap data.
    """
    try:
        # CoinGecko API endpoint for global market cap chart
        url = "https://api.coingecko.com/api/v3/global/market_cap_chart"
        
        # Convert dates to timestamps
        from datetime import datetime
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        params = {
            'vs_currency': 'usd',
            'days': 'max',
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        market_cap_data = data['market_cap']
        
        # Convert to DataFrame
        df = pd.DataFrame(market_cap_data, columns=['timestamp', 'market_cap'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Check if data is empty
        if df.empty:
            print("No market cap data found for the specified date range.")
            return None
        
        # Save to CSV file
        filename = f"market_cap_data_{start_date}_to_{end_date}.csv"
        df.to_csv(filename)
        print(f"Market cap data saved to {filename}")
        
        return df
    except Exception as e:
        print(f"An error occurred while fetching market cap data: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Fetch market cap data from Jan 2022 to Sep 2023
    fetch_market_cap()