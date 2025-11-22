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
        # CoinGecko API endpoint for global market cap history
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': int(pd.to_datetime(start_date).timestamp()),
            'to': int(pd.to_datetime(end_date).timestamp())
        }
        
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract market cap data (total market cap is not directly available, using Bitcoin's as a proxy for simplicity)
        # Note: For total market cap, you might need a different endpoint or service
        market_caps = [entry[1] for entry in data['market_caps']]
        timestamps = [entry[0] for entry in data['market_caps']]
        
        # Create DataFrame
        market_cap_data = pd.DataFrame({
            'market_cap': market_caps
        }, index=pd.to_datetime(timestamps, unit='ms'))
        
        # Resample to daily data if needed (API returns data points, ensure daily)
        market_cap_data = market_cap_data.resample('D').last().dropna()
        
        # Check if data is empty
        if market_cap_data.empty:
            print("No market cap data found for the specified date range.")
            return None
        
        # Save to CSV file
        filename = f"market_cap_data_{start_date}_to_{end_date}.csv"
        market_cap_data.to_csv(filename)
        print(f"Market cap data saved to {filename}")
        
        return market_cap_data
    except Exception as e:
        print(f"An error occurred while fetching market cap data: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Fetch market cap data from Jan 2022 to Sep 2023
    fetch_market_cap()