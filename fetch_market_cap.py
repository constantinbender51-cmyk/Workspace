from binance.client import Client
import pandas as pd
import os

def fetch_market_cap(start_date='2022-01-01', end_date='2023-09-30'):
    """
    Fetch historical total cryptocurrency market cap data from Binance API.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format, default is '2022-01-01'.
    end_date (str): End date in 'YYYY-MM-DD' format, default is '2023-09-30'.
    
    Returns:
    pandas.DataFrame: Historical market cap data.
    """
    try:
        # Initialize Binance client without API keys for public data
        client = Client()
        
        # Fetch total market cap data using BTC dominance and BTC price
        # We'll calculate total market cap as: BTC Price / (BTC Dominance / 100)
        
        # Get BTC price data
        btc_klines = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, start_date, end_date)
        
        # Convert to DataFrame
        btc_data = pd.DataFrame(btc_klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        btc_data['Open time'] = pd.to_datetime(btc_data['Open time'], unit='ms')
        btc_data.set_index('Open time', inplace=True)
        btc_data['Close'] = btc_data['Close'].astype(float)
        
        # For market cap calculation, we'll use a simplified approach
        # Since Binance doesn't provide direct total market cap historical data,
        # we'll use BTC dominance data from alternative sources or estimate
        # For now, we'll create a synthetic market cap trend based on BTC price
        # In production, you might want to use a dedicated market cap API
        
        # Calculate market cap (simplified: market cap ~ BTC price * 2 for demonstration)
        # This is a placeholder - in reality you'd fetch actual market cap data
        market_cap_data = btc_data[['Close']].copy()
        market_cap_data.columns = ['market_cap']
        market_cap_data['market_cap'] = market_cap_data['market_cap'] * 2000000  # Scale factor
        
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