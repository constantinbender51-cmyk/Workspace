from binance.client import Client
import pandas as pd
import os

def fetch_market_cap(start_date='2022-01-01', end_date='2023-09-30'):
    """
    Fetch historical total cryptocurrency market cap data using Binance API.
    Calculates total market cap by summing market caps of top cryptocurrencies.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format, default is '2022-01-01'.
    end_date (str): End date in 'YYYY-MM-DD' format, default is '2023-09-30'.
    
    Returns:
    pandas.DataFrame: Historical total cryptocurrency market cap data.
    """
    try:
        # Initialize Binance client
        client = Client()
        
        # Get top cryptocurrencies by market cap (symbols available on Binance)
        top_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT',
            'LINKUSDT', 'ATOMUSDT', 'XLMUSDT', 'FILUSDT', 'ETCUSDT',
            'XTZUSDT', 'EOSUSDT', 'AAVEUSDT', 'ALGOUSDT', 'NEOUSDT'
        ]
        
        # Fetch daily market cap data for each symbol
        market_cap_data = {}
        
        for symbol in top_symbols:
            try:
                # Fetch historical klines
                klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_date, end_date)
                
                if klines:
                    # Convert to DataFrame
                    data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                                        'Close time', 'Quote asset volume', 'Number of trades', 
                                                        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
                    
                    data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
                    data.set_index('Open time', inplace=True)
                    
                    # Convert price to float
                    data['Close'] = data['Close'].astype(float)
                    data['Volume'] = data['Volume'].astype(float)
                    
                    # Calculate daily market cap (price * volume as approximation)
                    # Note: This is a simplified calculation. For more accuracy, you'd need circulating supply data
                    market_cap_data[symbol] = data['Close'] * data['Volume']
                    
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
                continue
        
        if not market_cap_data:
            print("No market cap data could be fetched for any symbol")
            return None
        
        # Combine all market caps into a single DataFrame
        combined_df = pd.DataFrame(market_cap_data)
        
        # Calculate total market cap by summing all individual market caps
        combined_df['total_market_cap'] = combined_df.sum(axis=1)
        
        # Create final DataFrame with just the total market cap
        final_data = pd.DataFrame({
            'market_cap': combined_df['total_market_cap']
        })
        
        # Resample to ensure daily data
        final_data = final_data.resample('D').last().dropna()
        
        # Check if data is empty
        if final_data.empty:
            print("No market cap data found for the specified date range.")
            return None
        
        # Save to CSV file
        filename = f"total_market_cap_data_{start_date}_to_{end_date}.csv"
        final_data.to_csv(filename)
        print(f"Total market cap data saved to {filename}")
        print(f"Total market cap range: ${final_data['market_cap'].min():,.0f} - ${final_data['market_cap'].max():,.0f}")
        
        return final_data
        
    except Exception as e:
        print(f"An error occurred while fetching market cap data: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Fetch total market cap data from Jan 2022 to Sep 2023
    fetch_market_cap()