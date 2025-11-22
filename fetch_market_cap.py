import time
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
                    data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
                
                    data.set_index('Open time', inplace=True)
                if klines:
                    
                    # Convert to DataFrame
                    # Convert price to float
                    data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    data['Close'] = data['Close'].astype(float)
            time.sleep(1)  # Add delay to respect Binance rate limits
                    data['Volume'] = data['Volume'].astype(float)
                                                        'Close time', 'Quote asset volume', 'Number of trades', 
                    time.sleep(1)  # Add delay to respect Binance rate limits
                                                        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
                    
                    data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
                    data.set_index('Open time', inplace=True)
                    
                    # Convert price to float
                    data['Close'] = data['Close'].astype(float)
                    data['Volume'] = data['Volume'].astype(float)
                    # Get circulating supply for accurate market cap calculation
                    # For Binance, we'll use the price and estimate supply from known data
                    # This prevents the inflated 'price * volume' calculation
                    symbol_supplies = {
                        'BTCUSDT': 19400000,   # Approx BTC circulating supply
                        'ETHUSDT': 120000000,  # Approx ETH circulating supply
                        'BNBUSDT': 151000000,  # Approx BNB circulating supply
                        'XRPUSDT': 53000000000, # Approx XRP circulating supply
                        'ADAUSDT': 34000000000, # Approx ADA circulating supply
                        'DOGEUSDT': 132000000000, # Approx DOGE circulating supply
                        'MATICUSDT': 9300000000, # Approx MATIC circulating supply
                        'DOTUSDT': 1200000000,  # Approx DOT circulating supply
                        'LTCUSDT': 72000000,    # Approx LTC circulating supply
                        'BCHUSDT': 19000000,    # Approx BCH circulating supply
                        'LINKUSDT': 510000000,  # Approx LINK circulating supply
                        'ATOMUSDT': 290000000,  # Approx ATOM circulating supply
                        'XLMUSDT': 26000000000, # Approx XLM circulating supply
                        'FILUSDT': 390000000,   # Approx FIL circulating supply
                        'ETCUSDT': 140000000,   # Approx ETC circulating supply
                        'XTZUSDT': 900000000,   # Approx XTZ circulating supply
                        'EOSUSDT': 1100000000,  # Approx EOS circulating supply
                        'AAVEUSDT': 14000000,   # Approx AAVE circulating supply
                        'ALGOUSDT': 7000000000, # Approx ALGO circulating supply
                        'NEOUSDT': 70000000     # Approx NEO circulating supply
                    }
                    
                    # Calculate actual market cap: price * circulating supply
                    base_symbol = symbol.replace('USDT', '')
                    if base_symbol in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT', 'LTC', 'BCH', 
                                      'LINK', 'ATOM', 'XLM', 'FIL', 'ETC', 'XTZ', 'EOS', 'AAVE', 'ALGO', 'NEO']:
                        supply = symbol_supplies[symbol]
                        market_cap_data[symbol] = data['Close'] * supply
                    
                    # Calculate daily market cap (price * volume as approximation)
                    # Note: This is a simplified calculation. For more accuracy, you'd need circulating supply data
                    market_cap_data[symbol] = data['Close'] * data['Volume']
                    # Calculate actual market cap: price * circulating supply
                    base_symbol = symbol.replace('USDT', '')
                    if base_symbol in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT', 'LTC', 'BCH', 
                                      'LINK', 'ATOM', 'XLM', 'FIL', 'ETC', 'XTZ', 'EOS', 'AAVE', 'ALGO', 'NEO']:
                        supply = symbol_supplies[symbol]
                        market_cap_data[symbol] = data['Close'] * supply
                    else:
                        # Fallback: use price * volume for symbols without supply data
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
        print(f"Total market cap range: ${final_data['market_cap'].min():,.2f} - ${final_data['market_cap'].max():,.2f}")
        print(f"Sample market cap values (first 5): {final_data['market_cap'].head().tolist()}")
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