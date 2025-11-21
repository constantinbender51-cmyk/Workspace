#!/usr/bin/env python3

import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_btc_data_daily(start_date, end_date):
    """Fetch daily BTC candles from Binance for specified date range"""
    print(f"Fetching BTC data from {start_date} to {end_date}...")
    
    client = Client()
    
    # Fetch daily candles
    klines = client.get_historical_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_1DAY,
        start_str=start_date,
        end_str=end_date
    )
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['timestamp'].dt.date
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Calculate daily returns
    df['btc_return'] = df['close'].pct_change()
    
    # Keep only relevant columns
    df = df[['date', 'close', 'btc_return']].dropna()
    
    return df

def generate_mock_sp500_data(btc_dates):
    """Generate mock S&P 500 data for demonstration"""
    print("Generating mock S&P 500 data...")
    
    # Base S&P 500 price
    base_price = 4500
    
    # Create DataFrame with same dates as BTC data
    sp500_data = []
    
    for i, date in enumerate(btc_dates):
        # Simulate S&P 500 price with some correlation to BTC
        if i == 0:
            price = base_price
        else:
            # Add some random movement correlated with BTC
            btc_change = np.random.normal(0, 0.02)  # Simulated BTC influence
            market_noise = np.random.normal(0, 0.01)
            price = sp500_data[i-1]['close'] * (1 + btc_change * 0.3 + market_noise)
        
        sp500_data.append({
            'date': date,
            'close': price
        })
    
    df = pd.DataFrame(sp500_data)
    df['sp500_return'] = df['close'].pct_change()
    
    return df.dropna()

def calculate_covariance(btc_data, sp500_data):
    """Calculate covariance between BTC and S&P 500 returns"""
    
    # Merge data on date
    merged_data = pd.merge(btc_data, sp500_data, on='date', suffixes=('_btc', '_sp500'))
    
    if len(merged_data) < 2:
        print("Not enough data to calculate covariance")
        return None
    
    # Calculate covariance
    covariance = np.cov(merged_data['btc_return'], merged_data['sp500_return'])[0, 1]
    
    # Calculate correlation for additional insight
    correlation = np.corrcoef(merged_data['btc_return'], merged_data['sp500_return'])[0, 1]
    
    # Calculate descriptive statistics
    btc_mean_return = merged_data['btc_return'].mean()
    sp500_mean_return = merged_data['sp500_return'].mean()
    btc_volatility = merged_data['btc_return'].std()
    sp500_volatility = merged_data['sp500_return'].std()
    
    return {
        'covariance': covariance,
        'correlation': correlation,
        'btc_mean_return': btc_mean_return,
        'sp500_mean_return': sp500_mean_return,
        'btc_volatility': btc_volatility,
        'sp500_volatility': sp500_volatility,
        'data_points': len(merged_data),
        'period': f"{merged_data['date'].min()} to {merged_data['date'].max()}"
    }

def main():
    """Main function to calculate BTC-S&P500 covariance"""
    
    # Define date range (November 22-23, 2023)
    start_date = '2023-11-22'
    end_date = '2023-11-23'
    
    print("=" * 60)
    print("BTC-S&P 500 Covariance Calculator")
    print("=" * 60)
    
    try:
        # Fetch BTC data
        btc_data = fetch_btc_data_daily(start_date, end_date)
        
        if len(btc_data) == 0:
            print("No BTC data available for the specified date range")
            return
        
        print(f"Fetched {len(btc_data)} days of BTC data")
        
        # Generate mock S&P 500 data
        sp500_data = generate_mock_sp500_data(btc_data['date'].tolist())
        print(f"Generated {len(sp500_data)} days of S&P 500 data")
        
        # Calculate covariance
        results = calculate_covariance(btc_data, sp500_data)
        
        if results:
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"Period: {results['period']}")
            print(f"Data Points: {results['data_points']}")
            print(f"\nCovariance: {results['covariance']:.8f}")
            print(f"Correlation: {results['correlation']:.4f}")
            print(f"\nBTC Statistics:")
            print(f"  Mean Daily Return: {results['btc_mean_return']:.4%}")
            print(f"  Daily Volatility: {results['btc_volatility']:.4%}")
            print(f"\nS&P 500 Statistics:")
            print(f"  Mean Daily Return: {results['sp500_mean_return']:.4%}")
            print(f"  Daily Volatility: {results['sp500_volatility']:.4%}")
            
            # Interpretation
            print(f"\nInterpretation:")
            if results['covariance'] > 0:
                print("  Positive covariance: BTC and S&P 500 tend to move together")
            elif results['covariance'] < 0:
                print("  Negative covariance: BTC and S&P 500 tend to move in opposite directions")
            else:
                print("  Zero covariance: No clear relationship between BTC and S&P 500")
                
            if abs(results['correlation']) > 0.7:
                print("  Strong correlation between assets")
            elif abs(results['correlation']) > 0.3:
                print("  Moderate correlation between assets")
            else:
                print("  Weak correlation between assets")
        
        # Display sample data
        print(f"\n" + "=" * 60)
        print("SAMPLE DATA (First 5 rows)")
        print("=" * 60)
        merged_sample = pd.merge(btc_data.head(), sp500_data.head(), on='date')
        print(merged_sample[['date', 'close_btc', 'btc_return', 'close_sp500', 'sp500_return']].to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        print("This might be due to:")
        print("1. No trading data available for the specified dates")
        print("2. Network connectivity issues")
        print("3. Binance API limitations")

if __name__ == "__main__":
    main()