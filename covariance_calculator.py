#!/usr/bin/env python3

import pandas as pd
import numpy as np
from binance.client import Client
import yfinance as yf
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

def fetch_sp500_data(start_date, end_date):
    """Fetch real S&P 500 data from Yahoo Finance"""
    print(f"Fetching S&P 500 data from {start_date} to {end_date}...")
    
    # Use SPY ETF as proxy for S&P 500
    ticker = "SPY"
    
    try:
        # Download historical data
        sp500 = yf.download(ticker, start=start_date, end=end_date)
        
        if sp500.empty:
            print("No S&P 500 data available for the specified date range")
            return None
        
        # Reset index and prepare data
        sp500 = sp500.reset_index()
        sp500['date'] = sp500['Date'].dt.date
        sp500['sp500_return'] = sp500['Close'].pct_change()
        
        # Keep only relevant columns
        sp500 = sp500[['date', 'Close', 'sp500_return']].rename(columns={'Close': 'close'})
        sp500 = sp500.dropna()
        
        print(f"Fetched {len(sp500)} days of S&P 500 data")
        return sp500
        
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        return None

def calculate_covariance(btc_data, sp500_data):
    """Calculate covariance between BTC and S&P 500 returns"""
    
    # Merge data on date
    merged_data = pd.merge(btc_data, sp500_data, on='date', suffixes=('_btc', '_sp500'))
    
    if len(merged_data) < 2:
        print("Not enough overlapping data to calculate covariance")
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
    
    # Calculate additional metrics
    btc_total_return = (merged_data['close_btc'].iloc[-1] / merged_data['close_btc'].iloc[0] - 1)
    sp500_total_return = (merged_data['close_sp500'].iloc[-1] / merged_data['close_sp500'].iloc[0] - 1)
    
    return {
        'covariance': covariance,
        'correlation': correlation,
        'btc_mean_return': btc_mean_return,
        'sp500_mean_return': sp500_mean_return,
        'btc_volatility': btc_volatility,
        'sp500_volatility': sp500_volatility,
        'btc_total_return': btc_total_return,
        'sp500_total_return': sp500_total_return,
        'data_points': len(merged_data),
        'period': f"{merged_data['date'].min()} to {merged_data['date'].max()}",
        'merged_data': merged_data
    }

def print_results(results):
    """Print formatted results"""
    print("\n" + "=" * 60)
    print("BTC-S&P 500 COVARIANCE ANALYSIS")
    print("=" * 60)
    print(f"Period: {results['period']}")
    print(f"Data Points: {results['data_points']}")
    
    print(f"\nCORRELATION METRICS:")
    print(f"Covariance: {results['covariance']:.8f}")
    print(f"Correlation: {results['correlation']:.4f}")
    
    print(f"\nBITCOIN STATISTICS:")
    print(f"  Total Return: {results['btc_total_return']:.4%}")
    print(f"  Mean Daily Return: {results['btc_mean_return']:.4%}")
    print(f"  Daily Volatility: {results['btc_volatility']:.4%}")
    
    print(f"\nS&P 500 STATISTICS:")
    print(f"  Total Return: {results['sp500_total_return']:.4%}")
    print(f"  Mean Daily Return: {results['sp500_mean_return']:.4%}")
    print(f"  Daily Volatility: {results['sp500_volatility']:.4%}")
    
    # Interpretation
    print(f"\nINTERPRETATION:")
    if results['covariance'] > 0:
        print("  ✓ Positive covariance: BTC and S&P 500 tend to move together")
    elif results['covariance'] < 0:
        print("  ✓ Negative covariance: BTC and S&P 500 tend to move in opposite directions")
    else:
        print("  ✓ Zero covariance: No clear relationship between BTC and S&P 500")
    
    if abs(results['correlation']) > 0.7:
        strength = "Strong"
    elif abs(results['correlation']) > 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    print(f"  ✓ {strength} correlation between assets")
    
    if results['correlation'] > 0:
        print("  ✓ Positive correlation: When one asset goes up, the other tends to go up")
    elif results['correlation'] < 0:
        print("  ✓ Negative correlation: When one asset goes up, the other tends to go down")
    
    # Portfolio diversification insight
    if abs(results['correlation']) < 0.3:
        print("  ✓ Good diversification potential: Low correlation suggests diversification benefits")
    elif abs(results['correlation']) > 0.7:
        print("  ✓ Limited diversification: High correlation means both assets move together")

def main():
    """Main function to calculate BTC-S&P500 covariance"""
    
    # Define date range (November 22-23, 2023)
    start_date = '2023-11-22'
    end_date = '2023-11-23'
    
    print("=" * 60)
    print("BTC-S&P 500 Covariance Calculator")
    print("Using real data from Binance (BTC) and Yahoo Finance (S&P 500)")
    print("=" * 60)
    
    try:
        # Fetch BTC data
        btc_data = fetch_btc_data_daily(start_date, end_date)
        
        if len(btc_data) == 0:
            print("No BTC data available for the specified date range")
            return
        
        print(f"Fetched {len(btc_data)} days of BTC data")
        
        # Fetch real S&P 500 data
        sp500_data = fetch_sp500_data(start_date, end_date)
        
        if sp500_data is None or len(sp500_data) == 0:
            print("No S&P 500 data available for the specified date range")
            return
        
        # Calculate covariance
        results = calculate_covariance(btc_data, sp500_data)
        
        if results:
            print_results(results)
            
            # Display sample data
            print(f"\n" + "=" * 60)
            print("SAMPLE DATA")
            print("=" * 60)
            sample_data = results['merged_data'].head()
            print(sample_data[['date', 'close_btc', 'btc_return', 'close_sp500', 'sp500_return']].to_string(index=False, float_format='%.4f'))
            
            # Summary
            print(f"\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"The covariance between Bitcoin and S&P 500 during this period was {results['covariance']:.8f}")
            print(f"This indicates a {('positive' if results['covariance'] > 0 else 'negative')} relationship")
            print(f"The correlation coefficient of {results['correlation']:.4f} suggests a {('strong' if abs(results['correlation']) > 0.7 else 'moderate' if abs(results['correlation']) > 0.3 else 'weak')} linear relationship")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify the date range has trading data")
        print("3. Try a different date range if needed")

if __name__ == "__main__":
    main()