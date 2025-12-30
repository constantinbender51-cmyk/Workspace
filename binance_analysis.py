import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def download_data(file_id, output_filename='ohlcv_data.csv'):
    """
    Downloads the file from Google Drive using gdown.
    """
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if os.path.exists(output_filename):
        print(f"File {output_filename} already exists. Skipping download.")
        return

    print(f"Downloading file from Google Drive...")
    gdown.download(url, output_filename, quiet=False)
    print("Download complete.")

def run_strategy(csv_file):
    # 1. Load Data
    print("\nLoading data...")
    # Assumes CSV has headers like timestamp/date, open, high, low, close, volume
    # Trying to parse standard formats
    try:
        df = pd.read_csv(csv_file)
        
        # Identify the date column (usually the first one or named 'timestamp'/'date')
        date_col = df.columns[0] 
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Data loaded: {len(df)} 1-minute rows.")

    # 2. Resample to 5-Minute Bars
    # We take the last close, first open, max high, min low, sum volume
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Ensure columns exist before resampling
    available_cols = {k: v for k, v in ohlc_dict.items() if k in df.columns}
    
    print("Resampling to 5-minute intervals...")
    df_5m = df.resample('5min').agg(available_cols).dropna()

    # 3. Calculate Indicators
    # 40 SMA on the 5-minute Close
    df_5m['sma_40'] = df_5m['close'].rolling(window=40).mean()

    # 4. Define Logic
    # 1 day = 1440 minutes. 
    # In 5-minute bars, 1 day = 1440 / 5 = 288 bars.
    holding_period = 288

    # Create a 'Signal' column: 1 for Long, -1 for Short
    # We shift(1) to avoid lookahead bias? 
    # The prompt says "shorts when price below 40 SMA". 
    # Usually we execute AT the close when the condition is met.
    
    df_5m['signal'] = 0
    df_5m.loc[df_5m['close'] > df_5m['sma_40'], 'signal'] = 1  # Long
    df_5m.loc[df_5m['close'] < df_5m['sma_40'], 'signal'] = -1 # Short

    # 5. Calculate Returns
    # We enter at the Close of the current candle.
    # We exit at the Close of the candle 'holding_period' steps in the future.
    # Forward Return = (Future_Price - Current_Price) / Current_Price
    
    # Shift(-holding_period) gets the price 24 hours in the future
    future_close = df_5m['close'].shift(-holding_period)
    
    # Percentage change over the holding period
    df_5m['trade_return'] = (future_close - df_5m['close']) / df_5m['close']
    
    # Strategy Return = Signal * Trade Return
    # (If we are Short (-1) and price drops (negative return), we make profit)
    df_5m['strategy_return'] = df_5m['signal'] * df_5m['trade_return']

    # Drop NaN values created by the SMA and the future lookahead
    strategy_data = df_5m.dropna(subset=['strategy_return'])

    # 6. Analyze Results
    total_trades = len(strategy_data)
    cumulative_return = strategy_data['strategy_return'].cumsum()
    
    avg_return = strategy_data['strategy_return'].mean()
    win_rate = len(strategy_data[strategy_data['strategy_return'] > 0]) / total_trades * 100

    print("\n--- Strategy Results ---")
    print(f"Total Trades (5-min intervals): {total_trades}")
    print(f"Average Return per Trade: {avg_return*100:.4f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Cumulative Return (Sum): {strategy_data['strategy_return'].sum()*100:.2f}%")

    # 7. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_return.index, cumulative_return.values, label='Cumulative Return')
    plt.title('Strategy Performance: 5-min SMA(40) Crossover (1 Day Hold)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (non-compounded)')
    plt.legend()
    plt.grid(True)
    
    # Save plot strictly for the user to view later if needed, or show()
    plt.savefig('strategy_performance.png')
    print("Performance plot saved as 'strategy_performance.png'")
    # plt.show() # Uncomment if running in an environment with display

if __name__ == "__main__":
    # The file ID from the provided Drive link
    # https://drive.google.com/file/d/1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o/view?usp=drivesdk
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    FILENAME = 'btc_1m_data.csv' # Assuming BTC or similar based on typical datasets

    download_data(FILE_ID, FILENAME)
    run_strategy(FILENAME)