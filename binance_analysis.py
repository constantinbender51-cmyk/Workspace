import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for server environments without display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, send_file
import io
import time
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
VALIDATION_SMA = 120  # The specific SMA to validate
HORIZON = 30
PORT = 8080

def fetch_binance_data():
    """Fetches daily OHLCV from Binance starting Jan 1, 2018."""
    print(f"Fetching data for {SYMBOL} since {SINCE_STR}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(SINCE_STR)
    
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            # Update 'since' to the last timestamp + 1ms to avoid duplicates
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # Rate limit handling
            time.sleep(0.1)
            
            # If we reached current time (approx), stop
            if last_timestamp >= exchange.milliseconds() - 24*60*60*1000:
                break
                
            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nTotal candles fetched: {len(all_ohlcv)}")
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    
    # Calculate daily returns
    df['return'] = df['close'].pct_change()
    return df

def get_detailed_signals(df, period, horizon):
    """
    Analyzes a SPECIFIC SMA period and returns details for every signal.
    """
    prices = df['close'].values
    returns = df['return'].values
    dates = df.index
    
    # Calculate SMA
    sma = df['close'].rolling(window=period).mean().values
    
    # Weights: Linear Decay (1 - day/30)
    # Day 1 weight = 0.966, Day 30 weight = 0
    day_indices = np.arange(1, horizon + 1)
    weights = 1 - (day_indices / horizon)
    
    signal_details = []
    
    # Identify signals
    # We iterate to allow easy extraction of the 30-day window per signal
    # (Vectorization is faster for summary, but loop is better for detailed extraction)
    
    for i in range(period, len(prices) - horizon):
        prev_price = prices[i-1]
        curr_price = prices[i]
        prev_sma = sma[i-1]
        curr_sma = sma[i]
        
        signal_type = None
        if prev_price < prev_sma and curr_price > curr_sma:
            signal_type = 'LONG'
        elif prev_price > prev_sma and curr_price < curr_sma:
            signal_type = 'SHORT'
            
        if signal_type:
            # Get the 30-day forward returns
            # returns[i+1] is the return of the day AFTER the signal
            fwd_returns = returns[i+1 : i+1+horizon]
            
            if len(fwd_returns) < horizon:
                continue
                
            # Calculate Score
            if signal_type == 'LONG':
                weighted_sum = np.sum(fwd_returns * weights)
            else:
                weighted_sum = np.sum(fwd_returns * -1 * weights)
                
            signal_details.append({
                'date': dates[i],
                'index': i,
                'price': curr_price,
                'type': signal_type,
                'score': weighted_sum,
                'window_end_date': dates[i+horizon]
            })
            
    return pd.DataFrame(signal_details), sma

# --- Execution ---
print("Starting Data Fetch & Analysis...")
df_data = fetch_binance_data()
df_signals, sma_values = get_detailed_signals(df_data, VALIDATION_SMA, HORIZON)
df_data['SMA'] = sma_values
print("Analysis Complete.")

@app.route('/')
def plot_validation():
    """Generates the validation plot for SMA 120."""
    
    if df_signals.empty:
        return "No signals found to validate."

    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # --- Plot 1: Price, SMA, and Signal Windows ---
    # Use log scale for BTC to see 2018 and 2024 clearly
    ax1.semilogy(df_data.index, df_data['close'], color='gray', alpha=0.5, linewidth=1, label='Price')
    ax1.semilogy(df_data.index, df_data['SMA'], color='blue', linewidth=1.5, label=f'SMA {VALIDATION_SMA}')
    
    # Highlight Signal Windows
    for _, row in df_signals.iterrows():
        color = 'green' if row['type'] == 'LONG' else 'red'
        # Shade the 30-day window
        ax1.axvspan(row['date'], row['window_end_date'], color=color, alpha=0.1)
        # Mark the entry point
        marker = '^' if row['type'] == 'LONG' else 'v'
        ax1.scatter(row['date'], row['price'], color=color, marker=marker, s=100, zorder=5)

    ax1.set_title(f'SMA {VALIDATION_SMA} Validation: Signal Windows ({HORIZON} Days)', fontsize=14)
    ax1.set_ylabel('Price (Log Scale)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # --- Plot 2: Individual Signal Scores ---
    
    # Separate Longs and Shorts for color coding
    longs = df_signals[df_signals['type'] == 'LONG']
    shorts = df_signals[df_signals['type'] == 'SHORT']
    
    ax2.bar(longs['date'], longs['score'], color='green', alpha=0.7, width=10, label='Long Score')
    ax2.bar(shorts['date'], shorts['score'], color='red', alpha=0.7, width=10, label='Short Score')
    
    # Plot Average Score Line
    avg_score = df_signals['score'].mean()
    ax2.axhline(avg_score, color='black', linestyle='--', linewidth=2, label=f'Avg Score: {avg_score:.4f}')
    
    # Zero line
    ax2.axhline(0, color='gray', linewidth=0.8)

    ax2.set_title(f'Individual Dependability Scores per Signal (Linear Decay Weight)', fontsize=12)
    ax2.set_ylabel('Score (Weighted Return)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format Date Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Save to buffer
    output = io.BytesIO()
    plt.tight_layout()
    fig.savefig(output, format='png')
    plt.close(fig) 
    output.seek(0)
    
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
