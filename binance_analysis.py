import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
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
HORIZON = 30 # FIXED evaluation window length
PORT = 8080

def fetch_binance_data():
    """Fetches daily OHLCV from Binance starting Jan 1, 2018."""
    print(f"Fetching data for {SYMBOL} since {SINCE_STR}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(SINCE_STR)
    
    all_ohlcv = []
    # Binance rate limits require pagination
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            time.sleep(0.1)
            
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
    df['return'] = df['close'].pct_change()
    return df

def precompute_forward_matrix(returns_array, horizon):
    """
    Creates a matrix where row 't' contains returns for [t+1, t+2, ... t+horizon].
    Accepts the returns as a NumPy array.
    """
    n = len(returns_array)
    matrix = np.full((n, horizon), np.nan)
    
    for day in range(1, horizon + 1):
        # Shift returns back by 'day' steps
        shifted = np.roll(returns_array, -day)
        # Set the invalid end elements to NaN
        shifted[-day:] = np.nan
        matrix[:, day-1] = shifted
        
    return matrix

def get_fixed_window_signals_and_score(df, period, horizon):
    """
    Identifies simple crossover signals and calculates the metric using a fixed 30-day window.
    """
    prices = df['close'].values
    returns = df['return'].values
    dates = df.index
    
    # 1. Precompute forward return matrix (passing the NumPy array directly)
    print("Precomputing forward returns matrix...")
    fwd_matrix = precompute_forward_matrix(returns, horizon)
    
    # 2. Calculate SMA
    sma = df['close'].rolling(window=period).mean().values
    
    # Weights: Linear Decay (1 - day/30)
    days = np.arange(1, horizon + 1)
    weights = 1 - (days / horizon)
    
    signal_details = []
    
    # 3. Detect Simple Crossover Signals
    # We iterate up to the point where we can still look 30 days ahead.
    valid_limit = len(prices) - horizon
    
    for i in range(period, valid_limit):
        prev_price = prices[i-1]
        curr_price = prices[i]
        prev_sma = sma[i-1]
        curr_sma = sma[i]
        
        signal_type = None
        
        # Simple Close Price Crossover Signal Logic
        if prev_price < prev_sma and curr_price > curr_sma:
            signal_type = 'LONG'
        elif prev_price > prev_sma and curr_price < curr_sma:
            signal_type = 'SHORT'
        else:
            continue

        # Get the FIXED 30-day forward returns
        # fwd_matrix[i] returns the array of 30 future returns for index i
        fwd_returns = fwd_matrix[i] 
        
        # Calculate Score (Fixed-Window Logic)
        if signal_type == 'LONG':
            weighted_sum = np.sum(fwd_returns * weights)
        else:
            # Short score = Sum(-R * W)
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
df_signals, sma_values = get_fixed_window_signals_and_score(
    df_data, VALIDATION_SMA, HORIZON
)
df_data['SMA'] = sma_values
print("Analysis Complete.")

@app.route('/')
def plot_validation_fixed_poc():
    """Generates the validation plot for the fixed 30-day logic."""
    
    if df_signals.empty:
        return f"No signals found using SMA {VALIDATION_SMA} with simple crossover logic."

    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # --- Plot 1: Price, SMA, and Fixed 30-Day Windows ---
    ax1.semilogy(df_data.index, df_data['close'], color='gray', alpha=0.4, linewidth=1, label='Price')
    ax1.semilogy(df_data.index, df_data['SMA'], color='blue', linewidth=1.5, alpha=0.8, label=f'SMA {VALIDATION_SMA}')
    
    # Highlight Fixed 30-Day Signal Windows
    for _, row in df_signals.iterrows():
        color = 'green' if row['type'] == 'LONG' else 'red'
        # Shade the FIXED 30-day window
        ax1.axvspan(row['date'], row['window_end_date'], color=color, alpha=0.1)
        
        # Marker
        marker = '^' if row['type'] == 'LONG' else 'v'
        ax1.scatter(row['date'], row['price'], color=color, marker=marker, s=80, zorder=5)

    ax1.set_title(f'POC: Metric Behavior with SMA {VALIDATION_SMA} & Fixed {HORIZON}-Day Window', fontsize=14)
    ax1.set_ylabel('Price (Log Scale)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # --- Plot 2: Individual Signal Scores ---
    longs = df_signals[df_signals['type'] == 'LONG']
    shorts = df_signals[df_signals['type'] == 'SHORT']
    
    # Plot bars
    ax2.bar(longs['date'], longs['score'], color='green', alpha=0.8, width=5, label='Long Score')
    ax2.bar(shorts['date'], shorts['score'], color='red', alpha=0.8, width=5, label='Short Score')
    
    # Plot Average
    avg_score = df_signals['score'].mean()
    ax2.axhline(avg_score, color='black', linestyle='--', linewidth=2, label=f'Avg Dependability: {avg_score:.4f}')
    ax2.axhline(0, color='gray', linewidth=0.8)

    ax2.set_title('Individual Weighted Scores (Metric Output)', fontsize=12)
    ax2.set_ylabel('Weighted Score (Metric Output)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text box for stats
    stats_text = (
        f"Total Signals: {len(df_signals)}\n"
        f"Avg Score: {avg_score:.4f}"
    )
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    output = io.BytesIO()
    plt.tight_layout()
    fig.savefig(output, format='png')
    plt.close(fig) 
    output.seek(0)
    
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
