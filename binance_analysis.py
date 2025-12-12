import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for server environments without display
import matplotlib.pyplot as plt
from flask import Flask, send_file
import io
import time
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
SMA_START = 10
SMA_END = 400
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
            
            # Rate limit handling (Binance is usually generous, but safety first)
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

def precompute_forward_matrix(returns, horizon):
    """
    Creates a matrix where row 't' contains returns for [t+1, t+2, ... t+horizon].
    This allows O(1) lookups during the SMA loop.
    """
    values = returns.values
    n = len(values)
    # Create a 2D matrix of shape (n, horizon) filled with NaN
    matrix = np.full((n, horizon), np.nan)
    
    for day in range(1, horizon + 1):
        # Shift returns back by 'day' steps
        # If today is t, we want return at t+day in column (day-1)
        shifted = np.roll(values, -day)
        # Set the invalid end elements to NaN
        shifted[-day:] = np.nan
        matrix[:, day-1] = shifted
        
    return matrix

def calculate_dependability_scores(df):
    """Calculates the dependability score for SMA 10-400."""
    prices = df['close'].values
    returns = df['return']
    
    # 1. Precompute forward return matrix
    print("Precomputing forward returns matrix...")
    fwd_matrix = precompute_forward_matrix(returns, HORIZON)
    
    # 2. Precompute Weights: Linear Decay (1 - d/30)
    days = np.arange(1, HORIZON + 1)
    weights = 1 - (days / HORIZON) 
    
    # Initialize results dictionary
    results = {
        'sma': [], 
        'score_total': [], 
        'score_long': [], 
        'score_short': [], 
        'num_signals': []
    }
    
    print("Calculating scores for SMAs 10-400...")
    for period in range(SMA_START, SMA_END + 1):
        # Calculate SMA
        sma = df['close'].rolling(window=period).mean().values
        
        # Identify Crosses (Vectorized)
        prev_close = prices[:-1]
        curr_close = prices[1:]
        prev_sma = sma[:-1]
        curr_sma = sma[1:]
        
        # Boolean arrays for crossovers
        long_signals = (prev_close < prev_sma) & (curr_close > curr_sma)
        short_signals = (prev_close > prev_sma) & (curr_close < curr_sma)
        
        # Indices in the original array (add 1 because of shifting)
        long_indices = np.where(long_signals)[0] + 1
        short_indices = np.where(short_signals)[0] + 1
        
        # Filter indices near the end of data (not enough future days)
        valid_limit = len(prices) - HORIZON
        long_indices = long_indices[long_indices < valid_limit]
        short_indices = short_indices[short_indices < valid_limit]
        
        # Initialize scores for this SMA
        avg_long = np.nan
        avg_short = np.nan
        avg_total = np.nan
        count = len(long_indices) + len(short_indices)

        if count == 0:
            results['sma'].append(period)
            results['score_total'].append(0)
            results['score_long'].append(0)
            results['score_short'].append(0)
            results['num_signals'].append(0)
            continue

        # --- Calculate Long Scores ---
        if len(long_indices) > 0:
            long_fwd = fwd_matrix[long_indices]
            # Sum(R * Weight)
            long_weighted_sums = np.sum(long_fwd * weights, axis=1)
            # We want the average dependability per signal
            avg_long = np.mean(long_weighted_sums)
            
        # --- Calculate Short Scores ---
        if len(short_indices) > 0:
            short_fwd = fwd_matrix[short_indices]
            # Sum(-R * Weight) -> Profit from price drop
            short_weighted_sums = np.sum(short_fwd * -1 * weights, axis=1)
            avg_short = np.mean(short_weighted_sums)
            
        # --- Calculate Total Score ---
        # Combine all weighted sum values to get true weighted average
        all_sums = []
        if len(long_indices) > 0:
            all_sums.extend(long_weighted_sums)
        if len(short_indices) > 0:
            all_sums.extend(short_weighted_sums)
            
        if all_sums:
            avg_total = np.mean(all_sums)
        else:
            avg_total = 0

        # Fill NaNs with 0 for plotting if no signals of that type occurred
        results['sma'].append(period)
        results['score_total'].append(avg_total if not np.isnan(avg_total) else 0)
        results['score_long'].append(avg_long if not np.isnan(avg_long) else 0)
        results['score_short'].append(avg_short if not np.isnan(avg_short) else 0)
        results['num_signals'].append(count)
        
    return pd.DataFrame(results)

# --- Execution & Caching ---
# We calculate once on startup
print("Starting Data Fetch & Analysis...")
df_data = fetch_binance_data()
df_results = calculate_dependability_scores(df_data)
print("Analysis Complete.")

@app.route('/')
def plot_png():
    """Generates the plot and serves it."""
    
    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Dependability Score (Long vs Short vs Total)
    ax1.plot(df_results['sma'], df_results['score_total'], color='black', linewidth=2, alpha=0.8, label='Total')
    ax1.plot(df_results['sma'], df_results['score_long'], color='#2ecc71', linewidth=1.5, linestyle='--', label='Long Only')
    ax1.plot(df_results['sma'], df_results['score_short'], color='#e74c3c', linewidth=1.5, linestyle='--', label='Short Only')
    
    ax1.set_title(f'Signal Dependability: Long vs Short (BTC/USDT 2018-Present)\nMetric: Linear Decay (1 - day/30)', fontsize=14)
    ax1.set_ylabel('Dependability Score', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add Zero Line
    ax1.axhline(0, color='gray', linewidth=0.8, alpha=0.5)

    # Highlight max Total score
    max_idx = df_results['score_total'].idxmax()
    best_sma = df_results.loc[max_idx, 'sma']
    best_score = df_results.loc[max_idx, 'score_total']
    ax1.annotate(f'Best Total: SMA {best_sma}\nScore: {best_score:.4f}', 
                 xy=(best_sma, best_score), 
                 xytext=(best_sma+20, best_score),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Plot 2: Number of Signals
    ax2.bar(df_results['sma'], df_results['num_signals'], color='#3498db', alpha=0.6, width=1.0)
    ax2.set_title('Frequency of Signals', fontsize=12)
    ax2.set_xlabel('SMA Period', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Save to buffer
    output = io.BytesIO()
    plt.tight_layout()
    fig.savefig(output, format='png')
    plt.close(fig) # Clear memory
    output.seek(0)
    
    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
