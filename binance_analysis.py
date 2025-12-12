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
    # Days 1 to 30. Weight for day 1 = 29/30? No, 1 - 1/30 = 0.96. 
    # Or should it be 1 - (d-1)/30 to start at 1.0? 
    # User said "1-day/30". If day=30, weight=0. 
    # Let's use weights that don't hit zero exactly at the end to keep the last day relevant?
    # Actually, 1 - d/30 is fine.
    days = np.arange(1, HORIZON + 1)
    weights = 1 - (days / HORIZON) 
    # Avoid zero weight if desired, but formula is formula.
    # Note: If day=30, weight=0. That renders the 30th day useless. 
    # Often "linear decay" implies triangular window. We'll stick to the user's "1 - day/30" literal.
    
    results = {'sma': [], 'score': [], 'num_signals': []}
    
    print("Calculating scores for SMAs 10-400...")
    for period in range(SMA_START, SMA_END + 1):
        # Calculate SMA
        sma = df['close'].rolling(window=period).mean().values
        
        # Identify Crosses (Vectorized)
        # prev_price < prev_sma AND curr_price > curr_sma (Long)
        # prev_price > prev_sma AND curr_price < curr_sma (Short)
        
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
        
        # Combine valid indices
        # We need to filter indices that don't have enough future data (where fwd_matrix row contains NaN)
        valid_limit = len(prices) - HORIZON
        
        long_indices = long_indices[long_indices < valid_limit]
        short_indices = short_indices[short_indices < valid_limit]
        
        if len(long_indices) == 0 and len(short_indices) == 0:
            results['sma'].append(period)
            results['score'].append(0)
            results['num_signals'].append(0)
            continue

        # Get forward returns for these days
        # Shape: (num_signals, 30)
        long_fwd = fwd_matrix[long_indices]
        short_fwd = fwd_matrix[short_indices]
        
        # Calculate Weighted Sums
        # Long Score: Sum(R * Weight)
        long_scores = np.sum(long_fwd * weights, axis=1) if len(long_indices) > 0 else np.array([])
        
        # Short Score: Sum(R * -1 * Weight) -> Sum(-R * Weight)
        short_scores = np.sum(short_fwd * -1 * weights, axis=1) if len(short_indices) > 0 else np.array([])
        
        # Total Dependability: Average of all signal scores
        all_scores = np.concatenate([long_scores, short_scores])
        avg_score = np.mean(all_scores)
        
        results['sma'].append(period)
        results['score'].append(avg_score)
        results['num_signals'].append(len(all_scores))
        
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Dependability Score
    ax1.plot(df_results['sma'], df_results['score'], color='#2ecc71', linewidth=2)
    ax1.set_title(f'Signal Dependability Score (BTC/USDT 2018-Present)\nMetric: Linear Decay (1 - day/30) over 30 days', fontsize=14)
    ax1.set_ylabel('Dependability Score (Weighted Avg Return)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Highlight max score
    max_idx = df_results['score'].idxmax()
    best_sma = df_results.loc[max_idx, 'sma']
    best_score = df_results.loc[max_idx, 'score']
    ax1.annotate(f'Best SMA: {best_sma}\nScore: {best_score:.4f}', 
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
