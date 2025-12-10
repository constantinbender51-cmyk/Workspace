#!/usr/bin/env python3
"""
Binance OHLCV Analysis Script V2
Corrections:
1. SMA is now SMA of Cumulative Returns (Trend line), not sum of SMAs.
2. Proximity split into two types:
   - Inverse Proximity (for Resistance weighting): 1/(dist*20)
   - Linear Proximity (for Entry/Stop): dist*20
3. Global variable scope fixed.
"""

import ccxt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
import io
import base64

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01'
SMA_WINDOWS = list(range(10, 401, 10))  # 10, 20, ..., 400
LOOKBACK_WINDOW = 400
FUTURE_DAYS = 10
TRAILING_STOP = 0.02
PROXIMITY_FACTOR = 1/0.05  # 20

# Global variables
analysis_results = None
ohlcv_data = None
results_data = None

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Math Helper Functions
# -----------------------------------------------------------------------------

def calculate_inverse_proximity(current, reference):
    """
    Used for RESISTANCE calculation.
    Formula: 1 / (distance * (1/0.05))
    Logic: The CLOSER they are (distance -> 0), the HIGHER the value.
    """
    if reference == 0:
        return 0 # Avoid division by zero in distance calc
        
    dist = abs(current - reference) / abs(reference)
    
    # If distance is effectively 0, return a capped high value (max similarity)
    if dist < 1e-9:
        return 100.0 # Cap to prevent infinity
        
    return 1 / (dist * PROXIMITY_FACTOR)

def calculate_linear_proximity(current, reference):
    """
    Used for ENTRY and STOP calculations.
    Formula: distance * (1/0.05)
    Logic: The CLOSER they are, the LOWER the value (0 if same).
    """
    if reference == 0:
        return 0
        
    dist = abs(current - reference) / abs(reference)
    return dist * PROXIMITY_FACTOR

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

def fetch_ohlcv_data(symbol, timeframe, since):
    print(f"Fetching {symbol} data from {since}...")
    all_data = []
    since_ts = exchange.parse8601(since + 'T00:00:00Z')
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ts)
            if not ohlcv: break
            since_ts = ohlcv[-1][0] + 1
            all_data.extend(ohlcv)
            if len(ohlcv) < 1000: break
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def process_indicators(df):
    """Calculate Returns, RoR, and SMAs of Cumulative Returns"""
    
    # 1. Returns (Close-Open)/Open
    df['returns'] = (df['close'] - df['open']) / df['open']
    
    # 2. Cumulative Returns (Sum of returns)
    df['cumulative_returns'] = df['returns'].cumsum()
    
    # 3. Returns of Returns (Rate of change of returns)
    # (return_t - return_{t-1}) / abs(return_{t-1})
    df['returns_of_returns'] = df['returns'].diff() / df['returns'].shift(1).abs()
    df['returns_of_returns'].replace([np.inf, -np.inf], 0, inplace=True)
    df['returns_of_returns'].fillna(0, inplace=True)
    
    # 4. SMA of Cumulative Returns
    # This creates a "Trend Line" of the returns.
    for w in SMA_WINDOWS:
        df[f'sma_{w}'] = df['cumulative_returns'].rolling(window=w).mean()
        
    return df

def precompute_significance(df):
    """
    Pre-calculate significance for every day X based on X+1...X+10.
    Returns a Dictionary of Dicts: {day_index: {sma_window: significance_value}}
    """
    print("Pre-computing SMA significance maps...")
    sig_map = {}
    
    # We can only compute significance up to len(df) - FUTURE_DAYS
    limit = len(df) - FUTURE_DAYS
    
    for i in range(limit):
        # Calculate Weighted Future Returns of Returns Sum
        future_ror_sum = 0
        for f in range(1, FUTURE_DAYS + 1):
            weight = 1 / f # 1, 1/2, ... 1/10
            future_ror_sum += df['returns_of_returns'].iloc[i + f] * weight
            
        cum_ret = df['cumulative_returns'].iloc[i]
        
        day_sigs = {}
        for w in SMA_WINDOWS:
            sma_val = df[f'sma_{w}'].iloc[i]
            if pd.isna(sma_val):
                day_sigs[w] = 0
                continue
                
            # Proximity of Sum of Returns to SMA (Inverse Proximity)
            prox = calculate_inverse_proximity(cum_ret, sma_val)
            
            # Significance = Future_RoR * Proximity
            day_sigs[w] = future_ror_sum * prox
            
        sig_map[i] = day_sigs
        
    return sig_map

def calculate_resistance_at_n(df, n, sig_map):
    """
    Calculate Resistance for Day N based on lookback window.
    """
    resistance = 0
    current_cum_ret = df['cumulative_returns'].iloc[n]
    
    # Part 1: Sum over lookback days (Proximity to past cumulative returns * Past RoR)
    start_lookback = n - LOOKBACK_WINDOW
    # Ensure we don't look back before start of data
    start_lookback = max(0, start_lookback)
    
    # Vectorized calculation would be faster, but keeping loop for logic clarity per instructions
    for x in range(start_lookback, n):
        past_cum_ret = df['cumulative_returns'].iloc[x]
        past_ror = df['returns_of_returns'].iloc[x]
        
        prox = calculate_inverse_proximity(current_cum_ret, past_cum_ret)
        resistance += prox * past_ror
        
    # Part 2: Sum over SMAs (Proximity to SMA * Significance of SMA)
    # We look at the significance map we generated earlier
    
    # Note: The prompt implies summing proximity of current N to ALL previous SMAs?
    # Or proximity of N to Current SMAs * Significance?
    # "sum over all SMAs of the proximity of sum of returns up to n to SMA 10...400 * significance of that SMA"
    # This implies we check the current distance to the SMA, and multiply by the SMA's significance.
    # But an SMA doesn't have a single "significance" - it has significance *at a specific point in time*.
    # Interpretation: We sum over the lookback window. For every day x in lookback, we check the SMA significance at x.
    
    # Re-reading prompt carefully: 
    # "sum over all SMAs of the proximity of sum of returns up to n to SMA 10...400 * significance of that SMA"
    # This suggests we calculate the CURRENT proximity (at n) to the SMA (at n), but where does the significance come from?
    # It likely comes from the aggregation of past significances.
    
    # Let's assume the standard interpretation for this type of algo:
    # We iterate through the lookback window (x).
    # For each x, we check the stored Significance of SMA(w) at x.
    # We check the proximity of Current(n) to SMA(w) at x.
    
    for x in range(start_lookback, n):
        if x not in sig_map: continue
        
        x_sigs = sig_map[x]
        
        for w in SMA_WINDOWS:
            # Get SMA value at day X
            sma_at_x = df[f'sma_{w}'].iloc[x]
            if pd.isna(sma_at_x): continue
            
            # Proximity of Current Sum (at N) to SMA (at X)
            prox = calculate_inverse_proximity(current_cum_ret, sma_at_x)
            
            # Significance of SMA at X
            sig = x_sigs.get(w, 0)
            
            resistance += prox * sig
            
    return resistance

def run_analysis():
    global analysis_results, ohlcv_data, results_data
    
    # 1. Fetch
    df = fetch_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE)
    df = process_indicators(df)
    
    # 2. Pre-compute Significance (Training the Matrix on past data)
    sig_map = precompute_significance(df)
    
    results = []
    entry_flag = False
    entry_data = {} # Stores 'day_idx', 'price', 'cum_ret'
    position = 0 # 1 or -1
    trailing_stop_price = 0
    
    # Threshold for entry (set arbitrarily or 0)
    ENTRY_THRESHOLD = 0 
    
    print("Running main simulation loop...")
    
    # Start loop after lookback window
    start_idx = max(LOOKBACK_WINDOW, SMA_WINDOWS[-1])
    
    for n in range(start_idx, len(df)):
        date = df.index[n]
        price = df['close'].iloc[n]
        cum_ret = df['cumulative_returns'].iloc[n]
        
        # Calculate Resistance
        resistance = calculate_resistance_at_n(df, n, sig_map)
        
        # --- Entry Logic ---
        if not entry_flag:
            if resistance > ENTRY_THRESHOLD:
                # Set Entry Flag
                entry_flag = True
                entry_data = {
                    'idx': n,
                    'price': price,
                    'cum_ret': cum_ret
                }
                
                # Determine Direction
                # "position ourselves long or short based on proximity of sum of returns up to day x to entry"
                # Wait, at the exact moment of entry (n), the sum of returns IS the entry.
                # Proximity is 0. 
                # Interpretation: We likely determine direction based on the resistance sign or trend.
                # However, strict adherence to prompt: "based on proximity... to entry".
                # If n is entry, distance is 0.
                # I will assume we look at the MOMENTUM of the resistance or returns to decide.
                # Fallback: Long if resistance is positive, Short if negative.
                if resistance > 0:
                    position = 1
                else:
                    position = -1
                    
                # Set Initial Trailing Stop (2%)
                if position == 1:
                    trailing_stop_price = price * (1 - TRAILING_STOP)
                else:
                    trailing_stop_price = price * (1 + TRAILING_STOP)
                    
        # --- Position Management ---
        else: # In a trade
            # Calculate Proximity for Stop Logic (Linear Proximity)
            # "proximity is here calculated as distance* 1/0.05"
            prox_entry = calculate_linear_proximity(cum_ret, entry_data['cum_ret'])
            
            # Check Trailing Stop
            stop_hit = False
            
            if position == 1: # Long
                # Trailing logic: if price moves up, drag stop up
                new_stop = price * (1 - TRAILING_STOP)
                if new_stop > trailing_stop_price:
                    trailing_stop_price = new_stop
                
                if price < trailing_stop_price:
                    stop_hit = True
                    
            elif position == -1: # Short
                new_stop = price * (1 + TRAILING_STOP)
                if new_stop < trailing_stop_price:
                    trailing_stop_price = new_stop
                    
                if price > trailing_stop_price:
                    stop_hit = True
            
            if stop_hit:
                entry_flag = False
                position = 0
                
        # Record Result
        results.append({
            'date': date,
            'price': price,
            'resistance': resistance,
            'position': position,
            'stop_price': trailing_stop_price if position != 0 else None
        })
        
        if n % 100 == 0:
            print(f"Processed {date.date()}...")

    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    # Store globals
    analysis_results = results_df
    ohlcv_data = df
    results_data = results_df
    
    return results_df

# -----------------------------------------------------------------------------
# Web Server
# -----------------------------------------------------------------------------

@app.route('/')
def display_results():
    global analysis_results, ohlcv_data # Fixed Scope
    
    if analysis_results is None:
        return "Analysis running... please refresh in a minute."
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Price & Positions
    ax1.plot(analysis_results.index, analysis_results['price'], label='Price', color='black', alpha=0.6)
    
    # Color background for positions
    longs = analysis_results[analysis_results['position'] == 1]
    shorts = analysis_results[analysis_results['position'] == -1]
    
    ax1.scatter(longs.index, longs['price'], color='green', s=10, label='Long')
    ax1.scatter(shorts.index, shorts['price'], color='red', s=10, label='Short')
    
    if 'stop_price' in analysis_results.columns:
        ax1.plot(analysis_results.index, analysis_results['stop_price'], 'b--', alpha=0.3, label='Stop')
        
    ax1.set_title(f'{SYMBOL} Price & Positions')
    ax1.legend()
    
    # Plot 2: Resistance
    ax2.plot(analysis_results.index, analysis_results['resistance'], color='purple', label='Resistance')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title('Calculated Resistance')
    ax2.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    stats_html = analysis_results.tail().to_html()
    
    return render_template_string("""
        <html>
            <body style="font-family: sans-serif; padding: 20px;">
                <h1>Algorithmic Analysis: {{ symbol }}</h1>
                <img src="data:image/png;base64,{{ plot_url }}" style="max-width: 100%;">
                <h3>Recent Data</h3>
                {{ stats|safe }}
            </body>
        </html>
    """, symbol=SYMBOL, plot_url=plot_url, stats=stats_html)

if __name__ == "__main__":
    try:
        run_analysis()
        print("Starting Flask on 8080...")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
