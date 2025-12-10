#!/usr/bin/env python3
"""
Binance OHLCV Analysis Script V2.1
Updates:
1. Fixed fetching: Explicitly requests 1000 candles per page to ensure full history.
2. Enhanced Visualization: Added 4-panel debug plots to verify math/proximity.
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
    """
    if reference == 0:
        return 0
        
    dist = abs(current - reference) / abs(reference)
    
    # Cap very high similarity to prevent infinity
    if dist < 1e-9:
        return 100.0 
        
    return 1 / (dist * PROXIMITY_FACTOR)

def calculate_linear_proximity(current, reference):
    """
    Used for ENTRY and STOP calculations.
    Formula: distance * (1/0.05)
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
    limit = 1000  # Explicitly set limit to max allowed by Binance
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ts, limit=limit)
            if not ohlcv: 
                break
            
            since_ts = ohlcv[-1][0] + 1
            all_data.extend(ohlcv)
            
            print(f"Fetched {len(ohlcv)} candles, last date: {pd.to_datetime(ohlcv[-1][0], unit='ms')}")
            
            # If we got fewer candles than the limit, we've reached the end
            if len(ohlcv) < limit:
                break
                
            time.sleep(0.1) # Respect rate limits
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Total candles fetched: {len(df)}")
    return df

def process_indicators(df):
    """Calculate Returns, RoR, and SMAs of Cumulative Returns"""
    
    # 1. Returns
    df['returns'] = (df['close'] - df['open']) / df['open']
    
    # 2. Cumulative Returns (The Curve)
    df['cumulative_returns'] = df['returns'].cumsum()
    
    # 3. Returns of Returns (Acceleration)
    df['returns_of_returns'] = df['returns'].diff() / df['returns'].shift(1).abs()
    df['returns_of_returns'].replace([np.inf, -np.inf], 0, inplace=True)
    df['returns_of_returns'].fillna(0, inplace=True)
    
    # 4. SMA of Cumulative Returns (Trend Lines)
    for w in SMA_WINDOWS:
        df[f'sma_{w}'] = df['cumulative_returns'].rolling(window=w).mean()
        
    return df

def precompute_significance(df):
    """
    Pre-calculate significance for every day X based on X+1...X+10.
    """
    print("Pre-computing SMA significance maps...")
    sig_map = {}
    limit = len(df) - FUTURE_DAYS
    
    for i in range(limit):
        # Calculate Weighted Future Returns of Returns Sum
        future_ror_sum = 0
        for f in range(1, FUTURE_DAYS + 1):
            weight = 1 / f
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
    start_lookback = max(0, n - LOOKBACK_WINDOW)
    
    for x in range(start_lookback, n):
        # Part 1: Proximity to past cumulative returns
        past_cum_ret = df['cumulative_returns'].iloc[x]
        past_ror = df['returns_of_returns'].iloc[x]
        prox = calculate_inverse_proximity(current_cum_ret, past_cum_ret)
        resistance += prox * past_ror
        
        # Part 2: Proximity to SMAs using historic significance
        if x in sig_map:
            x_sigs = sig_map[x]
            for w in SMA_WINDOWS:
                sma_at_x = df[f'sma_{w}'].iloc[x]
                if pd.isna(sma_at_x): continue
                
                # Proximity of Current Sum (at N) to SMA (at X)
                prox_sma = calculate_inverse_proximity(current_cum_ret, sma_at_x)
                sig = x_sigs.get(w, 0)
                
                resistance += prox_sma * sig
            
    return resistance

def run_analysis():
    global analysis_results, ohlcv_data, results_data
    
    df = fetch_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE)
    if df.empty:
        print("No data fetched. Check connection or symbol.")
        return pd.DataFrame()

    df = process_indicators(df)
    sig_map = precompute_significance(df)
    
    results = []
    entry_flag = False
    entry_data = {} 
    position = 0 
    trailing_stop_price = 0
    
    # Threshold for entry 
    ENTRY_THRESHOLD = 0.5 # Slightly non-zero to avoid noise
    
    print("Running main simulation loop...")
    start_idx = max(LOOKBACK_WINDOW, SMA_WINDOWS[-1])
    
    for n in range(start_idx, len(df)):
        date = df.index[n]
        price = df['close'].iloc[n]
        cum_ret = df['cumulative_returns'].iloc[n]
        
        resistance = calculate_resistance_at_n(df, n, sig_map)
        
        # --- Entry Logic ---
        if not entry_flag:
            if abs(resistance) > ENTRY_THRESHOLD:
                entry_flag = True
                entry_data = {'idx': n, 'price': price, 'cum_ret': cum_ret}
                
                # Direction based on Resistance polarity
                if resistance > 0:
                    position = 1
                    trailing_stop_price = price * (1 - TRAILING_STOP)
                else:
                    position = -1
                    trailing_stop_price = price * (1 + TRAILING_STOP)
        
        # --- Position Management ---
        else: 
            prox_entry = calculate_linear_proximity(cum_ret, entry_data['cum_ret'])
            stop_hit = False
            
            if position == 1: # Long
                new_stop = price * (1 - TRAILING_STOP)
                if new_stop > trailing_stop_price: trailing_stop_price = new_stop
                if price < trailing_stop_price: stop_hit = True
                    
            elif position == -1: # Short
                new_stop = price * (1 + TRAILING_STOP)
                if new_stop < trailing_stop_price: trailing_stop_price = new_stop
                if price > trailing_stop_price: stop_hit = True
            
            if stop_hit:
                entry_flag = False
                position = 0
                
        results.append({
            'date': date,
            'price': price,
            'resistance': resistance,
            'position': position,
            'stop_price': trailing_stop_price if position != 0 else None,
            'cumulative_returns': cum_ret,
            'ror': df['returns_of_returns'].iloc[n]
        })
        
        if n % 200 == 0:
            print(f"Processed {date.date()}...")

    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    # Add SMAs to results for plotting
    for w in [50, 100, 200, 400]:
        results_df[f'sma_{w}'] = df[f'sma_{w}'].reindex(results_df.index)

    analysis_results = results_df
    ohlcv_data = df
    return results_df

# -----------------------------------------------------------------------------
# Web Server
# -----------------------------------------------------------------------------

@app.route('/')
def display_results():
    global analysis_results
    
    if analysis_results is None:
        return "Analysis running... please refresh in a minute."
        
    # Create 4 Panel Visualization
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    
    # 1. Price & Signals
    ax1.plot(analysis_results.index, analysis_results['price'], 'k', label='Price', alpha=0.6)
    longs = analysis_results[analysis_results['position'] == 1]
    shorts = analysis_results[analysis_results['position'] == -1]
    ax1.scatter(longs.index, longs['price'], c='g', marker='^', s=30, label='Long', zorder=5)
    ax1.scatter(shorts.index, shorts['price'], c='r', marker='v', s=30, label='Short', zorder=5)
    if 'stop_price' in analysis_results.columns:
        ax1.plot(analysis_results.index, analysis_results['stop_price'], 'b--', alpha=0.3, label='Stop')
    ax1.set_title(f'{SYMBOL} Price Action & Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # 2. The Math Check: Cumulative Returns vs SMAs
    ax2.plot(analysis_results.index, analysis_results['cumulative_returns'], 'b', linewidth=1.5, label='Cum. Returns')
    ax2.plot(analysis_results.index, analysis_results['sma_50'], 'orange', linestyle='--', alpha=0.8, label='SMA 50 (Cum)')
    ax2.plot(analysis_results.index, analysis_results['sma_400'], 'red', linestyle='--', alpha=0.8, label='SMA 400 (Cum)')
    ax2.set_title('Logic Verification: Cumulative Returns vs SMAs (The "Distance" Basis)')
    ax2.set_ylabel('Cum. Returns %')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    # 3. Resistance Oscillator
    ax3.plot(analysis_results.index, analysis_results['resistance'], 'purple', label='Resistance')
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_title('Calculated Resistance Output')
    ax3.set_ylabel('Resistance')
    ax3.grid(True, alpha=0.2)
    
    # 4. Input Factor: Returns of Returns
    ax4.plot(analysis_results.index, analysis_results['ror'], 'gray', alpha=0.6, label='RoR')
    ax4.set_title('Input Factor: Returns of Returns (Acceleration)')
    ax4.set_ylabel('RoR')
    ax4.set_ylim(-5, 5) # Clip outliers for readability
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    stats = analysis_results.tail(10).to_html()
    
    return render_template_string("""
        <html>
            <head>
                <style>
                    body { font-family: sans-serif; padding: 20px; background: #f0f0f0; }
                    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                    img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                    table { border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.9em; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Algorithmic Analysis: {{ symbol }}</h1>
                    <img src="data:image/png;base64,{{ plot_url }}">
                    <h3>Latest Data Points</h3>
                    {{ stats|safe }}
                </div>
            </body>
        </html>
    """, symbol=SYMBOL, plot_url=plot_url, stats=stats)

if __name__ == "__main__":
    try:
        run_analysis()
        print("Starting Flask on 8080...")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
