#!/usr/bin/env python3
"""
Binance OHLCV Analysis Script V2.2
Updates:
1. Proximity Capped at 1.0 for Resistance (1 / (dist*20)).
2. Leverage Calculation: dist*20, capped at 5.0.
3. Daily Position Sizing: Return = Daily_Ret * Leverage.
4. Added Strategy Equity & Leverage plots.
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
PROXIMITY_FACTOR = 20.0  # 1/0.05

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
    Formula: 1 / (distance * 20)
    Constraint: Max 1.0
    Logic: If distance < 5%, returns 1.0. If distance > 5%, decays.
    """
    if reference == 0:
        return 0
        
    dist = abs(current - reference) / abs(reference)
    
    # Avoid division by zero if dist is exactly 0
    if dist < 1e-9:
        return 1.0
        
    raw_prox = 1.0 / (dist * PROXIMITY_FACTOR)
    return min(1.0, raw_prox)

def calculate_leverage(current, reference):
    """
    Used for LEVERAGE calculation on active days.
    Formula: distance * 20
    Constraint: Max 5.0
    Logic: Leverage increases as price moves away from entry.
    """
    if reference == 0:
        return 0
        
    dist = abs(current - reference) / abs(reference)
    raw_lev = dist * PROXIMITY_FACTOR
    return min(5.0, raw_lev)

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

def fetch_ohlcv_data(symbol, timeframe, since):
    print(f"Fetching {symbol} data from {since}...")
    all_data = []
    since_ts = exchange.parse8601(since + 'T00:00:00Z')
    limit = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ts, limit=limit)
            if not ohlcv: 
                break
            
            since_ts = ohlcv[-1][0] + 1
            all_data.extend(ohlcv)
            
            if len(ohlcv) < limit:
                break
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Total candles fetched: {len(df)}")
    return df

def process_indicators(df):
    # 1. Returns
    df['returns'] = (df['close'] - df['open']) / df['open']
    
    # 2. Cumulative Returns
    df['cumulative_returns'] = df['returns'].cumsum()
    
    # 3. Returns of Returns
    df['returns_of_returns'] = df['returns'].diff() / df['returns'].shift(1).abs()
    df['returns_of_returns'].replace([np.inf, -np.inf], 0, inplace=True)
    df['returns_of_returns'].fillna(0, inplace=True)
    
    # 4. SMA of Cumulative Returns
    for w in SMA_WINDOWS:
        df[f'sma_{w}'] = df['cumulative_returns'].rolling(window=w).mean()
        
    return df

def precompute_significance(df):
    print("Pre-computing SMA significance maps...")
    sig_map = {}
    limit = len(df) - FUTURE_DAYS
    
    for i in range(limit):
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
                
            prox = calculate_inverse_proximity(cum_ret, sma_val)
            day_sigs[w] = future_ror_sum * prox
            
        sig_map[i] = day_sigs
        
    return sig_map

def calculate_resistance_at_n(df, n, sig_map):
    resistance = 0
    current_cum_ret = df['cumulative_returns'].iloc[n]
    start_lookback = max(0, n - LOOKBACK_WINDOW)
    
    for x in range(start_lookback, n):
        # Part 1: Past Cumulative Returns
        past_cum_ret = df['cumulative_returns'].iloc[x]
        past_ror = df['returns_of_returns'].iloc[x]
        prox = calculate_inverse_proximity(current_cum_ret, past_cum_ret)
        resistance += prox * past_ror
        
        # Part 2: SMAs
        if x in sig_map:
            x_sigs = sig_map[x]
            for w in SMA_WINDOWS:
                sma_at_x = df[f'sma_{w}'].iloc[x]
                if pd.isna(sma_at_x): continue
                
                prox_sma = calculate_inverse_proximity(current_cum_ret, sma_at_x)
                sig = x_sigs.get(w, 0)
                
                resistance += prox_sma * sig
            
    return resistance

def run_analysis():
    global analysis_results, ohlcv_data, results_data
    
    df = fetch_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE)
    if df.empty: return pd.DataFrame()

    df = process_indicators(df)
    sig_map = precompute_significance(df)
    
    results = []
    
    # State Variables
    entry_flag = False
    entry_cum_ret = 0
    position = 0          # 1 for Long, -1 for Short
    trailing_stop_price = 0
    
    # Strategy Performance Tracking
    strategy_equity = 1.0  # Starting at 100%
    
    ENTRY_THRESHOLD = 0.5 
    
    print("Running main simulation loop...")
    start_idx = max(LOOKBACK_WINDOW, SMA_WINDOWS[-1])
    
    for n in range(start_idx, len(df)):
        date = df.index[n]
        price = df['close'].iloc[n]
        cum_ret = df['cumulative_returns'].iloc[n]
        day_return = df['returns'].iloc[n]
        
        resistance = calculate_resistance_at_n(df, n, sig_map)
        
        leverage = 0
        strat_day_ret = 0
        
        # --- Entry Logic ---
        if not entry_flag:
            # Check for Entry
            if abs(resistance) > ENTRY_THRESHOLD:
                entry_flag = True
                entry_cum_ret = cum_ret # Store entry reference
                
                # Determine Direction
                if resistance > 0:
                    position = 1
                    trailing_stop_price = price * (1 - TRAILING_STOP)
                else:
                    position = -1
                    trailing_stop_price = price * (1 + TRAILING_STOP)
                    
        # --- Position Management ---
        else:
            # We are in a trade (including the day triggered if we entered at Open, 
            # but usually we enter next day. For simplicity, we process 'Active' status).
            
            # 1. Calculate Leverage based on distance from ENTRY
            leverage = calculate_leverage(cum_ret, entry_cum_ret)
            
            # 2. Calculate Strategy Return
            strat_day_ret = day_return * position * leverage
            strategy_equity *= (1 + strat_day_ret)
            
            # 3. Check Trailing Stop
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
                leverage = 0 # Leverage drops to 0 on exit
        
        # Capture Data
        results.append({
            'date': date,
            'price': price,
            'resistance': resistance,
            'position': position,
            'leverage': leverage,
            'strategy_equity': strategy_equity,
            'cumulative_returns': cum_ret,
            'stop_price': trailing_stop_price if entry_flag else None
        })
        
        if n % 200 == 0:
            print(f"Processed {date.date()}... Eq: {strategy_equity:.2f}")

    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    # Add SMAs for plotting
    results_df['sma_50'] = df['sma_50'].reindex(results_df.index)
    results_df['sma_400'] = df['sma_400'].reindex(results_df.index)

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
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 22), sharex=True)
    
    # 1. Strategy Performance
    ax1.plot(analysis_results.index, analysis_results['strategy_equity'], 'g', linewidth=2, label='Strategy Equity')
    # Normalized Buy & Hold for comparison
    bnh = analysis_results['price'] / analysis_results['price'].iloc[0]
    ax1.plot(analysis_results.index, bnh, 'gray', alpha=0.5, label='Buy & Hold (Norm)')
    ax1.set_title('Strategy Performance vs Buy & Hold')
    ax1.set_yscale('log') # Log scale is often better for equity
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Leverage Used
    ax2.plot(analysis_results.index, analysis_results['leverage'], 'b', label='Daily Leverage')
    ax2.fill_between(analysis_results.index, analysis_results['leverage'], alpha=0.3, color='blue')
    ax2.set_title('Leverage (Distance from Entry * 20, Max 5)')
    ax2.set_ylabel('Leverage Multiplier')
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.3)
    
    # 3. Price & Signals
    ax3.plot(analysis_results.index, analysis_results['price'], 'k', alpha=0.6, label='Price')
    longs = analysis_results[analysis_results['position'] == 1]
    shorts = analysis_results[analysis_results['position'] == -1]
    ax3.scatter(longs.index, longs['price'], c='g', s=10, label='Long Active')
    ax3.scatter(shorts.index, shorts['price'], c='r', s=10, label='Short Active')
    ax3.set_title(f'{SYMBOL} Price Action & Position Status')
    ax3.legend()
    
    # 4. Resistance
    ax4.plot(analysis_results.index, analysis_results['resistance'], 'purple', label='Resistance')
    ax4.axhline(0, color='gray', linestyle='--')
    ax4.set_title('Resistance Oscillator (Trigger > 0.5)')
    ax4.grid(True, alpha=0.3)
    
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
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Trading Algorithm Results</h1>
                    <p><strong>Logic:</strong> Proximity capped at 1.0. Leverage = Dist * 20 (Max 5).</p>
                    <img src="data:image/png;base64,{{ plot_url }}">
                    <h3>Recent Metrics</h3>
                    {{ stats|safe }}
                </div>
            </body>
        </html>
    """, plot_url=plot_url, stats=stats)

if __name__ == "__main__":
    try:
        run_analysis()
        print("Starting Flask on 8080...")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
