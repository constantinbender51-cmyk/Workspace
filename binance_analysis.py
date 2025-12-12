import ccxt
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
HORIZON = 30 # Fixed evaluation window length for all signals
PORT = 8080

# Periods to test
SMA_PERIODS = range(10, 300, 20)
RSI_PERIODS = range(10, 30, 5) 
MACD_FAST_EMA = 12
MACD_SLOW_EMA = 26
MACD_SIGNAL_PERIODS = range(5, 15, 2)
EMA_FAST = 50
EMA_SLOW_PERIODS = range(100, 300, 50)

# --- Data Caching ---
analysis_results = []
df_data = None
fwd_matrix = None
weights = None
data_loaded = False

# --- Data Fetching ---
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
    # Daily return calculation (Close[i] / Close[i-1] - 1)
    df['return'] = df['close'].pct_change()
    return df

# --- Core Metric Functions ---

def precompute_forward_matrix(returns_array, horizon):
    """
    Creates a matrix where row 't' contains returns for [t+1, t+2, ... t+horizon].
    Ensures NO LOOKAHEAD BIAS.
    """
    n = len(returns_array)
    matrix = np.full((n, horizon), np.nan)
    
    for day in range(1, horizon + 1):
        # Shift the returns array back by 'day' steps.
        shifted = np.roll(returns_array, -day)
        shifted[-day:] = np.nan
        matrix[:, day-1] = shifted
        
    return matrix

def calculate_dependability_score(signal_indices, is_long, fwd_matrix, weights, horizon):
    """Calculates the average weighted expectancy for a set of signals."""
    
    # Filter indices to ensure enough forward data exists
    # fwd_matrix.shape[0] is the length of the data series.
    valid_limit = fwd_matrix.shape[0] - horizon
    valid_indices = signal_indices[signal_indices < valid_limit]
    
    if len(valid_indices) == 0:
        return 0.0, 0
    
    # Extract forward returns matrix (Shape: N_signals x HORIZON)
    fwd_returns = fwd_matrix[valid_indices]
    
    # Determine direction modifier: 1 for long, -1 for short (profitability positive)
    direction_mod = 1.0 if is_long else -1.0
    
    # Calculate weighted sum for each signal
    # Weighted Sum = Sum(R * Direction * W)
    weighted_sums = np.sum(fwd_returns * direction_mod * weights, axis=1)
    
    # Dependability Score = Average weighted expectancy across all signals
    avg_score = np.mean(weighted_sums)
    
    return avg_score, len(valid_indices)

# --- Indicator Signal Logic ---

def analyze_price_sma_crossover(df, period, fwd_matrix, weights, horizon):
    """Generates signals when Price crosses SMA."""
    sma = df['close'].rolling(window=period).mean().values
    prices = df['close'].values
    
    # Long: Prev Price < Prev SMA AND Curr Price > Curr SMA
    long_signals = np.where((prices[:-1] < sma[:-1]) & (prices[1:] > sma[1:]))[0] + 1
    # Short: Prev Price > Prev SMA AND Curr Price < Curr SMA
    short_signals = np.where((prices[:-1] > sma[:-1]) & (prices[1:] < sma[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    total_count = long_count + short_count
    total_score = (long_score * long_count + short_score * short_count) / total_count if total_count > 0 else 0
    
    return {
        'indicator': f'Price/SMA {period}', 
        'score': total_score, 
        'count': total_count
    }

def analyze_double_ema_crossover(df, fast_period, slow_period, fwd_matrix, weights, horizon):
    """Generates signals when Fast EMA crosses Slow EMA."""
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean().values
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean().values
    
    # Long: Prev Fast < Prev Slow AND Curr Fast > Curr Slow
    long_signals = np.where((fast_ema[:-1] < slow_ema[:-1]) & (fast_ema[1:] > slow_ema[1:]))[0] + 1
    # Short: Prev Fast > Prev Slow AND Curr Fast < Curr Slow
    short_signals = np.where((fast_ema[:-1] > slow_ema[:-1]) & (fast_ema[1:] < slow_ema[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    total_count = long_count + short_count
    total_score = (long_score * long_count + short_score * short_count) / total_count if total_count > 0 else 0
    
    return {
        'indicator': f'EMA {fast_period}/{slow_period}', 
        'score': total_score, 
        'count': total_count
    }

def analyze_rsi_centerline_crossover(df, period, fwd_matrix, weights, horizon):
    """Generates signals when RSI crosses the 50 centerline."""
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # Handle division by zero warning/error if loss is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    centerline = 50
    
    # Long: Prev RSI < 50 AND Curr RSI > 50
    long_signals = np.where((rsi_values[:-1] < centerline) & (rsi_values[1:] > centerline))[0] + 1
    # Short: Prev RSI > 50 AND Curr RSI < 50
    short_signals = np.where((rsi_values[:-1] > centerline) & (rsi_values[1:] < centerline))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    total_count = long_count + short_count
    total_score = (long_score * long_count + short_score * short_count) / total_count if total_count > 0 else 0
    
    return {
        'indicator': f'RSI {period} (Crossover)', 
        'score': total_score, 
        'count': total_count
    }

def analyze_macd_signal_crossover(df, fast, slow, signal_period, fwd_matrix, weights, horizon):
    """Generates signals when MACD line crosses its signal line."""
    
    # Calculate MACD
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    macd_values = macd_line.values
    signal_values = signal_line.values
    
    # Long: Prev MACD < Prev Signal AND Curr MACD > Curr Signal
    long_signals = np.where((macd_values[:-1] < signal_values[:-1]) & (macd_values[1:] > signal_values[1:]))[0] + 1
    # Short: Prev MACD > Prev Signal AND Curr MACD < Curr Signal
    short_signals = np.where((macd_values[:-1] > signal_values[:-1]) & (macd_values[1:] < signal_values[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    total_count = long_count + short_count
    total_score = (long_score * long_count + short_score * short_count) / total_count if total_count > 0 else 0
    
    return {
        'indicator': f'MACD ({fast}/{slow}/{signal_period})', 
        'score': total_score, 
        'count': total_count
    }

# --- Main Logic ---

def load_data_and_run_analysis():
    """Fetches data and runs all indicator analysis."""
    global df_data, fwd_matrix, weights, analysis_results, data_loaded
    
    if data_loaded:
        print("Data already loaded, skipping fetch.")
        return

    try:
        df_data = fetch_binance_data()
        returns_array = df_data['return'].values
        
        # Initialize global metric components
        day_indices = np.arange(1, HORIZON + 1)
        weights = 1 - (day_indices / HORIZON)
        fwd_matrix = precompute_forward_matrix(returns_array, HORIZON)
        
        results = []
        
        print("\n--- Running Analysis ---")
        
        # 1. Price/SMA Crossover
        print("Running Price/SMA Crossover analysis...")
        for p in SMA_PERIODS:
            results.append(analyze_price_sma_crossover(df_data, p, fwd_matrix, weights, HORIZON))

        # 2. Double EMA Crossover (Fixed Fast, varied Slow)
        print("Running Double EMA Crossover analysis...")
        for p_slow in EMA_SLOW_PERIODS:
            results.append(analyze_double_ema_crossover(df_data, EMA_FAST, p_slow, fwd_matrix, weights, HORIZON))

        # 3. RSI Centerline Crossover
        print("Running RSI Centerline Crossover analysis...")
        for p in RSI_PERIODS:
            results.append(analyze_rsi_centerline_crossover(df_data, p, fwd_matrix, weights, HORIZON))
            
        # 4. MACD Signal Crossover
        print("Running MACD Signal Crossover analysis...")
        for p_signal in MACD_SIGNAL_PERIODS:
             results.append(analyze_macd_signal_crossover(df_data, MACD_FAST_EMA, MACD_SLOW_EMA, p_signal, fwd_matrix, weights, HORIZON))

        analysis_results = sorted(results, key=lambda x: x['score'], reverse=True)
        data_loaded = True
        print("\n--- Analysis Complete ---")
        
    except Exception as e:
        print(f"FATAL ERROR during data load or analysis: {e}")
        analysis_results = []
        data_loaded = False


# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the HTML table of results."""
    # Ensure data is loaded before rendering
    if not data_loaded:
        load_data_and_run_analysis()
        
    if not analysis_results:
         return "<h1 class='text-red-500'>Error: Analysis failed or returned no results. Check server logs.</h1>"
         
    # Generate the table content
    table_rows = ""
    rank = 1
    for item in analysis_results:
        # Determine color based on score
        score = item['score']
        color_class = 'bg-red-100' if score < 0 else 'bg-green-100' if score > 0 else 'bg-gray-100'
        
        # Highlight top 3
        if rank <= 3 and score > 0:
            color_class = 'bg-yellow-200 font-bold'
            
        table_rows += f"""
        <tr class="border-b hover:bg-gray-50 {color_class if rank > 3 else ''}">
            <td class="px-6 py-3 font-medium text-gray-900 whitespace-nowrap">{rank}</td>
            <td class="px-6 py-3 font-bold">{item['indicator']}</td>
            <td class="px-6 py-3 text-right" style="color: {'green' if score >= 0 else 'red'};">
                {score:.5f}
            </td>
            <td class="px-6 py-3 text-right">{item['count']}</td>
        </tr>
        """
        rank += 1

    # HTML structure with Tailwind CSS
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Signal Dependability Scorecard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #f7f9fb; }}
            .container {{ max-width: 1200px; }}
            th {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        </style>
    </head>
    <body>
        <div class="container mx-auto p-4 sm:p-8">
            <h1 class="text-3xl font-bold mb-2 text-gray-800">Signal Dependability Scorecard</h1>
            <p class="text-gray-600 mb-6">
                Average Weighted Expectancy (BTC/USDT 2018-Present). Calculated over a **Fixed {HORIZON}-Day Window** using **Linear Decay (1 - d/{HORIZON})**.
            </p>

            <div class="shadow-lg rounded-xl overflow-hidden bg-white">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left text-gray-500 rounded-tl-xl">Rank</th>
                            <th scope="col" class="px-6 py-3 text-left text-gray-500">Indicator / Parameters</th>
                            <th scope="col" class="px-6 py-3 text-right text-gray-500">Avg. Dependability Score</th>
                            <th scope="col" class="px-6 py-3 text-right text-gray-500 rounded-tr-xl">Total Signals (N)</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-200">
                        {table_rows}
                    </tbody>
                </table>
            </div>
            
            <p class="text-sm text-gray-500 mt-4">
                Score Interpretation: The metric measures the average expected weighted return per signal, prioritizing returns on Day 1 (weight ≈ 1) and diminishing to Day {HORIZON} (weight ≈ 0).
            </p>
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    # Initial load is outside the request cycle to avoid blocking the first request
    load_data_and_run_analysis()
    
    print(f"Starting web server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
