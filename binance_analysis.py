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

# --- Expanded Search Space ---
SMA_PERIODS = range(20, 500, 30)        # Price Crossover (Trend)
RSI_PERIODS = range(10, 40, 5)          # Centerline Crossover (Momentum)
EMA_FAST = 50
EMA_SLOW_PERIODS = range(100, 400, 50)  # Double EMA Crossover (Trend)
MACD_FAST_EMA = 12
MACD_SLOW_EMA = 26
MACD_SIGNAL_PERIODS = range(5, 20, 5)   # MACD Signal Crossover (Momentum)
STOCH_K_PERIODS = range(10, 40, 10)     # Stochastic %K/%D Crossover (Momentum)
STOCH_D_PERIOD = 3
BB_PERIODS = [20, 50, 100]              # Bollinger Bands (Volatility/Reversion)
BB_STDEVS = [1.5, 2.0, 2.5]             # Std Dev Multipliers

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

def combine_results(long_score, long_count, short_score, short_count, indicator_name):
    """Helper function to combine long and short scores into a total score."""
    total_count = long_count + short_count
    total_score = (long_score * long_count + short_score * short_count) / total_count if total_count > 0 else 0
    return {
        'indicator': indicator_name, 
        'score': total_score, 
        'count': total_count
    }

def analyze_price_sma_crossover(df, period, fwd_matrix, weights, horizon):
    """Generates signals when Price crosses SMA."""
    sma = df['close'].rolling(window=period).mean().values
    prices = df['close'].values
    
    long_signals = np.where((prices[:-1] < sma[:-1]) & (prices[1:] > sma[1:]))[0] + 1
    short_signals = np.where((prices[:-1] > sma[:-1]) & (prices[1:] < sma[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)
    
    return combine_results(long_score, long_count, short_score, short_count, f'Price/SMA {period}')

def analyze_double_ema_crossover(df, fast_period, slow_period, fwd_matrix, weights, horizon):
    """Generates signals when Fast EMA crosses Slow EMA."""
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean().values
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean().values
    
    long_signals = np.where((fast_ema[:-1] < slow_ema[:-1]) & (fast_ema[1:] > slow_ema[1:]))[0] + 1
    short_signals = np.where((fast_ema[:-1] > slow_ema[:-1]) & (fast_ema[1:] < slow_ema[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    return combine_results(long_score, long_count, short_score, short_count, f'EMA {fast_period}/{slow_period}')

def analyze_rsi_centerline_crossover(df, period, fwd_matrix, weights, horizon):
    """Generates signals when RSI crosses the 50 centerline."""
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    centerline = 50
    
    long_signals = np.where((rsi_values[:-1] < centerline) & (rsi_values[1:] > centerline))[0] + 1
    short_signals = np.where((rsi_values[:-1] > centerline) & (rsi_values[1:] < centerline))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    return combine_results(long_score, long_count, short_score, short_count, f'RSI {period} (Crossover)')

def analyze_macd_signal_crossover(df, fast, slow, signal_period, fwd_matrix, weights, horizon):
    """Generates signals when MACD line crosses its signal line."""
    
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    macd_values = macd_line.values
    signal_values = signal_line.values
    
    long_signals = np.where((macd_values[:-1] < signal_values[:-1]) & (macd_values[1:] > signal_values[1:]))[0] + 1
    short_signals = np.where((macd_values[:-1] > signal_values[:-1]) & (macd_values[1:] < signal_values[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    return combine_results(long_score, long_count, short_score, short_count, f'MACD ({fast}/{slow}/{signal_period})')
    
def analyze_stochastic_crossover(df, k_period, d_period, fwd_matrix, weights, horizon):
    """Generates signals when %K crosses %D line."""
    
    # Calculate %K
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    k_line = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D (SMA of %K)
    d_line = k_line.rolling(window=d_period).mean()
    
    k_values = k_line.values
    d_values = d_line.values
    
    # Long: Prev K < Prev D AND Curr K > Curr D
    long_signals = np.where((k_values[:-1] < d_values[:-1]) & (k_values[1:] > d_values[1:]))[0] + 1
    # Short: Prev K > Prev D AND Curr K < Curr D
    short_signals = np.where((k_values[:-1] > d_values[:-1]) & (k_values[1:] < d_values[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    return combine_results(long_score, long_count, short_score, short_count, f'Stoch ({k_period}/{d_period})')

def analyze_bollinger_reversion(df, period, stddev, fwd_matrix, weights, horizon):
    """Generates mean reversion signals when price crosses back INTO the bands."""
    
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    upper_band = (sma + std * stddev).values
    lower_band = (sma - std * stddev).values
    prices = df['close'].values
    
    # Long Reversion: Prev Price < Lower Band AND Curr Price > Lower Band
    long_signals = np.where((prices[:-1] < lower_band[:-1]) & (prices[1:] > lower_band[1:]))[0] + 1
    # Short Reversion: Prev Price > Upper Band AND Curr Price < Upper Band
    short_signals = np.where((prices[:-1] > upper_band[:-1]) & (prices[1:] < upper_band[1:]))[0] + 1
    
    long_score, long_count = calculate_dependability_score(long_signals, True, fwd_matrix, weights, horizon)
    short_score, short_count = calculate_dependability_score(short_signals, False, fwd_matrix, weights, horizon)

    # Note: Long signals profit from price *rising*, Short signals profit from price *falling*.
    # Reversion signals assume a move opposite to the crossover.
    
    # For Long signal (crossing lower band), profit is expected up (Long True)
    # For Short signal (crossing upper band), profit is expected down (Short True)

    return combine_results(long_score, long_count, short_score, short_count, f'BB ({period}/{stddev}) Reversion')

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
        
        print("\n--- Running Comprehensive Analysis ---")
        
        # 1. Price/SMA Crossover (Trend)
        for p in SMA_PERIODS:
            results.append(analyze_price_sma_crossover(df_data, p, fwd_matrix, weights, HORIZON))

        # 2. Double EMA Crossover (Trend)
        for p_slow in EMA_SLOW_PERIODS:
            results.append(analyze_double_ema_crossover(df_data, EMA_FAST, p_slow, fwd_matrix, weights, HORIZON))

        # 3. RSI Centerline Crossover (Momentum)
        for p in RSI_PERIODS:
            results.append(analyze_rsi_centerline_crossover(df_data, p, fwd_matrix, weights, HORIZON))
            
        # 4. MACD Signal Crossover (Momentum)
        for p_signal in MACD_SIGNAL_PERIODS:
             results.append(analyze_macd_signal_crossover(df_data, MACD_FAST_EMA, MACD_SLOW_EMA, p_signal, fwd_matrix, weights, HORIZON))

        # 5. Stochastic Oscillator (%K/%D Crossover)
        for k_period in STOCH_K_PERIODS:
             results.append(analyze_stochastic_crossover(df_data, k_period, STOCH_D_PERIOD, fwd_matrix, weights, HORIZON))

        # 6. Bollinger Band Reversion (Volatility)
        for period in BB_PERIODS:
             for stddev in BB_STDEVS:
                results.append(analyze_bollinger_reversion(df_data, period, stddev, fwd_matrix, weights, HORIZON))


        analysis_results = sorted(results, key=lambda x: x['score'], reverse=True)
        data_loaded = True
        print("\n--- Comprehensive Analysis Complete ---")
        
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
        
        # Base colors (used if not in top 3)
        base_color = 'bg-red-100' if score < 0 else 'bg-gray-50'
        
        # Highlight top 3 if score > 0
        if rank <= 3 and score > 0:
            row_class = 'bg-yellow-200 font-bold'
        else:
            row_class = base_color
            
        table_rows += f"""
        <tr class="border-b hover:bg-gray-100 {row_class}">
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
    load_data_and_run_analysis()
    
    print(f"Starting web server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT)
