import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import datetime as dt
import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
from matplotlib.ticker import ScalarFormatter

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '1 Jan, 2018'
SMA_PERIOD = 120
PLOT_FILE = 'strategy_results.png'
SERVER_PORT = 8080
RESULTS_DIR = 'results'

# --- 1. Data Fetching Utilities ---

def date_to_milliseconds(date_str):
    """Convert date string to UTC timestamp in milliseconds"""
    # Helper to convert human-readable date to Binance's millisecond timestamp
    # Note: Using a fixed date format for consistency
    return int(dt.datetime.strptime(date_str, '%d %b, %Y').timestamp() * 1000)

def fetch_klines(symbol, interval, start_str):
    """
    Fetches historical klines (OHLCV) data from Binance.
    Handles the 1000 limit per request by looping.
    """
    print(f"-> Fetching {symbol} {interval} data starting from {start_str}...")
    
    # Convert start date to millisecond timestamp
    start_ts = date_to_milliseconds(start_str)
    
    # Base URL for klines endpoint
    base_url = 'https://api.binance.com/api/v3/klines'
    
    all_data = []
    
    # Loop to fetch data in chunks of 1000 candles
    while True:
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'limit': 1000
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            klines = response.json()
            
            if not klines:
                break

            all_data.extend(klines)
            
            # The next request should start from the close time of the last candle + 1 millisecond
            start_ts = klines[-1][0] + 1
            
            # Print progress
            print(f"   Fetched up to: {dt.datetime.fromtimestamp(start_ts / 1000).strftime('%Y-%m-%d')}")
            
            # To respect Binance's rate limits
            time.sleep(0.5) 
            
            # Check if we have reached the current time (last candle in the response is the newest available)
            if len(klines) < 1000:
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
        
    print(f"-> Data fetch complete. Total candles: {len(all_data)}")
    
    # Convert list of lists into a DataFrame
    df = pd.DataFrame(all_data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])

    # Clean and format DataFrame
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close'] = pd.to_numeric(df['Close'])
    df = df.set_index('Open Time')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    return df.dropna()

# --- 2. Backtesting Logic ---

def run_backtest(df):
    """
    Applies the SMA strategy and calculates returns.
    """
    print(f"-> Running backtest on {len(df)} candles...")
    
    # 2.1 Calculate the 120-day Simple Moving Average (SMA)
    # The SMA is calculated based on the Close price of the current day.
    df[f'SMA_{SMA_PERIOD}'] = df['Close'].rolling(window=SMA_PERIOD).mean()
    
    # 2.2 Calculate daily returns (log returns are often preferred for backtesting)
    df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2.3 Generate Trading Signals (Position)
    # Rule: Price > SMA = Long (+1), Price < SMA = Short (-1)
    # Signal on day T is based on the comparison of Close(T-1) vs. SMA(T-1)
    
    # Calculate the position for the *next* day based on today's closing data
    # We use numpy.where for vectorized conditional assignment
    df['Position'] = np.where(
        df['Close'] > df[f'SMA_{SMA_PERIOD}'],  # Condition: Current Close > Current SMA
        1,                                      # Value if True (Long)
        -1                                      # Value if False (Short)
    )

    # Shift the position column forward by one day. 
    # This means the position calculated on day T is actually held on day T+1, 
    # which correctly avoids look-ahead bias.
    df['Position'] = df['Position'].shift(1)
    
    # The first SMA_PERIOD entries will have NaN SMA, and the first return will be NaN,
    # and the first Position will be NaN. We drop these.
    df = df.dropna()
    
    # 2.4 Calculate Strategy Returns
    # Strategy Return = Daily_Return * Position
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    
    # 2.5 Calculate Cumulative Returns (Equity Curve)
    # Cumulative returns are calculated using the exponent of the cumulative log returns
    df['Cumulative_Strategy_Return'] = np.exp(df['Strategy_Return'].cumsum())
    df['Buy_and_Hold_Return'] = np.exp(df['Daily_Return'].cumsum())
    
    # Calculate basic stats
    total_return = (df['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    bh_return = (df['Buy_and_Hold_Return'].iloc[-1] - 1) * 100
    
    print("-" * 40)
    print("Backtest Summary:")
    print(f"Strategy Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Total Return: {bh_return:.2f}%")
    print("-" * 40)
    
    return df

# --- 3. Plotting Results ---

def plot_results(df):
    """
    Generates and saves the plot of the strategy's equity curve.
    Uses a logarithmic Y-scale.
    """
    print(f"-> Generating plot in '{RESULTS_DIR}/{PLOT_FILE}'...")
    
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, PLOT_FILE)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plotting the equity curves
    df['Cumulative_Strategy_Return'].plot(ax=ax, label=f'SMA {SMA_PERIOD} Strategy', color='#10B981', linewidth=2)
    df['Buy_and_Hold_Return'].plot(ax=ax, label='Buy & Hold (Benchmark)', color='#EF4444', linestyle='--', linewidth=1.5)
    
    # Set Y-axis to Logarithmic Scale as requested
    ax.set_yscale('log')
    
    # Style and Labels
    ax.set_title(f'{SYMBOL} 120-Day SMA Strategy Equity Curve (Log Scale Since 2018)', fontsize=16, color='white')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Cumulative Return (Log Scale - Multiplier)', fontsize=12, color='white')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5, color='#374151', which='both')
    
    # Ensure non-scientific notation on the log scale
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Save the plot
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("-> Plot saved successfully.")

# --- 4. Web Server ---

def serve_results():
    """
    Starts a simple HTTP server to host the results page.
    """
    print(f"\n==========================================================")
    print(f"🚀 Starting Web Server on http://localhost:{SERVER_PORT}/")
    print(f"==========================================================")
    
    # Custom Handler to serve the HTML page
    class ResultsHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            # If the root path is requested, serve the custom HTML content
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                # HTML content to display the plot
                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Trading Strategy Results</title>
                    <script src="https://cdn.tailwindcss.com"></script>
                    <style>
                        body {{ font-family: 'Inter', sans-serif; background-color: #1F2937; color: #F3F4F6; }}
                        .container {{ max-width: 1024px; }}
                        .plot-container {{ background-color: #374151; border-radius: 0.5rem; padding: 1rem; }}
                    </style>
                </head>
                <body class="p-8">
                    <div class="container mx-auto p-4 bg-gray-800 shadow-xl rounded-xl">
                        <h1 class="text-3xl font-bold mb-4 text-green-400">Backtesting Results: {SYMBOL} SMA-120</h1>
                        <p class="text-gray-300 mb-6">
                            The backtest calculated daily returns for the strategy: Long if yesterday's Close > 120-Day SMA, Short if Close < 120-Day SMA. The Y-axis is set to a <strong>Logarithmic Scale</strong> to better show compounded returns.
                        </p>
                        <div class="plot-container">
                            <img src="{RESULTS_DIR}/{PLOT_FILE}" alt="Strategy Cumulative Returns Plot" class="w-full h-auto rounded-lg shadow-2xl">
                        </div>
                        <div class="mt-6 p-4 bg-gray-700 rounded-lg">
                            <p class="text-xl font-semibold">Server Running on Port {SERVER_PORT}</p>
                            <p class="text-sm text-gray-400">Close the terminal window to stop the server.</p>
                        </div>
                    </div>
                </body>
                </html>
                """
                self.wfile.write(bytes(html_content, "utf8"))
            
            # If the plot image or results directory is requested, serve it from the local path
            elif self.path.startswith('/' + RESULTS_DIR):
                SimpleHTTPRequestHandler.do_GET(self)
            
            # Fallback for other files (like favicon.ico)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'404 Not Found')

        # We must change the directory the server looks in to be the current directory
        def translate_path(self, path):
            # This ensures the server can find files in the current directory and subdirectories
            return os.path.join(os.getcwd(), path.lstrip('/'))

    
    # Change the working directory of the server to the root of the script execution
    with HTTPServer(("", SERVER_PORT), ResultsHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n-> Server stopped by user.")

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Fetch data
    df_data = fetch_klines(SYMBOL, INTERVAL, START_DATE)
    
    if df_data.empty:
        print("Error: Could not retrieve data. Exiting.")
    else:
        # 2. Run backtest and get results
        results_df = run_backtest(df_data)
        
        # 3. Plot results
        plot_results(results_df)

        # 4. Start web server in the main thread
        serve_results()
