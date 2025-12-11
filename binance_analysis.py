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
SMA_PERIOD_120 = 120 # SMA used for the trading rule and blue proximity
SMA_PERIOD_360 = 360 # SMA used for green/red distance shading
PLOT_FILE = 'strategy_results.png'
SERVER_PORT = 8080
RESULTS_DIR = 'results'
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data
PROXIMITY_SCALER = 50.0 # Proximity/Distance scaler (1/0.02)

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

# --- Sharpe Ratio Calculation ---

def calculate_sharpe_ratio(returns, annualization_factor=ANNUALIZATION_FACTOR, risk_free_rate=0):
    """
    Calculates the Annualized Sharpe Ratio.
    """
    if returns.empty:
        return 0.0
        
    # Calculate mean daily excess return
    excess_return = returns - risk_free_rate
    mean_excess_return = excess_return.mean()

    # Calculate standard deviation of daily returns
    std_dev = returns.std()
    
    if std_dev == 0:
        return 0.0

    # Annualize: mean * T / (std * sqrt(T))
    sharpe = (mean_excess_return * annualization_factor) / (std_dev * np.sqrt(annualization_factor))
    return sharpe

# --- 2. Backtesting Logic ---

def run_backtest(df):
    """
    Applies the SMA strategy, calculates returns, and computes the SMA Proximity.
    """
    print(f"-> Running backtest on {len(df)} candles...")
    
    # 2.1 Calculate the 120-day SMA (used for trading rule and blue shading)
    df[f'SMA_{SMA_PERIOD_120}'] = df['Close'].rolling(window=SMA_PERIOD_120).mean()
    
    # 2.1.5 Calculate the 360-day SMA (used for green/red distance shading)
    df[f'SMA_{SMA_PERIOD_360}'] = df['Close'].rolling(window=SMA_PERIOD_360).mean()
    
    # 2.2 Calculate daily returns (log returns are often preferred for backtesting)
    df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Look-ahead Bias Prevention (Explicit shift) ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA'] = df[f'SMA_{SMA_PERIOD_120}'].shift(1)
    
    # 2.3 Generate Trading Signals (Position for day T)
    # Rule: Position on day T is based on (Close(T-1) vs 120 SMA(T-1))
    df['Position'] = np.where(
        df['Yesterday_Close'] > df['Yesterday_SMA'],  # Condition: Close(T-1) > SMA_120(T-1)
        1,                                           # Value if True (Long position held on Day T)
        -1                                           # Value if False (Short position held on Day T)
    )
    
    # --- 2.7 Calculate SMA Proximity Metrics ---
    
    # --- 120 SMA Proximity (for bottom plot and blue shading - distance <= 2%) ---
    # Calculate X% (Absolute decimal distance: e.g., 0.01 for 1% distance)
    df['SMA_Distance_Decimal_120'] = np.abs((df['Close'] - df[f'SMA_{SMA_PERIOD_120}']) / df[f'SMA_{SMA_PERIOD_120}']) 
    
    # Proximity formula: P = min(1.0, 1 / (X% * 50.0))
    denominator_120 = df['SMA_Distance_Decimal_120'] * PROXIMITY_SCALER
    proximity_base_120 = np.where(denominator_120 == 0, 100.0, 1.0 / denominator_120)
    df['SMA_Proximity'] = np.minimum(proximity_base_120, 1.0) 

    # --- 360 SMA Distance Metric (for green/red shading - distance > 2%) ---
    # Calculate X% (Absolute decimal distance: e.g., 0.01 for 1% distance)
    df['SMA_Distance_Decimal_360'] = np.abs((df['Close'] - df[f'SMA_{SMA_PERIOD_360}']) / df[f'SMA_{SMA_PERIOD_360}']) 
    
    # New Metric: Distance * 50.0. (X% * 50.0). This metric is > 1.0 when price is > 2% away.
    df['SMA_Distance_Metric_360'] = df['SMA_Distance_Decimal_360'] * PROXIMITY_SCALER
    
    # Drop the temporary columns for cleanliness and keep the final SMA columns and proximities
    df = df.drop(columns=['Yesterday_Close', 'Yesterday_SMA', 'SMA_Distance_Decimal_120', 'SMA_Distance_Decimal_360'])
    
    # The first SMA_PERIOD_360 entries will have NaN SMA. We drop rows with any NaNs.
    df = df.dropna()
    
    # 2.4 Calculate Strategy Returns
    # Strategy Return = Daily_Return(T) * Position(T)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    
    # 2.5 Calculate Cumulative Returns (Equity Curve)
    df['Cumulative_Strategy_Return'] = np.exp(df['Strategy_Return'].cumsum())
    df['Buy_and_Hold_Return'] = np.exp(df['Daily_Return'].cumsum())
    
    # 2.6 Calculate Metrics
    total_return = (df['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    bh_return = (df['Buy_and_Hold_Return'].iloc[-1] - 1) * 100
    
    strategy_sharpe = calculate_sharpe_ratio(df['Strategy_Return'])
    bh_sharpe = calculate_sharpe_ratio(df['Daily_Return'])
    
    print("-" * 40)
    print("Backtest Summary:")
    print(f"Strategy Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Total Return: {bh_return:.2f}%")
    print("-" * 40)
    print(f"Strategy Annualized Sharpe: {strategy_sharpe:.2f}")
    print(f"Buy & Hold Annualized Sharpe: {bh_sharpe:.2f}")
    print("-" * 40)
    
    return df

# --- 3. Plotting Results ---

def plot_results(df):
    """
    Generates and saves the plot of the strategy's Close Price, SMA, 
    equity curve, and SMA Proximity.
    """
    print(f"-> Generating plot in '{RESULTS_DIR}/{PLOT_FILE}'...")
    
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, PLOT_FILE)

    plt.style.use('dark_background')
    
    # Create a figure with three subplots: 3 rows, 1 column. 
    # Ratio: 4 parts for Price, 3 parts for Equity Curve, 1 part for Proximity
    fig = plt.figure(figsize=(14, 15)) 
    gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[4, 3, 1])
    
    # --- Top Subplot (ax1): Price and SMAs (Linear Scale) ---
    ax1 = fig.add_subplot(gs[0])
    
    # --- Shading for 120 SMA Max Proximity (BLUE) ---
    # Condition: Price is within 2% of 120 SMA (Proximity >= 1.0)
    proximity_120_max = df['SMA_Proximity'] >= 1.0
    start_indices_120 = df.index[proximity_120_max & (~proximity_120_max.shift(1, fill_value=False))]
    end_indices_120 = df.index[proximity_120_max & (~proximity_120_max.shift(-1, fill_value=False))]

    shaded_label_120 = False
    for start_date, end_date in zip(start_indices_120, end_indices_120):
        end_dt = end_date + dt.timedelta(days=1)
        if not shaded_label_120:
            ax1.axvspan(start_date, end_dt, color='#3B82F6', alpha=0.2, label=f'{SMA_PERIOD_120} SMA Close Days (Blue)')
            shaded_label_120 = True
        else:
            ax1.axvspan(start_date, end_dt, color='#3B82F6', alpha=0.2)
            
    # --- Shading for 360 SMA Distance Metric (GREEN/RED) ---
    # Condition: Metric > 1.0 (Distance > 2%)
    distance_far = df['SMA_Distance_Metric_360'] > 1.0
    
    # Green Condition: Far Distance AND Price is ABOVE 360 SMA
    green_condition = distance_far & (df['Close'] > df[f'SMA_{SMA_PERIOD_360}'])
    start_indices_green = df.index[green_condition & (~green_condition.shift(1, fill_value=False))]
    end_indices_green = df.index[green_condition & (~green_condition.shift(-1, fill_value=False))]
    
    shaded_label_green = False
    for start_date, end_date in zip(start_indices_green, end_indices_green):
        end_dt = end_date + dt.timedelta(days=1)
        if not shaded_label_green:
            # Using Green color
            ax1.axvspan(start_date, end_dt, color='#10B981', alpha=0.3, label=f'{SMA_PERIOD_360} Far & Above (Green)')
            shaded_label_green = True
        else:
            ax1.axvspan(start_date, end_dt, color='#10B981', alpha=0.3)

    # Red Condition: Far Distance AND Price is BELOW 360 SMA
    red_condition = distance_far & (df['Close'] < df[f'SMA_{SMA_PERIOD_360}'])
    start_indices_red = df.index[red_condition & (~red_condition.shift(1, fill_value=False))]
    end_indices_red = df.index[red_condition & (~red_condition.shift(-1, fill_value=False))]
    
    shaded_label_red = False
    for start_date, end_date in zip(start_indices_red, end_indices_red):
        end_dt = end_date + dt.timedelta(days=1)
        if not shaded_label_red:
            # Using Red color
            ax1.axvspan(start_date, end_dt, color='#EF4444', alpha=0.3, label=f'{SMA_PERIOD_360} Far & Below (Red)') 
            shaded_label_red = True
        else:
            ax1.axvspan(start_date, end_dt, color='#EF4444', alpha=0.3)
            
    # Plotting the Close Price and SMAs (must be plotted *after* shading)
    df['Close'].plot(ax=ax1, label='Close Price', color='#9CA3AF', linewidth=1.5, alpha=0.9, zorder=3)
    df[f'SMA_{SMA_PERIOD_120}'].plot(ax=ax1, label=f'SMA {SMA_PERIOD_120}', color='#3B82F6', linewidth=2, zorder=4)
    df[f'SMA_{SMA_PERIOD_360}'].plot(ax=ax1, label=f'SMA {SMA_PERIOD_360}', color='#FACC15', linewidth=2, linestyle='--', zorder=4)

    # Style and Labels for ax1
    ax1.set_title(f'{SYMBOL} Price and SMAs (Linear Scale)', fontsize=16, color='white')
    ax1.set_xlabel('') 
    ax1.set_ylabel('Price (USDT)', fontsize=12, color='white')
    # Combine all labels and ensure only unique ones are used
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper left', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#374151', which='both')

    # --- Middle Subplot (ax2): Equity Curve (Log Scale) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1) # Share X-axis with ax1
    
    # Plotting the equity curves
    df['Cumulative_Strategy_Return'].plot(ax=ax2, label=f'SMA {SMA_PERIOD_120} Strategy', color='#10B981', linewidth=2)
    df['Buy_and_Hold_Return'].plot(ax=ax2, label='Buy & Hold (Benchmark)', color='#EF4444', linestyle='--', linewidth=1.5)
    
    # Set Y-axis to Logarithmic Scale
    ax2.set_yscale('log')
    
    # Style and Labels for ax2
    ax2.set_title('Cumulative Return (Log Scale)', fontsize=14, color='white')
    ax2.set_xlabel('')
    ax2.set_ylabel('Cumulative Return (Multiplier)', fontsize=12, color='white')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#374151', which='both')
    ax2.yaxis.set_major_formatter(ScalarFormatter()) # Ensure non-scientific notation
    
    # --- Bottom Subplot (ax3): SMA Proximity ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1) # Share X-axis with ax1
    
    # Plotting SMA Proximity (120 SMA)
    df['SMA_Proximity'].plot(ax=ax3, label=f'SMA {SMA_PERIOD_120} Proximity (Capped at 1.0)', color='#F59E0B', linewidth=1)
    
    # Style and Labels for ax3
    ax3.set_title(f'SMA {SMA_PERIOD_120} Proximity Metric (1.0 when price is 2% or less from SMA)', fontsize=14, color='white')
    ax3.set_xlabel('Date', fontsize=12, color='white')
    ax3.set_ylabel('Proximity Score', fontsize=12, color='white')
    ax3.set_ylim(0, 1.1) # Set limit for capped indicator
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.5, color='#374151')
    
    # Remove tick labels from the upper plots' x-axis for a cleaner look
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

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
                        <h1 class="text-3xl font-bold mb-4 text-green-400">Backtesting Results: {SYMBOL} SMA-120/360</h1>
                        <p class="text-gray-300 mb-6">
                            The plot now shows three SMAs: Close Price, 120-Day SMA (solid blue), and 360-Day SMA (dashed yellow).
                            <br>
                            - Blue shading indicates **close** proximity (<= 2%) to the 120-Day SMA.
                            <br>
                            - **Green shading** indicates **far distance** (> 2%) from the 360-Day SMA & Price is **Above** 360 SMA.
                            <br>
                            - **Red shading** indicates **far distance** (> 2%) from the 360-Day SMA & Price is **Below** 360 SMA.
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
