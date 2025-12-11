import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import datetime as dt
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from matplotlib.ticker import ScalarFormatter

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '1 Jan, 2018'
SMA_PERIOD_120 = 120 
PLOT_FILE = 'strategy_results.png'
SERVER_PORT = 8080 
RESULTS_DIR = 'results'
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data

# --- Rolling Optimization Parameters ---
OPTIMIZATION_STEP = 90 # Re-optimize every 90 days (approx. 3 months)
GRID_START = 0.010     # 1.0%
GRID_END = 0.080       # 8.0%
GRID_STEP = 0.001      # 0.1%

# --- 1. Data Fetching Utilities ---

def date_to_milliseconds(date_str):
    """Convert date string to UTC timestamp in milliseconds"""
    return int(dt.datetime.strptime(date_str, '%d %b, %Y').timestamp() * 1000)

def fetch_klines(symbol, interval, start_str):
    """Fetches historical klines data from Binance."""
    print(f"-> Fetching {symbol} {interval} data starting from {start_str}...")
    start_ts = date_to_milliseconds(start_str)
    base_url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    
    while True:
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'limit': 1000
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status() 
            klines = response.json()
            
            if not klines:
                break

            all_data.extend(klines)
            start_ts = klines[-1][0] + 1
            time.sleep(0.5) 
            
            if len(klines) < 1000:
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
        
    print(f"-> Data fetch complete. Total candles: {len(all_data)}")
    
    df = pd.DataFrame(all_data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])

    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close'] = pd.to_numeric(df['Close'])
    df = df.set_index('Open Time')
    df = df[['Open', 'Close']].astype(float)
    
    return df.dropna()

# --- 2. Metric Calculations ---

def calculate_sharpe_ratio(returns, annualization_factor=ANNUALIZATION_FACTOR, risk_free_rate=0):
    """Calculates the Annualized Sharpe Ratio."""
    if returns.empty or len(returns) <= 1:
        return -np.inf # Return a very low Sharpe for invalid windows
    excess_return = returns - risk_free_rate
    mean_excess_return = excess_return.mean()
    std_dev = excess_return.std()
    
    if std_dev == 0:
        return 0.0

    sharpe = (mean_excess_return * annualization_factor) / (std_dev * np.sqrt(annualization_factor))
    return sharpe

def calculate_max_drawdown(cumulative_returns):
    """Calculates the Maximum Drawdown (MDD) in percentage."""
    if cumulative_returns.empty:
        return 0.0
    peak = cumulative_returns.cummax()
    drawdown = (peak - cumulative_returns) / peak
    return drawdown.max() * 100

def generate_metrics(cumulative_returns, daily_returns, strategy_name):
    """Aggregates all metrics for a given strategy."""
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd = calculate_max_drawdown(cumulative_returns)
    
    return {
        'Strategy': strategy_name,
        'Total Return (%)': f'{total_return:.2f}',
        'Annualized Sharpe': f'{sharpe:.2f}',
        'Max Drawdown (%)': f'{max_dd:.2f}',
        'Cumulative Returns': cumulative_returns
    }

# --- 3. Strategy Core Logic ---

def run_strategy_core(df_window, center_distance):
    """
    Applies the dynamic sizing strategy to a DataFrame window for a specific center distance.
    Returns the daily strategy returns for that window.
    """
    df = df_window.copy()
    
    df[f'SMA_{SMA_PERIOD_120}'] = df['Close'].rolling(window=SMA_PERIOD_120).mean()
    df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Prepare Lagged Data (Look-ahead Bias Prevention) ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA_120'] = df[f'SMA_{SMA_PERIOD_120}'].shift(1)
    
    df = df.dropna()
    
    # 1. Calculate Distance (D): Absolute decimal distance from the SMA (lagged)
    df['Distance'] = np.abs((df['Yesterday_Close'] - df['Yesterday_SMA_120']) / df['Yesterday_SMA_120'])

    # 2. Calculate Multiplier (M) using the provided formula:
    # M = 1 / ( (1 / (D * Scaler)) + (D * Scaler) - 1 )
    distance_scaler = 1.0 / center_distance
    scaled_distance = df['Distance'] * distance_scaler
    
    epsilon = 1e-6 
    denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
    df['Multiplier'] = np.where(denominator == 0, 0, 1.0 / denominator)

    # 3. Determine Direction (Long/Short)
    df['Direction'] = np.where(
        df['Yesterday_Close'] > df['Yesterday_SMA_120'],
        1,
        -1
    )

    # 4. Final Position Size = Direction * Multiplier
    df['Position_Size'] = df['Direction'] * df['Multiplier']
    
    # 5. Calculate Strategy Returns
    df['Strategy_Return'] = df['Daily_Return'] * df['Position_Size']
    
    return df['Strategy_Return']

def run_grid_search(df_optimization_window):
    """
    Performs a grid search on the given optimization window to find the optimal center distance.
    """
    # Define grid search space (1.0% to 8.0% in 0.1% steps)
    center_distances = [round(c, 3) for c in np.arange(GRID_START, GRID_END + GRID_STEP/2, GRID_STEP)]

    best_sharpe = -np.inf
    optimal_center = GRID_START # Default to min if search fails
    
    # Run search
    for center_distance in center_distances:
        # Calculate returns for the optimization window
        strategy_returns = run_strategy_core(df_optimization_window, center_distance)
        
        # Calculate Sharpe Ratio on the returns
        sharpe = calculate_sharpe_ratio(strategy_returns)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            optimal_center = center_distance
            
    return optimal_center

def run_rolling_optimization_backtest(df_full):
    """
    Performs the rolling optimization and backtesting process.
    """
    print("-> Starting Rolling Optimization Backtest...")

    # Ensure enough data exists for initial SMA calculation and first optimization window
    min_data_needed = SMA_PERIOD_120 + OPTIMIZATION_STEP 
    if len(df_full) < min_data_needed:
        raise ValueError(f"Not enough data for rolling optimization. Need at least {min_data_needed} days.")

    # Index where optimization begins (after warmup + first step)
    start_index = SMA_PERIOD_120 
    end_index = len(df_full)

    # List to store daily returns from all execution windows
    all_strategy_returns = []
    
    # Dictionary to store the optimal center found at each rebalance date
    optimal_centers_history = {}
    
    # Loop starts at the first rebalance point after the SMA warmup
    current_rebalance_point = start_index

    while current_rebalance_point < end_index:
        
        # --- 1. Define Optimization Window (All past data) ---
        # The window includes data from the start up to the current rebalance point (exclusive of data to be traded)
        df_optimization_window = df_full.iloc[:current_rebalance_point]
        
        # --- 2. Run Optimization ---
        optimal_center = run_grid_search(df_optimization_window)
        optimal_centers_history[df_full.index[current_rebalance_point]] = optimal_center
        
        print(f"   Optimization up to {df_full.index[current_rebalance_point].strftime('%Y-%m-%d')} completed. Optimal Center: {optimal_center*100:.1f}%")

        # --- 3. Define Execution Window ---
        # The strategy trades the next OPTIMIZATION_STEP (90 days)
        execution_start = current_rebalance_point
        execution_end = min(current_rebalance_point + OPTIMIZATION_STEP, end_index)
        
        df_execution_window = df_full.iloc[execution_start:execution_end]
        
        # --- 4. Run Strategy on Execution Window with Optimal Parameter ---
        # We need the SMA and shifted returns/close for the execution window as well.
        # To calculate returns correctly, we need one prior row for the shift() function to work.
        
        # We pass the execution window and its preceding row to run_strategy_core
        df_full_with_preceding = df_full.iloc[execution_start - 1 : execution_end]
        
        # strategy_returns will return data only for the execution period (start to end)
        strategy_returns = run_strategy_core(df_full_with_preceding, optimal_center)
        
        # Store the returns for stitching
        all_strategy_returns.append(strategy_returns)
        
        # --- 5. Advance Rebalance Point ---
        current_rebalance_point += OPTIMIZATION_STEP
        
    # --- 6. Final Metric Calculation ---
    
    # Stitch all daily returns together
    df_returns = pd.concat(all_strategy_returns)
    
    # Calculate Buy & Hold returns for the same final period
    df_returns['Cumulative_Strategy_Return'] = np.exp(df_returns.sum().cumsum())
    
    # Recalculate B&H over the same period for fair comparison
    df_full = df_full.iloc[df_returns.index[0].to_datetime64() <= df_full.index]
    df_full['Daily_Return'] = np.log(df_full['Close'] / df_full['Close'].shift(1))
    df_full['Cumulative_Buy_and_Hold'] = np.exp(df_full['Daily_Return'].cumsum())
    
    # Combine final cumulative returns
    df_final = pd.DataFrame({
        'Cumulative_Strategy_Return': df_returns['Cumulative_Strategy_Return'],
        'Cumulative_Buy_and_Hold': df_full['Cumulative_Buy_and_Hold'].loc[df_returns.index]
    })

    # --- Generate Metrics ---
    metrics = []
    
    # B&H
    metrics.append(generate_metrics(
        df_final['Cumulative_Buy_and_Hold'], df_full['Daily_Return'].loc[df_returns.index], 'Buy & Hold (Benchmark)'
    ))
    
    # Strategy
    metrics.append(generate_metrics(
        df_returns.sum(), df_returns.sum(), f'Rolling Dynamic Strategy (90-day re-opt)'
    ))
    
    # Print comparison table
    comparison_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != 'Cumulative Returns'} 
        for m in metrics
    ])
    
    print("\n" + "=" * 60)
    print("BACKTEST METRICS COMPARISON (Rolling Optimization)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    return df_final, df_full, metrics

# --- 4. Plotting Results ---

def plot_results(df_final, df_full, metrics):
    """
    Generates and saves the plot of the strategy, benchmark equity curves, and price/SMA.
    """
    print(f"-> Generating plot in '{RESULTS_DIR}/{PLOT_FILE}'...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, PLOT_FILE)

    plt.style.use('dark_background')
    
    # Create a figure with two subplots: Price/SMA (top) and Equity Curve (bottom)
    fig = plt.figure(figsize=(14, 10)) 
    gs = fig.add_gridspec(2, 1, hspace=0.25, height_ratios=[1, 1])
    
    # --- Top Subplot (ax1): Price and SMA (Linear Scale) ---
    ax1 = fig.add_subplot(gs[0])
    
    # Plotting the Close Price and SMA
    df_full['Close'].plot(ax=ax1, label='Close Price', color='#9CA3AF', linewidth=1.5, alpha=0.9, zorder=3)
    df_full[f'SMA_{SMA_PERIOD_120}'].plot(ax=ax1, label=f'SMA {SMA_PERIOD_120}', color='#3B82F6', linewidth=2, zorder=4)
    
    # Style and Labels for ax1
    ax1.set_title(f'{SYMBOL} Price and SMA (Linear Scale)', fontsize=16, color='white')
    ax1.set_xlabel('') 
    ax1.set_ylabel('Price (USDT)', fontsize=12, color='white')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#374151', which='both')
    
    # --- Bottom Subplot (ax2): Equity Curve (Log Scale) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plotting the equity curves
    df_final['Cumulative_Buy_and_Hold'].plot(ax=ax2, label=metrics[0]['Strategy'], color='#EF4444', linestyle='--', linewidth=1.5)
    df_final['Cumulative_Strategy_Return'].plot(ax=ax2, label=metrics[1]['Strategy'], color='#3B82F6', linewidth=2.5)
    
    # Set Y-axis to Logarithmic Scale
    ax2.set_yscale('log')
    
    # Style and Labels for ax2
    ax2.set_title('Cumulative Return (Log Scale)', fontsize=14, color='white')
    ax2.set_xlabel('Date', fontsize=12, color='white')
    ax2.set_ylabel('Cumulative Return (Log Scale - Multiplier)', fontsize=12, color='white')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#374151', which='both')
    ax2.yaxis.set_major_formatter(ScalarFormatter()) 

    # Remove tick labels from the upper plot's x-axis for a cleaner look
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Save the plot
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("-> Plot saved successfully.")

# --- 5. Web Server ---

def serve_results(metrics):
    """
    Starts a simple HTTP server to host the results page.
    """
    print(f"\n==========================================================")
    print(f"🚀 Starting Web Server on http://localhost:{SERVER_PORT}/")
    print(f"==========================================================")
    
    metric_rows = ""
    for m in metrics:
        metric_rows += f"""
        <tr class="bg-gray-700 hover:bg-gray-600">
            <td class="px-6 py-3 font-medium text-white">{m['Strategy']}</td>
            <td class="px-6 py-3">{m['Total Return (%)']}%</td>
            <td class="px-6 py-3">{m['Annualized Sharpe']}</td>
            <td class="px-6 py-3">{m['Max Drawdown (%)']}%</td>
        </tr>
        """

    class ResultsHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
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
                        <h1 class="text-3xl font-bold mb-4 text-green-400">Backtest Results: {SYMBOL} Rolling Optimization</h1>
                        
                        <div class="mb-8">
                            <h2 class="text-xl font-semibold mb-3 text-gray-200">Strategy Metrics Comparison</h2>
                            <p class="text-gray-400 mb-4">The dynamic sizing parameter (center distance) is re-optimized every 90 days using all past data.</p>
                            <div class="relative overflow-x-auto shadow-md sm:rounded-lg">
                                <table class="w-full text-sm text-left text-gray-400">
                                    <thead class="text-xs uppercase bg-gray-700 text-gray-400">
                                        <tr>
                                            <th scope="col" class="px-6 py-3">Strategy</th>
                                            <th scope="col" class="px-6 py-3">Total Return</th>
                                            <th scope="col" class="px-6 py-3">Sharpe Ratio</th>
                                            <th scope="col" class="px-6 py-3">Max Drawdown</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {metric_rows}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div class="plot-container">
                            <h2 class="text-xl font-semibold mb-3 text-gray-200">Price Action & Cumulative Returns</h2>
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
            
            elif self.path.startswith('/' + RESULTS_DIR):
                SimpleHTTPRequestHandler.do_GET(self)
            
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'404 Not Found')

        def translate_path(self, path):
            return os.path.join(os.getcwd(), path.lstrip('/'))

    
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
        # 2. Run backtest and get comparison metrics
        results_df_final, df_full, comparison_metrics = run_rolling_optimization_backtest(df_data)
        
        # 3. Plot results
        plot_results(results_df_final, df_full, comparison_metrics)

        # 4. Start web server in the main thread
        serve_results(comparison_metrics)
