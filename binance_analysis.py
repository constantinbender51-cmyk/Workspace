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
GRID_START = 0.001     # 0.1% (Start of search)
GRID_END = 0.080       # 8.0%
GRID_STEP = 0.001      # 0.1%
MIN_SHARPE_VALIDITY_DAYS = 365 # Minimum days required for optimization window to run grid search
INITIAL_DEFAULT_CENTER = 0.013 # Default center distance (1.3%) used before 365 days are available

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
    # Need at least 2 data points for std dev
    if returns.empty or len(returns) <= 1:
        return -np.inf 
    excess_return = returns - risk_free_rate
    mean_excess_return = excess_return.mean()
    std_dev = excess_return.std()
    
    if std_dev == 0:
        # Avoid division by zero, set Sharpe to zero if no volatility
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
    The input df_window must contain the preceding data (120 days + 1 lag) for valid calculation.
    """
    df = df_window.copy()
    
    # 1. Calculate base components (SMA is calculated over the entire df_window which now includes history)
    df[f'SMA_{SMA_PERIOD_120}'] = df['Close'].rolling(window=SMA_PERIOD_120).mean()
    df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. Prepare Lagged Data (Look-ahead Bias Prevention)
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA_120'] = df[f'SMA_{SMA_PERIOD_120}'].shift(1)
    
    # 3. Drop rows that don't have all required data points
    required_cols = [f'SMA_{SMA_PERIOD_120}', 'Yesterday_Close', 'Yesterday_SMA_120', 'Daily_Return']
    df_clean = df.dropna(subset=required_cols).copy()

    if df_clean.empty:
        return pd.Series([], dtype=np.float64)
    
    # 4. Calculate Position Sizing Multiplier
    df_clean['Distance'] = np.abs((df_clean['Yesterday_Close'] - df_clean['Yesterday_SMA_120']) / df_clean['Yesterday_SMA_120'])
    
    distance_scaler = 1.0 / center_distance
    scaled_distance = df_clean['Distance'] * distance_scaler
    
    epsilon = 1e-6 
    denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
    df_clean['Multiplier'] = np.where(denominator == 0, 0, 1.0 / denominator)

    # 5. Determine Direction (Long/Short) and Final Returns
    df_clean['Direction'] = np.where(
        df_clean['Yesterday_Close'] > df_clean['Yesterday_SMA_120'],
        1,
        -1
    )
    df_clean['Position_Size'] = df_clean['Direction'] * df_clean['Multiplier']
    df_clean['Strategy_Return'] = df_clean['Daily_Return'] * df_clean['Position_Size']
    
    # Return all valid strategy returns
    return df_clean['Strategy_Return']

def run_grid_search(df_optimization_window):
    """
    Performs a grid search on the given optimization window to find the optimal center distance.
    """
    # Define grid search space (0.1% to 8.0% in 0.1% steps)
    center_distances = [round(c, 3) for c in np.arange(GRID_START, GRID_END + GRID_STEP/2, GRID_STEP)]

    best_sharpe = -np.inf
    optimal_center = GRID_START
    
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

    # The index where trading begins (after SMA + 1 day lag)
    FIRST_VALID_TRADE_INDEX = SMA_PERIOD_120 + 1 
    
    # The index where the first optimization-based trade will start
    current_rebalance_index = FIRST_VALID_TRADE_INDEX 

    if len(df_full) < current_rebalance_index:
        raise ValueError(f"Not enough data for initial SMA calculation. Need at least {current_rebalance_index} days.")

    end_index = len(df_full)
    all_strategy_returns = []
    
    while current_rebalance_index < end_index:
        
        # --- Determine Center Distance for Current Block ---
        
        # Optimization window includes all data up to the execution start index
        df_optimization_history = df_full.iloc[:current_rebalance_index]
        
        # Check if the optimization window has enough history for a meaningful Sharpe calculation
        if len(df_optimization_history) >= MIN_SHARPE_VALIDITY_DAYS:
            # OPTIMIZE: Include 120 prior days for SMA calculation within the optimization history
            opt_start_index = max(0, current_rebalance_index - MIN_SHARPE_VALIDITY_DAYS - SMA_PERIOD_120 - 1)
            df_opt_window = df_full.iloc[opt_start_index:current_rebalance_index]
            optimal_center = run_grid_search(df_opt_window)
            opt_status = "OPTIMIZED"
        else:
            # DEFAULT: Use the default center distance until enough data is available
            optimal_center = INITIAL_DEFAULT_CENTER
            opt_status = "DEFAULT"
            
        # 3. EXECUTION WINDOW (The next OPTIMIZATION_STEP days - Out-of-Sample)
        execution_start_index = current_rebalance_index
        execution_end_index = min(execution_start_index + OPTIMIZATION_STEP, end_index)
        
        # Execution data MUST include the preceding 120 days (or all data if earlier) for SMA calculation
        exec_data_start_index = max(0, execution_start_index - SMA_PERIOD_120 - 1)
        df_execution_data = df_full.iloc[exec_data_start_index : execution_end_index]
        
        # 4. Run Strategy on Execution Window with Optimal Parameter
        strategy_returns = run_strategy_core(df_execution_data, optimal_center)
        
        # 5. Extract only the Out-of-Sample returns for stitching
        # Find the index where the execution window starts in the generated returns
        execution_period_start_date = df_full.index[execution_start_index]
        strategy_returns_only_execution = strategy_returns.loc[strategy_returns.index >= execution_period_start_date]
        
        if not strategy_returns_only_execution.empty:
            all_strategy_returns.append(strategy_returns_only_execution)
            print(f"   Rebalance up to {df_full.index[current_rebalance_index-1].strftime('%Y-%m-%d')} | Center ({opt_status}): {optimal_center*100:.1f}% | Executed trades: {len(strategy_returns_only_execution)} days")
        else:
             print(f"   Rebalance up to {df_full.index[current_rebalance_index-1].strftime('%Y-%m-%d')} | WARNING: No valid returns generated for execution window. Skipping this period.")

        # 6. Advance to the next rebalance point
        current_rebalance_index = execution_end_index
        
    # --- 7. Final Metric Calculation ---
    
    # Stitch all daily returns together
    if not all_strategy_returns:
         raise IndexError("No objects to concatenate: No valid returns were generated throughout the backtest.")

    df_returns_stitched = pd.concat(all_strategy_returns).rename('Strategy_Return')
    
    # Calculate strategy cumulative returns
    strategy_cumulative_returns = np.exp(df_returns_stitched.cumsum())
    
    # Recalculate B&H over the same final period for fair comparison
    start_date = df_returns_stitched.index[0]
    
    # Use loc slicing to get the relevant period for B&H
    df_full_sliced = df_full.loc[start_date:].copy()
    
    df_full_sliced['Daily_Return'] = np.log(df_full_sliced['Close'] / df_full_sliced['Close'].shift(1))
    df_full_sliced['Cumulative_Buy_and_Hold'] = np.exp(df_full_sliced['Daily_Return'].cumsum())
    
    # Combine final cumulative returns, ensuring matching indices
    bh_daily_returns = df_full_sliced['Daily_Return'].loc[df_returns_stitched.index].dropna()
    strategy_daily_returns = df_returns_stitched.loc[bh_daily_returns.index]
    
    df_final = pd.DataFrame({
        'Cumulative_Strategy_Return': strategy_cumulative_returns.loc[strategy_daily_returns.index],
        'Cumulative_Buy_and_Hold': df_full_sliced['Cumulative_Buy_and_Hold'].loc[bh_daily_returns.index]
    })
    
    # --- Generate Metrics ---
    metrics = []
    
    # B&H
    metrics.append(generate_metrics(
        df_final['Cumulative_Buy_and_Hold'].loc[bh_daily_returns.index], bh_daily_returns, 'Buy & Hold (Benchmark)'
    ))
    
    # Strategy
    metrics.append(generate_metrics(
        df_final['Cumulative_Strategy_Return'].loc[strategy_daily_returns.index], strategy_daily_returns, f'Rolling Dynamic Strategy (90-day re-opt)'
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
    df_full[f'SMA_{SMA_PERIOD_120}'] = df_full['Close'].rolling(window=SMA_PERIOD_120).mean()
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
                            <p class="text-gray-400 mb-4">The dynamic sizing parameter (center distance) is re-optimized every 90 days using all past data (0.1% to 8.0% grid search). A default of 1.3% is used until 365 days of optimization history is available.</p>
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
        try:
            results_df_final, df_full, comparison_metrics = run_rolling_optimization_backtest(df_data)
        except ValueError as e:
            print(f"Error during backtest: {e}")
            exit()
        except IndexError as e:
            print(f"Error during backtest: {e}")
            exit()
        
        # 3. Plot results
        plot_results(results_df_final, df_full, comparison_metrics)

        # 4. Start web server in the main thread
        serve_results(comparison_metrics)
