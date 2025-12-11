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
PLOT_FILE = 'strategy_results.png'
SERVER_PORT = 8080 
RESULTS_DIR = 'results'
ANNUALIZATION_FACTOR = 365 # Used for annualizing Sharpe Ratio for daily data

# --- FINAL OPTIMIZED STRATEGY PARAMETERS (from GA) ---
# [C, L, E, S, P, M]
OPTIMAL_CENTER_DISTANCE = 0.01523558594313476
OPTIMAL_LEVERAGE = 1.8332066558220212
OPTIMAL_EXPONENT = 0.39561422613201874
OPTIMAL_STOP_LOSS_PERCENT = 0.07215336078895496
OPTIMAL_REENTRY_PROXIMITY_PERCENT = 0.015644741942864374
OPTIMAL_SMA_PERIOD = 120 # Fixed by GA, confirmed as 120

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
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
    df = df.set_index('Open Time')
    df = df[['Open', 'High', 'Low', 'Close']]
    
    return df.dropna()

# --- 2. Metric Calculations ---

def calculate_sharpe_ratio(returns, annualization_factor=ANNUALIZATION_FACTOR, risk_free_rate=0):
    """Calculates the Annualized Sharpe Ratio."""
    if returns.empty or len(returns) <= 1:
        return 0.0 
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

# --- 3. Backtesting Logic (Final Run) ---

def run_backtest_final(df_raw):
    """
    Applies the final strategy with all fixed and optimized parameters.
    """
    df = df_raw.copy()
    
    center = OPTIMAL_CENTER_DISTANCE
    leverage = OPTIMAL_LEVERAGE
    exponent = OPTIMAL_EXPONENT
    sl_percent = OPTIMAL_STOP_LOSS_PERCENT
    reentry_prox = OPTIMAL_REENTRY_PROXIMITY_PERCENT
    sma_period = OPTIMAL_SMA_PERIOD

    
    print(f"-> Running FINAL backtest (C={center*100:.2f}%, L={leverage:.2f}x, E={exponent:.2f}, SL={sl_percent*100:.2f}%, Re-entry={reentry_prox*100:.2f}%) on {len(df)} candles...")
    
    # 1. Calculate SMA and Raw Daily Returns
    df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()
    df['Daily_Return_Raw'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Look-ahead Prevention & Core Indicators ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA'] = df[f'SMA_{sma_period}'].shift(1)
    df['Proximity_to_SMA'] = np.abs((df['Yesterday_Close'] - df['Yesterday_SMA']) / df['Yesterday_SMA'])
    
    df['Center_Distance'] = center 
    df = df.dropna().copy()
    
    if df.empty:
        raise ValueError("Insufficient data after cleanup for backtesting.")

    # Initialize Series for returns
    strategy_returns = pd.Series(index=df.index, dtype=float)
    sl_cooldown = False # State variable
    
    # Iterate through the DataFrame for day-by-day logic (required for state management)
    for i in range(len(df)):
        index = df.index[i]
        
        entry_price = df.loc[index, 'Yesterday_Close']
        yesterday_sma = df.loc[index, 'Yesterday_SMA']
        proximity = df.loc[index, 'Proximity_to_SMA']
        
        # --- 1. Determine Base Position and Direction ---
        direction = np.where(entry_price > yesterday_sma, 1, -1)
        distance_d = np.abs((entry_price - yesterday_sma) / yesterday_sma)
        
        # Multiplier (M) calculation
        distance_scaler = 1.0 / center
        scaled_distance = distance_d * distance_scaler
        epsilon = 1e-10 
        
        # Denominator: (1/Scaled_Dist) + Scaled_Dist - 1
        denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
        
        # Multiplier = 1 / (Denominator)^E
        multiplier = np.where(denominator <= 0, 0, 1.0 / (denominator ** exponent))
        
        position_size_base = direction * multiplier

        # --- 2. Apply Re-entry/Cooldown Filter (State Management) ---
        
        if sl_cooldown:
            if proximity <= reentry_prox:
                sl_cooldown = False 
                
        if sl_cooldown:
            position_size_base = 0.0
            daily_return = 0.0
            
        else:
            # --- 3. Stop Loss Logic ---
            
            current_low = df.loc[index, 'Low']
            current_high = df.loc[index, 'High']
            raw_return = df.loc[index, 'Daily_Return_Raw']
            
            stop_price = np.where(
                direction == 1,
                entry_price * (1 - sl_percent),
                entry_price * (1 + sl_percent)
            )
            
            if sl_percent > 0.0 and (
                (direction == 1 and current_low <= stop_price) or 
                (direction == -1 and current_high >= stop_price)
            ):
                sl_return = np.log(stop_price / entry_price)
                daily_return = sl_return
                sl_cooldown = True

            else:
                daily_return = raw_return

            # --- 4. Final Strategy Return ---
            strategy_return = daily_return * position_size_base * leverage
            strategy_returns[index] = strategy_return
        
        # Store the daily return
        strategy_returns[index] = daily_return * position_size_base * leverage


    # --- 5. Final Metric Calculation ---
    
    df['Strategy_Return'] = strategy_returns
    df['Cumulative_Strategy_Return'] = np.exp(df['Strategy_Return'].cumsum())

    # ----------------------------------------------------
    # Benchmark: Buy & Hold (B&H)
    # ----------------------------------------------------
    df['Cumulative_Buy_and_Hold'] = np.exp(df['Daily_Return_Raw'].cumsum())

    # --- Generate Metrics ---
    metrics = []
    
    # B&H
    metrics.append(generate_metrics(
        df['Cumulative_Buy_and_Hold'], df['Daily_Return_Raw'], 'Buy & Hold (Benchmark)'
    ))
    
    # Strategy
    metrics.append(generate_metrics(
        df['Cumulative_Strategy_Return'], df['Strategy_Return'], f'Optimized Strategy (SMA {sma_period})'
    ))
    
    # Print comparison table
    comparison_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != 'Cumulative Returns'} 
        for m in metrics
    ])
    
    print("\n" + "=" * 60)
    print("FINAL BACKTEST METRICS (Optimized Strategy)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    return df, metrics

# --- 4. Plotting Results ---

def plot_results(df, metrics):
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
    df['Close'].plot(ax=ax1, label='Close Price', color='#9CA3AF', linewidth=1.5, alpha=0.9, zorder=3)
    
    # Calculate SMA period dynamically
    sma_period = OPTIMAL_SMA_PERIOD
    df[f'SMA_{sma_period}'].plot(ax=ax1, label=f'SMA {sma_period}', color='#3B82F6', linewidth=2, zorder=4)
    
    # Style and Labels for ax1
    ax1.set_title(f'{SYMBOL} Price and SMA (Linear Scale)', fontsize=16, color='white')
    ax1.set_xlabel('') 
    ax1.set_ylabel('Price (USDT)', fontsize=12, color='white')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#374151', which='both')
    
    # --- Bottom Subplot (ax2): Equity Curve (Log Scale) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plotting the equity curves
    df['Cumulative_Buy_and_Hold'].plot(ax=ax2, label=metrics[0]['Strategy'], color='#EF4444', linestyle='--', linewidth=1.5)
    df['Cumulative_Strategy_Return'].plot(ax=ax2, label=metrics[1]['Strategy'], color='#3B82F6', linewidth=2.5)
    
    # Set Y-axis to Logarithmic Scale
    ax2.set_yscale('log')
    
    # Style and Labels for ax2
    leverage = OPTIMAL_LEVERAGE
    sl_percent = OPTIMAL_STOP_LOSS_PERCENT
    reentry_prox = OPTIMAL_REENTRY_PROXIMITY_PERCENT
    ax2.set_title(f'Cumulative Return (Log Scale) - Optimized Strategy ({leverage:.2f}x) w/ {sl_percent*100:.2f}% SL', fontsize=14, color='white')
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
        
    optimal_c = OPTIMAL_CENTER_DISTANCE * 100
    optimal_l = OPTIMAL_LEVERAGE
    optimal_e = OPTIMAL_EXPONENT
    optimal_s = OPTIMAL_STOP_LOSS_PERCENT * 100
    optimal_p = OPTIMAL_REENTRY_PROXIMITY_PERCENT * 100
    optimal_m = OPTIMAL_SMA_PERIOD

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
                        <h1 class="text-3xl font-bold mb-4 text-green-400">Backtest Results: {SYMBOL} Final Optimized Strategy</h1>
                        
                        <div class="mb-8">
                            <h2 class="text-xl font-semibold mb-3 text-gray-200">Strategy Metrics Comparison</h2>
                            <p class="text-gray-400 mb-2">Final backtest run using 6 parameters optimized by Genetic Algorithm:</p>
                            <ul class="text-sm text-gray-400 mb-4 list-disc list-inside ml-4">
                                <li>SMA Period (M): {optimal_m} days</li>
                                <li>Center Distance (C): {optimal_c:.2f}%</li>
                                <li>Leverage (L): {optimal_l:.2f}x</li>
                                <li>Exponent (E): {optimal_e:.2f}</li>
                                <li>Stop Loss (S): {optimal_s:.2f}%</li>
                                <li>Re-entry Proximity (P): {optimal_p:.2f}%</li>
                            </ul>
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
        results_df, comparison_metrics = run_backtest_final(df_data)
        
        # 3. Plot results
        plot_results(results_df, comparison_metrics)

        # 4. Start web server in the main thread
        serve_results(comparison_metrics)
