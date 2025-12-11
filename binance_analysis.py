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
            print(f"   Fetched up to: {dt.datetime.fromtimestamp(start_ts / 1000).strftime('%Y-%m-%d')}")
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
    df = df[['Close']].astype(float)
    
    return df.dropna()

# --- 2. Metric Calculations ---

def calculate_sharpe_ratio(returns, annualization_factor=ANNUALIZATION_FACTOR, risk_free_rate=0):
    """Calculates the Annualized Sharpe Ratio."""
    if returns.empty:
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

# --- 3. Backtesting Logic ---

def run_backtest(df):
    """
    Applies the single 120 SMA trading strategy and generates metrics.
    """
    print(f"-> Running single 120 SMA backtest on {len(df)} candles...")
    
    # Calculate 120 SMA
    df[f'SMA_{SMA_PERIOD_120}'] = df['Close'].rolling(window=SMA_PERIOD_120).mean()
    
    # Calculate daily returns (log returns)
    df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Prepare Lagged Data (Look-ahead Bias Prevention) ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA_120'] = df[f'SMA_{SMA_PERIOD_120}'].shift(1)
    
    # Drop NaNs from SMAs and shifting
    df = df.dropna()
    
    # ----------------------------------------------------
    # Strategy: Simple 120 SMA Crossover (Long/Short)
    # ----------------------------------------------------
    df['Position'] = np.where(
        df['Yesterday_Close'] > df['Yesterday_SMA_120'],
        1,
        -1
    )
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    df['Cumulative_Strategy_Return'] = np.exp(df['Strategy_Return'].cumsum())

    # ----------------------------------------------------
    # Benchmark: Buy & Hold (B&H)
    # ----------------------------------------------------
    df['Cumulative_Buy_and_Hold'] = np.exp(df['Daily_Return'].cumsum())

    # --- Generate Metrics ---
    metrics = []
    
    # B&H
    metrics.append(generate_metrics(
        df['Cumulative_Buy_and_Hold'], df['Daily_Return'], 'Buy & Hold (Benchmark)'
    ))
    
    # Strategy
    metrics.append(generate_metrics(
        df['Cumulative_Strategy_Return'], df['Strategy_Return'], f'Strategy (120 SMA Crossover)'
    ))
    
    # Print comparison table
    comparison_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != 'Cumulative Returns'} 
        for m in metrics
    ])
    
    print("\n" + "=" * 60)
    print("BACKTEST METRICS COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    return df, metrics

# --- 4. Plotting Results ---

def plot_results(df, metrics):
    """
    Generates and saves the plot of the strategy and benchmark equity curves.
    """
    print(f"-> Generating plot in '{RESULTS_DIR}/{PLOT_FILE}'...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, PLOT_FILE)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plotting the equity curves
    df['Cumulative_Buy_and_Hold'].plot(ax=ax, label=metrics[0]['Strategy'], color='#EF4444', linestyle='--', linewidth=1.5)
    df['Cumulative_Strategy_Return'].plot(ax=ax, label=metrics[1]['Strategy'], color='#3B82F6', linewidth=2.5)
    
    # Set Y-axis to Logarithmic Scale
    ax.set_yscale('log')
    
    # Style and Labels
    ax.set_title(f'{SYMBOL} 120 SMA Crossover vs. Buy & Hold (Log Scale Since 2018)', fontsize=16, color='white')
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
                        <h1 class="text-3xl font-bold mb-4 text-green-400">Backtest Results: {SYMBOL} 120 SMA Crossover</h1>
                        
                        <div class="mb-8">
                            <h2 class="text-xl font-semibold mb-3 text-gray-200">Strategy Metrics Comparison</h2>
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
                            <h2 class="text-xl font-semibold mb-3 text-gray-200">Cumulative Returns (Log Scale)</h2>
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
        results_df, comparison_metrics = run_backtest(df_data)
        
        # 3. Plot results
        plot_results(results_df, comparison_metrics)

        # 4. Start web server in the main thread
        serve_results(comparison_metrics)
