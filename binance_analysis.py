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
REGIME_FILTER_SMA = 360 # Macro-regime filter (SMA 360)

# --- FINAL OPTIMIZED PARAMETER SETS ---

# Parameters optimized for the BEAR Market (2022-2024)
BEAR_PARAMS = {
    'C': 0.06845880041198227,         # Center Distance
    'L': 6.1440852287730205,          # Leverage
    'E': 0.8417039838296924,          # Exponent
    'S': 0.032163070773567434,        # Stop Loss
    'P': 0.029098279423892606,        # Re-entry Proximity
    'M': 157                          # Strategy SMA Period
}

# Parameters optimized for the BULL Market (2024-2025)
BULL_PARAMS = {
    'C': 0.001,
    'L': 1.9936150688902803,
    'E': 0.17170399111395088,
    'S': 0.038560454561416095,
    'P': 0.01,
    'M': 143
}

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

# --- 3. Backtesting Logic (Regime Switching) ---

def run_backtest_regime_switching(df_raw):
    """
    Applies the strategy, switching parameters based on the SMA 360 regime filter.
    """
    df = df_raw.copy()
    
    # 1. Calculate SMAs and Raw Daily Returns
    df[f'SMA_{REGIME_FILTER_SMA}'] = df['Close'].rolling(window=REGIME_FILTER_SMA).mean()
    df['Daily_Return_Raw'] = np.log(df['Close'] / df['Close'].shift(1))

    # Pre-calculate strategy-specific SMAs (M=143 and M=157)
    df[f'SMA_{BULL_PARAMS["M"]}'] = df['Close'].rolling(window=BULL_PARAMS["M"]).mean()
    df[f'SMA_{BEAR_PARAMS["M"]}'] = df['Close'].rolling(window=BEAR_PARAMS["M"]).mean()

    # --- Prepare LAG INDICATORS (CRITICAL FOR LOOKAHEAD PREVENTION AND FIXING ERROR) ---
    df['Yesterday_Close'] = df['Close'].shift(1)
    df['Yesterday_SMA_360'] = df[f'SMA_{REGIME_FILTER_SMA}'].shift(1)
    
    # Lagged SMAs for the strategy itself
    df[f'LAG_SMA_{BULL_PARAMS["M"]}'] = df[f'SMA_{BULL_PARAMS["M"]}'].shift(1)
    df[f'LAG_SMA_{BEAR_PARAMS["M"]}'] = df[f'SMA_{BEAR_PARAMS["M"]}'].shift(1)


    # Drop initial NaNs (max lag is 360 + 1)
    df = df.dropna().copy()
    
    if df.empty:
        raise ValueError("Insufficient data after cleanup for backtesting.")

    # Initialize Series for returns
    strategy_returns = pd.Series(index=df.index, dtype=float)
    sl_cooldown = False # State variable
    
    print(f"-> Running FINAL Regime Switching Backtest (Filter: SMA {REGIME_FILTER_SMA})...")

    # Iterate through the DataFrame for day-by-day logic
    for i in range(len(df)):
        index = df.index[i]
        
        entry_price = df.loc[index, 'Yesterday_Close']
        yesterday_sma_360 = df.loc[index, 'Yesterday_SMA_360']
        
        # --- 1. DETERMINE REGIME AND PARAMETERS ---
        # Bull Regime: Yesterday's Close Price > SMA 360
        is_bull_regime = entry_price > yesterday_sma_360
        
        current_params = BULL_PARAMS if is_bull_regime else BEAR_PARAMS
        
        center = current_params['C']
        leverage = current_params['L']
        exponent = current_params['E']
        sl_percent = current_params['S']
        reentry_prox = current_params['P']
        sma_period = current_params['M']
        
        # Access the correct LAG SMAs using the optimized M value
        strategy_sma_lag_key = f'LAG_SMA_{sma_period}'
        yesterday_strategy_sma = df.loc[index, strategy_sma_lag_key]
        
        # Proximity uses the current regime's strategy SMA
        proximity = np.abs((entry_price - yesterday_strategy_sma) / yesterday_strategy_sma)
        
        # --- 2. Calculate Base Position and Direction ---
        # Direction uses the current regime's strategy SMA
        direction = np.where(entry_price > yesterday_strategy_sma, 1, -1)
        distance_d = np.abs((entry_price - yesterday_strategy_sma) / yesterday_strategy_sma)
        
        # Multiplier (M) calculation
        distance_scaler = 1.0 / center
        scaled_distance = distance_d * distance_scaler
        epsilon = 1e-10 
        
        denominator = (1.0 / np.maximum(scaled_distance, epsilon)) + scaled_distance - 1.0
        multiplier = np.where(denominator <= 0, 0, 1.0 / (denominator ** exponent))
        position_size_base = direction * multiplier

        # --- 3. Apply Re-entry/Cooldown Filter (State Management) ---
        
        if sl_cooldown:
            if proximity <= reentry_prox:
                sl_cooldown = False 
                
        if sl_cooldown:
            position_size_base = 0.0
            daily_return = 0.0
            
        else:
            # --- 4. Stop Loss Logic ---
            
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

        # --- 5. Final Strategy Return ---
        strategy_return = daily_return * position_size_base * leverage
        strategy_returns[index] = strategy_return


    # --- 6. Final Metric Calculation ---
    
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
        df['Cumulative_Strategy_Return'], df['Strategy_Return'], f'Regime Switching (SMA {REGIME_FILTER_SMA} Filter)'
    ))
    
    # Print comparison table
    comparison_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != 'Cumulative Returns'} 
        for m in metrics
    ])
    
    print("\n" + "=" * 60)
    print("FINAL BACKTEST METRICS (Regime Switching)")
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
    
    # Plot the macro regime filter SMA 360
    df[f'SMA_{REGIME_FILTER_SMA}'].plot(ax=ax1, label=f'Regime Filter SMA {REGIME_FILTER_SMA}', color='#F97316', linewidth=2, zorder=4)
    
    # Style and Labels for ax1
    ax1.set_title(f'{SYMBOL} Price and Macro-Regime Filter (SMA {REGIME_FILTER_SMA})', fontsize=16, color='white')
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
    ax2.set_title(f'Cumulative Return (Log Scale) - Regime Switching Strategy (Filter: SMA {REGIME_FILTER_SMA})', fontsize=14, color='white')
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
        
    bull_str = f"Bull (C={BULL_PARAMS['C']*100:.2f}%, L={BULL_PARAMS['L']:.2f}x, E={BULL_PARAMS['E']:.2f}, SL={BULL_PARAMS['S']*100:.2f}%, P={BULL_PARAMS['P']*100:.2f}%, M={BULL_PARAMS['M']}d)"
    bear_str = f"Bear (C={BEAR_PARAMS['C']*100:.2f}%, L={BEAR_PARAMS['L']:.2f}x, E={BEAR_PARAMS['E']:.2f}, SL={BEAR_PARAMS['S']*100:.2f}%, P={BEAR_PARAMS['P']*100:.2f}%, M={BEAR_PARAMS['M']}d)"

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
                        <h1 class="text-3xl font-bold mb-4 text-green-400">Backtest Results: {SYMBOL} Regime Switching Strategy</h1>
                        
                        <div class="mb-8">
                            <h2 class="text-xl font-semibold mb-3 text-gray-200">Strategy Metrics Comparison</h2>
                            <p class="text-gray-400 mb-2">Strategy uses SMA {REGIME_FILTER_SMA} to define the macro-regime (Bull: Price > SMA, Bear: Price &le; SMA):</p>
                            <ul class="text-xs text-gray-400 mb-4 list-disc list-inside ml-4">
                                <li>**BULL Parameters (Optimized for 2024-2025)**: {bull_str}</li>
                                <li>**BEAR Parameters (Optimized for 2022-2024)**: {bear_str}</li>
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
        results_df, comparison_metrics = run_backtest_regime_switching(df_data)
        
        # 3. Plot results
        plot_results(results_df, comparison_metrics)

        # 4. Start web server in the main thread
        serve_results(comparison_metrics)
