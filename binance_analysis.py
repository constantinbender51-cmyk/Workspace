import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
from flask import Flask, render_template_string
import ccxt # CCXT is now mandatory

# --- Configuration Constants ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' # Daily candles
START_DATE = '2018-01-01'
SMA_WINDOWS = [50, 200, 400]
DC_WINDOW = 20  # Donchian Channel Period
POSITION_SIZE = 1  # Unit of asset traded
PORT = 8080

# --- Data Fetching (CCXT with Pagination) ---

def fetch_binance_data(symbol, timeframe, start_date):
    """
    Fetches historical OHLCV data from Binance using CCXT with pagination.
    This function is now the ONLY method for data retrieval.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True, # Automatically respect API rate limits
    })

    since_timestamp = exchange.parse8601(start_date + 'T00:00:00Z')
    all_ohlcv = []
    limit = 1000
    
    print(f"--- FETCHING DATA: {symbol} ({timeframe}) from {start_date} via Binance/CCXT ---")

    while True:
        try:
            # Fetch batch of OHLCV data
            ohlcv_batch = exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=since_timestamp, 
                limit=limit
            )
            
            if not ohlcv_batch:
                print("End of data reached.")
                break
                
            all_ohlcv.extend(ohlcv_batch)
            
            # Use the timestamp of the last candle fetched to set 'since' for the next request.
            # We add 1 millisecond to prevent fetching the last candle of the previous batch.
            since_timestamp = ohlcv_batch[-1][0] + 1 
            
            print(f"Fetched {len(all_ohlcv)} candles up to {exchange.iso8601(ohlcv_batch[-1][0])}")
            
            # If the number of candles returned is less than the limit, we've caught up to the present.
            if len(ohlcv_batch) < limit:
                break
                
            # Sleep to ensure we don't violate rate limits
            time.sleep(exchange.rateLimit / 1000) 
            
        except Exception as e:
            print(f"An error occurred during CCXT fetching: {e}")
            raise # Re-raise to be caught by the main handler
            
    if not all_ohlcv:
        raise ValueError("Could not fetch any data from Binance. Check connection or symbol.")
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df.sort_index()
    
    print(f"Total candles fetched: {len(df)}")
    return df

# --- Indicator Calculation ---

def calculate_indicators(df):
    """Calculates SMAs and Donchian Channels."""
    for window in SMA_WINDOWS:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

    df['DC_Upper'] = df['High'].rolling(window=DC_WINDOW).max()
    df['DC_Lower'] = df['Low'].rolling(window=DC_WINDOW).min()
    df['DC_Mid'] = (df['DC_Upper'] + df['DC_Lower']) / 2
    
    return df

# --- Backtesting and Equity Calculation ---

def run_backtest(df):
    """
    Implements a simple SMA 200 crossover strategy (Long/Flat) and calculates equity.
    Rule: Long (Position=1) when Close > SMA 200, Flat (Position=0) otherwise.
    """
    df['Position'] = np.where(df['Close'] > df['SMA_200'], POSITION_SIZE, 0)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1).fillna(0)
    df['Equity'] = (1 + df['Strategy_Return']).cumprod()
    df['Buy_Hold_Equity'] = (1 + df['Daily_Return']).cumprod()
    
    # Normalize starting equity to 1 at the point the longest SMA (SMA 400) is available
    normalization_index = df.first_valid_index()
    
    df['Equity'] = df['Equity'] / df['Equity'].loc[normalization_index]
    df['Buy_Hold_Equity'] = df['Buy_Hold_Equity'] / df['Buy_Hold_Equity'].loc[normalization_index]

    return df.dropna()

# --- Plotting ---

def create_plot(df):
    """Generates the main analysis plot using matplotlib, encoded as base64."""
    
    # Use data after the longest indicator (SMA 400) has stabilized
    df_plot = df.iloc[400:] 

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Price and Indicator Plot (Top Panel) ---
    ax1 = axes[0]
    ax1.plot(df_plot.index, df_plot['Close'], label='Price', color='#1f77b4', linewidth=1.5, alpha=0.9)
    ax1.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50', color='green', linestyle='--', alpha=0.7)
    ax1.plot(df_plot.index, df_plot['SMA_200'], label='SMA 200', color='red', linestyle='-', linewidth=2)
    ax1.plot(df_plot.index, df_plot['SMA_400'], label='SMA 400', color='orange', linestyle=':', alpha=0.7)
    
    # Donchian Channels
    ax1.plot(df_plot.index, df_plot['DC_Upper'], label=f'DC {DC_WINDOW} Upper', color='c', linestyle='-.', alpha=0.5)
    ax1.plot(df_plot.index, df_plot['DC_Lower'], label=f'DC {DC_WINDOW} Lower', color='c', linestyle='-.', alpha=0.5)
    ax1.fill_between(df_plot.index, df_plot['DC_Lower'], df_plot['DC_Upper'], color='cyan', alpha=0.05)
    
    # Highlighting Trades (Change in Position)
    entry_dates = df_plot[(df_plot['Position'].diff() > 0)].index
    exit_dates = df_plot[(df_plot['Position'].diff() < 0)].index
    
    ax1.scatter(entry_dates, df_plot.loc[entry_dates, 'Close'], marker='^', color='lime', s=100, label='Long Entry', zorder=5)
    ax1.scatter(exit_dates, df_plot.loc[exit_dates, 'Close'], marker='v', color='fuchsia', s=100, label='Exit', zorder=5)

    ax1.set_title(f'{SYMBOL} Price, Indicators, and Trading Signals (Binance/CCXT)', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_yscale('log')

    # --- Equity Plot (Bottom Panel) ---
    ax2 = axes[1]
    
    final_strategy_return = (df_plot['Equity'].iloc[-1] - 1) * 100
    final_bh_return = (df_plot['Buy_Hold_Equity'].iloc[-1] - 1) * 100
    
    ax2.plot(df_plot.index, df_plot['Equity'], label='SMA 200 Strategy Equity', color='blue', linewidth=2)
    ax2.plot(df_plot.index, df_plot['Buy_Hold_Equity'], label='Buy & Hold Benchmark', color='gray', linestyle='--', alpha=0.7)
    
    ax2.set_title(f'Strategy Equity Curve (Final Return: {final_strategy_return:.2f}%) vs B&H ({final_bh_return:.2f}%)', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Return (Normalized)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    # Encode plot for HTML embedding
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64, final_strategy_return

# --- Flask Web Server ---

app = Flask(__name__)

@app.route('/')
def analysis_dashboard():
    """Main route to run analysis and serve the plot."""
    data_source = "Binance (CCXT)"
    
    try:
        # 1. Get Data
        df_raw = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
        
        # 2. Calculate Indicators
        df_ind = calculate_indicators(df_raw)
        
        # 3. Run Backtest
        df_final = run_backtest(df_ind)
        
        # Determine current position status
        current_position = df_final['Position'].iloc[-1]
        status = "LONG" if current_position == POSITION_SIZE else "FLAT/SHORT"

        # 4. Create Plot
        img_base64, total_strategy_return = create_plot(df_final)
        
        # Determine the last 10 days of the data for a small table snapshot
        summary_df = df_final[['Close', 'SMA_200', 'DC_Upper', 'DC_Lower', 'Position', 'Equity']].tail(10)
        summary_table = summary_df.to_html(classes='table-auto w-full text-sm text-left', 
                                           float_format=lambda x: f'{x:,.2f}')
        
        # HTML Template with Tailwind CSS for modern look and responsiveness
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SMA 200 Trading Backtest</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                /* Custom styles for the generated table */
                .table-auto th, .table-auto td {{
                    padding: 8px 12px;
                    border: 1px solid #e5e7eb;
                    text-align: right;
                }}
                .table-auto th {{
                    text-align: left;
                    background-color: #f3f4f6;
                    font-weight: 600;
                }}
                .data-source-tag {{
                    font-size: 0.75rem;
                    padding: 0.25rem 0.5rem;
                    border-radius: 0.5rem;
                    font-weight: bold;
                    margin-left: 1rem;
                }}
            </style>
        </head>
        <body class="bg-gray-100 p-4 sm:p-8 font-sans">
            <div class="max-w-7xl mx-auto bg-white p-6 sm:p-10 rounded-xl shadow-2xl">
                <h1 class="text-4xl font-extrabold text-blue-700 mb-6 border-b-4 border-blue-200 pb-2">
                    Crypto Trading Backtest Dashboard
                    <span class="data-source-tag bg-blue-200 text-blue-800">{data_source}</span>
                </h1>

                <div class="grid md:grid-cols-3 gap-6 mb-8">
                    <div class="bg-blue-50 p-4 rounded-lg shadow-md">
                        <p class="text-lg font-medium text-blue-600">Symbol</p>
                        <p class="text-2xl font-bold text-blue-800">{SYMBOL} ({TIMEFRAME})</p>
                    </div>
                    <div class="bg-green-50 p-4 rounded-lg shadow-md">
                        <p class="text-lg font-medium text-green-600">Strategy Return</p>
                        <p class="text-2xl font-bold text-green-800">{total_strategy_return:.2f}%</p>
                    </div>
                    <div class="bg-red-50 p-4 rounded-lg shadow-md">
                        <p class="text-lg font-medium text-red-600">Current Position</p>
                        <p class="text-2xl font-bold text-red-800">{status}</p>
                    </div>
                </div>

                <h2 class="text-2xl font-semibold text-gray-700 mb-4">Price & Equity Plot (Since {START_DATE})</h2>
                <div class="bg-gray-50 p-2 rounded-lg shadow-inner mb-8 overflow-hidden">
                    <img src="data:image/png;base64,{img_base64}" alt="Trading Strategy Backtest Plot" class="w-full h-auto rounded-lg"/>
                </div>

                <h2 class="text-2xl font-semibold text-gray-700 mb-4">Last 10 Days Data Snapshot</h2>
                <div class="overflow-x-auto rounded-lg shadow-md border border-gray-200">
                    {summary_table}
                </div>
                
                <p class="mt-8 text-sm text-gray-600 border-t pt-4">
                    **Note:** This backtest uses a simple Close > SMA 200 rule for daily position sizing (long/flat). Transaction costs and slippage are not included.
                </p>
            </div>
        </body>
        </html>
        """
        return render_template_string(html_content)

    except Exception as e:
        # Generic error handling for failed CCXT import or data fetching issues
        error_message = f"Analysis Failed. Please ensure the 'ccxt' library is installed (`pip install ccxt`), and check your network connection or API limits. Details: {e}"
        print(f"FATAL ERROR: {error_message}")
        return render_template_string(f"""
            <div class="p-8 text-center bg-white rounded-xl shadow-lg m-auto max-w-lg">
                <h1 class="text-3xl font-bold text-red-600 mb-4">Analysis Failed</h1>
                <p class="text-gray-700">The script could not fetch data from Binance. This is usually due to the 'ccxt' library not being installed or a network/API issue.</p>
                <p class="mt-6 p-4 bg-red-100 text-red-800 rounded-lg text-left overflow-x-auto"><strong>Error:</strong> {error_message}</p>
            </div>
        """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)
