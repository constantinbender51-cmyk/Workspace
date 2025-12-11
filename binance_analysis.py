import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
from flask import Flask, render_template_string
import ccxt 
import sys
import traceback

# --- Configuration Constants ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d' # Daily candles
START_DATE = '2018-01-01'
SMA_WINDOWS = [50, 200, 400]
DC_WINDOW = 20  # Donchian Channel Period
POSITION_SIZE = 1  # Unit of asset traded
PORT = 8080

# --- Global Variables to store results (populated once at startup) ---
GLOBAL_DATA_SOURCE = "Binance (CCXT)"
GLOBAL_IMG_BASE64 = ""
GLOBAL_SUMMARY_TABLE = ""
GLOBAL_TOTAL_RETURN = 0.0
GLOBAL_STATUS = "PENDING"
GLOBAL_ERROR = None # Store fatal error if analysis fails

# --- Data Fetching (CCXT with Pagination) ---

def fetch_binance_data(symbol, timeframe, start_date):
    """
    Fetches historical OHLCV data from Binance using CCXT with pagination.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True, # Automatically respect API rate limits
    })

    since_timestamp = exchange.parse8601(start_date + 'T00:00:00Z')
    all_ohlcv = []
    limit = 1000
    
    print(f"--- STARTING DATA FETCH: {symbol} ({timeframe}) from {start_date} via Binance/CCXT ---")

    while True:
        try:
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
            since_timestamp = ohlcv_batch[-1][0] + 1 
            
            print(f"Fetched total {len(all_ohlcv)} candles up to {exchange.iso8601(ohlcv_batch[-1][0])}")
            
            if len(ohlcv_batch) < limit:
                break
                
            time.sleep(exchange.rateLimit / 1000) 
            
        except Exception as e:
            print(f"An error occurred during CCXT fetching: {e}")
            raise # Re-raise for the analysis setup function to catch
            
    if len(all_ohlcv) < 400: # Need at least 400 for the SMA 400
        raise ValueError(f"Not enough data fetched. Required >400, received {len(all_ohlcv)}.")
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df.sort_index()
    
    print(f"--- SUCCESS: Total candles fetched: {len(df)} ---")
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
    Implements a simple SMA 200 Long/Short crossover strategy and calculates equity.
    Rule: Long (Position=1) when Close > SMA 200. Short (Position=-1) when Close <= SMA 200.
    """
    # 1. Define Position: 1 (Long) if Close > SMA 200, -1 (Short) otherwise
    df['Position'] = np.where(df['Close'] > df['SMA_200'], 
                              POSITION_SIZE, 
                              -POSITION_SIZE)
    
    # 2. Calculate Daily Strategy Returns
    # Daily return: Close to Close
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Strategy return = (Daily Asset Return) * (Position held from previous day)
    # This correctly handles Long (return * 1) and Short (return * -1). 
    # A negative asset return (price drop) results in a positive strategy return when short (- * - = +).
    df['Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1).fillna(0)
    
    # 3. Calculate Cumulative Equity
    df['Equity'] = (1 + df['Strategy_Return']).cumprod()
    df['Buy_Hold_Equity'] = (1 + df['Daily_Return']).cumprod()
    
    # Normalize starting equity to 1 at the point the longest SMA (SMA 400) is available
    df_clean = df.dropna()
    if df_clean.empty:
        raise ValueError("DataFrame is empty after dropping NaN values from indicators.")
        
    normalization_index = df_clean.index[0]
    
    # Use .loc to avoid SettingWithCopyWarning, though technically safe here
    df_clean.loc[:, 'Equity'] = df_clean['Equity'] / df_clean['Equity'].loc[normalization_index]
    df_clean.loc[:, 'Buy_Hold_Equity'] = df_clean['Buy_Hold_Equity'] / df_clean['Buy_Hold_Equity'].loc[normalization_index]

    return df_clean

# --- Plotting ---

def create_plot(df):
    """Generates the main analysis plot using matplotlib, encoded as base64."""
    
    # Use data after the longest indicator (SMA 400) has stabilized
    df_plot = df.iloc[400:].copy() # Use a copy for manipulation

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Price and Indicator Plot (Top Panel) ---
    ax1 = axes[0]
    
    # 1. Background Coloring for Position (Long/Short)
    # Determine segments of continuous position
    positions = df_plot['Position'].unique()
    
    # Iterate through positions and plot background spans
    for pos in positions:
        df_pos = df_plot[df_plot['Position'] == pos]
        
        # Determine color
        if pos == POSITION_SIZE:
            color = 'green'
            alpha = 0.08
        elif pos == -POSITION_SIZE:
            color = 'red'
            alpha = 0.08
        else: # Should not happen with this 1/-1 strategy
            continue 
            
        # Group adjacent dates for cleaner spans
        in_position = df_pos.index.to_series().diff().dt.days == 1
        segment_starts = df_pos.index[~in_position]
        segment_ends = df_pos.index[in_position.shift(-1, fill_value=False)]
        
        # If the last point is a start, it's a segment end too.
        if df_pos.index[-1] not in segment_ends and df_pos.index[-1] in segment_starts:
             segment_ends = segment_ends.append(pd.Index([df_pos.index[-1]]))

        # Correct for cases where the very first day is a segment start
        if df_plot.index[0] in segment_starts and df_plot.index[0] in df_pos.index:
            pass
        else:
             segment_starts = df_pos.index[~in_position]
             
        # Plot spans
        for start, end in zip(segment_starts, segment_ends):
             # Ensure end date is slightly later than the actual candle to span the whole day visually
             end_adjusted = end + pd.Timedelta(days=1)
             ax1.axvspan(start, end_adjusted, facecolor=color, alpha=alpha, zorder=0)

    # 2. Price and Indicators
    ax1.plot(df_plot.index, df_plot['Close'], label='Price', color='#1f77b4', linewidth=1.5, alpha=0.9, zorder=1)
    ax1.plot(df_plot.index, df_plot['SMA_50'], label='SMA 50', color='green', linestyle='--', alpha=0.7, zorder=1)
    ax1.plot(df_plot.index, df_plot['SMA_200'], label='SMA 200', color='red', linestyle='-', linewidth=2, zorder=1)
    ax1.plot(df_plot.index, df_plot['SMA_400'], label='SMA 400', color='orange', linestyle=':', alpha=0.7, zorder=1)
    
    # Donchian Channels
    ax1.plot(df_plot.index, df_plot['DC_Upper'], label=f'DC {DC_WINDOW} Upper', color='c', linestyle='-.', alpha=0.5, zorder=1)
    ax1.plot(df_plot.index, df_plot['DC_Lower'], label=f'DC {DC_WINDOW} Lower', color='c', linestyle='-.', alpha=0.5, zorder=1)
    ax1.fill_between(df_plot.index, df_plot['DC_Lower'], df_plot['DC_Upper'], color='cyan', alpha=0.05, zorder=0)
    
    # Highlighting Trades (Change in Position)
    entry_long_dates = df_plot[(df_plot['Position'].diff() == POSITION_SIZE*2)].index # Transition from -1 to 1
    exit_long_dates = df_plot[(df_plot['Position'].diff() == -POSITION_SIZE*2)].index # Transition from 1 to -1
    
    ax1.scatter(entry_long_dates, df_plot.loc[entry_long_dates, 'Close'], marker='^', color='darkgreen', s=100, label='Long Entry', zorder=5)
    ax1.scatter(exit_long_dates, df_plot.loc[exit_long_dates, 'Close'], marker='v', color='darkred', s=100, label='Short Entry/Exit Long', zorder=5)


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

# --- Setup Function (Runs once at startup) ---

def setup_analysis():
    """Runs the full backtest analysis and populates global variables."""
    global GLOBAL_IMG_BASE64, GLOBAL_SUMMARY_TABLE, GLOBAL_TOTAL_RETURN, GLOBAL_STATUS, GLOBAL_ERROR
    
    try:
        # 1. Get Data
        df_raw = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
        
        # 2. Calculate Indicators
        df_ind = calculate_indicators(df_raw)
        
        # 3. Run Backtest
        df_final = run_backtest(df_ind)
        
        # 4. Determine current status
        current_position = df_final['Position'].iloc[-1]
        GLOBAL_STATUS = "LONG" if current_position == POSITION_SIZE else "SHORT"

        # 5. Create Plot
        img_base64, total_strategy_return = create_plot(df_final)
        
        # 6. Populate Global Results
        GLOBAL_IMG_BASE64 = img_base64
        GLOBAL_TOTAL_RETURN = total_strategy_return
        
        # Create summary table
        summary_df = df_final[['Close', 'SMA_200', 'DC_Upper', 'DC_Lower', 'Position', 'Equity']].tail(10)
        GLOBAL_SUMMARY_TABLE = summary_df.to_html(classes='table-auto w-full text-sm text-left', 
                                                  float_format=lambda x: f'{x:,.2f}')
        
        print("--- ANALYSIS COMPLETE ---")

    except Exception as e:
        # Catch and store any errors during startup analysis
        GLOBAL_ERROR = f"Fatal Analysis Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING ANALYSIS SETUP ---\n{GLOBAL_ERROR}", file=sys.stderr)
        
# --- Flask Web Server ---

app = Flask(__name__)

@app.route('/')
def analysis_dashboard():
    """Renders the dashboard using pre-calculated global variables."""
    
    if GLOBAL_ERROR:
        return render_template_string(f"""
            <div class="p-8 text-center bg-white rounded-xl shadow-lg m-auto max-w-lg">
                <h1 class="text-3xl font-bold text-red-600 mb-4">Analysis Failed at Startup</h1>
                <p class="text-gray-700">The backtest and plotting could not be completed when the server started. This is usually due to a network error, Binance API limit, or missing data.</p>
                <p class="mt-6 p-4 bg-red-100 text-red-800 rounded-lg text-left overflow-x-auto text-sm whitespace-pre-wrap"><strong>Details:</strong> {GLOBAL_ERROR}</p>
            </div>
        """)
        
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
                <span class="data-source-tag bg-blue-200 text-blue-800">{GLOBAL_DATA_SOURCE}</span>
            </h1>

            <div class="grid md:grid-cols-3 gap-6 mb-8">
                <div class="bg-blue-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-blue-600">Symbol</p>
                    <p class="text-2xl font-bold text-blue-800">{SYMBOL} ({TIMEFRAME})</p>
                </div>
                <div class="bg-green-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-green-600">Strategy Return</p>
                    <p class="text-2xl font-bold text-green-800">{GLOBAL_TOTAL_RETURN:.2f}%</p>
                </div>
                <div class="bg-red-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-red-600">Current Position</p>
                    <p class="text-2xl font-bold text-red-800">{GLOBAL_STATUS}</p>
                </div>
            </div>

            <h2 class="text-2xl font-semibold text-gray-700 mb-4">Price & Equity Plot (Since {START_DATE})</h2>
            <div class="bg-gray-50 p-2 rounded-lg shadow-inner mb-8 overflow-hidden">
                <img src="data:image/png;base64,{GLOBAL_IMG_BASE64}" alt="Trading Strategy Backtest Plot" class="w-full h-auto rounded-lg"/>
            </div>

            <h2 class="text-2xl font-semibold text-gray-700 mb-4">Last 10 Days Data Snapshot</h2>
            <div class="overflow-x-auto rounded-lg shadow-md border border-gray-200">
                {GLOBAL_SUMMARY_TABLE}
            </div>
            
            <p class="mt-8 text-sm text-gray-600 border-t pt-4">
                **Note:** This backtest uses a Long/Short strategy based on Close vs. SMA 200, executed daily (Close-to-Close). Transaction costs and slippage are not included.
            </p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == '__main__':
    # 1. Run the entire data analysis and calculation process once at startup
    setup_analysis()
    
    # 2. Start the Flask server to serve the pre-calculated results
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)
