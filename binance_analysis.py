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
SMA_WINDOWS_PLOT = [120] 
DC_WINDOW = 20  # Donchian Channel Period
POSITION_SIZE = 1  # Unit of asset traded (+1 for Long, -1 for Short)
PORT = 8080
ANNUAL_TRADING_DAYS = 252 # Used for annualizing Sharpe Ratio

# --- Global Variables to store results (populated once at startup) ---
GLOBAL_DATA_SOURCE = "Binance (CCXT)"
GLOBAL_IMG_BASE64 = ""      # Image for SMA 120 Strategy Plot
GLOBAL_ANALYSIS_IMG = ""    # Image for Sharpe/Equity Scan Plot
GLOBAL_SUMMARY_TABLE = ""
GLOBAL_TOTAL_RETURN = 0.0
GLOBAL_STATUS = "PENDING"
GLOBAL_ERROR = None
GLOBAL_TOP_10_MD = ""       # Markdown for the top 10 results

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
            raise 
            
    if len(all_ohlcv) < 400: 
        raise ValueError(f"Not enough data fetched. Required >400, received {len(all_ohlcv)}.")
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df.sort_index()
    
    print(f"--- SUCCESS: Total candles fetched: {len(df)} ---")
    return df

# --- Indicator Calculation ---

def calculate_indicators(df, windows_to_calculate):
    """Calculates specified SMAs and Donchian Channels."""
    # Ensure all required SMAs for the plot are calculated
    for window in windows_to_calculate:
        if f'SMA_{window}' not in df.columns:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    
    # Calculate D-Channels (not plotted, but useful if the strategy changes)
    df['DC_Upper'] = df['High'].rolling(window=DC_WINDOW).max()
    df['DC_Lower'] = df['Low'].rolling(window=DC_WINDOW).min()
    df['DC_Mid'] = (df['DC_Upper'] + df['DC_Lower']) / 2
    
    return df

# --- Backtesting and Strategy Return Calculation (Reusable) ---

def get_strategy_returns(df_raw, sma_window):
    """
    Calculates returns for a Long/Short strategy based on a single given SMA window.
    """
    df = df_raw.copy()
    sma_col = f'SMA_{sma_window}'
    
    # 1. Calculate the specific SMA needed for this run
    if sma_col not in df.columns:
        df[sma_col] = df['Close'].rolling(window=sma_window).mean()
    
    # 2. Strategy Logic: Long (+1) if Close > SMA, Short (-1) if Close <= SMA
    df.loc[:, 'Next_Day_Position'] = np.where(df['Close'] > df[sma_col], 
                                              POSITION_SIZE,      # +1 for Long
                                              -POSITION_SIZE)     # -1 for Short
    
    # Position HELD for the day t return is based on the decision from day t-1.
    df.loc[:, 'Position'] = df['Next_Day_Position'].shift(1).fillna(0)
    
    # 3. Calculate Strategy Return
    df['Daily_Return'] = df['Close'].pct_change()
    daily_returns_clean = df['Daily_Return'].fillna(0)
    
    # Strategy Return = Daily Asset Return * Held Position
    df.loc[:, 'Strategy_Return'] = daily_returns_clean * df['Position']
    
    # 4. Cleanup and Normalization
    df_clean = df.dropna(subset=['Position', sma_col])
    if df_clean.empty:
        return pd.DataFrame() 
    
    normalization_index = df_clean.index[0]
    
    # Calculate Strategy Equity
    strategy_returns_tradable = df_clean['Strategy_Return'].loc[normalization_index:]
    df_clean.loc[:, 'Equity'] = (1 + strategy_returns_tradable).cumprod().reindex(df_clean.index, fill_value=1.0)
    
    # Calculate Buy & Hold Benchmark 
    start_price_bh = df_clean['Close'].loc[normalization_index]
    bh_equity_series = (df_clean['Close'] / start_price_bh).reindex(df_clean.index, fill_value=1.0)
    df_clean.loc[:, 'Buy_Hold_Equity'] = bh_equity_series

    return df_clean

# --- Comprehensive SMA Scan and Sharpe Ratio Calculation ---

def calculate_sharpe_ratios_scan(df_raw, min_sma, max_sma):
    """
    Iterates through SMA windows and calculates the Annualized Sharpe Ratio and Final Equity for each.
    Returns a DataFrame with ['SMA_Window', 'Sharpe_Ratio', 'Final_Equity'].
    """
    results = []
    
    print(f"\n--- Starting SMA Scan (Sharpe Ratio Calculation from SMA {min_sma} to {max_sma}) ---")
    
    for sma_w in range(min_sma, max_sma + 1):
        df_strategy = get_strategy_returns(df_raw, sma_w)
        
        if df_strategy.empty:
            continue
            
        returns = df_strategy['Strategy_Return']
        
        avg_daily_return = returns.mean()
        std_dev_daily_return = returns.std()
        
        if std_dev_daily_return > 0:
            sharpe_ratio = (avg_daily_return / std_dev_daily_return) * np.sqrt(ANNUAL_TRADING_DAYS)
        else:
            sharpe_ratio = 0.0 
            
        final_equity = df_strategy['Equity'].iloc[-1]
        
        results.append({
            'SMA_Window': sma_w,
            'Sharpe_Ratio': sharpe_ratio,
            'Final_Equity': final_equity
        })
        
        if sma_w % 50 == 0:
            print(f"Processed SMA {sma_w}...")

    results_df = pd.DataFrame(results)
    
    print(f"--- SMA Scan Complete. Found {len(results_df)} valid windows. ---")
    return results_df

# --- Analysis Visualization (Heatmap Replacement) ---

def create_analysis_visualization(results_df):
    """
    Generates a figure with two plots: Sharpe Ratio vs. SMA and Final Equity vs. SMA.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Top Plot: Sharpe Ratio ---
    ax1 = axes[0]
    ax1.plot(results_df['SMA_Window'], results_df['Sharpe_Ratio'], 
             color='#1f77b4', linewidth=1.5, alpha=0.8)
    
    # Highlight the peak Sharpe Ratio
    best_sma = results_df.iloc[0]['SMA_Window']
    best_sharpe = results_df.iloc[0]['Sharpe_Ratio']
    
    ax1.scatter(best_sma, best_sharpe, color='red', s=50, zorder=5)
    ax1.axhline(best_sharpe, color='red', linestyle='--', alpha=0.4, linewidth=1)

    ax1.set_ylabel('Annualized Sharpe Ratio', fontsize=10)
    ax1.set_title('SMA Crossover Strategy Performance (SMA 1 to 400)', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Bottom Plot: Final Equity ---
    ax2 = axes[1]
    ax2.plot(results_df['SMA_Window'], results_df['Final_Equity'], 
             color='#2ca02c', linewidth=1.5, alpha=0.8)
    
    # Highlight the peak Final Equity
    best_equity_row = results_df.sort_values(by='Final_Equity', ascending=False).iloc[0]
    best_equity_sma = best_equity_row['SMA_Window']
    best_equity = best_equity_row['Final_Equity']
    
    ax2.scatter(best_equity_sma, best_equity, color='red', s=50, zorder=5)
    ax2.axhline(best_equity, color='red', linestyle='--', alpha=0.4, linewidth=1)
    
    ax2.set_xlabel('SMA Window (Days)', fontsize=10)
    ax2.set_ylabel('Final Equity Multiplier (x)', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    # Encode plot for HTML embedding
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# --- Plotting Main Strategy (SMA 120) ---

def create_plot(df, strategy_sma):
    """Generates the main SMA 120 strategy plot (Close Price + Equity Curve)."""
    
    # Use data after the longest required indicator (SMA 400 is the limit)
    df_plot = df.iloc[400:].copy() 
    strategy_sma_col = f'SMA_{strategy_sma}'

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, # Reduced height for better dashboard fit
                             gridspec_kw={'height_ratios': [3, 2]})
    
    # --- Price and SMA (Top Panel) ---
    ax1 = axes[0]
    
    # 1. Background Coloring
    pos_series = df_plot['Position']
    change_indices = pos_series.index[pos_series.diff() != 0]
    segment_starts = pd.Index([pos_series.index[0]]).append(change_indices)
    segment_ends = change_indices.append(pd.Index([pos_series.index[-1]]))
    
    for start, end in zip(segment_starts, segment_ends):
        current_pos = pos_series.loc[start]
        if current_pos == POSITION_SIZE: color = 'green'; alpha = 0.08
        elif current_pos == -POSITION_SIZE: color = 'red'; alpha = 0.08
        else: continue
            
        end_adjusted = end + pd.Timedelta(days=1)
        ax1.axvspan(start, end_adjusted, facecolor=color, alpha=alpha, zorder=0)

    # 2. Price and SMA 120
    ax1.plot(df_plot.index, df_plot['Close'], label='Price (Close)', color='#1f77b4', linewidth=1.5, alpha=0.9, zorder=1)
    ax1.plot(df_plot.index, df_plot[strategy_sma_col], 
             label=f'SMA {strategy_sma}', 
             color='#FF6347', linestyle='-', linewidth=2.5, zorder=2) 
    
    ax1.set_ylabel('Price (Log Scale)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_yscale('log')

    # --- Equity Plot (Bottom Panel) ---
    ax2 = axes[1]
    
    final_strategy_return = (df_plot['Equity'].iloc[-1] - 1) * 100
    
    plotted_returns = df_plot['Strategy_Return']
    sharpe_plotted = (plotted_returns.mean() / plotted_returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
    
    ax2.plot(df_plot.index, df_plot['Equity'], label=f'Strategy Equity (Sharpe: {sharpe_plotted:.2f})', color='blue', linewidth=3)
    ax2.plot(df_plot.index, df_plot['Buy_Hold_Equity'], label='Buy & Hold Benchmark', color='gray', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Cumulative Return', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64, final_strategy_return

# --- Setup Function (Runs once at startup) ---

def setup_analysis():
    """Runs the full backtest analysis, SMA scan, and populates global variables."""
    global GLOBAL_IMG_BASE64, GLOBAL_ANALYSIS_IMG, GLOBAL_SUMMARY_TABLE, GLOBAL_TOTAL_RETURN, GLOBAL_STATUS, GLOBAL_ERROR, GLOBAL_TOP_10_MD
    
    # --- Part 1: Data Fetching and Indicator Calculation ---
    try:
        df_raw = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
        
        # Calculate only SMA 120 and D-Channels for the main plot
        df_ind = calculate_indicators(df_raw, SMA_WINDOWS_PLOT)
        
    except Exception as e:
        GLOBAL_ERROR = f"Fatal Data/Indicator Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING DATA SETUP ---\n{GLOBAL_ERROR}", file=sys.stderr)
        return

    # --- Part 2: Comprehensive Sharpe Ratio Scan (SMA 1 to 400) ---
    try:
        results_df = calculate_sharpe_ratios_scan(df_raw, min_sma=1, max_sma=400)
        
        # Generate the visualization
        GLOBAL_ANALYSIS_IMG = create_analysis_visualization(results_df.sort_values(by='SMA_Window', ascending=True))
        
        # Generate Markdown table for the Top 10 results (for console and reference)
        top_10_df = results_df.sort_values(by='Sharpe_Ratio', ascending=False).head(10).copy()
        top_10_df['Sharpe_Ratio'] = top_10_df['Sharpe_Ratio'].apply(lambda x: f'{x:.3f}')
        top_10_df['Final_Equity'] = top_10_df['Final_Equity'].apply(lambda x: f'{x:.2f}x')
        top_10_df = top_10_df.rename(columns={
            'SMA_Window': 'SMA Window', 
            'Sharpe_Ratio': 'Sharpe Ratio', 
            'Final_Equity': 'Final Equity (x)'
        })
        
        GLOBAL_TOP_10_MD = f"""
## Top 10 SMA Crossover Strategies (Sharpe Ratio)
{top_10_df.to_markdown(index=False)}
"""
        print(GLOBAL_TOP_10_MD)


    except Exception as e:
        GLOBAL_ERROR = f"Fatal Sharpe Scan Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING SHARPE SCAN ---\n{GLOBAL_ERROR}", file=sys.stderr)
        return

    # --- Part 3: Main Strategy Backtest (SMA 120) ---
    try:
        MAIN_SMA_WINDOW = 120
        df_final = get_strategy_returns(df_ind, MAIN_SMA_WINDOW)
        
        # 4. Determine current status
        current_position = df_final['Position'].iloc[-1]
        GLOBAL_STATUS = "LONG" if current_position == POSITION_SIZE else "SHORT"

        # 5. Create Plot
        GLOBAL_IMG_BASE64, GLOBAL_TOTAL_RETURN = create_plot(df_final, MAIN_SMA_WINDOW)
        
        # 6. Populate Global Results
        # Create summary table (simplified)
        summary_df = df_final[['Close', f'SMA_{MAIN_SMA_WINDOW}', 'Position', 'Equity']].tail(10)
        summary_df = summary_df.rename(columns={f'SMA_{MAIN_SMA_WINDOW}': 'SMA 120'})
        
        GLOBAL_SUMMARY_TABLE = summary_df.to_html(classes='table-auto w-full text-sm text-left', 
                                                  float_format=lambda x: f'{x:,.2f}')
        
        print("--- ANALYSIS COMPLETE ---")

    except Exception as e:
        GLOBAL_ERROR = f"Fatal Main Backtest Error (SMA 120): {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING MAIN BACKTEST ---\n{GLOBAL_ERROR}", file=sys.stderr)
        return


# --- Flask Web Server ---
from flask import Flask, render_template_string

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
        <title>SMA Strategy Optimization</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
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
                SMA Crossover Strategy Optimization Dashboard
                <span class="data-source-tag bg-blue-200 text-blue-800">{GLOBAL_DATA_SOURCE}</span>
            </h1>

            <div class="grid md:grid-cols-3 gap-6 mb-8">
                <div class="bg-blue-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-blue-600">Symbol</p>
                    <p class="text-2xl font-bold text-blue-800">{SYMBOL} ({TIMEFRAME})</p>
                </div>
                <div class="bg-green-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-green-600">SMA 120 Strategy Return</p>
                    <p class="text-2xl font-bold text-green-800">{GLOBAL_TOTAL_RETURN:.2f}%</p>
                </div>
                <div class="bg-red-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-red-600">Current Position (SMA 120)</p>
                    <p class="text-2xl font-bold text-red-800">{GLOBAL_STATUS}</p>
                </div>
            </div>
            
            <div class="grid lg:grid-cols-2 gap-8 mb-8">
                
                <div class="order-1">
                    <h2 class="text-2xl font-semibold text-gray-700 mb-4">Current Strategy (SMA 120)</h2>
                    <div class="bg-gray-50 p-2 rounded-lg shadow-inner overflow-hidden">
                        <img src="data:image/png;base64,{GLOBAL_IMG_BASE64}" alt="Trading Strategy Backtest Plot" class="w-full h-auto rounded-lg"/>
                    </div>
                </div>

                <div class="order-2">
                    <h2 class="text-2xl font-semibold text-gray-700 mb-4">SMA Optimization Scan (1 to 400)</h2>
                    <div class="bg-gray-50 p-2 rounded-lg shadow-inner overflow-hidden">
                        <img src="data:image/png;base64,{GLOBAL_ANALYSIS_IMG}" alt="SMA Sharpe Ratio and Equity Analysis Plot" class="w-full h-auto rounded-lg"/>
                    </div>
                </div>
            </div>
            
            <h2 class="text-2xl font-semibold text-gray-700 mt-8 mb-4">Recent SMA 120 Data</h2>
            <div class="overflow-x-auto rounded-lg shadow-md border border-gray-200">
                {GLOBAL_SUMMARY_TABLE}
            </div>
            
            <p class="mt-8 text-sm text-gray-600 border-t pt-4">
                **Strategy:** Long if Close > SMA, Short if Close $\le$ SMA. The optimization chart shows how Sharpe Ratio and final compounded equity change across all moving average periods.
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
