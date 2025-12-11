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
MAX_SMA_SCAN = 400 # Maximum SMA for the scan and the global analysis start point
SCAN_DELAY = 5.0 # Seconds delay after every 50 SMA calculations to respect API limits

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
    *** CALLED ONLY ONCE ***
    Fetches historical OHLCV data from Binance using CCXT with pagination.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True, # Automatically respect API rate limits
    })

    since_timestamp = exchange.parse8601(start_date + 'T00:00:00Z')
    all_ohlcv = []
    limit = 1000
    
    print(f"--- STARTING API DATA FETCH: {symbol} ({timeframe}) from {start_date} via Binance/CCXT ---")

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
            
    # Check minimum data needed for the longest SMA scan (MAX_SMA_SCAN)
    if len(all_ohlcv) < MAX_SMA_SCAN + 10: 
        raise ValueError(f"Not enough data fetched. Required >{MAX_SMA_SCAN + 10}, received {len(all_ohlcv)}.")
        
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
    # Ensure all required SMAs for the plot and comparison are calculated
    required_smas = set(windows_to_calculate + [40, 120]) # Ensure 40 and 120 are calculated
    
    for window in required_smas:
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
    Enforces a global start time for equity/Sharpe calculation (MAX_SMA_SCAN day).
    *** Operates only on the in-memory df_raw (no API calls) ***
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
    
    # 4. Enforce Global Start for Tradable Period (Index MAX_SMA_SCAN)
    
    # Returns/Equity must start on the first day the longest SMA decision is fully informed (Day MAX_SMA_SCAN + 1).
    tradable_start_index = MAX_SMA_SCAN 
    
    if len(df) <= tradable_start_index:
        return pd.DataFrame() 
        
    # Slice the DataFrame to the tradable period (Day 401 onwards)
    df_tradable = df.iloc[tradable_start_index:].copy()
    
    # 5. Calculate Cumulative Equity (starting fresh from 1.0)
    
    # Strategy Equity: Calculated using geometric return.
    strategy_returns_tradable = df_tradable['Strategy_Return']
    df_tradable.loc[:, 'Equity'] = (1 + strategy_returns_tradable).cumprod()
    
    # Buy & Hold Benchmark: Calculated using the price ratio from the global start price.
    start_price_bh = df_tradable['Close'].iloc[0]
    df_tradable.loc[:, 'Buy_Hold_Equity'] = df_tradable['Close'] / start_price_bh

    return df_tradable

# --- Comprehensive SMA Scan and Sharpe Ratio Calculation ---

def calculate_sharpe_ratios_scan(df_raw, min_sma, max_sma):
    """
    Iterates through SMA windows and calculates the Annualized Sharpe Ratio and Final Equity for each.
    """
    results = []
    
    print(f"\n--- Starting SMA Scan (Sharpe Ratio Calculation from SMA {min_sma} to {max_sma}) ---")
    
    for sma_w in range(min_sma, max_sma + 1):
        # Passes the sliced data (no API call)
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
            # --- SAFEGUARD: Delay to avoid rate limiting during heavy scan ---
            time.sleep(SCAN_DELAY)
            # -----------------------------------------------------------------

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
    best_sharpe_row = results_df.sort_values(by='Sharpe_Ratio', ascending=False).iloc[0]
    best_sma = best_sharpe_row['SMA_Window']
    best_sharpe = best_sharpe_row['Sharpe_Ratio']
    
    ax1.scatter(best_sma, best_sharpe, color='red', s=50, zorder=5)
    ax1.axhline(best_sharpe, color='red', linestyle='--', alpha=0.4, linewidth=1)

    ax1.set_ylabel('Annualized Sharpe Ratio', fontsize=10)
    ax1.set_title(f'SMA Crossover Strategy Performance (SMA 1 to {MAX_SMA_SCAN} - 70% Data Sample)', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(1, MAX_SMA_SCAN)
    
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
    ax2.set_xlim(1, MAX_SMA_SCAN)
    
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
    """Generates the main SMA 120 strategy plot (Close Price + Equity Curve)
       and highlights days price is between SMA 40 & 120."""
    
    df_plot = df.copy() 
    sma_120_col = f'SMA_{strategy_sma}'
    sma_40_col = 'SMA_40'

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 2]})
    
    # --- Price and SMA (Top Panel) ---
    ax1 = axes[0]
    
    # 1. Background Coloring: Strategy Position (+1 or -1)
    pos_series = df_plot['Position'] # Base position used for continuous trading
    change_indices = pos_series.index[pos_series.diff() != 0]
    segment_starts = pd.Index([pos_series.index[0]]).append(change_indices)
    segment_ends = change_indices.append(pd.Index([pos_series.index[-1]]))
    
    # Light green/red background for Long/Short positions of SMA 120
    for start, end in zip(segment_starts, segment_ends):
        current_pos = pos_series.loc[start]
        
        color = None
        alpha = 0.08
        
        if current_pos == POSITION_SIZE:
            color = 'green'
        elif current_pos == -POSITION_SIZE:
            color = 'red'
            
        if color:
            end_adjusted = end + pd.Timedelta(days=1)
            ax1.axvspan(start, end_adjusted, facecolor=color, alpha=alpha, zorder=0)

    # 2. Highlight Price Between SMAs 40 and 120
    
    # Condition: Price is strictly between SMA 40 and SMA 120
    condition = (
        (df_plot['Close'] > df_plot[sma_40_col]) & (df_plot['Close'] < df_plot[sma_120_col]) |
        (df_plot['Close'] < df_plot[sma_40_col]) & (df_plot['Close'] > df_plot[sma_120_col])
    )
    
    between_sm_days = df_plot[condition].index
    
    # Draw a distinct red marker on these specific days
    for day in between_sm_days:
        # Use a strong red background for emphasis
        ax1.axvspan(day, day + pd.Timedelta(days=1), facecolor='#DC143C', alpha=0.3, zorder=1) 
    
    # 3. Price and SMAs 
    ax1.plot(df_plot.index, df_plot['Close'], label='Price (Close)', color='#1f77b4', linewidth=1.5, alpha=0.9, zorder=2)
    ax1.plot(df_plot.index, df_plot[sma_120_col], 
             label=f'SMA {strategy_sma}', 
             color='#FF6347', linestyle='-', linewidth=2.5, zorder=2) 
    ax1.plot(df_plot.index, df_plot[sma_40_col], 
             label=f'SMA 40', 
             color='#FFA07A', linestyle='--', linewidth=1.5, zorder=2) 
    
    ax1.set_ylabel('Price (Log Scale)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_yscale('log')
    ax1.set_title(f'SMA {strategy_sma} Crossover Strategy (Red Highlight: Price Between SMA 40 & 120)', fontsize=12)

    # --- Equity Plot (Bottom Panel) ---
    ax2 = axes[1]
    
    final_strategy_return = (df_plot['Equity'].iloc[-1] - 1) * 100
    
    plotted_returns = df_plot['Strategy_Return']
    sharpe_plotted = (plotted_returns.mean() / plotted_returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
    
    ax2.plot(df_plot.index, df_plot['Equity'], label=f'Strategy Equity (Sharpe: {sharpe_plotted:.2f})', color='blue', linewidth=3)
    ax2.plot(df_plot['Buy_Hold_Equity'], label='Buy & Hold Benchmark', color='gray', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Cumulative Return', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(

