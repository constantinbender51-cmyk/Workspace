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
DC_WINDOW = 20  # Donchian Channel Period (N)
STOP_LOSS_PCT = 0.05 # Fixed 5% stop loss from previous day's close (per user request)
# K_FACTOR will be dynamically set by the optimal value found in the grid search or manually provided
POSITION_SIZE_MAX = 1.0  # Maximum position size limit
PORT = 8080
ANNUAL_TRADING_DAYS = 252 # Used for annualizing Sharpe Ratio
MAX_SMA_SCAN = 120 # Maximum SMA for the scan and the global analysis start point
SCAN_DELAY = 5.0 # Seconds delay after every 50 SMA calculations to respect API limits

# --- Grid Search Range ---
# Range up to 5.00
K_FACTOR_RANGE = np.arange(0.01, 5.01, 0.01)

# --- Global Variables to store results (populated once at startup) ---
GLOBAL_DATA_SOURCE = "Binance (CCXT)"
GLOBAL_IMG_BASE64 = ""      # Image for Strategy Plot (using Optimal K)
GLOBAL_ANALYSIS_IMG = ""    # Image for Sharpe/Equity Scan (for SMA window 1-120)
GLOBAL_K_ANALYSIS_IMG = ""  # Image for K-Factor Scan
GLOBAL_SUMMARY_TABLE = ""
GLOBAL_TOTAL_RETURN = 0.0
GLOBAL_SHARPE = 0.0         # Annualized Sharpe Ratio for the Optimal K strategy
GLOBAL_STATUS = "PENDING"
GLOBAL_ERROR = None
GLOBAL_OPTIMAL_K = 0.0      # Optimal K found in the search
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
            
    # Check minimum data needed for the longest indicator (SMA 120)
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
    
    all_smas = set(windows_to_calculate + [120])
    
    for window in all_smas:
        if f'SMA_{window}' not in df.columns:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    
    # Calculate Donchian Channel
    df['DC_Upper'] = df['High'].rolling(window=DC_WINDOW).max()
    df['DC_Lower'] = df['Low'].rolling(window=DC_WINDOW).min()
    
    return df

# --- Backtesting and Strategy Return Calculation (Reusable for Scan) ---

def get_strategy_returns_scan(df_raw, sma_window):
    """
    Calculates returns for a simple Long/Short strategy based on a single given SMA window.
    Used ONLY for the initial Sharpe Scan (1 to 120).
    """
    df = df_raw.copy()
    sma_col = f'SMA_{sma_window}'
    
    # 1. Calculate the specific SMA needed for this run
    if sma_col not in df.columns:
        df[sma_col] = df['Close'].rolling(window=sma_window).mean()
    
    # 2. Strategy Logic: Long (+1) if Close > SMA, Short (-1) if Close <= SMA
    df.loc[:, 'Next_Day_Position'] = np.where(df['Close'] > df[sma_col], 
                                              POSITION_SIZE_MAX,      # +1 for Long
                                              -POSITION_SIZE_MAX)     # -1 for Short
    
    # Position HELD for the day t return is based on the decision from day t-1.
    df.loc[:, 'Position'] = df['Next_Day_Position'].shift(1).fillna(0)
    
    # 3. Calculate Strategy Return
    df['Daily_Return'] = df['Close'].pct_change()
    daily_returns_clean = df['Daily_Return'].fillna(0)
    
    # Strategy Return = Daily Asset Return * Held Position
    df.loc[:, 'Strategy_Return'] = daily_returns_clean * df['Position']
    
    # 4. Enforce Global Start for Tradable Period (Index MAX_SMA_SCAN)
    tradable_start_index = MAX_SMA_SCAN 
    
    if len(df) <= tradable_start_index:
        return pd.DataFrame() 
        
    df_tradable = df.iloc[tradable_start_index:].copy()
    
    # 5. Calculate Cumulative Equity (starting fresh from 1.0)
    strategy_returns_tradable = df_tradable['Strategy_Return']
    df_tradable.loc[:, 'Equity'] = (1 + strategy_returns_tradable).cumprod()
    
    # Buy & Hold Benchmark: Calculated using the price ratio from the global start price.
    start_price_bh = df_tradable['Close'].iloc[0]
    df_tradable.loc[:, 'Buy_Hold_Equity'] = df_tradable['Close'] / start_price_bh

    return df_tradable

# --- Dynamic Position Size Strategy (Core Logic adapted for K_FACTOR) ---

def calculate_dynamic_position(df_ind_raw, k_factor):
    """
    Calculates position size based on volatility (DC width relative to price),
    and sets the direction based on SMA 120, incorporating FIXED 5% Stop Loss logic.
    
    Size = 1 - (DC_Width / Close)^k_factor
    """
    df = df_ind_raw.copy()
    
    # 1. Calculate Relative Volatility for Position Sizing
    df['DC_Width'] = df['DC_Upper'] - df['DC_Lower']
    df['Relative_Width'] = df['DC_Width'] / df['Close']
    
    # 2. Calculate Base Position Size (1 - (Relative_Width ^ k_factor)), clipped at lower bound 0
    valid_conditions = (df['Close'] > 0) & (df['Relative_Width'] > 0)

    df['Size_Decider'] = np.where(
        valid_conditions,
        (1 - np.power(df['Relative_Width'], k_factor)).clip(lower=0),
        0.0 
    )
    
    # 3. Determine Direction Signal (+1 or -1) from SMA 120
    sma_col = 'SMA_120'
    df['Direction_Decider'] = np.where(df['Close'] > df[sma_col], 1, -1)
    
    # 4. Final Held Position = Direction Decided Yesterday * Size Decided Yesterday
    df['Held_Position'] = (df['Direction_Decider'] * df['Size_Decider']).shift(1).fillna(0)
    
    # Calculate Previous Close for SL calculation (used for current day's trade)
    df['Previous_Close'] = df['Close'].shift(1)
    
    # Calculate daily asset return
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    
    # --- Stop Loss Logic (SL FIXED at 5% from Previous Close) ---
    
    LOSS_RATE_FIXED = STOP_LOSS_PCT # 0.05
    
    # A. Long SL Hit Check: Low < SL Price (Prev Close * (1 - SL%))
    SL_Price_Long = df['Previous_Close'] * (1 - LOSS_RATE_FIXED)
    is_long_sl_hit = (df['Held_Position'] > 0) & (df['Low'] < SL_Price_Long)

    # B. Short SL Hit Check: High > SL Price (Prev Close * (1 + SL%))
    SL_Price_Short = df['Previous_Close'] * (1 + LOSS_RATE_FIXED)
    is_short_sl_hit = (df['Held_Position'] < 0) & (df['High'] > SL_Price_Short)

    df['SL_Hit'] = is_long_sl_hit | is_short_sl_hit

    # C. Calculate Strategy Return:
    # Default return: Daily asset return * Held Position
    df['Strategy_Return'] = df['Daily_Return'] * df['Held_Position']

    # If SL is hit, the return is capped at the maximum allowed loss (FIXED 5% rate)
    # Loss = -FIXED_RATE * |Held Position|
    
    # Apply SL for long positions
    loss_long_capped = -LOSS_RATE_FIXED * df['Held_Position']
    df.loc[is_long_sl_hit, 'Strategy_Return'] = loss_long_capped
    
    # Apply SL for short positions
    loss_short_capped = -LOSS_RATE_FIXED * np.abs(df['Held_Position'])
    df.loc[is_short_sl_hit, 'Strategy_Return'] = loss_short_capped
    
    # --- End Stop Loss Logic ---
    
    # 5. Enforce Global Start for Tradable Period (Index MAX_SMA_SCAN)
    tradable_start_index = MAX_SMA_SCAN 
    
    if len(df) <= tradable_start_index:
        return pd.DataFrame() 
        
    df_tradable = df.iloc[tradable_start_index:].copy()

    # 6. Calculate Cumulative Equity
    df_tradable.loc[:, 'Equity'] = (1 + df_tradable['Strategy_Return']).cumprod()
    
    # Buy & Hold Benchmark
    start_price_bh = df_tradable['Close'].iloc[0]
    df_tradable.loc[:, 'Buy_Hold_Equity'] = df_tradable['Close'] / start_price_bh
    
    return df_tradable[['Close', sma_col, 'DC_Upper', 'DC_Lower', 'Held_Position', 'SL_Hit', 'Strategy_Return', 'Equity', 'Buy_Hold_Equity']]

# --- K-Factor Grid Search Function ---

def run_k_grid_search(df_ind_raw):
    """Runs a grid search for the optimal K_FACTOR based on Sharpe Ratio."""
    print(f"\n--- Starting K-Factor Grid Search (0.01 to {K_FACTOR_RANGE[-1]:.2f}) ---")
    results = []
    
    for k in K_FACTOR_RANGE:
        k = round(k, 2) # Ensure clean decimal representation
        df_strategy = calculate_dynamic_position(df_ind_raw, k)
        
        if df_strategy.empty:
            continue
            
        returns = df_strategy['Strategy_Return']
        std_dev_daily_return = returns.std()
        
        if std_dev_daily_return > 0:
            avg_daily_return = returns.mean()
            sharpe_ratio = (avg_daily_return / std_dev_daily_return) * np.sqrt(ANNUAL_TRADING_DAYS)
        else:
            sharpe_ratio = 0.0 
            
        final_equity = df_strategy['Equity'].iloc[-1]
            
        results.append({
            'K_Factor': k,
            'Sharpe_Ratio': sharpe_ratio,
            'Final_Equity': final_equity
        })
        
        if (k * 100) % 50 == 0: # Print every 0.5 step
            print(f"Processed K={k:.2f}...")

    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        # If search fails, we still rely on the user's defined K=0.35
        return 0.35, results_df, "" 

    # Note: We still calculate the actual best K from the run but will force 0.35 for the main strategy run.
    best_k_row_actual = results_df.sort_values(by='Sharpe_Ratio', ascending=False).iloc[0]
    
    # Generate Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(results_df['K_Factor'], results_df['Sharpe_Ratio'], 
            color='#5a3d90', linewidth=2)
    
    # Highlight the calculated best K on the plot
    ax.scatter(best_k_row_actual['K_Factor'], best_k_row_actual['Sharpe_Ratio'], color='red', s=50, zorder=5)
    ax.text(best_k_row_actual['K_Factor'], best_k_row_actual['Sharpe_Ratio'] * 1.05, 
            f'Calculated Optimal K: {best_k_row_actual["K_Factor"]:.3f}', fontsize=9, color='red', ha='center')

    ax.set_xlabel('K-Factor (Exponent)', fontsize=10)
    ax.set_ylabel('Annualized Sharpe Ratio', fontsize=10)
    ax.set_title(f'Sharpe Ratio vs. K-Factor Optimization (SMA 120 Signal)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return best_k_row_actual['K_Factor'], results_df, img_base64

# --- Comprehensive SMA Scan and Sharpe Ratio Calculation ---

def calculate_sharpe_ratios_scan(df_raw, min_sma, max_sma):
    """
    Iterates through SMA windows and calculates the Annualized Sharpe Ratio and Final Equity for each.
    """
    results = []
    
    print(f"\n--- Starting SMA Scan (SMA {min_sma} to {max_sma}) ---")
    
    for sma_w in range(min_sma, max_sma + 1):
        df_strategy = get_strategy_returns_scan(df_raw, sma_w)
        
        if df_strategy.empty:
            continue
            
        returns = df_strategy['Strategy_Return']
        std_dev_daily_return = returns.std()
        
        if std_dev_daily_return > 0:
            avg_daily_return = returns.mean()
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
            time.sleep(SCAN_DELAY)

    results_df = pd.DataFrame(results)
    print(f"--- SMA Scan Complete. Found {len(results_df)} valid windows. ---")
    return results_df

# --- Analysis Visualization (Shared) ---

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

# --- Plotting Dynamic Position Strategy ---

def create_plot(df, sma_window, k_factor):
    """Generates the main strategy plot (Close Price + Equity Curve) for the dynamic position size strategy."""
    
    df_plot = df.copy() 
    sma_col = f'SMA_{sma_window}'
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 2]})
    
    # --- Price and SMA (Top Panel) ---
    ax1 = axes[0]
    
    # 1. Background Coloring: Strategy Position (+/- size)
    pos_series = df_plot['Held_Position'] 
    change_indices = pos_series.index[pos_series.diff() != 0]
    segment_starts = pd.Index([pos_series.index[0]]).append(change_indices)
    segment_ends = change_indices.append(pd.Index([pos_series.index[-1]]))
    
    # Color intensity corresponds to position size magnitude
    for start, end in zip(segment_starts, segment_ends):
        current_pos = pos_series.loc[start]
        
        color = None
        alpha = np.abs(current_pos) * 0.15 # Max alpha 0.15
        
        if current_pos > 0:
            color = 'green'
        elif current_pos < 0:
            color = 'red'
        
        if color and alpha > 0.01: # Don't plot near-zero positions
            end_adjusted = end + pd.Timedelta(days=1)
            ax1.axvspan(start, end_adjusted, facecolor=color, alpha=alpha, zorder=0)
            
    # 2. Highlight Stop Loss days with a black marker
    sl_days = df_plot[df_plot['SL_Hit']].index
    for day in sl_days:
        ax1.axvspan(day, day + pd.Timedelta(days=1), facecolor='black', alpha=0.4, zorder=1)

    # 3. Price and SMAs/DC
    ax1.plot(df_plot.index, df_plot['Close'], label='Price (Close)', color='#1f77b4', linewidth=1.5, alpha=0.9, zorder=2)
    
    # Plotting the main SMA 120 and DC for context
    ax1.plot(df_plot.index, df_plot[sma_col], 
             label=f'SMA {sma_window}', 
             color='#FF6347', linestyle='-', linewidth=2.5, zorder=2) 
    ax1.plot(df_plot.index, df_plot['DC_Upper'], label='DC Upper (20)', color='#9400D3', linestyle=':', linewidth=1.0, alpha=0.5, zorder=2)
    ax1.plot(df_plot.index, df_plot['DC_Lower'], label='DC Lower (20)', color='#9400D3', linestyle=':', linewidth=1.0, alpha=0.5, zorder=2)
    
    ax1.set_ylabel('Price (Log Scale)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_yscale('log')
    ax1.set_title(f'SMA {sma_window} Signal w/ Dynamic Sizing (K={k_factor:.3f}) - Black Overlay: SL Hit @ 5%', fontsize=12)

    # --- Equity Plot (Bottom Panel) ---
    ax2 = axes[1]
    
    final_strategy_return = (df_plot['Equity'].iloc[-1] - 1) * 100
    
    plotted_returns = df_plot['Strategy_Return']
    sharpe_plotted = (plotted_returns.mean() / plotted_returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
    
    ax2.plot(df_plot.index, df_plot['Equity'], label=f'Strategy Equity (Sharpe: {sharpe_plotted:.3f})', color='blue', linewidth=3)
    ax2.plot(df_plot['Buy_Hold_Equity'], label='Buy & Hold Benchmark', color='gray', linestyle='--', alpha=0.7)
    
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
    """Runs the full backtest analysis, SMA scan, K-factor search, and populates global variables."""
    global GLOBAL_IMG_BASE64, GLOBAL_ANALYSIS_IMG, GLOBAL_K_ANALYSIS_IMG, GLOBAL_SUMMARY_TABLE, GLOBAL_TOTAL_RETURN, GLOBAL_STATUS, GLOBAL_ERROR, GLOBAL_TOP_10_MD, GLOBAL_SHARPE, GLOBAL_OPTIMAL_K
    
    MAIN_SMA_WINDOW = 120 
    # Use the K value provided by the user for the final strategy run
    FORCED_OPTIMAL_K = 0.35 

    # --- Part 1: Data Fetching and Indicator Calculation ---
    try:
        df_raw = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
        split_idx = int(len(df_raw) * 0.70)
        df_raw_sliced = df_raw.iloc[:split_idx]
        print(f"--- DATA SLICED: Running analysis on first {len(df_raw_sliced)} candles (70% of total) ---")
        
        if len(df_raw_sliced) < MAX_SMA_SCAN + 10:
             raise ValueError(f"70% data slice is too short ({len(df_raw_sliced)} days). Max SMA {MAX_SMA_SCAN} requires more data.")
        
        df_ind = calculate_indicators(df_raw_sliced, [])
        
    except Exception as e:
        GLOBAL_ERROR = f"Fatal Data/Indicator Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING DATA SETUP ---\n{GLOBAL_ERROR}", file=sys.stderr)
        return

    # --- Part 2: Comprehensive SMA Scan (SMA 1 to MAX_SMA_SCAN) ---
    try:
        results_df_sma = calculate_sharpe_ratios_scan(df_raw_sliced, min_sma=1, max_sma=MAX_SMA_SCAN)
        GLOBAL_ANALYSIS_IMG = create_analysis_visualization(results_df_sma.sort_values(by='SMA_Window', ascending=True))
        
        top_10_df = results_df_sma.sort_values(by='Sharpe_Ratio', ascending=False).head(10).copy()
        top_10_df['Sharpe_Ratio'] = top_10_df['Sharpe_Ratio'].apply(lambda x: f'{x:.3f}')
        top_10_df['Final_Equity'] = top_10_df['Final_Equity'].apply(lambda x: f'{x:.2f}x')
        top_10_df = top_10_df.rename(columns={'SMA_Window': 'SMA Window', 'Sharpe_Ratio': 'Sharpe Ratio', 'Final_Equity': 'Final Equity (x)'})
        GLOBAL_TOP_10_MD = f"""
## Top 10 SMA Crossover Strategies (Sharpe Ratio)
{top_10_df.to_markdown(index=False)}
"""
        print(GLOBAL_TOP_10_MD)


    except Exception as e:
        GLOBAL_ERROR = f"Fatal Sharpe Scan Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING SHARPE SCAN ---\n{GLOBAL_ERROR}", file=sys.stderr)
        return
        
    # --- Part 3: K-Factor Grid Search ---
    try:
        # Run search to generate plot, but the final K will be forced to 0.35
        _, results_df_k, GLOBAL_K_ANALYSIS_IMG = run_k_grid_search(df_ind)
        GLOBAL_OPTIMAL_K = FORCED_OPTIMAL_K
        
    except Exception as e:
        GLOBAL_ERROR = f"Fatal K-Factor Grid Search Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING K-FACTOR SEARCH ---\n{GLOBAL_ERROR}", file=sys.stderr)
        GLOBAL_OPTIMAL_K = FORCED_OPTIMAL_K
        return

    # --- Part 4: Run Main Strategy with Optimal K ---
    try:
        print(f"--- Running main strategy with forced Optimal K: {GLOBAL_OPTIMAL_K} ---")
        
        df_final = calculate_dynamic_position(df_ind, GLOBAL_OPTIMAL_K)
        
        returns = df_final['Strategy_Return']
        if returns.std() > 0:
            GLOBAL_SHARPE = (returns.mean() / returns.std()) * np.sqrt(ANNUAL_TRADING_DAYS)
        else:
            GLOBAL_SHARPE = 0.0

        current_position = df_final['Held_Position'].iloc[-1]
        
        if current_position > 0:
            GLOBAL_STATUS = f"LONG ({current_position:.2f})"
        elif current_position < 0:
            GLOBAL_STATUS = f"SHORT ({current_position:.2f})"
        else:
             GLOBAL_STATUS = "FLAT (0.00)"
             
        GLOBAL_IMG_BASE64, GLOBAL_TOTAL_RETURN = create_plot(df_final, MAIN_SMA_WINDOW, GLOBAL_OPTIMAL_K)
        
        # Add SL_Hit column to the summary table
        summary_df = df_final[['Close', f'SMA_{MAIN_SMA_WINDOW}', 'DC_Upper', 'DC_Lower', 'Held_Position', 'SL_Hit', 'Equity', 'Strategy_Return']].tail(10)
        summary_df['SL_Hit'] = summary_df['SL_Hit'].astype(int) # Convert boolean to 1/0 for cleaner display
        summary_df = summary_df.rename(columns={'Held_Position': 'Pos Held', 'Strategy_Return': 'Ret Final', f'SMA_{MAIN_SMA_WINDOW}': 'SMA 120', 'SL_Hit': 'SL Hit'})
        
        GLOBAL_SUMMARY_TABLE = summary_df.to_html(classes='table-auto w-full text-sm text-left', 
                                                  float_format=lambda x: f'{x:,.4f}') 
        
        print(f"--- Strategy Sharpe Ratio (K={GLOBAL_OPTIMAL_K}): {GLOBAL_SHARPE:.3f} ---")
        print("--- ANALYSIS COMPLETE ---")

    except Exception as e:
        GLOBAL_ERROR = f"Fatal Main Strategy Run Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"--- FATAL ERROR DURING MAIN STRATEGY RUN ---\n{GLOBAL_ERROR}", file=sys.stderr)
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
        
    # Define the mathematical expression for the HTML display separately.
    position_sizing_formula = r"$1 - \left( \frac{\text{DC Upper} - \text{DC Lower}}{\text{Close}} \right)^{k}$"
        
    # HTML Template with Tailwind CSS for modern look and responsiveness
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SMA 120 Dynamic Position</title>
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
                Dynamic Volatility Sizing Dashboard
                <span class="data-source-tag bg-blue-200 text-blue-800">{GLOBAL_DATA_SOURCE}</span>
            </h1>

            <div class="grid md:grid-cols-4 gap-6 mb-8">
                <div class="bg-blue-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-blue-600">Symbol</p>
                    <p class="text-2xl font-bold text-blue-800">{SYMBOL} ({TIMEFRAME})</p>
                </div>
                <div class="bg-yellow-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-yellow-600">Optimal K-Factor (User Defined)</p>
                    <p class="text-2xl font-bold text-yellow-800">{GLOBAL_OPTIMAL_K:.3f}</p>
                </div>
                <div class="bg-green-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-green-600">Strategy Sharpe Ratio (SL Included)</p>
                    <p class="text-2xl font-bold text-green-800">{GLOBAL_SHARPE:.3f}</p>
                </div>
                <div class="bg-red-50 p-4 rounded-lg shadow-md">
                    <p class="text-lg font-medium text-red-600">Current Position</p>
                    <p class="text-2xl font-bold text-red-800">{GLOBAL_STATUS}</p>
                </div>
            </div>
            
            <div class="grid lg:grid-cols-2 gap-8 mb-8">
                
                <div class="order-1">
                    <h2 class="text-2xl font-semibold text-gray-700 mb-4">Dynamic Position Sizing (K={GLOBAL_OPTIMAL_K:.3f})</h2>
                    <div class="bg-gray-50 p-2 rounded-lg shadow-inner overflow-hidden">
                        <img src="data:image/png;base64,{GLOBAL_IMG_BASE64}" alt="Trading Strategy Backtest Plot" class="w-full h-auto rounded-lg"/>
                    </div>
                </div>

                <div class="order-2">
                    <h2 class="text-2xl font-semibold text-gray-700 mb-4">K-Factor Optimization Plot (Sharpe vs K)</h2>
                    <div class="bg-gray-50 p-2 rounded-lg shadow-inner overflow-hidden">
                        <img src="data:image/png;base64,{GLOBAL_K_ANALYSIS_IMG}" alt="K-Factor Optimization Plot" class="w-full h-auto rounded-lg"/>
                    </div>
                </div>
            </div>
            
            <h2 class="text-2xl font-semibold text-gray-700 mt-8 mb-4">Recent Dynamic Strategy Data (SL Hit = 1 means Stop Loss Hit)</h2>
            <div class="overflow-x-auto rounded-lg shadow-md border border-gray-200">
                {GLOBAL_SUMMARY_TABLE}
            </div>
            
            <p class="mt-8 text-sm text-gray-600 border-t pt-4">
                **Strategy Logic:** Direction is determined by SMA 120 crossover. Position size is calculated dynamically (K={GLOBAL_OPTIMAL_K:.3f}): {position_sizing_formula}. **Stop Loss (SL) is implemented at a fixed {STOP_LOSS_PCT*100:.0f}% deviation from the previous close against the position.**
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
