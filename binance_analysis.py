import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
HORIZON = 30 # Decay window for signal contribution
INITIAL_CAPITAL = 10000.0

# --- Winning Signals (Top 5 Diverse) ---
WINNING_SIGNALS = [
    # (Type, Param1, Param2, Param3)
    ('EMA_CROSS', 50, 150, 0),         # EMA 50/150
    ('PRICE_SMA', 380, 0, 0),          # Price/SMA 380
    ('PRICE_SMA', 140, 0, 0),          # Price/SMA 140
    ('MACD_CROSS', 12, 26, 15),        # MACD (12/26/15)
    ('RSI_CROSS', 35, 0, 0),           # RSI 35 (Crossover)
]

# --- Data Fetching (Reused from original script) ---
def fetch_binance_data():
    """Fetches daily OHLCV from Binance starting Jan 1, 2018."""
    print(f"Fetching data for {SYMBOL} since {SINCE_STR}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(SINCE_STR)
    
    all_ohlcv = []
    # Binance rate limits require pagination
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            time.sleep(0.1)
            
            # Stop fetching if we are within the last day
            if (exchange.milliseconds() - last_timestamp) < (24*60*60*1000):
                break
                
            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nTotal candles fetched: {len(all_ohlcv)}")
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    # Daily return calculation (Close[i] / Close[i-1] - 1)
    df['return'] = df['close'].pct_change()
    
    # Drop the first row which has NaN return
    df.dropna(subset=['return'], inplace=True)
    return df

# --- Signal Generation Functions (Optimized for Strategy) ---

def generate_signals(df):
    """Generates signals for all 5 winning indicators."""
    df_signals = pd.DataFrame(index=df.index, dtype=int)
    
    close_prices = df['close']
    lows = df['low']
    highs = df['high']
    
    for sig_type, p1, p2, p3 in WINNING_SIGNALS:
        signal_col_name = f"{sig_type}_{p1}_{p2}_{p3}"
        
        if sig_type == 'EMA_CROSS':
            fast_ema = close_prices.ewm(span=p1, adjust=False).mean()
            slow_ema = close_prices.ewm(span=p2, adjust=False).mean()
            
            # Long: Fast crosses above Slow
            long_condition = (fast_ema.shift(1) < slow_ema.shift(1)) & (fast_ema > slow_ema)
            # Short: Fast crosses below Slow
            short_condition = (fast_ema.shift(1) > slow_ema.shift(1)) & (fast_ema < slow_ema)
            
        elif sig_type == 'PRICE_SMA':
            sma = close_prices.rolling(window=p1).mean()
            
            # Long: Price crosses above SMA
            long_condition = (close_prices.shift(1) < sma.shift(1)) & (close_prices > sma)
            # Short: Price crosses below SMA
            short_condition = (close_prices.shift(1) > sma.shift(1)) & (close_prices < sma)
            
        elif sig_type == 'MACD_CROSS':
            ema_fast = close_prices.ewm(span=p1, adjust=False).mean()
            ema_slow = close_prices.ewm(span=p2, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=p3, adjust=False).mean()
            
            # Long: MACD crosses above Signal Line
            long_condition = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
            # Short: MACD crosses below Signal Line
            short_condition = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)

        elif sig_type == 'RSI_CROSS':
            # Calculate RSI
            period = p1
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            centerline = 50

            # Long: RSI crosses above 50
            long_condition = (rsi.shift(1) < centerline) & (rsi > centerline)
            # Short: RSI crosses below 50
            short_condition = (rsi.shift(1) > centerline) & (rsi < centerline)
            
        else:
            df_signals[signal_col_name] = 0
            continue
            
        # Apply signals: +1 for Long, -1 for Short, 0 otherwise
        df_signals[signal_col_name] = np.where(long_condition, 1, np.where(short_condition, -1, 0))
        
    # Drop NaNs that resulted from indicator calculation period
    df_signals.dropna(inplace=True)
    return df_signals


# --- Core Conviction and Strategy Implementation ---

def run_conviction_backtest(df_data, df_signals):
    """
    Applies the 30-day decaying conviction strategy.
    """
    
    # Align dataframes and get core arrays
    df_data = df_data.loc[df_signals.index]
    returns = df_data['return'].values
    num_days = len(df_signals)
    
    # Initialize strategy tracking
    portfolio_value = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    
    # Tracking for each signal: stores the day index when the CURRENT active signal started.
    # Initialized to -1 (no active signal)
    signal_start_day = np.full(len(WINNING_SIGNALS), -1) 
    # Tracking the direction of the current active signal: 1 (Long), -1 (Short)
    signal_direction = np.full(len(WINNING_SIGNALS), 0) 
    
    # Backtest Loop
    for t in range(num_days):
        # 1. Check for New Signals & Update Start Day/Direction
        daily_conviction = 0.0
        
        for i, (sig_type, p1, p2, p3) in enumerate(WINNING_SIGNALS):
            current_signal = df_signals.iloc[t, i]
            
            # Check for a new signal (+1 or -1)
            if current_signal != 0:
                # If a new signal is generated, it starts decay.
                # It also effectively "flips" the previous decay by resetting the start day.
                signal_start_day[i] = t
                signal_direction[i] = current_signal
            
            # 2. Calculate Decay Contribution (1-d/30)
            
            # Only apply contribution if there is an active signal
            if signal_direction[i] != 0:
                # Days since the active signal started
                days_since_signal = t - signal_start_day[i] 
                
                # Apply linear decay: 1 - (d/HORIZON)
                decay_factor = max(0.0, 1.0 - (days_since_signal / HORIZON))
                
                # Contribution is Direction * Decay Factor (max 1, min 0)
                contribution = signal_direction[i] * decay_factor
                daily_conviction += contribution
        
        # 3. Calculate Daily P&L
        
        # Position Size = Conviction (Leverage)
        position_size = daily_conviction
        
        # PnL = Position Size * Daily Return
        # Note: Position size is capped by the number of signals (5) but can float with decay
        pnl = position_size * returns[t]
        daily_pnl[t] = pnl
        
        # 4. Update Portfolio Value
        if t == 0:
            portfolio_value[t] = INITIAL_CAPITAL * (1 + pnl)
        else:
            # PnL is based on a fraction of the capital (Leverage / Total Signals)
            # The strategy assumes you risk up to 5 units of capital when fully leveraged.
            # Daily P&L = (Daily Conviction / Max Conviction) * Capital * Daily Return
            # We will simplify by tracking portfolio return
            portfolio_value[t] = portfolio_value[t-1] * (1 + pnl)


    # Final results packaging
    total_return = (portfolio_value[-1] / portfolio_value[0]) - 1
    
    # Combine results for output
    results_df = df_data[['close', 'return']].copy()
    results_df['Conviction'] = daily_pnl / results_df['return'].where(results_df['return'] != 0, 0) # Calculate conviction used
    results_df['Daily_PnL_Unit'] = daily_pnl # P&L assuming 1 unit of capital
    
    return total_return, results_df
    
# --- Execution ---

if __name__ == '__main__':
    
    # Load and process data
    df_data = fetch_binance_data()
    df_signals = generate_signals(df_data)
    
    if df_signals.empty:
        print("Analysis failed: No valid signals generated.")
    else:
        # Run Backtest
        total_return, results_df = run_conviction_backtest(df_data, df_signals)
        
        # Display Results
        print("\n" + "="*50)
        print("     Conviction Trading Strategy Backtest Results     ")
        print(f"     Asset: {SYMBOL} ({TIMEFRAME} data) since {SINCE_STR}")
        print("="*50)
        
        # Get start/end dates
        start_date = results_df.index.min().strftime('%Y-%m-%d')
        end_date = results_df.index.max().strftime('%Y-%m-%d')
        
        # Buy and Hold Return
        bh_return = (results_df['close'].iloc[-1] / results_df['close'].iloc[0]) - 1
        
        print(f"Time Period: {start_date} to {end_date}")
        print(f"Total Trading Days: {len(results_df)}")
        print(f"Initial Capital (Simulated): ${INITIAL_CAPITAL:,.2f}")
        
        print("\n--- Performance Metrics ---")
        print(f"Conviction Strategy Total Return: {total_return * 100:.2f}%")
        print(f"Buy & Hold (BTC/USDT) Return:   {bh_return * 100:.2f}%")
        
        if total_return > bh_return:
            print("\nStrategy OUTPERFORMED Buy & Hold. Great start!")
        else:
            print("\nStrategy UNDERPERFORMED Buy & Hold.")
            
        print(f"Final Portfolio Value: ${INITIAL_CAPITAL * (1 + total_return):,.2f}")
        
        # Save detailed results to CSV for analysis
        results_df.to_csv('conviction_backtest_results.csv')
        print("\nDetailed daily results saved to 'conviction_backtest_results.csv'")

