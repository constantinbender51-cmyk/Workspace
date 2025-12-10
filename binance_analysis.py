import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. DATA FETCHING
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', since_year=2018):
    """
    Fetches OHLCV data from Binance.
    """
    print(f"Fetching {symbol} data from Binance starting {since_year}...")
    exchange = ccxt.binance()
    
    # Calculate timestamp for Jan 1, 2018
    since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
    
    all_ohlcv = []
    limit = 1000  # Binance limit per request
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1 # Move to next timestamp
            
            # Rate limit safety
            # time.sleep(0.1) 
            
            # Break if we reached near current time (simple check)
            if len(ohlcv) < limit:
                break
                
            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"\nTotal data points: {len(df)}")
    return df

# -----------------------------------------------------------------------------
# 2. STRATEGY CALCULATION CORE
# -----------------------------------------------------------------------------
def calculate_metrics(df):
    """
    Computes Returns, RoR, SMAs, and Future Decay Sums.
    """
    print("Calculating base metrics...")
    
    # 1. Intraday Returns: (Close - Open) / Open
    df['returns'] = (df['close'] - df['open']) / df['open']
    
    # 2. Cumulative Returns (for SMAs and Distance calculations)
    df['sum_returns'] = df['returns'].cumsum()
    
    # 3. Returns of Returns (Rate of Change of Returns)
    df['ror'] = df['returns'].diff()
    
    # 4. Future Decay Sum (FDS)
    # The significance of day x depends on RoR for x+1...x+10
    # Formula: Sum( RoR[t+k] / k ) for k=1 to 10
    # We use a rolling window looking forward (implemented via shifting backwards)
    df['fds'] = 0.0
    for k in range(1, 11):
        # shift(-k) brings future value to current row
        df['fds'] += df['ror'].shift(-k) / k
        
    # 5. Calculate SMAs 10 to 400 (Step 10) on Cumulative Returns
    sma_cols = []
    for window in range(10, 410, 10):
        col_name = f'sma_{window}'
        df[col_name] = df['sum_returns'].rolling(window=window).mean()
        sma_cols.append(col_name)
        
    return df, sma_cols

def get_proximity_resistance(value, reference):
    """
    Proximity for Resistance Calculation.
    Formula: 1 / ( (distance / reference) * (1/0.05) )
    Simplified: 1 / ( abs((val-ref)/ref) * 20 )
    """
    if reference == 0: return 0 # Avoid div by zero
    
    dist_pct = abs((value - reference) / reference)
    denominator = dist_pct * 20.0
    
    # Avoid division by zero if distance is extremely small (perfect match)
    if denominator < 1e-6:
        return 100.0 # Cap max proximity
        
    return 1.0 / denominator

def backtest_strategy(df, sma_cols, threshold=5.0, lookback=400):
    """
    Runs the rolling window logic for Resistance and Trading.
    """
    print("Running simulation (this may take a moment)...")
    
    # Arrays to store results
    resistances = []
    signals = []
    equity_curve = [10000.0] # Start with $10k
    position = 0 # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    trailing_high = 0.0
    
    # Convert to numpy for speed where possible, but loop is logic-heavy
    timestamps = df.index
    sum_r = df['sum_returns'].values
    returns = df['returns'].values
    fds = df['fds'].values # Future decay sum
    
    # Pre-extract SMA data into a matrix (Rows: Time, Cols: SMAs)
    # shape: (N, 40)
    sma_matrix = df[sma_cols].values
    
    # We need at least lookback + 10 days (for FDS) to start
    start_idx = lookback + 10
    
    # Align results with DataFrame length
    resistances = [np.nan] * len(df)
    entry_flags = [0] * len(df) # For visualization
    
    # Loop through data
    for n in range(start_idx, len(df) - 10):
        
        # ---------------------------------------
        # A. CALCULATE RESISTANCE
        # ---------------------------------------
        
        current_sum_r = sum_r[n]
        
        # 1. Price Memory Component
        # Sum over lookback days (n-lookback to n-1)
        # Proximity(Current, Past) * FDS_of_Past
        
        # Slicing history for vectorization
        hist_start = n - lookback
        hist_end = n
        
        hist_sum_r = sum_r[hist_start:hist_end]
        hist_fds = fds[hist_start:hist_end]
        
        # Vectorized Proximity Calculation for Price Memory
        # Handle division by zero/small ref safely
        with np.errstate(divide='ignore', invalid='ignore'):
            # dist_pct = abs((current - history) / history)
            # using epsilon to avoid zero division
            safe_hist_sum_r = np.where(hist_sum_r == 0, 1e-9, hist_sum_r) 
            dist_pcts = np.abs((current_sum_r - hist_sum_r) / safe_hist_sum_r)
            denoms = dist_pcts * 20.0
            # If denom is 0 (match), max proximity 100
            proximities = np.where(denoms < 1e-6, 100.0, 1.0 / denoms)
            
        term1_price_memory = np.sum(proximities * hist_fds)
        
        # 2. SMA Influence Component
        # We need the "Significance" of each SMA.
        # Significance of SMA w = Average of (Proximity(Past_SumR, Past_SMA_w) * Past_FDS) over lookback
        
        term2_sma_influence = 0
        
        # We iterate through the 40 SMAs
        for i, sma_val in enumerate(sma_matrix[n]):
            if np.isnan(sma_val): continue
            
            # Current Proximity to this SMA
            if sma_val == 0: sma_val = 1e-9
            curr_dist_pct = abs((current_sum_r - sma_val) / sma_val)
            curr_denom = curr_dist_pct * 20.0
            curr_prox = 100.0 if curr_denom < 1e-6 else 1.0 / curr_denom
            
            # Historical Significance of this SMA (Rolling calculation)
            # Slice historical data for this specific SMA column
            hist_sma_vals = sma_matrix[hist_start:hist_end, i]
            
            # Vectorized calculation of historical proximity for this SMA
            safe_hist_sma = np.where(hist_sma_vals == 0, 1e-9, hist_sma_vals)
            hist_dists = np.abs((hist_sum_r - safe_hist_sma) / safe_hist_sma)
            hist_denoms = hist_dists * 20.0
            hist_proxs = np.where(hist_denoms < 1e-6, 100.0, 1.0 / hist_denoms)
            
            # Raw Significance history = Prox * FDS
            raw_sigs = hist_proxs * hist_fds
            
            # SMA Significance = Sum (or Mean) of Raw Sigs. 
            # Prompt says "sum over all SMAs... * significance". 
            # Assuming Significance is the accumulated weight.
            sma_sig = np.sum(raw_sigs) 
            
            term2_sma_influence += curr_prox * sma_sig
            
        total_resistance = term1_price_memory + term2_sma_influence
        resistances[n] = total_resistance
        
        # ---------------------------------------
        # B. TRADING LOGIC
        # ---------------------------------------
        
        # Update Equity (Market to Market)
        current_price = df['close'].iloc[n]
        prev_price = df['close'].iloc[n-1]
        
        if position != 0:
            # Simple PnL: Position Size * %Change
            # Note: The logic below is simplified. Real PnL depends on position size scaling.
            # Assuming full equity is scaled by the "Signal Strength" factor?
            # Or just tracking % change of underlying asset?
            # Let's assume standard 1x leverage for PnL calculation to see direction correctness.
            
            pct_change = (current_price - prev_price) / prev_price
            equity_curve.append(equity_curve[-1] * (1 + position * pct_change))
            
            # Trailing Stop Logic
            # "High since entry"
            if position == 1:
                trailing_high = max(trailing_high, current_price)
                drawdown = (trailing_high - current_price) / trailing_high
                if drawdown >= 0.02: # 2% loss from high
                    position = 0
                    # print(f"Stop Loss triggered at {timestamps[n]}")
                    
            elif position == -1:
                # For short, "High" equity means "Low" price
                trailing_high = min(trailing_high, current_price)
                # Drawdown for short: (Price - Low) / Low
                drawdown = (current_price - trailing_high) / trailing_high
                if drawdown >= 0.02:
                    position = 0
                    # print(f"Stop Loss triggered at {timestamps[n]}")
        else:
            equity_curve.append(equity_curve[-1])

        # Entry Logic
        if position == 0 and total_resistance > threshold:
            # Set Entry Flag
            entry_flags[n] = 1
            entry_ref = current_sum_r
            entry_price_val = current_price
            
            # Position Sizing / Direction Logic
            # "positive and small if return is positive and price is close to entry"
            # "calculated as distance * 1/0.05" -> distance * 20
            
            curr_return = returns[n]
            
            # Distance from entry (Sum Returns)
            # At the exact moment of entry, distance is 0.
            # The logic implies we adjust position size dynamically? 
            # OR we determine initial position direction here?
            # Assuming we take a position immediately upon breach.
            
            dist = abs(current_sum_r - entry_ref)
            # proximity_signal = dist * 20.0
            # signal_strength = min(proximity_signal, 5.0)
            
            # Since distance is 0 at entry, signal is 0? 
            # This suggests the signal is dynamic. The strategy might be:
            # "Once threshold breached, stay in market and adjust size/direction daily"
            # But simpler interpretation: determine direction now.
            
            # If return is positive -> Long. Negative -> Short.
            if curr_return > 0:
                position = 1
                trailing_high = current_price
            else:
                position = -1
                trailing_high = current_price

    # Clean up results
    df['resistance'] = resistances
    df['entry_signal'] = entry_flags
    
    # Pad equity curve
    equity_curve = equity_curve[:len(df)]
    while len(equity_curve) < len(df):
        equity_curve.insert(0, 10000.0)
        
    df['equity'] = equity_curve
    
    return df

# -----------------------------------------------------------------------------
# 3. EXECUTION & PLOTTING
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Fetch
    try:
        df = fetch_binance_data('BTC/USDT', '1d', 2018)
    except Exception as e:
        print("Could not fetch from Binance (API issues or restriction). Generating Mock Data.")
        dates = pd.date_range(start="2018-01-01", periods=1000)
        df = pd.DataFrame(index=dates)
        df['open'] = 100 + np.cumsum(np.random.randn(1000))
        df['close'] = df['open'] + np.random.randn(1000)
        df['high'] = df[['open', 'close']].max(axis=1) + 1
        df['low'] = df[['open', 'close']].min(axis=1) - 1
        df['volume'] = 1000

    if not df.empty:
        # 2. Calc
        df, sma_cols = calculate_metrics(df)
        
        # 3. Backtest
        # Resistance Threshold set arbitrarily to 0.5 for testing; 
        # Real values depend heavily on the magnitude of returns of returns.
        df = backtest_strategy(df, sma_cols, threshold=2.0, lookback=400)
        
        # 4. Plot
        plt.figure(figsize=(12, 10))
        
        # Subplot 1: Price & SMAs
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['close'], label='Close Price', color='black', alpha=0.6)
        # Plot a few SMAs for visual check (rescaled to price for visual? No, SMAs are on SumReturns)
        plt.title('BTC/USDT Price')
        plt.legend()
        
        # Subplot 2: Resistance
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['resistance'], label='Calculated Resistance', color='orange')
        plt.axhline(y=2.0, color='r', linestyle='--', label='Threshold')
        plt.title('Algorithmic Resistance')
        plt.legend()
        
        # Subplot 3: Equity
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['equity'], label='Strategy Equity', color='green')
        plt.title('Equity Curve (Start $10k)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Output Stats
        total_return = (df['equity'].iloc[-11] - 10000) / 10000 * 100
        print(f"\nBacktest Complete.")
        print(f"Final Equity: ${df['equity'].iloc[-11]:.2f}") # -11 to account for lookahead cutoff
        print(f"Total Return: {total_return:.2f}%")
