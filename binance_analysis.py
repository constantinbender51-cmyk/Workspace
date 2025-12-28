import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION (MATCHING TUMBLER.PY)
# ==========================================
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"

# Strategy Parameters
SMA_PERIOD_1 = 32           # Primary logic
SMA_PERIOD_2 = 114          # Trend filter
STATIC_STOP_PCT = 0.043     # 4.3%
TAKE_PROFIT_PCT = 0.126     # 12.6%
LIMIT_OFFSET_PCT = 0.0002   # 0.02% (Simulated as slippage/better fill)

# III Parameters
III_WINDOW = 27
III_T_LOW = 0.058
III_T_HIGH = 0.259

# Leverage Tiers
LEV_LOW = 0.079    # Choppy
LEV_MID = 4.327    # Sweet spot
LEV_HIGH = 3.868   # Overextended

# Flat Regime Parameters
FLAT_REGIME_THRESHOLD = 0.356
BAND_WIDTH_PCT = 0.077

# Backtest Settings
INITIAL_CAPITAL = 10000.0
FEE_PCT = 0.0005  # 0.05% per trade (taker/market proxy)

# ==========================================
# DATA FETCHING
# ==========================================
def fetch_binance_data(symbol, start_date_str):
    filename = f"{symbol.lower()}_daily_{start_date_str}.csv"
    
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    print(f"Fetching data from Binance for {symbol} starting {start_date_str}...")
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ts,
            "limit": limit
        }
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            
            if not data:
                break
                
            all_data.extend(data)
            start_ts = data[-1][0] + 86400000 # Next day
            
            # Rate limit respect
            time.sleep(0.1)
            print(f"Fetched {len(all_data)} candles...", end='\r')
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print("\nData fetch complete.")
    
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols:
        df[c] = df[c].astype(float)
        
    final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    final_df.set_index('timestamp', inplace=True)
    final_df.to_csv(filename)
    return final_df

# ==========================================
# INDICATOR CALCULATIONS
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    
    # SMAs
    df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    
    # III Calculation (Matching tumbler.py logic)
    # log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling sum of abs log returns (Path Length)
    # Rolling abs sum of log returns (Net Direction)
    # III = Net Direction / Path Length? 
    # tumbler.py: (df_calc['log_ret'].rolling(w).sum().abs() / df_calc['log_ret'].abs().rolling(w).sum())
    
    w = III_WINDOW
    rolling_sum_log_ret = df['log_ret'].rolling(w).sum().abs()
    rolling_sum_abs_log_ret = df['log_ret'].abs().rolling(w).sum()
    
    df['iii'] = (rolling_sum_log_ret / rolling_sum_abs_log_ret).fillna(0)
    
    return df

# ==========================================
# BACKTEST ENGINE
# ==========================================
def run_backtest(df):
    print("Running Backtest...")
    
    # Initialize state variables
    cash = INITIAL_CAPITAL
    btc_position = 0.0
    equity_curve = []
    trade_history = []
    
    # Strategy State
    flat_regime_active = False
    
    # Pre-calculate indicators to speed up loop
    df = calculate_indicators(df)
    
    # We need to iterate carefully.
    # At index `i` (representing "Today"), we use indicators derived from Close of `i-1` (Yesterday).
    # We trade at Open of `i`.
    
    timestamps = df.index
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    sma1 = df['sma_1'].values
    sma2 = df['sma_2'].values
    iii_vals = df['iii'].values
    
    # Start loop after SMA2 period to ensure valid indicators
    start_idx = SMA_PERIOD_2 + 1
    
    for i in range(start_idx, len(df)):
        # Data for Decision (Yesterday's Close)
        prev_idx = i - 1
        
        # Current Market Data for Execution (Today)
        current_time = timestamps[i]
        current_open = opens[i]
        current_high = highs[i]
        current_low = lows[i]
        
        # Indicators from Yesterday (simulating decision made at 00:01 UTC)
        val_sma1 = sma1[prev_idx]
        val_sma2 = sma2[prev_idx]
        val_iii = iii_vals[prev_idx]
        prev_close = closes[prev_idx]
        
        # ----------------------------------------
        # 1. Update Flat Regime State
        # ----------------------------------------
        
        # Check Trigger (Enter Flat Regime)
        if val_iii < FLAT_REGIME_THRESHOLD:
            # If we were not flat, we enter flat regime
            # (In tumbler.py, checking trigger doesn't require prev state, it just sets it)
            # "If III < T: return True"
            potential_flat = True
        else:
            # If III is healthy, we maintain previous state unless manually released?
            # tumbler.py: "if iii < threshold: return True. else: return current_state"
            # Wait, tumbler.py: "return current_flat_regime" if not triggered.
            # This means once entered, it sticks until released.
            potential_flat = flat_regime_active

        # Check Release (Exit Flat Regime)
        # Release happens if Price enters 7.7% band around EITHER SMA
        if potential_flat:
            # Check bands using Prev Close and Prev SMAs (Decision time data)
            diff_sma1 = abs(prev_close - val_sma1)
            diff_sma2 = abs(prev_close - val_sma2)
            thresh_sma1 = val_sma1 * BAND_WIDTH_PCT
            thresh_sma2 = val_sma2 * BAND_WIDTH_PCT
            
            in_band = (diff_sma1 <= thresh_sma1) or (diff_sma2 <= thresh_sma2)
            
            if in_band:
                flat_regime_active = False # RELEASED
            else:
                flat_regime_active = True # MAINTAIN FLAT
        else:
            flat_regime_active = False

        # ----------------------------------------
        # 2. Determine Target Signal
        # ----------------------------------------
        signal = "FLAT"
        
        if flat_regime_active:
            signal = "FLAT"
        else:
            if prev_close > val_sma1 and prev_close > val_sma2:
                signal = "LONG"
            elif prev_close < val_sma1 and prev_close < val_sma2:
                signal = "SHORT"
            else:
                signal = "FLAT"
        
        # ----------------------------------------
        # 3. Determine Leverage & Position Size
        # ----------------------------------------
        leverage = 0.0
        if val_iii < III_T_LOW:
            leverage = LEV_LOW
        elif val_iii < III_T_HIGH:
            leverage = LEV_MID
        else:
            leverage = LEV_HIGH
            
        # ----------------------------------------
        # 4. Execution Logic (Open of Day)
        # ----------------------------------------
        
        # Calculate Equity before trade (Mark to Market at Open)
        # If we held a position overnight, its value changed from Prev Close to Curr Open
        # But for simplicity in this loop, we recalculate position value at Open.
        
        current_equity = cash
        if btc_position != 0:
            # PnL from position
            # Note: btc_position is size in BTC. 
            # If LONG (+), value = size * price
            # If SHORT (-), value = size * (entry_price - price)? 
            # Simplified futures math: PnL = size * (price - entry_price)
            # We track 'cash' as Collateral.
            pass

        # We need to manage the position continuously.
        # Let's track: entry_price, btc_size, side (1 or -1)
        
        # For simplicity, we will close position and re-open daily if signal persists,
        # OR just check for signal changes.
        # Given the "Daily Trade" nature of tumbler.py, it flattens and re-enters daily 
        # based on current portfolio value to maintain exact leverage target.
        # We will simulate "Close All -> Re-Enter" logic at Open.
        
        # A. Close existing position at Open Price
        if btc_position != 0:
            # Calculate PnL
            # btc_position is signed (+ for Long, - for Short)
            # Exit Value
            pnl = btc_position * (current_open - last_entry_price)
            
            # Apply fees (on notional value)
            exit_fee = abs(btc_position * current_open) * FEE_PCT
            
            cash += pnl - exit_fee
            btc_position = 0.0
        
        # B. Check for Intraday SL/TP if we were to enter
        # We assume we enter at Open Price
        
        if signal != "FLAT":
            # Target Notional
            target_notional = cash * leverage
            size = target_notional / current_open
            
            if signal == "SHORT":
                size = -size
            
            # Calculate SL/TP Prices
            entry_price = current_open
            
            sl_price = 0.0
            tp_price = 0.0
            
            if signal == "LONG":
                sl_price = entry_price * (1 - STATIC_STOP_PCT)
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            else: # SHORT
                sl_price = entry_price * (1 + STATIC_STOP_PCT)
                tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
            
            # Check Low/High for hits
            hit_sl = False
            hit_tp = False
            exit_price = 0.0
            
            if signal == "LONG":
                if current_low <= sl_price:
                    hit_sl = True
                    exit_price = sl_price
                elif current_high >= tp_price:
                    hit_tp = True
                    exit_price = tp_price
            else: # SHORT
                if current_high >= sl_price:
                    hit_sl = True
                    exit_price = sl_price
                elif current_low <= tp_price:
                    hit_tp = True
                    exit_price = tp_price
            
            # Executing the Trade
            if hit_sl:
                # Entered at Open, Exited at SL
                fee_entry = abs(size * entry_price) * FEE_PCT
                fee_exit = abs(size * exit_price) * FEE_PCT
                pnl = size * (exit_price - entry_price)
                cash += pnl - fee_entry - fee_exit
                
                trade_history.append({
                    'date': current_time, 'type': 'SL', 'pnl': pnl, 'equity': cash
                })
                
            elif hit_tp:
                # Entered at Open, Exited at TP
                fee_entry = abs(size * entry_price) * FEE_PCT
                fee_exit = abs(size * exit_price) * FEE_PCT
                pnl = size * (exit_price - entry_price)
                cash += pnl - fee_entry - fee_exit
                
                trade_history.append({
                    'date': current_time, 'type': 'TP', 'pnl': pnl, 'equity': cash
                })
                
            else:
                # Position Held until Close (or carried over, simplified as re-eval next day)
                # We enter here, pay fee, and keep position open for the "day"
                # Since we reset at next Open, we effectively hold till next Open.
                # However, to track equity curve daily, we mark to Close.
                
                fee_entry = abs(size * entry_price) * FEE_PCT
                cash -= fee_entry
                
                # We hold this position into the "state" for the next loop iteration (Close logic above)
                btc_position = size
                last_entry_price = entry_price
                
                # But for Equity Curve calculation, we value it at Close
                unrealized_pnl = size * (closes[i] - entry_price)
                equity_at_close = cash + unrealized_pnl
                
                # Note: We don't add equity_at_close to cash yet. 
                # The "cash" variable tracks realized balance.
                
                equity_curve.append({'date': current_time, 'equity': equity_at_close})
                continue # Skip the default append below

        # If Flat or SL/TP hit, equity is just cash
        equity_curve.append({'date': current_time, 'equity': cash})

    return pd.DataFrame(equity_curve).set_index('date')

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Get Data
    df = fetch_binance_data(SYMBOL, START_DATE)
    
    # 2. Run Backtest
    results = run_backtest(df)
    
    # 3. Calculate Metrics
    results['returns'] = results['equity'].pct_change()
    
    # Buy and Hold (Normalized)
    df_bh = df.loc[results.index]
    bh_norm = (df_bh['close'] / df_bh['close'].iloc[0]) * INITIAL_CAPITAL
    
    total_return = (results['equity'].iloc[-1] / INITIAL_CAPITAL) - 1
    bh_return = (bh_norm.iloc[-1] / INITIAL_CAPITAL) - 1
    
    # Sharpe (assuming daily risk free = 0)
    sharpe = np.sqrt(365) * (results['returns'].mean() / results['returns'].std())
    
    # Max Drawdown
    rolling_max = results['equity'].cummax()
    drawdown = (results['equity'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (Dual SMA + III + Flat Regime)")
    print("="*40)
    print(f"Start Date:       {results.index[0].date()}")
    print(f"End Date:         {results.index[-1].date()}")
    print(f"Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Equity:     ${results['equity'].iloc[-1]:,.2f}")
    print(f"Total Return:     {total_return*100:.2f}%")
    print(f"Buy & Hold Ret:   {bh_return*100:.2f}%")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Max Drawdown:     {max_dd*100:.2f}%")
    print("="*40)
    
    # 4. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['equity'], label='Strategy Equity', linewidth=1.5)
    plt.plot(results.index, bh_norm, label='Buy & Hold (BTC)', alpha=0.5, linewidth=1)
    
    # Log scale often looks better for crypto
    plt.yscale('log')
    
    plt.title(f'Tumbler Strategy vs BTC Buy & Hold ({START_DATE} to Now)')
    plt.ylabel('Equity (USD) - Log Scale')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('backtest_result.png')
    print("Plot saved to backtest_result.png")
    
    plt.show()
