import ccxt
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, send_file
import io
import threading

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
HORIZON = 380  # Decay window (days) - OPTIMAL VALUE
INITIAL_CAPITAL = 10000.0

# --- Winning Signals ---
WINNING_SIGNALS = [
    ('EMA_CROSS', 50, 150, 0),         # EMA 50/150
    ('PRICE_SMA', 380, 0, 0),          # Price/SMA 380
    ('PRICE_SMA', 140, 0, 0),          # Price/SMA 140
    ('MACD_CROSS', 12, 26, 15),        # MACD (12/26/15)
    ('RSI_CROSS', 35, 0, 0),           # RSI 35 (Crossover)
]

MAX_CONVICTION = len(WINNING_SIGNALS)

# --- Data Fetching ---
def fetch_binance_data():
    print(f"Fetching data for {SYMBOL} since {SINCE_STR}...")
    exchange = ccxt.binance()
    exchange.enableRateLimit = True 
    since = exchange.parse8601(SINCE_STR)

    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            if (exchange.milliseconds() - last_timestamp) < (24 * 60 * 60 * 1000):
                break

            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
        except Exception as e:
            print(f"\nError fetching data: {e}")
            time.sleep(5)
            continue

    print(f"\nTotal candles fetched: {len(all_ohlcv)}")

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)

    # Standard Close-to-Close Return
    df['return'] = df['close'].pct_change()
    
    df.dropna(subset=['return'], inplace=True)
    return df


# --- Signal Generation ---
def generate_signals(df):
    df_signals = pd.DataFrame(index=df.index, dtype=int)
    close = df['close']

    for sig_type, p1, p2, p3 in WINNING_SIGNALS:
        col = f"{sig_type}_{p1}_{p2}_{p3}"

        if sig_type == 'EMA_CROSS':
            fast = close.ewm(span=p1, adjust=False).mean()
            slow = close.ewm(span=p2, adjust=False).mean()
            long_cond = (fast.shift(1) < slow.shift(1)) & (fast > slow)
            short_cond = (fast.shift(1) > slow.shift(1)) & (fast < slow)

        elif sig_type == 'PRICE_SMA':
            sma = close.rolling(window=p1).mean()
            long_cond = (close.shift(1) < sma.shift(1)) & (close > sma)
            short_cond = (close.shift(1) > sma.shift(1)) & (close < sma)

        elif sig_type == 'MACD_CROSS':
            ema_fast = close.ewm(span=p1, adjust=False).mean()
            ema_slow = close.ewm(span=p2, adjust=False).mean()
            macd = ema_fast - ema_slow
            sig_line = macd.ewm(span=p3, adjust=False).mean()
            long_cond = (macd.shift(1) < sig_line.shift(1)) & (macd > sig_line)
            short_cond = (macd.shift(1) > sig_line.shift(1)) & (macd < sig_line)

        elif sig_type == 'RSI_CROSS':
            period = p1
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            center = 50
            long_cond = (rsi.shift(1) < center) & (rsi > center)
            short_cond = (rsi.shift(1) > center) & (rsi < center)
        else:
            df_signals[col] = 0
            continue

        df_signals[col] = np.where(long_cond, 1, np.where(short_cond, -1, 0))

    # Store RAW signals before shift
    df_signals_raw = df_signals.copy()

    # Shift signals forward for backtest
    df_signals = df_signals.shift(1)
    df_signals.fillna(0, inplace=True)
    df_signals = df_signals.astype(int)

    return df_signals, df_signals_raw


# --- Backtest implementation ---
def run_conviction_backtest(df_data, df_signals):
    horizon = HORIZON
    
    common_idx = df_data.index.intersection(df_signals.index)
    df = df_data.loc[common_idx].copy()
    signals = df_signals.loc[common_idx]

    num_days = len(df)
    daily_returns = df['return'].values 
    
    # We need Close, Low, and High for Stop Loss logic
    closes = df['close'].values
    lows = df['low'].values
    highs = df['high'].values
    
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    conviction_norm = np.zeros(num_days)
    sl_triggered = np.zeros(num_days, dtype=bool) # Track days where SL hit
    
    daily_contributions = np.zeros((num_days, len(WINNING_SIGNALS)))

    signal_start_day = np.full(MAX_CONVICTION, -1, dtype=int)
    signal_direction = np.zeros(MAX_CONVICTION, dtype=int)

    portfolio[0] = INITIAL_CAPITAL

    # Stop Loss Constants
    SL_THRESHOLD = 0.02
    SL_PENALTY = 0.0205 # 2% + 0.05% slippage

    for t in range(num_days):
        daily_sum = 0.0

        for i in range(len(WINNING_SIGNALS)):
            current_sig = signals.iloc[t, i]

            if current_sig != 0:
                signal_start_day[i] = t
                signal_direction[i] = current_sig
            
            if signal_direction[i] != 0:
                d = t - signal_start_day[i]
                if d < 0: 
                    decay = 0.0
                else:
                    decay = max(0.0, 1.0 - (d / horizon))
                
                contribution = signal_direction[i] * decay
                daily_contributions[t, i] = contribution
                daily_sum += contribution

                if decay == 0.0:
                    signal_direction[i] = 0
                    signal_start_day[i] = -1

        exposure = daily_sum / MAX_CONVICTION
        exposure = np.clip(exposure, -1.0, 1.0)

        if t > 0:
            # --- STOP LOSS LOGIC ---
            prev_close = closes[t-1]
            current_low = lows[t]
            current_high = highs[t]
            
            # Default to actual market return
            effective_return = daily_returns[t]
            
            # Check Stop Loss
            if exposure > 0:
                # Long: If Low drops 2% below prev close
                if (current_low / prev_close) - 1.0 <= -SL_THRESHOLD:
                    effective_return = -SL_PENALTY
                    sl_triggered[t] = True
                    
            elif exposure < 0:
                # Short: If High rises 2% above prev close
                if (current_high / prev_close) - 1.0 >= SL_THRESHOLD:
                    effective_return = SL_PENALTY # Positive return means loss for Short
                    sl_triggered[t] = True

            # Calculate PnL with effective return
            pnl_amt = portfolio[t-1] * exposure * effective_return
            portfolio[t] = portfolio[t-1] + pnl_amt
            daily_pnl[t] = pnl_amt
        else:
            portfolio[t] = INITIAL_CAPITAL

        conviction_norm[t] = exposure

    results = df[['close', 'return']].copy()
    results['Exposure'] = conviction_norm
    results['Daily_PnL'] = daily_pnl
    results['Portfolio_Value'] = portfolio
    results['SL_Triggered'] = sl_triggered
    
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)
    
    # Calculate Drawdown
    rolling_max = results['Portfolio_Value'].cummax()
    results['Drawdown'] = (results['Portfolio_Value'] - rolling_max) / rolling_max

    signal_names = [f"{s[0]}_{s[1]}" for s in WINNING_SIGNALS]
    for i, name in enumerate(signal_names):
        results[f"Contrib_{name}"] = daily_contributions[:, i]
    
    total_return = (portfolio[-1] / portfolio[0]) - 1.0
    return total_return, results, signal_names

def calculate_net_daily_signal_event(df_signals_raw, results_df):
    aligned_signals = df_signals_raw.loc[results_df.index.min():results_df.index.max()]
    net_signal_sum = aligned_signals.sum(axis=1)
    long_signal_dates = net_signal_sum[net_signal_sum > 0].index
    short_signal_dates = net_signal_sum[net_signal_sum < 0].index
    return long_signal_dates, short_signal_dates


def create_equity_plot(results_df, long_signal_dates, short_signal_dates, horizon):
    # Use 3 subplots: Price, Equity, Drawdown
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
    
    # --- Plot 1: Price Chart (Log Scale) ---
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1, label='BTC Price')
    ax1.set_yscale('log')
    ax1.set_title(f'BTC/USDT Price (Log Scale) - Horizon: {horizon}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc='upper left')

    # Signals
    long_prices = results_df.loc[long_signal_dates, 'close']
    short_prices = results_df.loc[short_signal_dates, 'close']
    ax1.scatter(long_signal_dates, long_prices, marker='^', color='green', s=50, label='Long Signal', zorder=5)
    ax1.scatter(short_signal_dates, short_prices, marker='v', color='red', s=50, label='Short Signal', zorder=5)
    
    # Mark Stop Loss Events
    sl_dates = results_df[results_df['SL_Triggered']].index
    sl_prices = results_df.loc[sl_dates, 'close']
    ax1.scatter(sl_dates, sl_prices, marker='x', color='purple', s=40, label='Stop Loss Hit', zorder=6)

    # --- Plot 2: Equity Curve (Log Scale) ---
    ax2.plot(results_df.index, results_df['Portfolio_Value'], 'b-', linewidth=1.5, label='Strategy Equity')
    
    bh_curve = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax2.plot(results_df.index, bh_curve, 'g--', alpha=0.8, linewidth=1, label='Buy & Hold')
    
    ax2.set_yscale('log')
    ax2.set_title('Strategy Equity vs Buy & Hold (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value ($)', fontsize=10)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(loc='upper left')
    
    # --- Plot 3: Drawdown ---
    ax3.fill_between(results_df.index, results_df['Drawdown'] * 100, 0, color='red', alpha=0.3)
    ax3.plot(results_df.index, results_df['Drawdown'] * 100, 'r-', linewidth=0.5)
    ax3.set_title('Strategy Drawdown (%)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Drawdown %', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def start_web_server(results_df, long_signal_dates, short_signal_dates, signal_names, best_horizon):
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        final_val = results_df['Portfolio_Value'].iloc[-1]
        total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
        
        strat_daily_rets = results_df['Strategy_Daily_Return']
        if strat_daily_rets.std() > 0:
            overall_sharpe = (strat_daily_rets.mean() / strat_daily_rets.std()) * np.sqrt(365)
        else:
            overall_sharpe = 0.0
            
        sl_hits = results_df['SL_Triggered'].sum()

        # --- Monthly Stats ---
        monthly_rows = ""
        monthly_groups = results_df.groupby(pd.Grouper(freq='M'))
        monthly_stats = []
        for name, group in monthly_groups:
            if len(group) < 5: continue
            
            m_daily_rets = group['Strategy_Daily_Return']
            m_start_val = group['Portfolio_Value'].iloc[0]
            m_end_val = group['Portfolio_Value'].iloc[-1


