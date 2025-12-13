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
from matplotlib.figure import Figure

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
HORIZON = 380  # Decay window (days)
INITIAL_CAPITAL = 10000.0
LEVERAGE = 5.0 # Fixed 5x leverage based on normalized conviction score (0 to 1)

# --- Winning Signals ---
WINNING_SIGNALS = [
    ('EMA_CROSS', 50, 150, 0),         # 0: EMA 50/150
    ('PRICE_SMA', 380, 0, 0),          # 1: Price/SMA 380
    ('PRICE_SMA', 140, 0, 0),          # 2: Price/SMA 140
    ('MACD_CROSS', 12, 26, 15),        # 3: MACD (12/26/15) - Standard Weight
    ('RSI_CROSS', 35, 0, 0),           # 4: RSI 35 (Crossover)
]

N_SIGNALS = len(WINNING_SIGNALS)
SIGNAL_NAMES = [f"{s[0]}_{s[1]}_{s[2]}" for s in WINNING_SIGNALS]
SIGNAL_COLORS = ['#3498DB', '#E67E22', '#F1C40F', '#2ECC71', '#E74C3C'] # Blue, Orange, Yellow, Green, Red

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
            # Remove short condition for long-only strategy
            short_cond = False

        elif sig_type == 'PRICE_SMA':
            sma = close.rolling(window=p1).mean()
            long_cond = (close.shift(1) < sma.shift(1)) & (close > sma)
            # Remove short condition for long-only strategy
            short_cond = False

        elif sig_type == 'MACD_CROSS':
            ema_fast = close.ewm(span=p1, adjust=False).mean()
            ema_slow = close.ewm(span=p2, adjust=False).mean()
            macd = ema_fast - ema_slow
            sig_line = macd.ewm(span=p3, adjust=False).mean()
            long_cond = (macd.shift(1) < sig_line.shift(1)) & (macd > sig_line)
            # Remove short condition for long-only strategy
            short_cond = False

        elif sig_type == 'RSI_CROSS':
            period = p1
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50) 
            center = 50
            long_cond = (rsi.shift(1) < center) & (rsi > center)
            # Remove short condition for long-only strategy
            short_cond = False
        else:
            df_signals[col] = 0
            continue

        # Only generate long signals (1) or neutral (0), no short signals (-1)
        df_signals[col] = np.where(long_cond, 1, 0)

    df_signals_shifted = df_signals.shift(1)
    df_signals_shifted.fillna(0, inplace=True)
    df_signals_shifted = df_signals_shifted.astype(int)

    return df_signals_shifted 

# --- Pre-calculate Signal Contributions (The "Decay" Matrix) ---
def precalculate_contributions(df_data, df_signals):
    """
    Creates a (Days, Signals) matrix where each cell is the decayed value (-1 to 1)
    of that signal for that day.
    """
    common_idx = df_data.index.intersection(df_signals.index)
    signals = df_signals.loc[common_idx]
    num_days = len(common_idx)
    
    contributions = np.zeros((num_days, N_SIGNALS))
    
    signal_start_day = np.full(N_SIGNALS, -1, dtype=int)
    signal_direction = np.zeros(N_SIGNALS, dtype=int)

    for t in range(num_days):
        for i in range(N_SIGNALS):
            current_sig = signals.iloc[t, i]
            
            if current_sig != 0:
                signal_start_day[i] = t
                signal_direction[i] = current_sig
            
            if signal_direction[i] != 0:
                d = t - signal_start_day[i]
                if d < 0: decay = 0.0
                else: decay = max(0.0, 1.0 - (d / HORIZON))
                
                contributions[t, i] = signal_direction[i] * decay
                
                if decay == 0.0:
                    signal_direction[i] = 0
                    signal_start_day[i] = -1
                    
    return contributions, common_idx

# --- Backtest Runner (Simple Conviction, MACD is Normal) ---
def run_simple_backtest(df_data, contributions, common_idx):
    df = df_data.loc[common_idx].copy()
    num_days = len(df)
    returns = df['return'].values
    
    # Initialize backtest arrays
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    
    # Simple Conviction: Sum of all decaying contributions, normalized by N_SIGNALS
    # (T,) vector of overall conviction score (-1.0 to 1.0)
    raw_conviction = np.sum(contributions, axis=1) / N_SIGNALS
    
    # Calculate Exposure: Raw Conviction * Fixed Leverage (Long Only)
    # Clip raw_conviction to be non-negative (0 to 1) for long-only strategy
    raw_conviction_long = np.clip(raw_conviction, 0, 1)
    exposure = raw_conviction_long * LEVERAGE
    exposure = np.clip(exposure, 0, LEVERAGE)  # Only positive exposure for long-only
    
    # Run Portfolio Loop
    portfolio[0] = INITIAL_CAPITAL
    is_bankrupt = False
    
    for t in range(1, num_days):
        exposure_t = exposure[t]
        
        if not is_bankrupt:
            pnl = portfolio[t-1] * exposure_t * returns[t]
            portfolio[t] = portfolio[t-1] + pnl
            daily_pnl[t] = pnl
            
            if portfolio[t] <= 0:
                portfolio[t] = 0
                is_bankrupt = True
        else:
            portfolio[t] = 0
            
    # Compile results
    results = df[['close', 'return']].copy()
    results['Exposure'] = exposure
    results['Raw_Conviction'] = raw_conviction
    results['Portfolio_Value'] = portfolio
    results['Daily_PnL'] = daily_pnl
    
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)
    
    rolling_max = results['Portfolio_Value'].cummax()
    results['Drawdown'] = np.where(rolling_max > 0, (results['Portfolio_Value'] - rolling_max) / rolling_max, -1.0)
    
    # Return results and the original contributions (as no scaling was applied)
    return results, contributions

def calculate_sharpe(results_df):
    d_rets = results_df['Strategy_Daily_Return']
    sharpe = (d_rets.mean()/d_rets.std())*np.sqrt(365) if d_rets.std()>0 else 0
    return sharpe

def create_equity_plot(results_df, contributions):
    fig = Figure(figsize=(12, 16))
    
    # 4 Subplots: Price, Signal Contributions, Equity, Drawdown
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    
    # --- Plot 1: Price ---
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1)
    ax1.set_yscale('log')
    ax1.set_title('BTC Price (Log)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Individual Signal Contributions (Stacked Area) ---
    
    # For long-only strategy, contributions should only be positive or zero
    pos_contributions = np.clip(contributions, 0, None)
    
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # Stackplot uses the raw contribution matrix (only positive for long-only)
    ax2.stackplot(results_df.index, pos_contributions.T, colors=SIGNAL_COLORS, labels=SIGNAL_NAMES, alpha=0.7)

    # Plot total used conviction (Conviction * N_SIGNALS)
    ax2.plot(results_df.index, results_df['Raw_Conviction'] * N_SIGNALS, 'k--', linewidth=1, alpha=0.5, label='Total Conviction Score')
    
    ax2.set_ylim(0, N_SIGNALS)  # Only positive range for long-only
    ax2.set_ylabel('Signal Contribution (Max +5)', fontsize=10)
    ax2.set_title('Individual Signal Contributions (Long Only)', fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    
    # --- Plot 3: Equity ---
    
    ax3.plot(results_df.index, results_df['Portfolio_Value'], 'b-', linewidth=1.5, label=f'Strategy ({LEVERAGE}x)')
    bh = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax3.plot(results_df.index, bh, 'g--', alpha=0.8, label='Buy & Hold')
    
    if (results_df['Portfolio_Value'] <= 0).any(): ax3.set_yscale('symlog', linthresh=1.0)
    else: ax3.set_yscale('log')
    
    ax3.legend(loc='upper left')
    ax3.set_title('Equity Curve', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Drawdown ---
    ax4.fill_between(results_df.index, results_df['Drawdown']*100, 0, color='red', alpha=0.3)
    ax4.plot(results_df.index, results_df['Drawdown']*100, 'r-', linewidth=0.5)
    ax4.set_title('Drawdown %', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

def start_web_server(results_df, contributions, overall_sharpe):
    app = Flask(__name__)
    
    final_val = results_df['Portfolio_Value'].iloc[-1]
    total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
    
    # Monthly Stats
    monthly_rows = ""
    monthly_groups = results_df.groupby(pd.Grouper(freq='M'))
    for name, group in reversed(list(monthly_groups)): 
        if len(group) < 5: continue
        m_daily_rets = group['Strategy_Daily_Return']
        m_start = group['Portfolio_Value'].iloc[0]
        m_end = group['Portfolio_Value'].iloc[-1]
        m_ret = (m_end / m_start) - 1.0 if m_start > 0 else 0.0
        m_sharpe = (m_daily_rets.mean()/m_daily_rets.std())*np.sqrt(365) if m_daily_rets.std()>0 else 0
        
        color = "green" if m_ret > 0 else "red"
        monthly_rows += f"<tr><td>{name.strftime('%Y-%m')}</td><td style='color:{color}'><b>{m_ret*100:.2f}%</b></td><td>{m_sharpe:.2f}</td></tr>"

    @app.route('/')
    def index():
        ts = int(time.time())
        return f'''
        <html>
        <head><title>Conviction Strategy Simple</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; margin: 40px; color: #333; }}
            .stats {{ background: #f9f9f9; padding: 20px; border-radius: 8px; display: inline-block; margin-bottom: 20px; }}
            img {{ max-width: 95%; border: 1px solid #ccc; margin: 20px 0; }}
            table {{ margin: 0 auto; border-collapse: collapse; width: 100%; max-width: 600px; }}
            td, th {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        </style>
        </head>
        <body>
            <h1>Conviction Strategy (Long Only, 5x Leverage)</h1>
            <div class="stats">
                <p><strong>Horizon:</strong> {HORIZON} days | <strong>Fixed Leverage:</strong> {LEVERAGE}x</p>
                <p><strong>Rule:</strong> All 5 signals contribute equally. LONG ONLY - No short positions.</p>
                <p><strong>Final:</strong> ${final_val:,.2f} | <strong>Return:</strong> {total_ret:.2f}%</p>
                <p><strong>Overall Sharpe:</strong> {overall_sharpe:.2f}</p>
            </div>
            
            <h2>Signal Contributions Breakdown</h2>
            <img src="/plot?v={ts}" />
            
            <h2>Monthly Returns</h2>
            <table><thead><tr><th>Month</th><th>Return</th><th>Sharpe</th></tr></thead><tbody>{monthly_rows}</tbody></table>
        </body>
        </html>
        '''

    @app.route('/plot')
    def plot():
        try:
            buf = create_equity_plot(results_df, contributions)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            print(f"Error creating plot: {e}")
            return f"Error creating plot: {e}", 500

    print(f"Server running on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    df_data = fetch_binance_data()
    df_signals = generate_signals(df_data)
    
    # 1. Precalculate Signal Contributions (Decay Matrix)
    contributions, common_idx = precalculate_contributions(df_data, df_signals)
    df_clean = df_data.loc[common_idx]
    
    # 2. Run Simple Backtest (No Rules)
    results, contributions_raw = run_simple_backtest(df_clean, contributions, common_idx)
    
    # 3. Calculate Sharpe
    sharpe = calculate_sharpe(results)
    
    start_web_server(results, contributions_raw, sharpe)
