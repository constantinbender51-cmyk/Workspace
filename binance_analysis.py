# conviction_backtest_no_lookahead.py
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from flask import Flask, send_file
import io
import threading

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
HORIZON = 30  # Decay window for signal contribution (days)
INITIAL_CAPITAL = 10000.0

# --- Winning Signals (Top 5 Diverse) ---
WINNING_SIGNALS = [
    ('EMA_CROSS', 50, 150, 0),         # EMA 50/150
    ('PRICE_SMA', 380, 0, 0),          # Price/SMA 380
    ('PRICE_SMA', 140, 0, 0),          # Price/SMA 140
    ('MACD_CROSS', 12, 26, 15),        # MACD (12/26/15)
    ('RSI_CROSS', 35, 0, 0),           # RSI 35 (Crossover)
]

MAX_CONVICTION = len(WINNING_SIGNALS)  # used to normalize to [-1,1] exposure


# --- Data Fetching ---
def fetch_binance_data():
    """Fetches daily OHLCV from Binance starting SINCE_STR."""
    print(f"Fetching data for {SYMBOL} since {SINCE_STR}...")
    exchange = ccxt.binance()
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
            time.sleep(0.1)

            # Stop fetching if close to current time
            if (exchange.milliseconds() - last_timestamp) < (24 * 60 * 60 * 1000):
                break

            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break

    print(f"\nTotal candles fetched: {len(all_ohlcv)}")

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)

    # intraday return: close / open - 1 (this will be used for open->close when entering at open)
    df['intraday_return'] = df['close'] / df['open'] - 1.0

    # close-to-close return kept for reference if needed
    df['return'] = df['close'].pct_change()

    # drop the first row which has NaNs for returns
    df.dropna(subset=['intraday_return'], inplace=True)
    return df


# --- Signal Generation ---
def generate_signals(df):
    """
    Generate signals using only information available at the close of each day 't'.
    Signals represent the *intent* to trade on the next day's open (t+1).
    """
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

    # --- IMPORTANT: shift signals forward by 1 day to avoid lookahead ---
    # A signal computed at close(t) will be executed at open(t+1).
    df_signals = df_signals.shift(1)

    # Drop NaNs created by shifting (first row(s))
    df_signals.dropna(inplace=True)
    # Convert floats to ints (shift created floats)
    df_signals = df_signals.astype(int)

    return df_signals


# --- Backtest implementation ---
def run_conviction_backtest(df_data, df_signals):
    """
    Executes the conviction backtest with:
     - signals shifted so they are executed at next day's open
     - PnL computed using intraday returns (open->close of the trading day)
     - linear 30-day decay starting when a signal is first active in the tradable index
    """

    # align dataframes: only keep rows where we have both market data and signals
    df = df_data.loc[df_signals.index].copy()
    signals = df_signals.loc[df.index]  # same index now

    num_days = len(df)
    intraday_returns = df['intraday_return'].values  # this is return for each day if entered at that day's open
    dates = df.index

    # track arrays
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    conviction_raw = np.zeros(num_days)   # raw sum of contributions (range approx [-MAX_CONVICTION, MAX_CONVICTION])
    conviction_norm = np.zeros(num_days)  # normalized exposure in [-1,1]

    # signal tracking per indicator
    signal_start_day = np.full(MAX_CONVICTION, -1, dtype=int)  # day index when current active signal started (in tradable index)
    signal_direction = np.zeros(MAX_CONVICTION, dtype=int)

    # initialize portfolio
    portfolio[0] = INITIAL_CAPITAL

    # backtest loop (day by day). On index t we are trading the open->close of day t
    for t in range(num_days):
        # 1) Update signal activation based on signals DataFrame (these signals were shifted so they correspond to trades on day t)
        daily_conviction_raw = 0.0

        for i, (sig_type, p1, p2, p3) in enumerate(WINNING_SIGNALS):
            current_sig = signals.iloc[t, i]  # this is the signal intended to be executed at open of this day

            if current_sig != 0:
                # If a new signal arrives (or continues), ensure start day is set if it wasn't active before
                if signal_direction[i] == 0:
                    # first day the signal will be active is this day t (because signals were shifted)
                    signal_start_day[i] = t
                    signal_direction[i] = current_sig
                else:
                    # If direction flipped, restart decay
                    if current_sig != signal_direction[i]:
                        signal_start_day[i] = t
                        signal_direction[i] = current_sig
                    # if same direction and already active, do nothing (decay continues)

            else:
                # If there is no current signal, we don't immediately clear previous signal's decay unless you want that behavior.
                # In this design, absence of signal simply means no new activation; previously active signals continue to decay
                # until their decay reaches zero. If you prefer immediate termination on signal=0, uncomment below:
                # signal_direction[i] = 0
                pass

            # compute contribution if there is an active direction from earlier (signal_direction non-zero)
            if signal_direction[i] != 0:
                d = t - signal_start_day[i]
                if d < 0:
                    # shouldn't happen but guard
                    decay = 0.0
                else:
                    decay = max(0.0, 1.0 - (d / HORIZON))

                contribution = signal_direction[i] * decay
                daily_conviction_raw += contribution

                # If decay dropped to zero, clear the tracked signal
                if decay == 0.0:
                    signal_direction[i] = 0
                    signal_start_day[i] = -1

        # 2) Normalize conviction to -1..1 exposure (so full agreement across signals => 100% long)
        exposure = daily_conviction_raw / MAX_CONVICTION
        exposure = float(np.clip(exposure, -1.0, 1.0))

        # 3) Compute PnL for day t using intraday returns (open->close)
        #    PnL fraction of capital = exposure * intraday_return
        pnl_fraction = exposure * intraday_returns[t]
        daily_pnl[t] = portfolio[t - 1] * pnl_fraction if t > 0 else portfolio[0] * pnl_fraction
        # update portfolio for next day
        if t == 0:
            portfolio[t] = portfolio[0] + daily_pnl[t]
        else:
            portfolio[t] = portfolio[t - 1] + daily_pnl[t]

        # store diagnostics
        conviction_raw[t] = daily_conviction_raw
        conviction_norm[t] = exposure

    # finalize output DataFrame
    results = df[['open', 'close', 'intraday_return', 'return']].copy()
    results['Conviction_Raw'] = conviction_raw
    results['Exposure'] = conviction_norm  # fraction of capital invested [-1,1]
    results['Daily_PnL'] = daily_pnl
    results['Portfolio_Value'] = portfolio

    total_return = (portfolio[-1] / portfolio[0]) - 1.0

    return total_return, results


def create_equity_plot(results_df):
    """
    Create matplotlib plot of price and equity curve with red and blue background colors.
    Returns the plot as a PNG image in bytes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Price chart with blue background
    ax1.set_facecolor('lightblue')
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=2, label='BTC Price')
    ax1.set_title('BTC/USDT Price', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Equity curve with red background
    ax2.set_facecolor('lightcoral')
    ax2.plot(results_df.index, results_df['Portfolio_Value'], 'g-', linewidth=2, label='Portfolio Value')
    ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Portfolio Value (USDT)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf




def start_web_server(results_df):
    """
    Start Flask web server on port 8080 to display the equity curve plot.
    """
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conviction Backtest Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .info {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .plot-container {{ text-align: center; margin-top: 20px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Conviction Strategy Backtest Results</h1>
                <div class="info">
                    <p><strong>Period:</strong> {start_date} to {end_date}</p>
                    <p><strong>Initial Capital:</strong> ${initial_capital:,.2f}</p>
                    <p><strong>Final Portfolio Value:</strong> ${final_value:,.2f}</p>
                    <p><strong>Total Return:</strong> {total_return:.2f}%</p>
                    <p><strong>Server running on port 8080</strong></p>
                </div>
                <div class="plot-container">
                    <h2>Price and Equity Curve</h2>
                    <p>Price chart (blue background) and equity curve (red background)</p>
                    <img src="/plot" alt="Equity Curve Plot">
                </div>
            </div>
        </body>
        </html>
        '''
        return html_template.format(
            start_date=results_df.index.min().strftime('%Y-%m-%d'),
            end_date=results_df.index.max().strftime('%Y-%m-%d'),
            initial_capital=INITIAL_CAPITAL,
            final_value=results_df['Portfolio_Value'].iloc[-1],
            total_return=((results_df['Portfolio_Value'].iloc[-1] / INITIAL_CAPITAL) - 1) * 100
        )
    
    @app.route('/plot')
    def plot():
        buf = create_equity_plot(results_df)
        return send_file(buf, mimetype='image/png')
    
    @app.route('/data')
    def data():
        return results_df.to_csv()
    
    print("\n" + "=" * 60)
    print("Starting web server on http://localhost:8080")
    print("=" * 60)
    
    # Run Flask in a separate thread to avoid blocking
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)).start()





if __name__ == '__main__':
    df_data = fetch_binance_data()
    df_signals = generate_signals(df_data)

    if df_signals.empty:
        print("No signals generated after shift — aborting.")
    else:
        total_return, results_df = run_conviction_backtest(df_data, df_signals)

        print("\n" + "=" * 60)
        print("Conviction Strategy Backtest (No Lookahead) - Executions at NEXT DAY OPEN")
        print("=" * 60)
        start_date = results_df.index.min().strftime('%Y-%m-%d')
        end_date = results_df.index.max().strftime('%Y-%m-%d')
        bh_return = (results_df['close'].iloc[-1] / results_df['close'].iloc[0]) - 1.0

        print(f"Time Period: {start_date} to {end_date}")
        print(f"Trading Days (post-warmup): {len(results_df)}")
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"Conviction Strategy Total Return: {total_return * 100:.2f}%")
        print(f"Buy & Hold (close-to-close) Return: {bh_return * 100:.2f}%")
        print(f"Final Portfolio Value: ${INITIAL_CAPITAL * (1 + total_return):,.2f}")

        # Calculate Sharpe ratio
        daily_returns = results_df['Daily_PnL'] / results_df['Portfolio_Value'].shift(1)
        daily_returns.iloc[0] = results_df['Daily_PnL'].iloc[0] / INITIAL_CAPITAL
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        if std_daily_return > 0:
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)  # Annualized with 252 trading days
        else:
            sharpe_ratio = 0.0
        print(f"Sharpe Ratio (annualized, risk-free=0): {sharpe_ratio:.4f}")

        results_df.to_csv('conviction_backtest_results_no_lookahead.csv')
        print("\nSaved daily results to 'conviction_backtest_results_no_lookahead.csv'")
        
        # Start web server after backtest completes
        start_web_server(results_df)

