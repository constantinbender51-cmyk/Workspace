import ccxt
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
            
            # Stop if we are within 24 hours of now
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
    """
    Generates signals. 1 = Long, -1 = Short, 0 = Neutral.
    Signals are shifted +1 day to prevent lookahead bias.
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

    # --- Shift signals forward ---
    df_signals = df_signals.shift(1)
    df_signals.fillna(0, inplace=True)
    df_signals = df_signals.astype(int)

    return df_signals


# --- Backtest implementation ---
def run_conviction_backtest(df_data, df_signals):
    common_idx = df_data.index.intersection(df_signals.index)
    df = df_data.loc[common_idx].copy()
    signals = df_signals.loc[common_idx]

    num_days = len(df)
    daily_returns = df['return'].values 
    dates = df.index

    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    conviction_raw = np.zeros(num_days)
    conviction_norm = np.zeros(num_days)

    signal_start_day = np.full(MAX_CONVICTION, -1, dtype=int)
    signal_direction = np.zeros(MAX_CONVICTION, dtype=int)

    portfolio[0] = INITIAL_CAPITAL

    for t in range(num_days):
        daily_sum = 0.0

        for i in range(len(WINNING_SIGNALS)):
            current_sig = signals.iloc[t, i]

            # Signal Recharging Logic
            if current_sig != 0:
                signal_start_day[i] = t
                signal_direction[i] = current_sig
            
            if signal_direction[i] != 0:
                d = t - signal_start_day[i]
                if d < 0: 
                    decay = 0.0
                else:
                    decay = max(0.0, 1.0 - (d / HORIZON))
                
                daily_sum += signal_direction[i] * decay

                if decay == 0.0:
                    signal_direction[i] = 0
                    signal_start_day[i] = -1

        exposure = daily_sum / MAX_CONVICTION
        exposure = np.clip(exposure, -1.0, 1.0)

        if t > 0:
            pnl_amt = portfolio[t-1] * exposure * daily_returns[t]
            portfolio[t] = portfolio[t-1] + pnl_amt
            daily_pnl[t] = pnl_amt
        else:
            portfolio[t] = INITIAL_CAPITAL

        conviction_raw[t] = daily_sum
        conviction_norm[t] = exposure

    results = df[['close', 'return']].copy()
    results['Exposure'] = conviction_norm
    results['Daily_PnL'] = daily_pnl
    results['Portfolio_Value'] = portfolio
    
    total_return = (portfolio[-1] / portfolio[0]) - 1.0
    return total_return, results


def create_equity_plot(results_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    exposure = results_df['Exposure']
    
    # Calculate days where position starts or flips direction
    # We check if the sign changes compared to the previous day
    # np.sign(x) gives 1 for positive, -1 for negative, 0 for zero
    position_change = np.sign(exposure).diff().fillna(0).abs() > 0
    
    # Exclude the first day unless exposure is non-zero
    if exposure.iloc[0] == 0:
        position_change.iloc[0] = False
    
    entry_dates = results_df.index[position_change]
    
    # --- Plot 1: Price Chart ---
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1.5, label='BTC Price')
    ax1.set_title('BTC/USDT Price', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # --- Plot 2: Equity Curve ---
    ax2.plot(results_df.index, results_df['Portfolio_Value'], 'k-', linewidth=1.5, label='Strategy Equity')
    
    # Add Buy & Hold for comparison
    bh_curve = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax2.plot(results_df.index, bh_curve, 'g--', alpha=0.6, label='Buy & Hold')
    
    ax2.set_title('Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value (USDT)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # --- Add Vertical Lines for Entries/Flips ---
    for date in entry_dates:
        # Determine color based on the exposure *after* the change
        current_exposure = exposure.loc[date]
        line_color = 'red' if current_exposure > 0 else 'blue'
        
        # Add the vertical line to both plots
        ax1.axvline(x=date, color=line_color, linestyle='--', linewidth=1, alpha=0.7)
        ax2.axvline(x=date, color=line_color, linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def start_web_server(results_df):
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        final_val = results_df['Portfolio_Value'].iloc[-1]
        total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
        
        # Build Table Rows (Reverse Chronological)
        table_rows = ""
        df_rev = results_df.sort_index(ascending=False)
        
        for date, row in df_rev.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            close = f"${row['close']:,.2f}"
            exposure_val = row['Exposure']
            
            # Formatting Exposure and Color
            if exposure_val > 0:
                exp_str = f"LONG {exposure_val*100:.1f}%"
                row_color = "color: #C0392B; font-weight: bold;" # Darker Red
            elif exposure_val < 0:
                exp_str = f"SHORT {abs(exposure_val*100):.1f}%"
                row_color = "color: #2980B9; font-weight: bold;" # Darker Blue
            else:
                exp_str = "NEUTRAL 0%"
                row_color = "color: #7f8c8d;" # Gray
            
            pnl = f"${row['Daily_PnL']:,.2f}"
            equity = f"${row['Portfolio_Value']:,.2f}"
            
            table_rows += f"""
            <tr>
                <td>{date_str}</td>
                <td>{close}</td>
                <td style="{row_color}">{exp_str}</td>
                <td>{pnl}</td>
                <td>{equity}</td>
            </tr>
            """

        html_template = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conviction Backtest</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; text-align: center; color: #333; }}
                .stats {{ margin: 20px auto; padding: 20px; background: #f9f9f9; max-width: 800px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                img {{ max-width: 95%; height: auto; border: 1px solid #ccc; margin-bottom: 30px; }}
                h2 {{ margin-top: 40px; }}
                .table-container {{ margin: 0 auto; max-width: 900px; max-height: 500px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #eee; position: sticky; top: 0; padding: 12px; border-bottom: 2px solid #ccc; }}
                td {{ padding: 10px; border-bottom: 1px solid #eee; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>Conviction Strategy Results</h1>
            
            <div class="stats">
                <p><strong>Initial Capital:</strong> ${INITIAL_CAPITAL:,.2f} | <strong>Final Value:</strong> ${final_val:,.2f}</p>
                <p><strong>Total Return:</strong> <span style="font-size: 1.2em; color: {'green' if total_ret > 0 else '#C0392B'};">{total_ret:.2f}%</span></p>
            </div>
            
            <img src="/plot" />
            
            <h2>Daily Position Log</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Close Price</th>
                            <th>Position / Conviction</th>
                            <th>Daily PnL</th>
                            <th>Total Equity</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        '''
        return html_template
    
    @app.route('/plot')
    def plot():
        return send_file(create_equity_plot(results_df), mimetype='image/png')
    
    print("\n" + "=" * 60)
    print("Server running on http://localhost:8080")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    df_data = fetch_binance_data()
    
    if df_data.empty:
        print("No data fetched.")
    else:
        df_signals = generate_signals(df_data)
        
        if df_signals.empty or len(df_signals) < 10:
             print("Not enough signals generated.")
        else:
            ret, res = run_conviction_backtest(df_data, df_signals)
            
            print(f"\nFinal Portfolio: ${res['Portfolio_Value'].iloc[-1]:,.2f}")
            print(f"Strategy Return: {ret*100:.2f}%")
            
            start_web_server(res)
