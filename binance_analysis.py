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
import base64

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
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

    # --- Store ORIGINAL (non-decayed) signals before shifting ---
    df_signals_raw = df_signals.copy()

    # --- Shift signals forward for backtesting ---
    df_signals = df_signals.shift(1)
    df_signals.fillna(0, inplace=True)
    df_signals = df_signals.astype(int)

    return df_signals, df_signals_raw


# --- Backtest implementation ---
def run_conviction_backtest(df_data, df_signals, horizon):
    """
    Runs the backtest with a specific horizon decay parameter.
    """
    common_idx = df_data.index.intersection(df_signals.index)
    df = df_data.loc[common_idx].copy()
    signals = df_signals.loc[common_idx]

    num_days = len(df)
    daily_returns = df['return'].values 
    dates = df.index

    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    conviction_norm = np.zeros(num_days)
    
    # Store detailed contributions: Shape [Days, Num_Signals]
    daily_contributions = np.zeros((num_days, len(WINNING_SIGNALS)))

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
                    # Use the passed horizon parameter
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
            pnl_amt = portfolio[t-1] * exposure * daily_returns[t]
            portfolio[t] = portfolio[t-1] + pnl_amt
            daily_pnl[t] = pnl_amt
        else:
            portfolio[t] = INITIAL_CAPITAL

        conviction_norm[t] = exposure

    results = df[['close', 'return']].copy()
    results['Exposure'] = conviction_norm
    results['Daily_PnL'] = daily_pnl
    results['Portfolio_Value'] = portfolio
    
    # Calculate Strategy Daily Return for Sharpe
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)

    # Add contribution columns to results for detailed analysis
    signal_names = [f"{s[0]}_{s[1]}" for s in WINNING_SIGNALS]
    for i, name in enumerate(signal_names):
        results[f"Contrib_{name}"] = daily_contributions[:, i]
    
    total_return = (portfolio[-1] / portfolio[0]) - 1.0
    return total_return, results, signal_names

def run_grid_search(df_data, df_signals):
    print("\nRunning Grid Search for Optimal Horizon (10-400 days)...")
    results = []
    
    # Test range from 10 to 400 in steps of 10
    search_space = range(10, 401, 10)
    
    for h in search_space:
        ret, res, _ = run_conviction_backtest(df_data, df_signals, horizon=h)
        
        # Calculate Sharpe
        daily_rets = res['Strategy_Daily_Return']
        if daily_rets.std() > 0:
            sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(365)
        else:
            sharpe = 0.0
            
        results.append({
            'Horizon': h,
            'Return': ret,
            'Sharpe': sharpe
        })
        print(f"Horizon {h}: Sharpe {sharpe:.2f} | Return {ret*100:.1f}%", end='\r')
        
    print("\nGrid Search Complete.")
    df_grid = pd.DataFrame(results)
    return df_grid

def calculate_net_daily_signal_event(df_signals_raw, results_df):
    aligned_signals = df_signals_raw.loc[results_df.index.min():results_df.index.max()]
    net_signal_sum = aligned_signals.sum(axis=1)
    long_signal_dates = net_signal_sum[net_signal_sum > 0].index
    short_signal_dates = net_signal_sum[net_signal_sum < 0].index
    return long_signal_dates, short_signal_dates


def create_equity_plot(results_df, long_signal_dates, short_signal_dates, horizon):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Plot 1: Price Chart ---
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1.5, label='BTC Price')
    ax1.set_title(f'BTC/USDT Price (Horizon: {horizon})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # --- Plot 2: Equity Curve ---
    ax2.plot(results_df.index, results_df['Portfolio_Value'], 'k-', linewidth=1.5, label='Strategy Equity')
    bh_curve = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax2.plot(results_df.index, bh_curve, 'g--', alpha=0.6, label='Buy & Hold')
    
    ax2.set_title('Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value (USDT)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # --- Add Vertical Lines ---
    for date in long_signal_dates:
        ax1.axvline(x=date, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=date, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
    for date in short_signal_dates:
        ax1.axvline(x=date, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=date, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def create_grid_plot(df_grid):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Horizon (Days)')
    ax1.set_ylabel('Sharpe Ratio', color=color, fontweight='bold')
    ax1.plot(df_grid['Horizon'], df_grid['Sharpe'], color=color, linewidth=2, marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:gray'
    ax2.set_ylabel('Total Return', color=color, fontweight='bold')  # we already handled the x-label with ax1
    ax2.plot(df_grid['Horizon'], df_grid['Return'], color=color, linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Grid Search: Horizon vs Performance")
    fig.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def start_web_server(results_df, long_signal_dates, short_signal_dates, signal_names, df_grid, best_horizon):
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

        # --- Grid Search Table (Top 5) ---
        top_5 = df_grid.sort_values(by='Sharpe', ascending=False).head(5)
        grid_rows = ""
        for _, row in top_5.iterrows():
            is_best = int(row['Horizon']) == best_horizon
            style = "background-color: #d4edda;" if is_best else ""
            grid_rows += f"""
            <tr style="{style}">
                <td>{int(row['Horizon'])}</td>
                <td>{row['Sharpe']:.2f}</td>
                <td>{row['Return']*100:.1f}%</td>
            </tr>
            """

        # --- Calculate Monthly Sharpe & Returns ---
        monthly_rows = ""
        monthly_groups = results_df.groupby(pd.Grouper(freq='M'))
        monthly_stats = []
        for name, group in monthly_groups:
            if len(group) < 5: continue
            
            m_daily_rets = group['Strategy_Daily_Return']
            m_start_val = group['Portfolio_Value'].iloc[0]
            m_end_val = group['Portfolio_Value'].iloc[-1]
            m_ret_total = (m_end_val / m_start_val) - 1.0
            
            if m_daily_rets.std() > 0:
                m_sharpe = (m_daily_rets.mean() / m_daily_rets.std()) * np.sqrt(365)
            else:
                m_sharpe = 0.0
                
            monthly_stats.append({
                'Date': name.strftime('%Y-%m'),
                'Return': m_ret_total,
                'Sharpe': m_sharpe
            })
            
        for stats in reversed(monthly_stats):
            ret_val = stats['Return']
            sharpe_val = stats['Sharpe']
            ret_color = "green" if ret_val > 0 else "red"
            sharpe_color = "green" if sharpe_val > 1 else ("orange" if sharpe_val > 0 else "red")
            monthly_rows += f"""
            <tr>
                <td>{stats['Date']}</td>
                <td style="color: {ret_color}; font-weight: bold;">{ret_val*100:.2f}%</td>
                <td style="color: {sharpe_color};">{sharpe_val:.2f}</td>
            </tr>
            """

        # --- Attribution Table (First 30 Days) ---
        first_month_df = results_df.iloc[:30]
        attribution_headers = "".join([f"<th>{name}</th>" for name in signal_names])
        attribution_rows = ""
        
        for date, row in first_month_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            exposure = row['Exposure']
            
            if exposure > 0: exp_style = "color: #C0392B; font-weight: bold;"
            elif exposure < 0: exp_style = "color: #2980B9; font-weight: bold;"
            else: exp_style = "color: #7f8c8d;"
                 
            sig_cells = ""
            for name in signal_names:
                val = row[f"Contrib_{name}"]
                if val > 0: color = "background-color: #ffe6e6; color: #C0392B;" 
                elif val < 0: color = "background-color: #e6f2ff; color: #2980B9;" 
                else: color = "color: #ccc;"
                sig_cells += f"<td style='{color}'>{val:.2f}</td>"
            
            attribution_rows += f"<tr><td>{date_str}</td><td style='{exp_style}'>{exposure:.2f}</td>{sig_cells}</tr>"

        # --- Daily Log Table ---
        table_rows = ""
        df_rev = results_df.sort_index(ascending=False)
        for date, row in df_rev.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            exposure_val = row['Exposure']
            if exposure_val > 0:
                exp_str = f"LONG {exposure_val*100:.1f}%"
                row_color = "color: #C0392B; font-weight: bold;"
            elif exposure_val < 0:
                exp_str = f"SHORT {abs(exposure_val*100):.1f}%"
                row_color = "color: #2980B9; font-weight: bold;"
            else:
                exp_str = "NEUTRAL 0%"
                row_color = "color: #7f8c8d;"
            
            table_rows += f"""
            <tr>
                <td>{date_str}</td>
                <td>${row['close']:,.2f}</td>
                <td style="{row_color}">{exp_str}</td>
                <td>${row['Daily_PnL']:,.2f}</td>
                <td>${row['Portfolio_Value']:,.2f}</td>
            </tr>
            """

        html_template = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conviction Strategy Results</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; text-align: center; color: #333; }}
                .stats {{ margin: 20px auto; padding: 20px; background: #f9f9f9; max-width: 900px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                img {{ max-width: 95%; height: auto; border: 1px solid #ccc; margin-bottom: 20px; }}
                h2 {{ margin-top: 40px; border-bottom: 2px solid #eee; display: inline-block; padding-bottom: 5px; }}
                .table-container {{ margin: 0 auto; max-width: 1000px; max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                th {{ background: #eee; position: sticky; top: 0; padding: 10px; border-bottom: 2px solid #ccc; }}
                td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .grid-section {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-bottom: 40px; }}
            </style>
        </head>
        <body>
            <h1>Conviction Strategy Results</h1>
            
            <div class="stats">
                <p>
                    <strong>Selected Horizon:</strong> {best_horizon} days |
                    <strong>Initial Capital:</strong> ${INITIAL_CAPITAL:,.2f} | 
                    <strong>Final Value:</strong> ${final_val:,.2f} | 
                    <strong>Return:</strong> <span style="color: {'green' if total_ret > 0 else 'red'};">{total_ret:.2f}%</span>
                </p>
                <p>
                    <strong>Overall Sharpe Ratio:</strong> <span style="font-size: 1.2em; font-weight: bold;">{overall_sharpe:.2f}</span>
                </p>
            </div>
            
            <h2>Grid Search Analysis</h2>
            <div class="grid-section">
                <div>
                    <img src="/grid_plot" style="max-width: 600px;" />
                </div>
                <div class="table-container" style="max-width: 300px; height: auto;">
                    <h3>Top 5 Settings</h3>
                    <table>
                        <thead>
                            <tr><th>Horizon</th><th>Sharpe</th><th>Return</th></tr>
                        </thead>
                        <tbody>{grid_rows}</tbody>
                    </table>
                </div>
            </div>

            <h2>Performance (Horizon: {best_horizon})</h2>
            <img src="/plot" />

            <h2>Monthly Performance</h2>
            <div class="table-container" style="max-width: 600px;">
                <table>
                    <thead><tr><th>Month</th><th>Total Return</th><th>Sharpe Ratio</th></tr></thead>
                    <tbody>{monthly_rows}</tbody>
                </table>
            </div>

            <h2>Attribution (First 30 Days)</h2>
            <div class="table-container">
                <table class="attrib-table">
                    <thead><tr><th>Date</th><th>Total Exp</th>{attribution_headers}</tr></thead>
                    <tbody>{attribution_rows}</tbody>
                </table>
            </div>
            
            <h2>Daily Position Log</h2>
            <div class="table-container">
                <table>
                    <thead><tr><th>Date</th><th>Close</th><th>Position</th><th>PnL</th><th>Equity</th></tr></thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </body>
        </html>
        '''
        return html_template
    
    @app.route('/plot')
    def plot():
        buf = create_equity_plot(results_df, long_signal_dates, short_signal_dates, best_horizon)
        return send_file(buf, mimetype='image/png')
    
    @app.route('/grid_plot')
    def grid_plot():
        buf = create_grid_plot(df_grid)
        return send_file(buf, mimetype='image/png')
    
    print("\n" + "=" * 60)
    print(f"Server running on http://localhost:8080 (Showing Best Horizon: {best_horizon})")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    df_data = fetch_binance_data()
    
    if df_data.empty:
        print("No data fetched.")
    else:
        df_signals, df_signals_raw = generate_signals(df_data)
        
        if df_signals.empty or len(df_signals) < 10:
             print("Not enough signals generated.")
        else:
            # 1. Run Grid Search
            df_grid = run_grid_search(df_data, df_signals)
            
            # 2. Find Best Horizon (Max Sharpe)
            best_row = df_grid.loc[df_grid['Sharpe'].idxmax()]
            best_h = int(best_row['Horizon'])
            print(f"\nOPTIMAL FOUND: Horizon {best_h} days (Sharpe: {best_row['Sharpe']:.2f})")
            
            # 3. Run Final Backtest with Best Horizon
            ret, res, sig_names = run_conviction_backtest(df_data, df_signals, horizon=best_h)
            
            long_dates, short_dates = calculate_net_daily_signal_event(df_signals_raw, res)
            
            print(f"Final Portfolio: ${res['Portfolio_Value'].iloc[-1]:,.2f}")
            print(f"Strategy Return: {ret*100:.2f}%")
            
            start_web_server(res, long_dates, short_dates, sig_names, df_grid, best_h)
