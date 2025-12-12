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
HORIZON = 380  # Decay window (days) - OPTIMAL VALUE
INITIAL_CAPITAL = 10000.0
LEVERAGE = 5.0  # Multiplier applied to conviction exposure
THRESHOLD_1 = 0.50
THRESHOLD_2 = 0.20

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
            rsi = rsi.fillna(50) 
            center = 50
            long_cond = (rsi.shift(1) < center) & (rsi > center)
            short_cond = (rsi.shift(1) > center) & (rsi < center)
        else:
            df_signals[col] = 0
            continue

        df_signals[col] = np.where(long_cond, 1, np.where(short_cond, -1, 0))

    df_signals_raw = df_signals.copy()

    # Shift signals forward for backtest
    df_signals = df_signals.shift(1)
    df_signals.fillna(0, inplace=True)
    df_signals = df_signals.astype(int)

    return df_signals, df_signals_raw


# --- Backtest implementation ---
def run_conviction_backtest(df_data, df_signals, threshold):
    """
    Runs the backtest with a specific Conviction Threshold.
    """
    horizon = HORIZON
    
    common_idx = df_data.index.intersection(df_signals.index)
    df = df_data.loc[common_idx].copy()
    signals = df_signals.loc[common_idx]

    num_days = len(df)
    daily_returns = df['return'].values 
    
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    conviction_norm = np.zeros(num_days)
    
    daily_contributions = np.zeros((num_days, len(WINNING_SIGNALS)))

    signal_start_day = np.full(MAX_CONVICTION, -1, dtype=int)
    signal_direction = np.zeros(MAX_CONVICTION, dtype=int)
    
    portfolio[0] = INITIAL_CAPITAL
    is_bankrupt = False

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

        # Calculate Normalized Conviction (-1 to 1)
        raw_conviction = daily_sum / MAX_CONVICTION
        
        # --- CONVICTION FILTER ---
        if abs(raw_conviction) < threshold:
            raw_conviction = 0.0
        
        # Apply LEVERAGE
        exposure = raw_conviction * LEVERAGE
        
        # Clip to max leverage
        exposure = np.clip(exposure, -LEVERAGE, LEVERAGE)

        if t > 0:
            if not is_bankrupt:
                effective_return = daily_returns[t]
                
                pnl_amt = portfolio[t-1] * exposure * effective_return
                portfolio[t] = portfolio[t-1] + pnl_amt
                daily_pnl[t] = pnl_amt
                
                if portfolio[t] <= 0:
                    portfolio[t] = 0
                    daily_pnl[t] = -portfolio[t-1]
                    is_bankrupt = True
            else:
                portfolio[t] = 0
                daily_pnl[t] = 0
                
        else:
            portfolio[t] = INITIAL_CAPITAL

        conviction_norm[t] = exposure
        
    results = df[['close', 'return']].copy()
    results['Exposure'] = conviction_norm
    results['Daily_PnL'] = daily_pnl
    results['Portfolio_Value'] = portfolio
    
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)
    
    rolling_max = results['Portfolio_Value'].cummax()
    # Handle division by zero for bankruptcy
    results['Drawdown'] = np.where(rolling_max > 0, (results['Portfolio_Value'] - rolling_max) / rolling_max, -1.0)

    signal_names = [f"{s[0]}_{s[1]}" for s in WINNING_SIGNALS]
    for i, name in enumerate(signal_names):
        results[f"Contrib_{name}"] = daily_contributions[:, i]
    
    total_return = (portfolio[-1] / portfolio[0]) - 1.0
    
    # Calculate Sharpe Ratio
    daily_rets = results['Strategy_Daily_Return']
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(365)
    else:
        sharpe = 0.0
    
    return total_return, sharpe, results, signal_names

def run_comparison_backtests(df_data, df_signals):
    # Run for Threshold 0.50 (Tighter Filter)
    ret_50, sharpe_50, res_50, sig_names = run_conviction_backtest(df_data, df_signals, threshold=THRESHOLD_1)
    
    # Run for Threshold 0.20 (Looser Filter)
    ret_20, sharpe_20, res_20, _ = run_conviction_backtest(df_data, df_signals, threshold=THRESHOLD_2)
    
    return {
        'main_result': res_50,
        'main_sharpe': sharpe_50,
        'main_ret': ret_50,
        'sig_names': sig_names,
        'comp_result_050': res_50['Portfolio_Value'],
        'comp_result_020': res_20['Portfolio_Value'],
        'comp_sharpe_020': sharpe_20,
    }

def calculate_net_daily_signal_event(df_signals_raw, results_df):
    aligned_signals = df_signals_raw.loc[results_df.index.min():results_df.index.max()]
    net_signal_sum = aligned_signals.sum(axis=1)
    long_signal_dates = net_signal_sum[net_signal_sum > 0].index
    short_signal_dates = net_signal_sum[net_signal_sum < 0].index
    return long_signal_dates, short_signal_dates


def create_equity_plot(results_df, long_signal_dates, short_signal_dates, horizon, threshold):
    # Use explicitly created Figure for thread safety in Flask
    fig = Figure(figsize=(12, 14))
    
    # Create subplots manually
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    
    # --- Plot 1: Price Chart (Log Scale) ---
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1, label='BTC Price')
    ax1.set_yscale('log')
    ax1.set_title(f'BTC/USDT Price (Log Scale) - Filter: {threshold:.2f}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc='upper left')

    # Signals
    long_prices = results_df.loc[long_signal_dates, 'close']
    short_prices = results_df.loc[short_signal_dates, 'close']
    ax1.scatter(long_signal_dates, long_prices, marker='^', color='green', s=50, label='Long Signal', zorder=5)
    ax1.scatter(short_signal_dates, short_prices, marker='v', color='red', s=50, label='Short Signal', zorder=5)
    
    # --- Plot 2: Equity Curve (Log Scale) ---
    ax2.plot(results_df.index, results_df['Portfolio_Value'], 'b-', linewidth=1.5, label=f'Strategy Equity ({LEVERAGE}x Lev, Filter {threshold:.2f})')
    
    # Calculate B&H curve
    bh_curve = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax2.plot(results_df.index, bh_curve, 'g--', alpha=0.8, linewidth=1, label='Buy & Hold (1x)')
    
    if (results_df['Portfolio_Value'] <= 0).any():
        ax2.set_yscale('symlog', linthresh=1.0) 
    else:
        ax2.set_yscale('log')
        
    ax2.set_title(f'Strategy Equity (Main Result) - {LEVERAGE}x Leverage', fontsize=12, fontweight='bold')
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

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

def create_comparison_plot(comp_result_050, comp_result_020, comp_sharpe_020, main_result_close):
    # Use explicitly created Figure
    fig = Figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    
    ax1.plot(comp_result_050.index, comp_result_050, 'b-', linewidth=2, label=f'Threshold 0.50 (Main)')
    ax1.plot(comp_result_020.index, comp_result_020, 'r--', linewidth=1.5, alpha=0.7, label=f'Threshold 0.20 (Sharpe {comp_sharpe_020:.2f})')
    
    # Calculate B&H curve
    bh_curve = main_result_close / main_result_close.iloc[0] * INITIAL_CAPITAL
    ax1.plot(bh_curve.index, bh_curve.values, 'g-', linewidth=1, alpha=0.8, label='Buy & Hold (1x)')

    if (comp_result_050.min() <= 0) or (comp_result_020.min() <= 0):
        ax1.set_yscale('symlog', linthresh=1.0) 
    else:
        ax1.set_yscale('log')

    ax1.set_title(f'Equity Comparison: Threshold 0.50 vs 0.20 ({LEVERAGE}x Leverage)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($) - Log Scale', fontsize=10)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc='upper left')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

def start_web_server(backtest_data, comp_sharpe_020, df_signals_raw):
    app = Flask(__name__)
    
    results_df = backtest_data['main_result']
    sig_names = backtest_data['sig_names']
    long_signal_dates, short_signal_dates = calculate_net_daily_signal_event(df_signals_raw, results_df)
    
    # Stats logic...
    best_threshold = THRESHOLD_1
    final_val = results_df['Portfolio_Value'].iloc[-1]
    total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
    strat_daily_rets = results_df['Strategy_Daily_Return']
    overall_sharpe = (strat_daily_rets.mean() / strat_daily_rets.std()) * np.sqrt(365) if strat_daily_rets.std() > 0 else 0.0
    is_bust = (final_val <= 0)

    # ... [HTML building code remains same as before, omitted for brevity but included in full file output] ...
    
    # --- Reconstructing the HTML string for completeness in file output ---
    monthly_rows = ""
    monthly_groups = results_df.groupby(pd.Grouper(freq='M'))
    monthly_stats = []
    for name, group in monthly_groups:
        if len(group) < 5: continue
        m_daily_rets = group['Strategy_Daily_Return']
        m_start_val = group['Portfolio_Value'].iloc[0]
        m_end_val = group['Portfolio_Value'].iloc[-1]
        m_ret_total = (m_end_val / m_start_val) - 1.0 if m_start_val > 0 else 0.0
        m_sharpe = (m_daily_rets.mean() / m_daily_rets.std()) * np.sqrt(365) if m_daily_rets.std() > 0 else 0.0
        monthly_stats.append({'Date': name.strftime('%Y-%m'), 'Return': m_ret_total, 'Sharpe': m_sharpe})
        
    for stats in reversed(monthly_stats):
        ret_val = stats['Return']
        sharpe_val = stats['Sharpe']
        ret_color = "green" if ret_val > 0 else "red"
        sharpe_color = "green" if sharpe_val > 1 else ("orange" if sharpe_val > 0 else "red")
        monthly_rows += f"<tr><td>{stats['Date']}</td><td style='color: {ret_color}; font-weight: bold;'>{ret_val*100:.2f}%</td><td style='color: {sharpe_color};'>{sharpe_val:.2f}</td></tr>"

    attribution_headers = "".join([f"<th>{name}</th>" for name in sig_names])
    attribution_rows = ""
    for date, row in results_df.iloc[:30].iterrows():
        date_str = date.strftime('%Y-%m-%d')
        exposure = row['Exposure']
        exp_style = "color: #C0392B; font-weight: bold;" if exposure > 0 else ("color: #2980B9; font-weight: bold;" if exposure < 0 else "color: #7f8c8d;")
        sig_cells = ""
        for name in sig_names:
            val = row[f"Contrib_{name}"]
            color = "background-color: #ffe6e6; color: #C0392B;" if val > 0 else ("background-color: #e6f2ff; color: #2980B9;" if val < 0 else "color: #ccc;")
            sig_cells += f"<td style='{color}'>{val:.2f}</td>"
        attribution_rows += f"<tr><td>{date_str}</td><td style='{exp_style}'>{exposure:.2f}x</td>{sig_cells}</tr>"

    table_rows = ""
    for date, row in results_df.sort_index(ascending=False).iterrows():
        date_str = date.strftime('%Y-%m-%d')
        exposure_val = row['Exposure']
        if exposure_val > 0:
            exp_str, row_color = f"LONG {exposure_val:.2f}x", "color: #C0392B; font-weight: bold;"
        elif exposure_val < 0:
            exp_str, row_color = f"SHORT {abs(exposure_val):.2f}x", "color: #2980B9; font-weight: bold;"
        else:
            exp_str, row_color = "NEUTRAL 0x", "color: #7f8c8d;"
        table_rows += f"<tr><td>{date_str}</td><td>${row['close']:,.2f}</td><td style='{row_color}'>{exp_str}</td><td>${row['Daily_PnL']:,.2f}</td><td>${row['Portfolio_Value']:,.2f}</td></tr>"
    
    bust_warning = "<h2 style='color: red; background: #fee; padding: 10px; border: 1px solid red;'>⚠️ STRATEGY BANKRUPT (LIQUIDATED) ⚠️</h2>" if is_bust else ""

    @app.route('/')
    def index():
        timestamp = int(time.time())
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
                th {{ background: #eee; position: sticky; top: 0; padding: 10px; border-bottom: 1px solid #ccc; }}
                td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .comparison-section {{ margin-top: 40px; }}
            </style>
        </head>
        <body>
            <h1>Conviction Strategy Results</h1>
            {bust_warning}
            <div class="stats">
                <p><strong>Horizon:</strong> {HORIZON} days | <strong>Leverage:</strong> {LEVERAGE}x | <strong>Filter:</strong> {best_threshold:.2f}</p>
                <p><strong>Final:</strong> ${final_val:,.2f} | <strong>Return:</strong> <span style="color: {'green' if total_ret > 0 else 'red'};">{total_ret:.2f}%</span></p>
                <p><strong>Sharpe:</strong> <span style="font-size: 1.2em; font-weight: bold;">{overall_sharpe:.2f}</span></p>
            </div>
            
            <div class="comparison-section">
                <h2>Comparison (0.50 vs 0.20)</h2>
                <img src="/comparison_plot?v={timestamp}" />
            </div>

            <h2>Detailed Performance (Filter: {best_threshold:.2f})</h2>
            <img src="/plot?v={timestamp}" />

            <h2>Monthly Performance</h2>
            <div class="table-container" style="max-width: 600px;">
                <table><thead><tr><th>Month</th><th>Return</th><th>Sharpe</th></tr></thead><tbody>{monthly_rows}</tbody></table>
            </div>
            
            <h2>Daily Position Log</h2>
            <div class="table-container">
                <table><thead><tr><th>Date</th><th>Close</th><th>Position</th><th>PnL</th><th>Equity</th></tr></thead><tbody>{table_rows}</tbody></table>
            </div>
        </body>
        </html>
        '''
        return html_template
    
    @app.route('/plot')
    def plot():
        try:
            buf = create_equity_plot(results_df, long_signal_dates, short_signal_dates, HORIZON, THRESHOLD_1)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            return f"Error creating plot: {e}", 500

    @app.route('/comparison_plot')
    def comparison_plot():
        try:
            buf = create_comparison_plot(backtest_data['comp_result_050'], backtest_data['comp_result_020'], comp_sharpe_020, results_df['close'])
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            return f"Error creating comparison plot: {e}", 500
    
    print("\n" + "=" * 60)
    print(f"Server running on http://localhost:8080")
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
            # 1. Run comparison backtests
            backtest_data = run_comparison_backtests(df_data, df_signals)
            
            # 2. Extract data for server startup
            res = backtest_data['main_result']
            ret = backtest_data['main_ret']
            
            print(f"\n--- Strategy Comparison Complete ---")
            print(f"Filter 0.50 (Main): Sharpe {backtest_data['main_sharpe']:.2f} | Return {ret*100:.2f}%")
            print(f"Filter 0.20 (Comp): Sharpe {backtest_data['comp_sharpe_020']:.2f}")
            print("====================================")
            
            start_web_server(backtest_data, backtest_data['comp_sharpe_020'], df_signals_raw)
