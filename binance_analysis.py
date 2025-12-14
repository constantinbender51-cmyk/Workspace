import ccxt
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker # Import ticker for safe formatting
from flask import Flask, send_file
import io
import threading
from matplotlib.figure import Figure

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
SINCE_STR = '2018-01-01 00:00:00'
INITIAL_CAPITAL = 10000.0
LEVERAGE = 1.0 # Fixed 1x leverage

# Grid Search Config
HORIZON_START = 5
HORIZON_END = 800
HORIZON_STEP = 5

# --- Signals (now referred to as ALL_SIGNALS) ---
ALL_SIGNALS = [
    ('EMA_CROSS', 50, 150, 0),         # 0: EMA 50/150
    ('PRICE_SMA', 380, 0, 0),          # 1: Price/SMA 380
    ('PRICE_SMA', 140, 0, 0),          # 2: Price/SMA 140
    ('MACD_CROSS', 12, 26, 15),        # 3: MACD (12/26/15)
    ('RSI_CROSS', 35, 0, 0),           # 4: RSI 35 (Crossover)
]

N_SIGNALS = len(ALL_SIGNALS)
SIGNAL_NAMES = [f"{s[0]}_{s[1]}_{s[2]}" for s in ALL_SIGNALS]
SIGNAL_COLORS = ['#3498DB', '#E67E22', '#F1C40F', '#2ECC71', '#E74C3C'] 

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

    for sig_type, p1, p2, p3 in ALL_SIGNALS:
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

    df_signals_shifted = df_signals.shift(1)
    df_signals_shifted.fillna(0, inplace=True)
    df_signals_shifted = df_signals_shifted.astype(int)

    return df_signals_shifted 

# --- Optimization & Backtest Logic ---

def calculate_contributions(signals_vec, horizon, num_days):
    """Calculates decay array for a given horizon."""
    contributions = np.zeros(num_days)
    signal_start_day = -1
    signal_direction = 0

    # Optimization: using a simple loop is fast enough for 2500 days
    # Vectorizing the inner logic of 'stateful decay' is complex, 
    # but since N is small, JIT/Cython isn't strictly needed for grid search.
    for t in range(num_days):
        current_sig = signals_vec[t]
        
        if current_sig != 0:
            signal_start_day = t
            signal_direction = current_sig
        
        if signal_direction != 0:
            d = t - signal_start_day
            if d < 0: 
                decay = 0.0
            else:
                # Linear decay
                decay = max(0.0, 1.0 - (d / horizon))
            
            contributions[t] = signal_direction * decay
            
            if decay == 0.0:
                signal_direction = 0
                signal_start_day = -1
    return contributions

def optimize_signal_horizon(df_data, df_signals_all, signal_index, signal_name):
    """Grid searches for the best horizon for a single signal."""
    
    # Extract data for this signal once
    signal_column = df_signals_all.iloc[:, signal_index:signal_index+1]
    common_idx = df_data.index.intersection(signal_column.index)
    
    # Working data
    signals_series = signal_column.loc[common_idx].iloc[:, 0]
    signals_vec = signals_series.values
    
    df_slice = df_data.loc[common_idx].copy()
    returns_vec = df_slice['return'].values
    num_days = len(common_idx)
    
    best_sharpe = -999.0
    best_horizon = HORIZON_START
    best_results = None
    
    horizons = range(HORIZON_START, HORIZON_END + 1, HORIZON_STEP)
    
    print(f"Optimizing {signal_name}: ", end="", flush=True)
    
    for h in horizons:
        # 1. Calculate Decay
        contributions = calculate_contributions(signals_vec, h, num_days)
        
        # 2. Exposure (Bi-directional, 1x)
        exposure = contributions * LEVERAGE
        exposure = np.clip(exposure, -LEVERAGE, LEVERAGE)
        
        # 3. Fast PnL for Optimization (Vectorized Approx)
        
        # Vectorized simulation for speed
        # Padding first element (t=0) as 0 return
        strat_daily_rets = exposure * returns_vec
        strat_daily_rets[0] = 0.0
        
        # Check std dev to avoid div by zero
        std_dev = np.std(strat_daily_rets)
        if std_dev > 0:
            sharpe = (np.mean(strat_daily_rets) / std_dev) * np.sqrt(365)
        else:
            sharpe = 0.0
            
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_horizon = h
            
    print(f"Best H={best_horizon}, Sharpe={best_sharpe:.2f}")

    # --- Re-run with Best Horizon to get Full Data ---
    final_contributions = calculate_contributions(signals_vec, best_horizon, num_days)
    final_exposure = np.clip(final_contributions * LEVERAGE, -LEVERAGE, LEVERAGE)
    
    portfolio = np.zeros(num_days)
    portfolio[0] = INITIAL_CAPITAL
    daily_pnl = np.zeros(num_days)
    
    for t in range(1, num_days):
        pnl = portfolio[t-1] * final_exposure[t] * returns_vec[t]
        portfolio[t] = portfolio[t-1] + pnl
        daily_pnl[t] = pnl
        if portfolio[t] <= 0: portfolio[t] = 0
            
    # Compile Final DF
    results = df_slice[['close', 'return']].copy()
    results['Exposure'] = final_exposure
    results['Raw_Conviction'] = final_contributions
    results['Portfolio_Value'] = portfolio
    results['Daily_PnL'] = daily_pnl
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)
    
    total_return = (portfolio[-1] / portfolio[0]) - 1.0
    
    return {
        'name': signal_name,
        'horizon': best_horizon,
        'sharpe': best_sharpe,
        'return': total_return,
        'df': results
    }

def create_single_equity_plot(result_entry, plot_index):
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    results_df = result_entry['df']
    h = result_entry['horizon']
    
    # Highlight Short periods
    ax_ymin, ax_ymax = 0.0, 1.0
    ax.fill_between(results_df.index, ax_ymin, ax_ymax, where=results_df['Raw_Conviction'] < 0, 
                     color='#FFC0CB', alpha=0.3, transform=ax.get_xaxis_transform(), label='Short Position')
                     
    ax.plot(results_df.index, results_df['Portfolio_Value'], color=SIGNAL_COLORS[plot_index % len(SIGNAL_COLORS)], 
            linewidth=1.5, label=f'Strategy (Sharpe: {result_entry["sharpe"]:.2f})')
    
    bh = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax.plot(results_df.index, bh, 'g--', alpha=0.8, linewidth=1, label='Buy & Hold')
    
    if (results_df['Portfolio_Value'] <= 0).any(): 
        ax.set_yscale('symlog', linthresh=1.0)
    else: 
        ax.set_yscale('log')
        
    # --- FIX: Safe Ticker Formatting for Log Scale ---
    # This overrides the default Matplotlib log formatter which uses math text (LaTeX)
    # and prevents the ParseFatalException seen in some environments.
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    
    ax.set_title(f'Equity Curve: {result_entry["name"]} (Opt Horizon: {h} days)', fontweight='bold')
    ax.set_ylabel('Portfolio Value (Log Scale)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

def start_web_server(all_results):
    app = Flask(__name__)
    
    stats_rows = ""
    for i, res in enumerate(all_results):
        sharpe_color = "green" if res['sharpe'] > 0 else "red"
        ret_color = "green" if res['return'] > 0 else "red"
        stats_rows += f"""
        <tr>
            <td>{res['name']}</td>
            <td><strong>{res['horizon']} days</strong></td>
            <td style='color: {sharpe_color}; font-weight: bold;'>{res['sharpe']:.2f}</td>
            <td style='color: {ret_color}; font-weight: bold;'>{res['return']*100:.2f}%</td>
            <td><img src="/plot/{i}" width="100%"></td>
        </tr>
        """

    @app.route('/')
    def index():
        ts = int(time.time())
        return f'''
        <html>
        <head><title>Optimized Signal Backtests</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; margin: 40px; color: #333; }}
            h1 {{ margin-bottom: 30px; }}
            .stats-table {{ margin: 0 auto; border-collapse: collapse; max-width: 1200px; }}
            .stats-table th, .stats-table td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            .stats-table th {{ background: #f0f0f0; }}
            .summary-table {{ width: 100%; }}
        </style>
        </head>
        <body>
            <h1>Optimized Single-Signal Performance</h1>
            <p>Grid Search performed for Decay Horizon (5 to 400 days). Bi-Directional Trading, 1x Leverage.</p>
            
            <table class="stats-table summary-table">
                <thead>
                    <tr>
                        <th style="width: 15%;">Signal</th>
                        <th style="width: 10%;">Optimal Horizon</th>
                        <th style="width: 10%;">Sharpe Ratio</th>
                        <th style="width: 10%;">Total Return</th>
                        <th style="width: 55%;">Equity Curve (Pink = Short)</th>
                    </tr>
                </thead>
                <tbody>
                    {stats_rows}
                </tbody>
            </table>
        </body>
        </html>
        '''

    @app.route('/plot/<int:plot_index>')
    def plot(plot_index):
        if 0 <= plot_index < len(all_results):
            try:
                buf = create_single_equity_plot(all_results[plot_index], plot_index)
                return send_file(buf, mimetype='image/png')
            except Exception as e:
                print(f"Error creating plot {plot_index}: {e}")
                return f"Error creating plot: {e}", 500
        return "Plot not found", 404

    print(f"Server running on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    df_data = fetch_binance_data()
    df_signals = generate_signals(df_data)
    
    all_results = []
    
    print(f"\n--- Running Grid Search ({HORIZON_START}-{HORIZON_END}, step {HORIZON_STEP}) ---")
    for i, sig_name in enumerate(SIGNAL_NAMES):
        result = optimize_signal_horizon(df_data, df_signals, i, sig_name)
        all_results.append(result)
        
    start_web_server(all_results)
