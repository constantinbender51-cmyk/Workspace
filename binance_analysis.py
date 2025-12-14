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
LEVERAGE = 1.0 # Fixed 1x leverage (no external multiplier)

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
SIGNAL_COLORS = ['#3498DB', '#E67E22', '#F1C40F', '#2ECC71', '#E74C3C'] # For visual consistency, though plots are independent

# --- Global Storage for Results ---
# Will store: [{'name': str, 'sharpe': float, 'return': float, 'df': DataFrame, 'contributions': np.array}]
INDIVIDUAL_RESULTS = []
GLOBAL_DF_DATA = None 

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

# --- Backtest Runner for Single Signal (Bi-directional, 1x Exposure) ---
def run_single_signal_backtest(df_data, df_signals_all, signal_index, signal_name):
    """Runs decay and backtest for one signal treated as a standalone system."""
    
    # 1. Pre-calculate Decay Contribution (Matrix will be N_DAYS x 1)
    signal_column = df_signals_all.iloc[:, signal_index:signal_index+1]
    
    common_idx = df_data.index.intersection(signal_column.index)
    signals = signal_column.loc[common_idx]
    df = df_data.loc[common_idx].copy()
    num_days = len(common_idx)
    returns = df['return'].values
    
    contributions = np.zeros(num_days)
    signal_start_day = -1
    signal_direction = 0

    # Calculate Decay
    for t in range(num_days):
        current_sig = signals.iloc[t, 0]
        
        if current_sig != 0:
            signal_start_day = t
            signal_direction = current_sig
        
        if signal_direction != 0:
            d = t - signal_start_day
            decay = max(0.0, 1.0 - (d / HORIZON)) if d >= 0 else 0.0
            contributions[t] = signal_direction * decay
            
            if decay == 0.0:
                signal_direction = 0
                signal_start_day = -1
                
    # 2. Exposure Logic (Bi-directional, 1x Exposure)
    
    raw_conviction = contributions 
    
    # Calculate Exposure: Raw Conviction * Fixed 1x Leverage
    exposure = raw_conviction * LEVERAGE
    # Clamp exposure to the leverage limits (-1.0x to +1.0x)
    exposure = np.clip(exposure, -LEVERAGE, LEVERAGE) 
    
    # 3. Run PnL Loop
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    portfolio[0] = INITIAL_CAPITAL
    is_bankrupt = False
    
    for t in range(1, num_days):
        exposure_t = exposure[t]
        
        if not is_bankrupt:
            # PnL = Previous Equity * Exposure * Return
            pnl = portfolio[t-1] * exposure_t * returns[t]
            portfolio[t] = portfolio[t-1] + pnl
            daily_pnl[t] = pnl
            
            if portfolio[t] <= 0:
                portfolio[t] = 0
                is_bankrupt = True
        else:
            portfolio[t] = 0
            
    # 4. Compile Results
    results = df[['close', 'return']].copy()
    results['Exposure'] = exposure
    results['Raw_Conviction'] = raw_conviction
    results['Portfolio_Value'] = portfolio
    results['Daily_PnL'] = daily_pnl
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)
    
    # Metrics
    total_return = (portfolio[-1] / portfolio[0]) - 1.0
    d_rets = results['Strategy_Daily_Return']
    sharpe = (d_rets.mean()/d_rets.std())*np.sqrt(365) if d_rets.std()>0 else 0
    
    return {'name': signal_name, 'sharpe': sharpe, 'return': total_return, 'df': results, 'contributions': contributions}

def create_single_equity_plot(result_entry, plot_index):
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    results_df = result_entry['df']
    
    # Highlight Short periods (when Raw_Conviction < 0)
    ax_ymin, ax_ymax = 0.0, 1.0
    ax.fill_between(results_df.index, ax_ymin, ax_ymax, where=results_df['Raw_Conviction'] < 0, 
                     color='#FFC0CB', alpha=0.3, transform=ax.get_xaxis_transform(), label='Short Position')
                     
    # Plot Strategy Equity
    ax.plot(results_df.index, results_df['Portfolio_Value'], color=SIGNAL_COLORS[plot_index % len(SIGNAL_COLORS)], 
            linewidth=1.5, label=f'Strategy (Sharpe: {result_entry["sharpe"]:.2f})')
    
    # Plot Buy & Hold
    bh = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax.plot(results_df.index, bh, 'g--', alpha=0.8, linewidth=1, label='Buy & Hold')
    
    if (results_df['Portfolio_Value'] <= 0).any(): ax.set_yscale('symlog', linthresh=1.0)
    else: ax.set_yscale('log')
    
    ax.set_title(f'Equity Curve: {result_entry["name"]} (Bi-Dir {LEVERAGE:.0f}x)', fontweight='bold')
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
    
    # --- Generate Stats Table ---
    stats_rows = ""
    for i, res in enumerate(all_results):
        sharpe_color = "green" if res['sharpe'] > 0 else "red"
        ret_color = "green" if res['return'] > 0 else "red"
        stats_rows += f"""
        <tr>
            <td>{res['name']}</td>
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
        <head><title>Individual Signal Backtests</title>
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
            <h1>Independent Single-Signal Performance</h1>
            <p>Each signal is tested individually with Bi-Directional Trading (Long/Short) and 1x Exposure.</p>
            
            <table class="stats-table summary-table">
                <thead>
                    <tr>
                        <th style="width: 20%;">Signal</th>
                        <th style="width: 10%;">Sharpe Ratio</th>
                        <th style="width: 15%;">Total Return</th>
                        <th style="width: 55%;">Equity Curve (Pink Background = Short Position)</th>
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
    
    print("\n--- Running Independent Backtests ---")
    for i, sig_name in enumerate(SIGNAL_NAMES):
        print(f"Testing Signal {i+1}/{N_SIGNALS}: {sig_name}...")
        result = run_single_signal_backtest(df_data, df_signals, i, sig_name)
        all_results.append(result)
        
    start_web_server(all_results)
