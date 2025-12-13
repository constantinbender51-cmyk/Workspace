import ccxt
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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
LEVERAGE = 5.0  # Multiplier applied to conviction exposure
CHOP_PERIOD = 14

# --- Winning Signals ---
WINNING_SIGNALS = [
    ('EMA_CROSS', 50, 150, 0),         # EMA 50/150
    ('PRICE_SMA', 380, 0, 0),          # Price/SMA 380
    ('PRICE_SMA', 140, 0, 0),          # Price/SMA 140
    ('MACD_CROSS', 12, 26, 15),        # MACD (12/26/15)
    ('RSI_CROSS', 35, 0, 0),           # RSI 35 (Crossover)
]

MAX_CONVICTION = len(WINNING_SIGNALS)

# --- Helper: Choppiness Index ---
def calculate_choppiness(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period).sum()
    
    max_hi = high.rolling(window=period).max()
    min_lo = low.rolling(window=period).min()
    
    numerator = atr_sum / (max_hi - min_lo)
    numerator = numerator.replace(0, np.nan) 
    
    chop = 100 * np.log10(numerator) / np.log10(period)
    return chop.fillna(50)

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
    df['chop'] = calculate_choppiness(df, CHOP_PERIOD)
    
    df.dropna(subset=['return', 'chop'], inplace=True)
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
    df_signals = df_signals.shift(1)
    df_signals.fillna(0, inplace=True)
    df_signals = df_signals.astype(int)

    return df_signals, df_signals_raw

# --- Pre-calculate Raw Conviction (Optimization) ---
def precalculate_conviction(df_data, df_signals):
    common_idx = df_data.index.intersection(df_signals.index)
    signals = df_signals.loc[common_idx]
    num_days = len(common_idx)
    
    raw_conviction = np.zeros(num_days)
    daily_contributions = np.zeros((num_days, len(WINNING_SIGNALS)))
    
    signal_start_day = np.full(MAX_CONVICTION, -1, dtype=int)
    signal_direction = np.zeros(MAX_CONVICTION, dtype=int)

    for t in range(num_days):
        daily_sum = 0.0
        for i in range(len(WINNING_SIGNALS)):
            current_sig = signals.iloc[t, i]
            if current_sig != 0:
                signal_start_day[i] = t
                signal_direction[i] = current_sig
            
            if signal_direction[i] != 0:
                d = t - signal_start_day[i]
                if d < 0: decay = 0.0
                else: decay = max(0.0, 1.0 - (d / HORIZON))
                
                contribution = signal_direction[i] * decay
                daily_contributions[t, i] = contribution
                daily_sum += contribution
                
                if decay == 0.0:
                    signal_direction[i] = 0
                    signal_start_day[i] = -1
        
        raw_conviction[t] = daily_sum / MAX_CONVICTION
    
    return raw_conviction, daily_contributions, common_idx

# --- Fast Simulator for Grid Search ---
def simulate_strategy(raw_conviction, returns, chop_values, a, b):
    # Formula: 1 - ((CHOP_Norm - a)^2)^b
    chop_norm = chop_values / 100.0
    term = (chop_norm - a) ** 2
    # Safe power operation
    scale_factor = 1.0 - (term ** b)
    scale_factor = np.clip(scale_factor, 0.0, 1.0)
    
    exposure = raw_conviction * LEVERAGE * scale_factor
    exposure = np.clip(exposure, -LEVERAGE, LEVERAGE)
    
    # Vectorized PnL calculation approx (compounding requires loop but approx is ok for grid)
    # For accurate Sharpe, we need daily PnL relative to equity.
    # We can do a quick loop here as N is small (~2000 days)
    
    portfolio = np.zeros(len(returns))
    portfolio[0] = INITIAL_CAPITAL
    
    # Fast loop using Numba would be better, but standard python loop is fine for 2k days
    # To optimize further, we can just calculate log returns:
    # Strat_Ret = Exposure * Market_Ret
    # Equity = Cumprod(1 + Strat_Ret)
    
    strat_daily_ret = exposure[:-1] * returns[1:] # Shift exposure to match return t+1
    # Pad first day
    strat_daily_ret = np.insert(strat_daily_ret, 0, 0.0)
    
    # Check bankruptcy
    cum_ret = np.cumprod(1 + strat_daily_ret)
    
    if np.any(cum_ret <= 0):
        return -1.0 # Penalize bankruptcy
        
    if np.std(strat_daily_ret) == 0:
        return 0.0
        
    sharpe = (np.mean(strat_daily_ret) / np.std(strat_daily_ret)) * np.sqrt(365)
    return sharpe

# --- Grid Search ---
def run_grid_search(df_data, raw_conviction):
    print("Running Grid Search for a (0-1) and b (0.01-0.5)...")
    
    common_idx = df_data.index
    # Align data
    if len(common_idx) != len(raw_conviction):
        # Truncate to match if needed (should be aligned by precalculate_conviction)
        print("Warning: Index mismatch in grid search, aligning...")
        limit = min(len(common_idx), len(raw_conviction))
        common_idx = common_idx[:limit]
        raw_conviction = raw_conviction[:limit]
        
    returns = df_data.loc[common_idx, 'return'].values
    chop_values = df_data.loc[common_idx, 'chop'].values
    
    a_values = np.linspace(0.0, 1.0, 21) # 0, 0.05 ... 1.0
    b_values = np.linspace(0.01, 0.51, 26) # 0.01, 0.03 ... 0.5
    
    results = []
    
    best_sharpe = -999
    best_params = (0.5, 0.1) # Default
    
    heatmap_data = np.zeros((len(b_values), len(a_values)))
    
    for i, b in enumerate(b_values):
        for j, a in enumerate(a_values):
            sharpe = simulate_strategy(raw_conviction, returns, chop_values, a, b)
            heatmap_data[i, j] = sharpe
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (a, b)
    
    print(f"Grid Search Complete. Best Sharpe: {best_sharpe:.4f} | a={best_params[0]:.2f}, b={best_params[1]:.2f}")
    
    return best_params, heatmap_data, a_values, b_values

# --- Full Backtest Run (Detailed) ---
def run_final_backtest(df_data, raw_conviction, contributions, common_idx, a, b):
    df = df_data.loc[common_idx].copy()
    num_days = len(df)
    daily_returns = df['return'].values 
    chop_values = df['chop'].values
    
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    conviction_norm = np.zeros(num_days)
    scaling_factors = np.zeros(num_days)
    
    portfolio[0] = INITIAL_CAPITAL
    is_bankrupt = False
    
    # Calculate Scaling Factors Vectorized
    chop_norm = chop_values / 100.0
    term = (chop_norm - a) ** 2
    scale_vec = 1.0 - (term ** b)
    scale_vec = np.clip(scale_vec, 0.0, 1.0)
    
    for t in range(num_days):
        scale_factor = scale_vec[t]
        scaling_factors[t] = scale_factor
        
        exposure = raw_conviction[t] * LEVERAGE * scale_factor
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
        
    results = df[['close', 'return', 'chop']].copy()
    results['Exposure'] = conviction_norm
    results['Scaling_Factor'] = scaling_factors
    results['Daily_PnL'] = daily_pnl
    results['Portfolio_Value'] = portfolio
    
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)
    rolling_max = results['Portfolio_Value'].cummax()
    results['Drawdown'] = np.where(rolling_max > 0, (results['Portfolio_Value'] - rolling_max) / rolling_max, -1.0)
    
    signal_names = [f"{s[0]}_{s[1]}" for s in WINNING_SIGNALS]
    for i, name in enumerate(signal_names):
        results[f"Contrib_{name}"] = contributions[:, i]
        
    return results, signal_names


def create_heatmap_plot(heatmap_data, a_values, b_values, best_params):
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Flip Y axis for correct visualization (low b at bottom)
    sns.heatmap(heatmap_data, ax=ax, cmap='viridis', 
                xticklabels=[f"{x:.2f}" for x in a_values], 
                yticklabels=[f"{y:.2f}" for y in b_values])
    
    ax.set_xlabel('a (Ideal Chop Level)', fontsize=12)
    ax.set_ylabel('b (Curve Steepness)', fontsize=12)
    ax.set_title(f'Grid Search Sharpe Ratio\nBest: a={best_params[0]:.2f}, b={best_params[1]:.2f}', fontsize=14, fontweight='bold')
    
    # Inverse y-axis to match values
    ax.invert_yaxis()
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

def create_equity_plot(results_df, horizon, a, b):
    fig = Figure(figsize=(12, 16))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    
    # Plot 1: Price
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1)
    ax1.set_yscale('log')
    ax1.set_title('BTC Price (Log)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scaling Factor
    ax2.plot(results_df.index, results_df['Scaling_Factor'], 'purple', linewidth=1, label=f'Factor (a={a:.2f}, b={b:.2f})')
    ax2.fill_between(results_df.index, results_df['Scaling_Factor'], 0, color='purple', alpha=0.1)
    ax2.set_ylim(0, 1.1)
    ax2b = ax2.twinx()
    ax2b.plot(results_df.index, results_df['chop'], 'gray', alpha=0.3, linewidth=0.5, label='Raw CHOP')
    ax2.legend(loc='upper left')
    ax2.set_title('Exposure Scaling Factor', fontweight='bold')
    
    # Plot 3: Equity
    ax3.plot(results_df.index, results_df['Portfolio_Value'], 'b-', linewidth=1.5, label='Strategy')
    bh = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax3.plot(results_df.index, bh, 'g--', alpha=0.8, label='Buy & Hold')
    if (results_df['Portfolio_Value'] <= 0).any(): ax3.set_yscale('symlog', linthresh=1.0)
    else: ax3.set_yscale('log')
    ax3.legend()
    ax3.set_title('Equity Curve', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown
    ax4.fill_between(results_df.index, results_df['Drawdown']*100, 0, color='red', alpha=0.3)
    ax4.set_title('Drawdown %', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

def start_web_server(results_df, heatmap_img_buf, best_params, main_sharpe):
    app = Flask(__name__)
    
    final_val = results_df['Portfolio_Value'].iloc[-1]
    total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
    
    # --- Monthly Stats ---
    monthly_rows = ""
    monthly_groups = results_df.groupby(pd.Grouper(freq='M'))
    for name, group in reversed(list(monthly_groups)): # Reverse order
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
        <head><title>Conviction Strategy</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; margin: 40px; color: #333; }}
            .stats {{ background: #f9f9f9; padding: 20px; border-radius: 8px; display: inline-block; margin-bottom: 20px; }}
            img {{ max-width: 95%; border: 1px solid #ccc; margin: 20px 0; }}
            table {{ margin: 0 auto; border-collapse: collapse; width: 100%; max-width: 600px; }}
            td, th {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        </style>
        </head>
        <body>
            <h1>Conviction Strategy Results</h1>
            <div class="stats">
                <p><strong>Horizon:</strong> {HORIZON} | <strong>Leverage:</strong> {LEVERAGE}x</p>
                <p><strong>Optimal Params:</strong> a={best_params[0]:.2f}, b={best_params[1]:.2f}</p>
                <p><strong>Final:</strong> ${final_val:,.2f} | <strong>Return:</strong> {total_ret:.2f}% | <strong>Sharpe:</strong> {main_sharpe:.2f}</p>
            </div>
            
            <h2>Parameter Grid Search (Heatmap)</h2>
            <p>X-axis: 'a' (Target Chop) | Y-axis: 'b' (Steepness)</p>
            <img src="/heatmap?v={ts}" />
            
            <h2>Equity Curve</h2>
            <img src="/plot?v={ts}" />
            
            <h2>Monthly Returns</h2>
            <table><thead><tr><th>Month</th><th>Return</th><th>Sharpe</th></tr></thead><tbody>{monthly_rows}</tbody></table>
        </body>
        </html>
        '''

    @app.route('/plot')
    def plot():
        buf = create_equity_plot(results_df, HORIZON, best_params[0], best_params[1])
        return send_file(buf, mimetype='image/png')
        
    @app.route('/heatmap')
    def heatmap():
        # Re-create heatmap buffer from stored data
        # Note: In a real app we'd store the image, but re-generating here is fine for this context
        # Actually, we need to pass the data. 
        # For simplicity, we just return the pre-generated buffer passed to this function.
        heatmap_img_buf.seek(0)
        return send_file(heatmap_img_buf, mimetype='image/png')

    print(f"Server running on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    df_data = fetch_binance_data()
    df_signals, _ = generate_signals(df_data)
    
    # 1. Precalculate Raw Conviction
    raw_conviction, contributions, common_idx = precalculate_conviction(df_data, df_signals)
    
    # Align df_data to common_idx for grid search
    df_grid_data = df_data.loc[common_idx].copy()
    
    # 2. Run Grid Search
    best_params, heatmap_data, a_vals, b_vals = run_grid_search(df_grid_data, raw_conviction)
    
    # 3. Create Heatmap Image
    heatmap_buf = create_heatmap_plot(heatmap_data, a_vals, b_vals, best_params)
    
    # 4. Run Final Backtest
    res, sig_names = run_final_backtest(df_grid_data, raw_conviction, contributions, common_idx, best_params[0], best_params[1])
    
    # Calc Sharpe for display
    daily_rets = res['Strategy_Daily_Return']
    sharpe = (daily_rets.mean()/daily_rets.std())*np.sqrt(365) if daily_rets.std() > 0 else 0
    
    start_web_server(res, heatmap_buf, best_params, sharpe)
