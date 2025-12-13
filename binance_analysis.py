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

# --- Winning Signals ---
WINNING_SIGNALS = [
    ('EMA_CROSS', 50, 150, 0),         # EMA 50/150
    ('PRICE_SMA', 380, 0, 0),          # Price/SMA 380
    ('PRICE_SMA', 140, 0, 0),          # Price/SMA 140
    ('MACD_CROSS', 12, 26, 15),        # MACD (12/26/15)
    ('RSI_CROSS', 35, 0, 0),           # RSI 35 (Crossover)
]

N_SIGNALS = len(WINNING_SIGNALS)

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

    df_signals = df_signals.shift(1)
    df_signals.fillna(0, inplace=True)
    df_signals = df_signals.astype(int)

    return df_signals

# --- Pre-calculate Signal Contributions (The "Decay" Matrix) ---
def precalculate_contributions(df_data, df_signals):
    """
    Creates a (Days, Signals) matrix where each cell is the decayed value (-1 to 1)
    of that signal for that day.
    """
    common_idx = df_data.index.intersection(df_signals.index)
    signals = df_signals.loc[common_idx]
    num_days = len(common_idx)
    
    # Matrix to store the calculated contribution of each signal over time
    contributions = np.zeros((num_days, N_SIGNALS))
    
    signal_start_day = np.full(N_SIGNALS, -1, dtype=int)
    signal_direction = np.zeros(N_SIGNALS, dtype=int)

    # We iterate once to build the static history of conviction
    for t in range(num_days):
        for i in range(N_SIGNALS):
            current_sig = signals.iloc[t, i]
            
            # New signal triggers logic reset
            if current_sig != 0:
                signal_start_day[i] = t
                signal_direction[i] = current_sig
            
            # Calculate decay
            if signal_direction[i] != 0:
                d = t - signal_start_day[i]
                if d < 0: decay = 0.0
                else: decay = max(0.0, 1.0 - (d / HORIZON))
                
                # Store
                contributions[t, i] = signal_direction[i] * decay
                
                if decay == 0.0:
                    signal_direction[i] = 0
                    signal_start_day[i] = -1
                    
    return contributions, common_idx

# --- Genetic Algorithm Implementation ---
def fitness_function(weights, contributions, returns):
    """
    Calculates Sharpe Ratio for a given set of weights.
    NO NORMALIZATION: Exposure is purely additive.
    """
    # Exposure = Sum(Signal_Contribution * Weight)
    # Shape: (T, N) @ (N,) -> (T,)
    exposure = contributions @ weights
    
    # Clip exposure to safeguard against craziness (e.g. > 5x)
    # If weights are 0-1, max possible is N_SIGNALS (5.0).
    exposure = np.clip(exposure, -N_SIGNALS, N_SIGNALS)
    
    strat_ret = exposure * returns
    
    if np.std(strat_ret) == 0:
        return 0.0
    
    # Annualized Sharpe
    sharpe = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(365)
    return sharpe

def run_ga_optimization(contributions, returns, split_idx):
    print(f"\nRunning GA Optimization on first {split_idx} days (70% Training Data)...")
    
    # Train Data
    train_contrib = contributions[:split_idx]
    train_returns = returns[:split_idx]
    
    # GA Parameters
    pop_size = 100
    generations = 30
    mutation_rate = 0.1
    mutation_scale = 0.1
    n_genes = N_SIGNALS
    
    # Initialize Population (random weights 0.0 to 1.0)
    population = np.random.rand(pop_size, n_genes)
    
    best_weights = None
    best_fitness = -999.0
    
    for gen in range(generations):
        fitness_scores = np.zeros(pop_size)
        
        # Evaluate
        for i in range(pop_size):
            fitness_scores[i] = fitness_function(population[i], train_contrib, train_returns)
            
        # Track Best
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_score = fitness_scores[gen_best_idx]
        
        if gen_best_score > best_fitness:
            best_fitness = gen_best_score
            best_weights = population[gen_best_idx].copy()
            
        # Selection (Tournament)
        new_pop = np.zeros_like(population)
        # Elitism: Keep best
        new_pop[0] = population[gen_best_idx]
        
        for i in range(1, pop_size):
            # Tournament size 3
            candidates_idx = np.random.choice(pop_size, 3, replace=False)
            winner_idx = candidates_idx[np.argmax(fitness_scores[candidates_idx])]
            parent = population[winner_idx]
            
            # Mutation
            child = parent.copy()
            if np.random.rand() < 0.5: # 50% chance to mutate a gene set
                 mask = np.random.rand(n_genes) < mutation_rate
                 noise = np.random.normal(0, mutation_scale, n_genes)
                 child[mask] += noise[mask]
                 child = np.clip(child, 0.0, 1.0)
            
            new_pop[i] = child
            
        population = new_pop
        
        if gen % 5 == 0:
            print(f"Gen {gen}: Best Sharpe = {best_fitness:.4f}")

    print(f"Optimization Complete. Best Training Sharpe: {best_fitness:.4f}")
    return best_weights

# --- Backtest Runner ---
def run_weighted_backtest(df_data, contributions, common_idx, weights):
    df = df_data.loc[common_idx].copy()
    num_days = len(df)
    returns = df['return'].values
    
    # Calculate Exposure (Additive, no normalization)
    exposure = contributions @ weights
    exposure = np.clip(exposure, -N_SIGNALS, N_SIGNALS)
    
    # Run Portfolio Loop
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    portfolio[0] = INITIAL_CAPITAL
    is_bankrupt = False
    
    for t in range(1, num_days):
        if not is_bankrupt:
            # PnL = Previous Equity * Exposure * Return
            pnl = portfolio[t-1] * exposure[t] * returns[t]
            portfolio[t] = portfolio[t-1] + pnl
            daily_pnl[t] = pnl
            
            if portfolio[t] <= 0:
                portfolio[t] = 0
                is_bankrupt = True
        else:
            portfolio[t] = 0
            
    results = df[['close', 'return']].copy()
    results['Exposure'] = exposure
    results['Portfolio_Value'] = portfolio
    results['Daily_PnL'] = daily_pnl
    results['Strategy_Daily_Return'] = results['Portfolio_Value'].pct_change().fillna(0)
    
    rolling_max = results['Portfolio_Value'].cummax()
    results['Drawdown'] = np.where(rolling_max > 0, (results['Portfolio_Value'] - rolling_max) / rolling_max, -1.0)
    
    # Add contributions for display
    # We multiply raw contribution by its weight
    weighted_contribs = contributions * weights
    signal_names = [f"{s[0]}_{s[1]}" for s in WINNING_SIGNALS]
    for i, name in enumerate(signal_names):
        results[f"W_Contrib_{name}"] = weighted_contribs[:, i]
        
    return results, signal_names

def create_equity_plot(results_df, split_date):
    fig = Figure(figsize=(12, 12))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    
    # Plot 1: Price
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1)
    ax1.set_yscale('log')
    ax1.set_title('BTC Price (Log)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=split_date, color='orange', linestyle='--', linewidth=2, label='Train/Test Split')
    ax1.legend()
    
    # Plot 2: Equity
    ax2.plot(results_df.index, results_df['Portfolio_Value'], 'b-', linewidth=1.5, label='Strategy (GA Opt)')
    bh = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax2.plot(results_df.index, bh, 'g--', alpha=0.8, label='Buy & Hold')
    ax2.axvline(x=split_date, color='orange', linestyle='--', linewidth=2)
    
    if (results_df['Portfolio_Value'] <= 0).any(): ax2.set_yscale('symlog', linthresh=1.0)
    else: ax2.set_yscale('log')
    
    ax2.legend(loc='upper left')
    ax2.set_title('Equity Curve (Train vs Test)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    ax3.fill_between(results_df.index, results_df['Drawdown']*100, 0, color='red', alpha=0.3)
    ax3.set_title('Drawdown %', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=split_date, color='orange', linestyle='--', linewidth=2)
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

def start_web_server(results_df, best_weights, split_date):
    app = Flask(__name__)
    
    final_val = results_df['Portfolio_Value'].iloc[-1]
    total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
    
    # Overall Sharpe
    d_rets = results_df['Strategy_Daily_Return']
    overall_sharpe = (d_rets.mean()/d_rets.std())*np.sqrt(365) if d_rets.std()>0 else 0
    
    # Test Set Sharpe (Out of Sample)
    test_df = results_df[results_df.index > split_date]
    t_rets = test_df['Strategy_Daily_Return']
    test_sharpe = (t_rets.mean()/t_rets.std())*np.sqrt(365) if t_rets.std()>0 else 0
    
    # Weights Table
    signal_names = [f"{s[0]}_{s[1]}" for s in WINNING_SIGNALS]
    weights_html = "<table border='1' cellpadding='5'><tr><th>Signal</th><th>Optimized Weight (Leverage)</th></tr>"
    for name, w in zip(signal_names, best_weights):
        weights_html += f"<tr><td>{name}</td><td>{w:.4f}x</td></tr>"
    
    # Calculate Theoretical Max Leverage
    max_lev = np.sum(best_weights)
    weights_html += f"<tr><td><b>Total Max Leverage</b></td><td><b>{max_lev:.2f}x</b></td></tr></table>"
    
    monthly_rows = ""
    monthly_groups = results_df.groupby(pd.Grouper(freq='M'))
    for name, group in reversed(list(monthly_groups)): 
        if len(group) < 5: continue
        m_daily_rets = group['Strategy_Daily_Return']
        m_start = group['Portfolio_Value'].iloc[0]
        m_end = group['Portfolio_Value'].iloc[-1]
        m_ret = (m_end / m_start) - 1.0 if m_start > 0 else 0.0
        m_sharpe = (m_daily_rets.mean()/m_daily_rets.std())*np.sqrt(365) if m_daily_rets.std()>0 else 0
        
        # Highlight Test vs Train months?
        bg = "#eaffea" if name > split_date else "#fff"
        
        color = "green" if m_ret > 0 else "red"
        monthly_rows += f"<tr style='background:{bg}'><td>{name.strftime('%Y-%m')}</td><td style='color:{color}'><b>{m_ret*100:.2f}%</b></td><td>{m_sharpe:.2f}</td></tr>"

    @app.route('/')
    def index():
        ts = int(time.time())
        return f'''
        <html>
        <head><title>Conviction Strategy GA</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; margin: 40px; color: #333; }}
            .stats {{ background: #f9f9f9; padding: 20px; border-radius: 8px; display: inline-block; margin-bottom: 20px; }}
            img {{ max-width: 95%; border: 1px solid #ccc; margin: 20px 0; }}
            table {{ margin: 0 auto; border-collapse: collapse; width: 100%; max-width: 600px; }}
            td, th {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        </style>
        </head>
        <body>
            <h1>Conviction Strategy (GA Optimized - Additive)</h1>
            <div class="stats">
                <p><strong>Horizon:</strong> {HORIZON} | <strong>Leverage:</strong> Dynamic (Weights)</p>
                <p><strong>Split Date (70%):</strong> {split_date.strftime('%Y-%m-%d')}</p>
                <p><strong>Final:</strong> ${final_val:,.2f} | <strong>Return:</strong> {total_ret:.2f}%</p>
                <p><strong>Overall Sharpe:</strong> {overall_sharpe:.2f} | <strong>Test Set Sharpe:</strong> {test_sharpe:.2f}</p>
            </div>
            
            <h2>Optimized Weights (0.0 - 1.0)</h2>
            <p>Sum of Weights = Total Leverage Capacity</p>
            {weights_html}
            
            <h2>Equity Curve (Train vs Test)</h2>
            <img src="/plot?v={ts}" />
            
            <h2>Monthly Returns (Green BG = Test Set)</h2>
            <table><thead><tr><th>Month</th><th>Return</th><th>Sharpe</th></tr></thead><tbody>{monthly_rows}</tbody></table>
        </body>
        </html>
        '''

    @app.route('/plot')
    def plot():
        buf = create_equity_plot(results_df, split_date)
        return send_file(buf, mimetype='image/png')
        
    print(f"Server running on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    df_data = fetch_binance_data()
    df_signals = generate_signals(df_data)
    
    # 1. Precalculate Signal Contributions (Decay Matrix)
    contributions, common_idx = precalculate_contributions(df_data, df_signals)
    df_clean = df_data.loc[common_idx]
    returns = df_clean['return'].values
    
    # 2. Determine Split Index (70%)
    split_idx = int(len(common_idx) * 0.70)
    split_date = common_idx[split_idx]
    
    # 3. Run GA Optimization on Training Data
    best_weights = run_ga_optimization(contributions, returns, split_idx)
    
    # 4. Run Backtest on FULL Data using Optimized Weights
    results, _ = run_weighted_backtest(df_clean, contributions, common_idx, best_weights)
    
    start_web_server(results, best_weights, split_date)
