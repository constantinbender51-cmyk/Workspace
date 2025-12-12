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

    # Standard Close-to-Close Return (Standard for holding strategies)
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
    # Signal at Close(t) applies to exposure for period t to t+1
    df_signals = df_signals.shift(1)
    df_signals.fillna(0, inplace=True)
    df_signals = df_signals.astype(int)

    return df_signals


# --- Backtest implementation ---
def run_conviction_backtest(df_data, df_signals):
    # Align Data
    common_idx = df_data.index.intersection(df_signals.index)
    df = df_data.loc[common_idx].copy()
    signals = df_signals.loc[common_idx]

    num_days = len(df)
    
    # We use Close-to-Close returns because we hold positions overnight (Conviction)
    daily_returns = df['return'].values 
    dates = df.index

    # Arrays for speed
    portfolio = np.zeros(num_days)
    daily_pnl = np.zeros(num_days)
    conviction_raw = np.zeros(num_days)
    conviction_norm = np.zeros(num_days)

    # State tracking
    signal_start_day = np.full(MAX_CONVICTION, -1, dtype=int)
    signal_direction = np.zeros(MAX_CONVICTION, dtype=int)

    portfolio[0] = INITIAL_CAPITAL

    for t in range(num_days):
        daily_sum = 0.0

        for i in range(len(WINNING_SIGNALS)):
            current_sig = signals.iloc[t, i]

            # --- CRITICAL FIX: RECHARGING LOGIC ---
            # If a signal fires (even if same direction), reset the decay.
            # This "recharges" conviction if the market confirms the trend again.
            if current_sig != 0:
                signal_start_day[i] = t
                signal_direction[i] = current_sig
            
            # Compute Decay
            if signal_direction[i] != 0:
                d = t - signal_start_day[i]
                if d < 0: 
                    decay = 0.0
                else:
                    decay = max(0.0, 1.0 - (d / HORIZON))
                
                daily_sum += signal_direction[i] * decay

                # Clean up expired signals
                if decay == 0.0:
                    signal_direction[i] = 0
                    signal_start_day[i] = -1

        # Normalize Exposure
        exposure = daily_sum / MAX_CONVICTION
        exposure = np.clip(exposure, -1.0, 1.0)

        # PnL Calculation
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Price Chart
    ax1.set_facecolor('#e6f2ff') # Light Blue
    ax1.plot(results_df.index, results_df['close'], 'k-', linewidth=1.5, label='BTC Price')
    ax1.set_title('BTC/USDT Price', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Equity Curve
    ax2.set_facecolor('#ffe6e6') # Light Red
    ax2.plot(results_df.index, results_df['Portfolio_Value'], 'b-', linewidth=1.5, label='Strategy Equity')
    
    # Add Buy & Hold for comparison
    bh_curve = results_df['close'] / results_df['close'].iloc[0] * INITIAL_CAPITAL
    ax2.plot(results_df.index, bh_curve, 'g--', alpha=0.6, label='Buy & Hold')
    
    ax2.set_title('Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value (USDT)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
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
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conviction Backtest</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; text-align: center; }}
                .stats {{ margin: 20px auto; padding: 20px; background: #f0f0f0; max-width: 600px; border-radius: 8px; }}
                img {{ max-width: 90%; height: auto; border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <h1>Strategy Results</h1>
            <div class="stats">
                <p><strong>Initial Capital:</strong> ${INITIAL_CAPITAL:,.2f}</p>
                <p><strong>Final Value:</strong> ${final_val:,.2f}</p>
                <p><strong>Total Return:</strong> {total_ret:.2f}%</p>
            </div>
            <img src="/plot" />
        </body>
        </html>
        '''
    
    @app.route('/plot')
    def plot():
        return send_file(create_equity_plot(results_df), mimetype='image/png')
    
    print("\n" + "=" * 60)
    print("Server running on http://localhost:8080")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    # Run in main thread to keep script alive
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    df_data = fetch_binance_data()
    
    if df_data.empty:
        print("No data fetched.")
    else:
        df_signals = generate_signals(df_data)
        
        # Ensure we have data to backtest
        if df_signals.empty or len(df_signals) < 10:
             print("Not enough signals generated.")
        else:
            ret, res = run_conviction_backtest(df_data, df_signals)
            
            print(f"\nFinal Portfolio: ${res['Portfolio_Value'].iloc[-1]:,.2f}")
            print(f"Strategy Return: {ret*100:.2f}%")
            
            # Start server (Blocking)
            start_web_server(res)
