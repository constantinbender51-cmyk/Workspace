import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
from io import BytesIO
import base64
from flask import Flask, render_template_string
import time

# --- 1. CONFIGURATION ---
SYMBOL = 'BTC/USDT'  # Binance standard symbol
START_DATE = '2018-01-01'
TIME_FRAME = '1d'

SMA_LONG_TERM = 400
SMA_SHORT_TERM = 120

# --- 2. DATA FETCHING (Using CCXT for Binance) ---

def get_data(symbol, since_date_str='2018-01-01'):
    """
    Fetches historical OHLCV data from Binance using CCXT, handling pagination.
    """
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True, # Important for respecting API limits
        })
        
        # Convert start date string to milliseconds timestamp
        since_ms = exchange.parse8601(since_date_str + 'T00:00:00Z')
        
        all_ohlcv = []
        limit = 1000  # Max limit per request on Binance
        
        print(f"Fetching data for {symbol} from Binance starting {since_date_str}...")
        
        while True:
            # Fetch data batch
            ohlcv = exchange.fetch_ohlcv(symbol, TIME_FRAME, since=since_ms, limit=limit)
            
            if not ohlcv:
                print("No more data available.")
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Set the new 'since' to the timestamp of the last candle fetched + 1 day (in ms)
            last_timestamp = ohlcv[-1][0]
            since_ms = last_timestamp + exchange.parse_timeframe(TIME_FRAME) * 1000 
            
            # Print status and sleep to respect rate limits
            print(f"Fetched {len(ohlcv)} candles. Latest date: {exchange.iso8601(last_timestamp)}")
            if len(ohlcv) < limit:
                break
                
            time.sleep(exchange.rateLimit / 1000) # Wait to respect the rate limit

        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df = df.drop(columns=['timestamp'])
        df.index.name = 'Date'
        df = df.sort_index(ascending=True)
        
        print(f"Data fetching complete. Total {len(df)} candles retrieved.")
        return df
    
    except ImportError:
        print("ERROR: CCXT library not found. Please run: pip install ccxt")
        return None
    except Exception as e:
        print(f"ERROR: Failed to fetch data from Binance: {e}")
        print("Check if the symbol is correct and the exchange is accessible.")
        return None


def run_backtest(df):
    """
    Calculates SMAs, implements the trading strategy, and computes metrics.
    """
    print("Running backtest and calculating SMAs...")
    
    # Calculate Simple Moving Averages
    df[f'SMA_{SMA_LONG_TERM}'] = df['close'].rolling(window=SMA_LONG_TERM).mean()
    df[f'SMA_{SMA_SHORT_TERM}'] = df['close'].rolling(window=SMA_SHORT_TERM).mean()

    # Drop initial NaN values created by rolling windows (first 400 days)
    df = df.dropna()

    # --- 3. CREATE DATASETS (Regimes) ---
    
    # Bullish Regime: Price > 400 SMA
    df_bull = df[df['close'] > df[f'SMA_{SMA_LONG_TERM}']].copy()
    
    # Bearish Regime: Price < 400 SMA
    df_bear = df[df['close'] < df[f'SMA_{SMA_LONG_TERM}']].copy()

    # --- 4. IMPLEMENT STRATEGY LOGIC ---

    # Combined Strategy Logic (Signal column in original DF)
    df['Signal'] = 0.0

    # Long Logic (Bullish Regime: Price > 400 SMA)
    # Trade Long if 120 SMA is below price (Price > 120 SMA)
    df.loc[(df['close'] > df[f'SMA_{SMA_LONG_TERM}']) & (df['close'] > df[f'SMA_{SMA_SHORT_TERM}']), 'Signal'] = 1.0

    # Short Logic (Bearish Regime: Price < 400 SMA)
    # Trade Short if 120 SMA is above price (Price < 120 SMA)
    df.loc[(df['close'] < df[f'SMA_{SMA_LONG_TERM}']) & (df['close'] < df[f'SMA_{SMA_SHORT_TERM}']), 'Signal'] = -1.0

    # Calculate Daily Returns
    df['Daily_Return'] = df['close'].pct_change()

    # Calculate Strategy Returns (Shift signals to act on the next day's return)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']

    # Drop the first row introduced by pct_change and shift
    df = df.dropna()

    # Calculate Cumulative Returns
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_BuyHold_Return'] = (1 + df['Daily_Return']).cumprod()


    # --- 5. METRICS CALCULATION ---
    
    # Total Number of Trading Days
    total_days = len(df)
    
    # Annualized Returns (Assuming 252 trading days for crypto for calculation simplicity)
    annualized_returns = (df['Cumulative_Strategy_Return'].iloc[-1] ** (365 / total_days)) - 1
    
    # Sharpe Ratio (Using risk-free rate of 0 for simplicity)
    daily_volatility = df['Strategy_Return'].std()
    sharpe_ratio = (df['Strategy_Return'].mean() / daily_volatility) * np.sqrt(365) # Using 365 for crypto
    
    # Max Drawdown
    cumulative_max = df['Cumulative_Strategy_Return'].cummax()
    drawdown = cumulative_max - df['Cumulative_Strategy_Return']
    max_drawdown = drawdown.max()
    
    # Trades and Win Rate
    # A trade is when the position changes from 0 or switches direction
    df['Position'] = df['Signal'].replace(0, method='ffill').fillna(0)
    
    trades = []
    trade_open = None
    
    # Identify trades by position change
    for index, row in df.iterrows():
        current_pos = row['Position']
        prev_pos = df['Position'].shift(1).loc[index]

        # Entry condition (position changes from 0 to 1 or -1)
        if current_pos != 0 and prev_pos == 0:
            # Exit previous trade if still open (shouldn't happen with the logic above, but safety check)
            if trade_open is not None:
                 trades.append({
                    'entry_date': trade_open['entry_date'],
                    'exit_date': index - pd.Timedelta(days=1),
                    'position': 'Long (Forced Exit)' if trade_open['position'] == 1 else 'Short (Forced Exit)',
                    'pnl': (df['close'].loc[index - pd.Timedelta(days=1)] / trade_open['entry_price'] - 1) * trade_open['position'],
                    'win': (df['close'].loc[index - pd.Timedelta(days=1)] / trade_open['entry_price'] - 1) * trade_open['position'] > 0
                })
            trade_open = {'entry_date': index, 'entry_price': row['close'], 'position': current_pos}

        # Exit condition (position changes from 1/-1 to 0 or switches direction)
        elif current_pos == 0 and prev_pos != 0 and trade_open is not None:
             pnl = (row['close'] / trade_open['entry_price'] - 1) * trade_open['position']
             trades.append({
                'entry_date': trade_open['entry_date'],
                'exit_date': index,
                'position': 'Long' if trade_open['position'] == 1 else 'Short',
                'pnl': pnl,
                'win': pnl > 0
            })
             trade_open = None


    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['win'])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Net Profit/Loss (Final Strategy Return - 1)
    net_pnl = df['Cumulative_Strategy_Return'].iloc[-1] - 1
    
    metrics = {
        "Total Days Tested": total_days,
        "Start Date (after SMAs)": df.index.min().strftime('%Y-%m-%d'),
        "End Date": df.index.max().strftime('%Y-%m-%d'),
        "Final Strategy PnL": f"{net_pnl * 100:.2f}%",
        "Final Buy & Hold PnL": f"{(df['Cumulative_BuyHold_Return'].iloc[-1] - 1) * 100:.2f}%",
        "Annualized Return": f"{annualized_returns * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%",
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Win Rate": f"{win_rate * 100:.2f}%",
    }
    
    # Prepare trade list for output
    trade_list_output = [f"{t['exit_date'].strftime('%Y-%m-%d')}: {t['position']} trade ended. PnL: {t['pnl'] * 100:.2f}%" for t in trades]
    
    # Prepare the two segregated datasets (for listing)
    bullish_days = len(df_bull)
    bearish_days = len(df_bear)

    regime_metrics = {
        "Bullish Regime Days (Price > 400 SMA)": bullish_days,
        "Bearish Regime Days (Price < 400 SMA)": bearish_days
    }

    return df, metrics, trade_list_output, regime_metrics


def generate_plot(df):
    """
    Generates a plot of the price, SMAs, and strategy equity curve.
    Returns the plot as a base64 encoded image string.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Price and SMA Plot ---
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', color='#4A90E2', linewidth=1)
    ax1.plot(df.index, df[f'SMA_{SMA_LONG_TERM}'], label=f'{SMA_LONG_TERM}-Day SMA', color='#F5A623', linestyle='--')
    ax1.plot(df.index, df[f'SMA_{SMA_SHORT_TERM}'], label=f'{SMA_SHORT_TERM}-Day SMA', color='#7ED321')
    
    # Highlight trade entry points
    ax1.scatter(df.index[df['Signal'] == 1], df['close'][df['Signal'] == 1], marker='^', color='green', label='Long Entry', alpha=1, s=50)
    ax1.scatter(df.index[df['Signal'] == -1], df['close'][df['Signal'] == -1], marker='v', color='red', label='Short Entry', alpha=1, s=50)

    ax1.set_title(f'{SYMBOL} Price and Strategy Signals ({TIME_FRAME})', fontsize=16)
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # --- Equity Curve Plot ---
    ax2 = axes[1]
    ax2.plot(df.index, df['Cumulative_Strategy_Return'], label='Strategy Equity', color='#50E3C2', linewidth=2)
    ax2.plot(df.index, df['Cumulative_BuyHold_Return'], label='Buy & Hold', color='#FF6347', linestyle='--', linewidth=1)
    
    ax2.set_title('Strategy vs. Buy & Hold Cumulative Returns', fontsize=14)
    ax2.set_ylabel('Cumulative Return', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot to an in-memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data

# --- 6. FLASK WEB SERVER SETUP ---

app = Flask(__name__)

# HTML template for the web page
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Binance Backtest: {{ symbol }}</title>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #1e1e1e; color: #fff; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background-color: #2c2c2c; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5); }
        h1 { color: #50E3C2; text-align: center; margin-bottom: 20px; }
        .datasource { text-align: center; color: #F5A623; margin-bottom: 30px; font-size: 1.1em; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .card { background-color: #383838; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3); }
        .metrics-list, .regime-list { list-style: none; padding: 0; }
        .metrics-list li, .regime-list li { padding: 8px 0; border-bottom: 1px solid #4a4a4a; display: flex; justify-content: space-between; }
        .metrics-list li:last-child, .regime-list li:last-child { border-bottom: none; }
        .value { font-weight: bold; color: #50E3C2; }
        img { max-width: 100%; height: auto; border-radius: 8px; margin-top: 20px; background-color: #fff; padding: 10px;}
        .trade-log { max-height: 400px; overflow-y: scroll; background-color: #1e1e1e; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.9em; }
        .trade-log p { margin: 4px 0; border-bottom: 1px dotted #4a4a4a; padding-bottom: 4px; }
        .trade-log p:last-child { border-bottom: none; }
        .positive { color: #7ED321; }
        .negative { color: #D0021B; }
        .plot-container { grid-column: 1 / -1; }
        .strategy-description { text-align: center; margin-top: 20px; font-size: 0.9em; color: #888; border-top: 1px solid #4a4a4a; padding-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ symbol }} 400/120 SMA Crossover Strategy Backtest</h1>
        <div class="datasource">Data Source: Binance via CCXT | Timeframe: {{ timeframe }}</div>
        
        <!-- Plotting Area -->
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plot_data }}" alt="Strategy Performance Plot">
        </div>

        <div class="grid">
            <!-- Metrics Card -->
            <div class="card">
                <h2>Performance Metrics</h2>
                <ul class="metrics-list">
                    {% for key, value in metrics.items() %}
                    <li>
                        <span>{{ key }}:</span>
                        <span class="value">{{ value }}</span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
            
            <!-- Regimes & Trade Log Card -->
            <div class="card">
                <h2>Regime Analysis</h2>
                <ul class="regime-list">
                    {% for key, value in regime_metrics.items() %}
                    <li>
                        <span>{{ key }}:</span>
                        <span class="value">{{ value }}</span>
                    </li>
                    {% endfor %}
                </ul>

                <h2>Trade List (Last 50)</h2>
                <div class="trade-log">
                    {% if trade_list %}
                        {% for trade in trade_list|slice:":50" %}
                            {% set color_class = 'value' %}
                            {% if 'PnL: -' in trade %}
                                {% set color_class = 'negative' %}
                            {% elif 'PnL: 0.00%' not in trade %}
                                {% set color_class = 'positive' %}
                            {% endif %}
                            <p class="{{ color_class }}">{{ trade }}</p>
                        {% endfor %}
                    {% else %}
                        <p>No completed trades found in the backtest period.</p>
                    {% endif %}
                </div>
            </div>
        </div>
        <div class="strategy-description">
            Strategy Logic:<br>
            - **Bullish Regime** (Price > 400 SMA): Enter **Long** when Price > 120 SMA.<br>
            - **Bearish Regime** (Price < 400 SMA): Enter **Short** when Price < 120 SMA.
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """
    Main route to run the backtest, generate the plot, and display the dashboard.
    """
    df = get_data(SYMBOL, START_DATE)
    
    if df is None or df.empty:
        error_msg = f"""
        <div style="background-color: #2c2c2c; color: #fff; padding: 40px; text-align: center; border-radius: 12px; margin: 50px auto; max-width: 600px;">
            <h1>Data Fetching Failed</h1>
            <p style="color: #F5A623;">Could not fetch sufficient historical data for {SYMBOL} from Binance.</p>
            <p style="font-size: 0.9em; margin-top: 20px;">Please ensure you have installed <code>ccxt</code> and check if the API is currently accessible. The script needs data starting from {START_DATE}.</p>
        </div>
        """
        return render_template_string(error_msg), 500

    # Run backtest
    # Check if there is enough data for the 400-day SMA
    if len(df) < SMA_LONG_TERM:
        error_msg = f"""
        <div style="background-color: #2c2c2c; color: #fff; padding: 40px; text-align: center; border-radius: 12px; margin: 50px auto; max-width: 600px;">
            <h1>Insufficient Data Error</h1>
            <p style="color: #F5A623;">Fetched only {len(df)} days of data. We need at least {SMA_LONG_TERM} days to calculate the 400-day SMA.</p>
            <p style="font-size: 0.9em; margin-top: 20px;">The script cannot run the backtest until more data is available or the required SMA window is reduced.</p>
        </div>
        """
        return render_template_string(error_msg), 500

    df_results, metrics, trade_list, regime_metrics = run_backtest(df)

    # Generate plot
    plot_data = generate_plot(df_results)

    # Render HTML
    return render_template_string(
        HTML_TEMPLATE,
        symbol=SYMBOL,
        timeframe=TIME_FRAME,
        metrics=metrics,
        regime_metrics=regime_metrics,
        trade_list=trade_list,
        plot_data=plot_data
    )

if __name__ == '__main__':
    print(f"\n--- Strategy Backtester Starting ---")
    print(f"Data Source: Binance ({SYMBOL})")
    print(f"Web server starting on http://127.0.0.1:8080/")
    
    # Required libraries: pip install pandas numpy matplotlib flask ccxt
    # Set debug=False and use_reloader=False for cleaner execution
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
