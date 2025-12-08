import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from flask import Flask, render_template_string, send_file
import io
import base64


# --- Flask App ---
app = Flask(__name__)
PORT = 8080# --- Configuration ---
SYMBOL = "BTCUSDT"
START_DATE = "2018-01-01"
INITIAL_CAPITAL = 1000.0
FEES_PCT = 0.0006  # 0.06% est. taker fee per trade (entry + exit)

# Strategy Parameters
SMA_PERIOD_1 = 57
SMA_PERIOD_2 = 124
SMA_PERIOD_365 = 365
BAND_WIDTH = 0.05
STATIC_STOP_PCT = 0.02
TAKE_PROFIT_PCT = 0.16
III_WINDOW = 35

# III Leverage Thresholds
III_T_LOW = 0.13
III_T_HIGH = 0.18
LEV_LOW = 0.5
LEV_MID = 4.5
LEV_HIGH = 2.45

def fetch_binance_data(symbol, start_date):
    """Fetches daily OHLC data from Binance public API."""
    print(f"Fetching data for {symbol} starting {start_date}...")
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while True:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ts,
            "limit": limit
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        start_ts = data[-1][0] + 86400000  # Move to next day
        
        # Stop if we reached current time
        if len(data) < limit:
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    float_cols = ["open", "high", "low", "close", "volume"]
    df[float_cols] = df[float_cols].astype(float)
    
    return df[["date", "open", "high", "low", "close"]].set_index("date")

def calculate_indicators(df):
    """Calculates SMAs and III."""
    df = df.copy()
    
    # SMAs
    df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    df['sma_365'] = df['close'].rolling(window=SMA_PERIOD_365).mean()
    
    # Bands
    df['upper_band'] = df['sma_1'] * (1 + BAND_WIDTH)
    df['lower_band'] = df['sma_1'] * (1 - BAND_WIDTH)
    
    # III Calculation (Rolling 35 days)
    # III = |Sum(LogReturns)| / Sum(|LogReturns|)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    def calc_rolling_iii(x):
        abs_net_dir = np.abs(x.sum())
        path_len = np.abs(x).sum()
        if path_len == 0: return 0
        return abs_net_dir / path_len

    # Use rolling apply for III (Note: this can be slow on huge datasets, but fine for daily)
    df['iii'] = df['log_ret'].rolling(window=III_WINDOW).apply(calc_rolling_iii, raw=True)
    
    return df

def get_leverage(iii):
    if pd.isna(iii): return 1.0
    if iii < III_T_LOW: return LEV_LOW
    elif iii < III_T_HIGH: return LEV_MID
    else: return LEV_HIGH

def run_backtest(df):
    """Executes the state machine and trade logic."""
    print("Running backtest strategy logic...")
    
    equity_curve = [INITIAL_CAPITAL]
    capital = INITIAL_CAPITAL
    position = None  # {type: 'LONG'/'SHORT', entry_price: float, size: float, stop: float, tp: float}
    
    # State Machine Variables
    cross_flag = 0
    
    # III Condition Tracking
    iii_condition_active = False
    iii_above_threshold = False 
    
    history = []
    condition_status = []  # Track when condition is active for plotting
    
    # Iterate row by row to replicate state machine accurately
    # Start loop after enough data for indicators
    start_idx = max(SMA_PERIOD_2, III_WINDOW) + 1
    
    for i in range(start_idx, len(df)):
        curr_date = df.index[i]
        curr_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # --- 1. UPDATE STATE MACHINE (Based on previous close vs SMA) ---
        # Note: Strategy Logic generates signal at T based on T-1 Close
        
        sma_1 = prev_row['sma_1']
        sma_2 = prev_row['sma_2']
        prev_close = df.iloc[i-2]['close'] # Close of T-2
        last_close = prev_row['close']     # Close of T-1
        
        # Detect Crosses (T-2 to T-1)
        if prev_close < sma_1 and last_close > sma_1:
            cross_flag = 1
        elif prev_close > sma_1 and last_close < sma_1:
            cross_flag = -1
            
        # Reset Flag if exited bands
        upper_band = prev_row['upper_band']
        lower_band = prev_row['lower_band']
        
        if last_close > upper_band or last_close < lower_band:
            cross_flag = 0
            
        # --- 2. GENERATE SIGNAL ---
        signal = "FLAT"
        
        # Long Logic
        if last_close > upper_band: signal = "LONG"
        elif last_close > sma_1 and cross_flag == 1: signal = "LONG"
        
        # Short Logic
        elif last_close < lower_band: signal = "SHORT"
        elif last_close < sma_1 and cross_flag == -1: signal = "SHORT"
        
        # Filter Logic (SMA 2)
        if signal == "LONG" and last_close < sma_2: signal = "FLAT"
        if signal == "SHORT" and last_close > sma_2: signal = "FLAT"
        
        # --- 3. EXECUTE TRADES ---
        # Today's price action
        today_open = curr_row['open']
        today_high = curr_row['high']
        today_low = curr_row['low']
        today_close = curr_row['close']
        
        # Determine Leverage for TODAY based on YESTERDAY's III
        lev = get_leverage(prev_row['iii'])
        

        
        # Check Existing Position (SL/TP or Signal Flip)
        if position:
            # Check SL/TP first (assuming they exist in the market)
            # Assumption: SL hit first if Low < SL, unless Open gap causes issues (simplified here)
            
            pnl_pct = 0
            exit_price = None
            exit_reason = None
            
            # LONG EXIT CHECKS
            if position['type'] == 'LONG':
                if today_low <= position['stop']:
                    exit_price = position['stop']
                    exit_reason = "STOP_LOSS"
                elif today_high >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = "TAKE_PROFIT"
                elif signal != "LONG": # Signal Flip or Flat
                    exit_price = today_open
                    exit_reason = "SIGNAL_CHANGE"
                
                if exit_price:
                    # Calculate PnL
                    # (Exit - Entry) / Entry * Leverage
                    raw_ret = (exit_price - position['entry_price']) / position['entry_price']
                    pnl_pct = raw_ret * position['leverage']
            
            # SHORT EXIT CHECKS
            elif position['type'] == 'SHORT':
                if today_high >= position['stop']:
                    exit_price = position['stop']
                    exit_reason = "STOP_LOSS"
                elif today_low <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = "TAKE_PROFIT"
                elif signal != "SHORT":
                    exit_price = today_open
                    exit_reason = "SIGNAL_CHANGE"
                    
                if exit_price:
                    # (Entry - Exit) / Entry * Leverage
                    raw_ret = (position['entry_price'] - exit_price) / position['entry_price']
                    pnl_pct = raw_ret * position['leverage']

            # Process Exit
            if exit_price:
                # Apply Fees
                fee_impact = FEES_PCT * position['leverage'] 
                net_pnl = pnl_pct - fee_impact
                
                capital = capital * (1 + net_pnl)
                position = None # Flat
                
                # No streak logic to update
                
                # If we exited due to signal change, we might re-enter immediately below
                # But for simplicity, we trade on Open, so Signal Change exit happens at Open
                # If signal is reversed, we can enter new position same bar? 
                # Yes, if we exited at Open, we can enter at Open.
        
        # Entry Logic (If flat)
        if position is None and signal != "FLAT":
            entry_price = today_open
            
            # No streak condition to check
            can_enter = True
            

            
            # Setup Stops/TP
            if signal == "LONG":
                sl_price = entry_price * (1 - STATIC_STOP_PCT)
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            else:
                sl_price = entry_price * (1 + STATIC_STOP_PCT)
                tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
            
            # Check if Candle immediately hits SL/TP (Intraday volatility)
            # Simplified: Assume entry successful, check SL/TP next loop OR same loop?
            # We must check same loop for validity.
            
            # Immediate Stop Check (bad luck entry)
            hit_sl = False
            if signal == "LONG" and today_low < sl_price: hit_sl = True
            if signal == "SHORT" and today_high > sl_price: hit_sl = True
            
            if hit_sl:
                # Immediate loss
                pnl_pct = -STATIC_STOP_PCT * lev
                fee_impact = FEES_PCT * lev
                capital = capital * (1 + pnl_pct - fee_impact)
                # No streak logic to update
            else:
                # Position Established
                position = {
                    'type': signal,
                    'entry_price': entry_price,
                    'stop': sl_price,
                    'tp': tp_price,
                    'leverage': lev,
                    'entry_date': curr_date
                }
                # Apply Entry Fee immediately? usually applied on PnL calc, but let's deduct from 'virtual' equity if we tracked realized.
                # We will deduct entry fee upon exit calculation for simplicity, or effectively reduce size.
                # Actually, standard is: Capital is locked.
                
        equity_curve.append(capital)
        history.append({'date': curr_date, 'equity': capital, 'signal': signal, 'leverage': lev if position else 0})

    result_df = pd.DataFrame(history).set_index('date')
    return result_df

def calculate_metrics(df):
    """Calculates Sortino, Sharpe, and Quarterly Returns."""
    df['returns'] = df['equity'].pct_change()
    
    # Annualization factor (Crypto trades 365 days)
    ann_factor = 365
    
    total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
    cagr = (df['equity'].iloc[-1] / df['equity'].iloc[0]) ** (365 / len(df)) - 1
    
    daily_std = df['returns'].std()
    downside_std = df[df['returns'] < 0]['returns'].std()
    
    sharpe = (df['returns'].mean() / daily_std) * np.sqrt(ann_factor) if daily_std > 0 else 0
    sortino = (df['returns'].mean() / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0
    
    # Quarterly Capital
    quarterly = df['equity'].resample('QE').last()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Final Capital': df['equity'].iloc[-1],
        'Quarterly': quarterly
    }

def project_growth(equity_df, days_forward=365):
    """Projects future equity using log-linear regression."""
    # Fit y = a * e^(bx)  <->  ln(y) = ln(a) + bx
    
    y = np.log(equity_df['equity'].values)
    x = np.arange(len(y))
    
    # Linear fit on log equity
    coeffs = np.polyfit(x, y, 1) # [slope, intercept]
    
    # Generate future dates
    last_date = equity_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_forward + 1)]
    future_x = np.arange(len(x), len(x) + days_forward)
    
    # Project
    predicted_log_y = coeffs[0] * future_x + coeffs[1]
    predicted_equity = np.exp(predicted_log_y)
    
    return pd.Series(predicted_equity, index=future_dates)

# --- Flask Routes ---

@app.route('/')
def home():
    """Main page with backtest results and plot."""
    # 1. Fetch
    df = fetch_binance_data(SYMBOL, START_DATE)
    
    # 2. Indicators
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    # 3. Backtest
    res_df = run_backtest(df)
    
    # 4. Metrics
    metrics = calculate_metrics(res_df)
    
    # 5. Projection
    projection = project_growth(res_df)
    
    # Generate plot
    plot_url = generate_plot(res_df, projection)
    
    # HTML template
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tumbler Strategy Backtest</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .metrics { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .metric-row { display: flex; flex-wrap: wrap; gap: 20px; }
            .metric-box { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .metric-label { color: #7f8c8d; font-size: 14px; }
            .plot-container { text-align: center; margin-top: 30px; }
            .plot-img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
            h1 { color: #2c3e50; }
            .condition-note { background: #ffe6e6; padding: 10px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tumbler Strategy Backtest</h1>
            <p>Symbol: {{ symbol }} | Period: {{ start_date }} to Present</p>
            

            
            <div class="metrics">
                <h2>Performance Metrics</h2>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-label">Initial Capital</div>
                        <div class="metric-value">${{ "%.2f"|format(initial_capital) }}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Final Capital</div>
                        <div class="metric-value">${{ "%.2f"|format(metrics['Final Capital']) }}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value">{{ "%.2f"|format(metrics['Total Return']*100) }}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">CAGR</div>
                        <div class="metric-value">{{ "%.2f"|format(metrics['CAGR']*100) }}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{{ "%.2f"|format(metrics['Sharpe Ratio']) }}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{{ "%.2f"|format(metrics['Sortino Ratio']) }}</div>
                    </div>
                </div>
            </div>
            
            <div class="plot-container">
                <h2>Equity Curve with Condition Highlighting</h2>
                <img src="{{ plot_url }}" alt="Equity Plot" class="plot-img">
            </div>
            
            <div style="margin-top: 30px; font-size: 12px; color: #7f8c8d;">
                <p>Server running on port {{ port }}. Data fetched from Binance API.</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return render_template_string(html_template,
                                 symbol=SYMBOL,
                                 start_date=START_DATE,
                                 initial_capital=INITIAL_CAPITAL,
                                 metrics=metrics,
                                 plot_url=plot_url,
                                 port=PORT)

@app.route('/plot')
def plot_only():
    """Direct endpoint to get the plot image."""
    # Run backtest to get data
    df = fetch_binance_data(SYMBOL, START_DATE)
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    res_df = run_backtest(df)
    projection = project_growth(res_df)
    
    # Create plot in memory
    img = io.BytesIO()
    generate_plot(res_df, projection, save_to=img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

def generate_plot(res_df, projection, save_to=None):
    """Generate the equity plot."""
    plt.figure(figsize=(14, 8))
    
    # Historical Equity
    plt.plot(res_df.index, res_df['equity'], label='Historical Equity', color='blue', linewidth=2)
    
    # Projection
    plt.plot(projection.index, projection.values, label='1-Year Projection', color='green', linestyle='--', linewidth=2)
    
    plt.title(f"Tumbler Strategy Equity Curve\nSMA({SMA_PERIOD_1}/{SMA_PERIOD_2}) | Dynamic Leverage | Port 8080", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Capital (USD) - Log Scale", fontsize=12)
    plt.yscale("log")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    if save_to:
        plt.savefig(save_to, format='png', dpi=100)
        plt.close()
        return None
    else:
        # Convert plot to base64 for HTML embedding
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        plt.close()
        img.seek(0)
        plot_url = 'data:image/png;base64,' + base64.b64encode(img.getvalue()).decode('utf-8')
        return plot_url

# --- Main Execution ---
if __name__ == '__main__':
    print(f"Starting Flask server on port {PORT}...")
    print(f"Open http://localhost:{PORT} in your browser")
    app.run(host='0.0.0.0', port=PORT, debug=False)
