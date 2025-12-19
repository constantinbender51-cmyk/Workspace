import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
from datetime import datetime
import time

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTCUSDT'
START_YEAR = 2018

# --- "FOUND SPECS" (Hardcoded Strategy Parameters) ---
# These match the optimized parameters provided in the conversation history.

# MACD: [(Fast, Slow, Sig), (Fast, Slow, Sig), (Fast, Slow, Sig)], [W1, W2, W3]
STRAT_MACD_1H = {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]}
STRAT_MACD_4H = {'params': [(6, 8, 4), (84, 324, 96), (22, 86, 14)], 'weights': [0.29, 0.58, 0.64]}
STRAT_MACD_1D = {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]}

# SMA: [P1, P2, P3], [W1, W2, W3]
STRAT_SMA_1H = {'params': [10, 80, 380], 'weights': [0.0, 1.0, 0.8]}
STRAT_SMA_4H = {'params': [20, 120, 260], 'weights': [0.4, 0.4, 1.0]}
STRAT_SMA_1D = {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}

# ENSEMBLE WEIGHTS (Contribution of each strategy to the final vote)
# Set to Equal Weights (1.0 each) as specific GA weights were not provided in text.
# Normalized automatically.
ENSEMBLE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
LABELS = ['MACD 1H', 'MACD 4H', 'MACD 1D', 'SMA 1H', 'SMA 4H', 'SMA 1D']

# --- DATA ENGINE ---

def fetch_binance_data(symbol, interval, start_year):
    """Fetches full history of 1H data to ensure accurate signal generation."""
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    limit = 1000
    all_data = []
    current_start = start_ts
    
    print(f"Fetching {interval} data from {start_year}...")
    
    while current_start < end_ts:
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_start, 'limit': limit}
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data: break
            all_data.extend(data)
            current_start = data[-1][0] + 1
            if len(data) < limit: break 
            time.sleep(0.05)
        except: break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['close']]

def calculate_macd_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite = np.zeros(len(prices))
    for (f, s, sig_p), w in zip(params, weights):
        fast = prices.ewm(span=f, adjust=False).mean()
        slow = prices.ewm(span=s, adjust=False).mean()
        macd = fast - slow
        sig_line = macd.ewm(span=sig_p, adjust=False).mean()
        composite += np.where(macd > sig_line, 1.0, -1.0) * w
    total_w = sum(weights)
    return composite / total_w if total_w > 0 else composite

def calculate_sma_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite = np.zeros(len(prices))
    for p, w in zip(params, weights):
        sma = prices.rolling(window=p).mean()
        composite += np.where(prices > sma, 1.0, -1.0) * w
    total_w = sum(weights)
    composite = np.nan_to_num(composite)
    return composite / total_w if total_w > 0 else composite

def run_backtest_daily():
    # 1. Fetch High Res Data (1H)
    # We need 1H data to accurately calculate the 1H and 4H strategy signals
    # BEFORE we downsample to Daily.
    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    
    # 2. Generate Derived Timeframes
    df_4h = df_1h.resample('4h').last().dropna()
    df_1d = df_1h.resample('1D').last().dropna()
    
    # 3. Calculate Raw Signals on Native Timeframes
    # (These match the "Found Specs" logic exactly)
    s1 = pd.Series(calculate_macd_pos(df_1h['close'], STRAT_MACD_1H), index=df_1h.index)
    s2 = pd.Series(calculate_macd_pos(df_4h['close'], STRAT_MACD_4H), index=df_4h.index)
    s3 = pd.Series(calculate_macd_pos(df_1d['close'], STRAT_MACD_1D), index=df_1d.index)
    s4 = pd.Series(calculate_sma_pos(df_1h['close'], STRAT_SMA_1H), index=df_1h.index)
    s5 = pd.Series(calculate_sma_pos(df_4h['close'], STRAT_SMA_4H), index=df_4h.index)
    s6 = pd.Series(calculate_sma_pos(df_1d['close'], STRAT_SMA_1D), index=df_1d.index)
    
    # 4. Align HTF Signals to 1H (Fixing Lookahead)
    # Shift indices to when the bar actually closes
    s2.index = s2.index + pd.Timedelta(hours=3) # 4H close
    s5.index = s5.index + pd.Timedelta(hours=3)
    s3.index = s3.index + pd.Timedelta(hours=23) # 1D close
    s6.index = s6.index + pd.Timedelta(hours=23)
    
    # Reindex to 1H Base to create a continuous signal stream
    # ffill() propagates the last known signal forward
    target_idx = df_1h.index
    s2 = s2.reindex(target_idx, method='ffill').fillna(0)
    s3 = s3.reindex(target_idx, method='ffill').fillna(0)
    s5 = s5.reindex(target_idx, method='ffill').fillna(0)
    s6 = s6.reindex(target_idx, method='ffill').fillna(0)
    
    # 5. DOWNSAMPLE TO DAILY (Trading Frequency: Once a Day)
    # We take the state of the signals at the END of each day (23:00 UTC)
    # This simulates a trader checking their bot once a day before close.
    
    # Combine into DataFrame
    df_signals_1h = pd.DataFrame({
        'S1': s1, 'S2': s2, 'S3': s3, 
        'S4': s4, 'S5': s5, 'S6': s6
    }, index=target_idx)
    
    # Resample to Daily, taking the LAST value of the day
    df_daily_signals = df_signals_1h.resample('1D').last().dropna()
    
    # Get Daily Prices (Close) for return calc
    df_daily_price = df_1d['close'].reindex(df_daily_signals.index)
    
    # 6. Apply Ensemble Weights
    weights = np.array(ENSEMBLE_WEIGHTS)
    total_w = np.sum(weights)
    
    # Matrix mult: (Days x 6) dot (6,) -> (Days,)
    raw_ensemble = df_daily_signals.dot(weights) / total_w
    
    # 7. Calculate Returns (Shift 1 Day)
    # We trade at the Open of D+1 based on Signal at Close of D
    # Or simplified: Return of D+1 * Signal of D
    final_pos = raw_ensemble.shift(1).fillna(0)
    
    daily_returns = df_daily_price.pct_change().fillna(0)
    strat_returns = final_pos * daily_returns
    
    # 8. Metrics
    cum_ret = (1 + strat_returns).cumprod()
    
    total_ret_pct = (cum_ret.iloc[-1] - 1) * 100
    std = strat_returns.std()
    sharpe = (strat_returns.mean() / std * np.sqrt(365)) if std > 0 else 0
    
    roll_max = cum_ret.cummax()
    dd = (cum_ret - roll_max) / roll_max
    max_dd = dd.min() * 100
    
    metrics = {
        'Total Return': f"{total_ret_pct:,.2f}%",
        'Sharpe Ratio': f"{sharpe:.4f}",
        'Max Drawdown': f"{max_dd:.2f}%",
        'Win Rate': f"{len(strat_returns[strat_returns > 0]) / len(strat_returns[strat_returns != 0]) * 100:.1f}%"
    }
    
    # 9. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    bh = (1 + daily_returns).cumprod()
    bh = bh / bh.iloc[0]
    
    ax1.plot(cum_ret.index, cum_ret, color='#00ff88', label='Ensemble (Daily Rebalance)')
    ax1.plot(bh.index, bh, color='white', alpha=0.3, label='BTC Buy & Hold')
    ax1.set_yscale('log')
    ax1.set_title("Strategy Performance (Daily Execution)")
    ax1.grid(True, alpha=0.1)
    ax1.legend()
    
    ax2.plot(final_pos.index, final_pos, color='#00e5ff', linewidth=0.5, drawstyle='steps-post')
    ax2.set_title("Net Leverage (-1.0 to 1.0)")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.1)
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e1e1e')
        for s in ax.spines.values(): s.set_color('#444')
        ax.tick_params(colors='white')
        
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Correlation
    corr = df_daily_signals.corr().round(2)
    corr_html = corr.to_html(classes='table table-dark table-sm', border=0)
    
    return metrics, plot_url, corr_html

@app.route('/')
def index():
    metrics, plot_url, corr_html = run_backtest_daily()
    
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Daily Ensemble Strategy</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #0f0f0f; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            .header-section { padding: 40px 0; border-bottom: 1px solid #333; margin-bottom: 30px; }
            .metric-card { background: #1a1a1a; border: 1px solid #333; padding: 20px; border-radius: 8px; text-align: center; }
            .metric-val { font-size: 2rem; font-weight: 700; margin-bottom: 5px; }
            .card { background: #1a1a1a; border: 1px solid #333; margin-bottom: 20px; }
            .guide-step { border-left: 3px solid #00ff88; padding-left: 15px; margin-bottom: 25px; }
            h2, h4, h5 { color: #f0f0f0; }
            .text-accent { color: #00ff88; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-section text-center">
                <h1 class="display-5 fw-bold text-accent">Final Strategy Report</h1>
                <p class="lead text-muted">Daily Execution • 6-Model Ensemble • Optimized Specs</p>
            </div>

            <!-- Executive Summary -->
            <div class="row mb-5">
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-val text-accent">{{ metrics['Total Return'] }}</div>
                        <div class="text-muted text-uppercase small">Total Return</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-val text-info">{{ metrics['Sharpe Ratio'] }}</div>
                        <div class="text-muted text-uppercase small">Sharpe Ratio</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-val text-danger">{{ metrics['Max Drawdown'] }}</div>
                        <div class="text-muted text-uppercase small">Max Drawdown</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="metric-val text-warning">{{ metrics['Win Rate'] }}</div>
                        <div class="text-muted text-uppercase small">Win Rate</div>
                    </div>
                </div>
            </div>

            <!-- Charts -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card p-2">
                        <img src="data:image/png;base64,{{ plot_url }}" class="img-fluid rounded">
                    </div>
                </div>
            </div>

            <div class="row">
                <!-- Implementation Guide -->
                <div class="col-lg-7">
                    <div class="card p-4 h-100">
                        <h4 class="mb-4 text-accent">🚀 Daily Execution Protocol</h4>
                        
                        <div class="guide-step">
                            <h5>1. Frequency: Once a Day</h5>
                            <p class="mb-0 text-muted">Execute logic daily at <strong>00:00 UTC</strong> (Daily Close).</p>
                        </div>

                        <div class="guide-step">
                            <h5>2. Data Requirement</h5>
                            <p class="mb-0 text-muted">
                                While trading is daily, the strategy logic requires intra-day data:
                                <br>- Fetch the last 24 1H candles.
                                <br>- Fetch the last 6 4H candles.
                                <br>- Fetch the last Daily candle.
                            </p>
                        </div>

                        <div class="guide-step">
                            <h5>3. Calculation Logic</h5>
                            <p class="text-muted">Compute the 6 signals using the specific parameters below. Average them:</p>
                            <code class="d-block bg-black p-3 rounded mb-2 text-warning">
                                Target_Leverage = (S1 + S2 + S3 + S4 + S5 + S6) / 6.0
                            </code>
                            <p class="mb-0 text-muted">This results in a target between -1.0 (Full Short) and 1.0 (Full Long).</p>
                        </div>
                    </div>
                </div>

                <!-- Specs -->
                <div class="col-lg-5">
                    <div class="card p-4 h-100">
                        <h5 class="mb-3">Found Specifications</h5>
                        <ul class="list-group list-group-flush bg-transparent">
                            <li class="list-group-item bg-transparent text-white border-secondary">
                                <strong>MACD 1H:</strong> 97/366/47, 15/40/11, 16/55/13
                            </li>
                            <li class="list-group-item bg-transparent text-white border-secondary">
                                <strong>MACD 4H:</strong> 6/8/4, 84/324/96, 22/86/14
                            </li>
                            <li class="list-group-item bg-transparent text-white border-secondary">
                                <strong>MACD 1D:</strong> 52/64/61, 5/6/4, 17/18/16
                            </li>
                            <li class="list-group-item bg-transparent text-white border-secondary">
                                <strong>SMA 1H:</strong> 10, 80, 380
                            </li>
                            <li class="list-group-item bg-transparent text-white border-secondary">
                                <strong>SMA 4H:</strong> 20, 120, 260
                            </li>
                            <li class="list-group-item bg-transparent text-white border-secondary">
                                <strong>SMA 1D:</strong> 40, 120, 390
                            </li>
                        </ul>
                        <div class="mt-3">
                            <small class="text-muted">Correlation Matrix:</small>
                            {{ corr_html | safe }}
                        </div>
                    </div>
                </div>
            </div>
            
            <footer class="text-center py-4 text-muted small">
                Generated by Daily Ensemble Strategy Engine • No Financial Advice
            </footer>
        </div>
    </body>
    </html>
    """, metrics=metrics, plot_url=plot_url, corr_html=corr_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
