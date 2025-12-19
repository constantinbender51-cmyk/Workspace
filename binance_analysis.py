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

# --- STRATEGY PARAMETERS ---

# MACD STRATEGIES
STRAT_MACD_1H = {
    'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)],
    'weights': [0.45, 0.43, 0.01]
}
STRAT_MACD_4H = {
    'params': [(6, 8, 4), (84, 324, 96), (22, 86, 14)],
    'weights': [0.29, 0.58, 0.64]
}
STRAT_MACD_1D = {
    'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)],
    'weights': [0.87, 0.92, 0.73]
}

# SMA STRATEGIES
STRAT_SMA_1H = {
    'params': [10, 80, 380],
    'weights': [0.0, 1.0, 0.8]
}
STRAT_SMA_4H = {
    'params': [20, 120, 260],
    'weights': [0.4, 0.4, 1.0]
}
STRAT_SMA_1D = {
    'params': [40, 120, 390],
    'weights': [0.6, 0.8, 0.4]
}

# --- DATA FETCHING ---
def fetch_binance_data(symbol, interval, start_year):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    limit = 1000
    all_data = []
    current_start = start_ts
    
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

# --- SIGNAL GENERATION LOGIC ---

def calculate_macd_pos(df, strat_config):
    prices = df['close']
    params = strat_config['params']
    weights = strat_config['weights']
    
    composite_signal = pd.Series(0.0, index=df.index)
    
    for (fast, slow, sig_p), w in zip(params, weights):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig_p, adjust=False).mean()
        
        sig = np.where(macd_line > signal_line, 1.0, -1.0)
        composite_signal += (sig * w)
        
    total_w = sum(weights)
    if total_w == 0: return composite_signal
    return composite_signal / total_w

def calculate_sma_pos(df, strat_config):
    prices = df['close']
    periods = strat_config['params']
    weights = strat_config['weights']
    
    composite_signal = pd.Series(0.0, index=df.index)
    
    for p, w in zip(periods, weights):
        sma = prices.rolling(window=p).mean()
        sig = np.where(prices > sma, 1.0, -1.0)
        sig = pd.Series(sig, index=df.index).fillna(0)
        composite_signal += (sig * w)
        
    total_w = sum(weights)
    if total_w == 0: return composite_signal
    return composite_signal / total_w

def calculate_metrics(returns):
    cum_ret = (1 + returns).cumprod()
    if cum_ret.empty: return {'Total Return': '0%', 'Sharpe Ratio': '0', 'Max Drawdown': '0%'}
    
    total_ret_pct = (cum_ret.iloc[-1] - 1) * 100
    
    sharpe = 0
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    max_dd_pct = drawdown.min() * 100
    
    return {
        'Total Return': f"{total_ret_pct:,.2f}%",
        'Sharpe Ratio': f"{sharpe:.4f}",
        'Max Drawdown': f"{max_dd_pct:.2f}%"
    }

# --- MAIN ANALYSIS ---

def run_ensemble_analysis():
    print("Fetching 1H data...")
    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    
    # Resample (using 'last' to represent closing price of the higher timeframe)
    df_4h = df_1h.resample('4h').last().dropna()
    df_1d = df_1h.resample('1D').last().dropna()
    
    # --- STEP 1: Calculate Native Signals ---
    pos_macd_1h = calculate_macd_pos(df_1h, STRAT_MACD_1H)
    pos_macd_4h = calculate_macd_pos(df_4h, STRAT_MACD_4H)
    pos_macd_1d = calculate_macd_pos(df_1d, STRAT_MACD_1D)
    
    pos_sma_1h  = calculate_sma_pos(df_1h, STRAT_SMA_1H)
    pos_sma_4h  = calculate_sma_pos(df_4h, STRAT_SMA_4H)
    pos_sma_1d  = calculate_sma_pos(df_1d, STRAT_SMA_1D)
    
    # --- STEP 2: Strict Index Alignment (Fixing Lookahead Bias) ---
    # We shift the INDEX of higher timeframes to the moment the bar closes in 1H time.
    # A 4H bar at 08:00 closes at 12:00. In 1H terms (open time), the last bar is 11:00.
    # So we map 08:00 -> 11:00 (+3 hours).
    # A 1D bar at 00:00 closes at 24:00. In 1H terms, the last bar is 23:00.
    # So we map 00:00 -> 23:00 (+23 hours).
    
    pos_macd_4h.index = pos_macd_4h.index + pd.Timedelta(hours=3)
    pos_sma_4h.index  = pos_sma_4h.index  + pd.Timedelta(hours=3)
    
    pos_macd_1d.index = pos_macd_1d.index + pd.Timedelta(hours=23)
    pos_sma_1d.index  = pos_sma_1d.index  + pd.Timedelta(hours=23)
    
    # --- STEP 3: Reindex to 1H Base (Forward Fill) ---
    target_idx = df_1h.index
    
    p1 = pos_macd_1h
    p2 = pos_macd_4h.reindex(target_idx, method='ffill').fillna(0)
    p3 = pos_macd_1d.reindex(target_idx, method='ffill').fillna(0)
    p4 = pos_sma_1h
    p5 = pos_sma_4h.reindex(target_idx, method='ffill').fillna(0)
    p6 = pos_sma_1d.reindex(target_idx, method='ffill').fillna(0)
    
    # --- STEP 4: Ensemble & Global Shift ---
    # Normalized Leverage (Average of 6 strategies -> Max 1.0)
    # Global shift(1) simulates trading at the OPEN of the NEXT bar based on signals from the CLOSE of the current bar.
    # Because we aligned HTF indices to the last 1H bar of their block, this shift correctly trades 
    # the 4H signal starting 1 hour after the 4H bar closes (simulating strictly available data).
    ensemble_pos = ((p1 + p2 + p3 + p4 + p5 + p6) / 6.0).shift(1).fillna(0)
    
    # --- STEP 5: Returns ---
    market_returns = df_1h['close'].pct_change().fillna(0)
    strategy_returns = ensemble_pos * market_returns
    
    # Analysis outputs
    metrics = calculate_metrics(strategy_returns)
    
    signals_df = pd.DataFrame({
        'MACD_1H': p1, 'MACD_4H': p2, 'MACD_1D': p3,
        'SMA_1H': p4, 'SMA_4H': p5, 'SMA_1D': p6,
        'Ensemble': ensemble_pos.shift(-1) 
    }).dropna()
    corr_matrix = signals_df.corr().round(2)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    cum_ret = (1 + strategy_returns).cumprod()
    buy_hold = (1 + market_returns).cumprod()
    if not buy_hold.empty: buy_hold = buy_hold / buy_hold.iloc[0]
    
    ax1.plot(cum_ret.index, cum_ret, label='Ensemble (Normalized)', color='#00ff88', linewidth=1.5)
    ax1.plot(buy_hold.index, buy_hold, label='BTC Buy & Hold', color='white', alpha=0.3, linewidth=1)
    ax1.set_yscale('log')
    ax1.set_title(f"Ensemble Strategy (Zero Lookahead)\nSharpe: {metrics['Sharpe Ratio']} | Return: {metrics['Total Return']}", color='white')
    ax1.grid(True, alpha=0.1)
    ax1.legend()
    
    ax2.plot(ensemble_pos.index, ensemble_pos, color='#00e5ff', linewidth=0.5, alpha=0.8)
    ax2.set_title("Net Exposure / Leverage (-1.0 to +1.0)", color='white')
    ax2.set_ylabel("Leverage", color='white')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.1)
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#444')
    
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_b64 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return metrics, corr_matrix, plot_b64

# --- FLASK ROUTES ---
@app.route('/')
def index():
    metrics, corr, plot_url = run_ensemble_analysis()
    corr_html = corr.to_html(classes='table table-dark table-sm table-bordered', border=0)
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ensemble Strategy Analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; padding: 20px; }
            .card { background: #1e1e1e; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            h2, h4 { color: #00ff88; }
            .metric-box { text-align: center; padding: 15px; border: 1px solid #333; border-radius: 8px; background: #252525; }
            .metric-val { font-size: 1.5rem; font-weight: bold; color: #fff; }
            .metric-lbl { color: #888; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="text-center mb-4">🧪 6-Strategy Ensemble (Zero Bias)</h2>
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="metric-box">
                        <div class="metric-val" style="color:#00ff88">{{ metrics['Total Return'] }}</div>
                        <div class="metric-lbl">Total Cumulative Return</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-box">
                        <div class="metric-val" style="color:#00e5ff">{{ metrics['Sharpe Ratio'] }}</div>
                        <div class="metric-lbl">Sharpe Ratio</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-box">
                        <div class="metric-val" style="color:#ff5555">{{ metrics['Max Drawdown'] }}</div>
                        <div class="metric-lbl">Max Drawdown</div>
                    </div>
                </div>
            </div>
            <div class="card p-2 mb-4">
                <img src="data:image/png;base64,{{ plot_url }}" class="img-fluid">
            </div>
            <div class="card p-4">
                <h4 class="mb-3">Strategy Correlation Matrix</h4>
                <div class="table-responsive">{{ corr_html | safe }}</div>
            </div>
        </div>
    </body>
    </html>
    """, metrics=metrics, corr_html=corr_html, plot_url=plot_url)

if __name__ == '__main__':
    print("Starting Ensemble Analysis Server...")
    app.run(host='0.0.0.0', port=8080)
