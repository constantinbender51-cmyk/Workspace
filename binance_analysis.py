import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_YEAR = 2018
PORT = 8080

# --- Strategy Parameters ---
# Strategy 1 (SMA 120)
S1_SMA = 120
S1_DECAY = 40
S1_STOP_PCT = 0.13  # 13% Stop Loss (Grid Search Winner)

# Strategy 2 (SMA 400)
S2_SMA = 400
S2_PROX_PCT = 0.05  # 5% Proximity
S2_STOP_PCT = 0.27  # 27% Stop Loss

app = Flask(__name__)

# --- Data Fetching ---
def fetch_data():
    print(f"Fetching {SYMBOL} data from Binance starting {START_YEAR}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds():
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Indicators
    df['Asset_Ret'] = df['close'].pct_change().fillna(0.0)
    df['SMA_120'] = df['close'].rolling(window=S1_SMA).mean()
    df['SMA_400'] = df['close'].rolling(window=S2_SMA).mean()
    df['Dist_Pct_400'] = (df['close'] - df['SMA_400']).abs() / df['SMA_400']
    
    df.dropna(inplace=True)
    print(f"Data ready: {len(df)} rows.")
    return df

# --- Run Strategy 1 ---
def run_s1(df):
    closes = df['close'].values
    smas = df['SMA_120'].values
    asset_rets = df['Asset_Ret'].values
    n = len(df)
    
    pos = np.zeros(n)
    strat_ret = np.zeros(n)
    
    curr_trend = 0
    last_sig = -9999
    
    trade_eq = 1.0
    max_trade_eq = 1.0
    stopped = False
    
    stop_thresh = 1.0 - S1_STOP_PCT
    
    # Pre-calc weights
    d_weights = np.zeros(S1_DECAY + 1)
    for d in range(1, S1_DECAY + 1):
        d_weights[d] = 1 - (d / float(S1_DECAY))**2
    
    for i in range(n):
        # YESTERDAY'S Position PnL
        prev = pos[i-1] if i > 0 else 0
        pnl = prev * asset_rets[i]
        strat_ret[i] = pnl
        
        # Track Stop
        if prev != 0 and np.sign(prev) == curr_trend:
            trade_eq *= (1 + pnl)
            if trade_eq > max_trade_eq: max_trade_eq = trade_eq
            if not stopped and trade_eq < (max_trade_eq * stop_thresh):
                stopped = True
        
        # TODAY'S Decision
        trend = 1 if closes[i] > smas[i] else -1
        
        if trend != curr_trend:
            curr_trend = trend
            last_sig = i
            trade_eq = 1.0
            max_trade_eq = 1.0
            stopped = False
            pos[i] = curr_trend * 1.0
        else:
            dt = i - last_sig
            if stopped:
                pos[i] = 0.0
            elif 1 <= dt <= S1_DECAY:
                pos[i] = curr_trend * d_weights[dt]
            else:
                pos[i] = 0.0
                
    return pos, strat_ret

# --- Run Strategy 2 ---
def run_s2(df):
    closes = df['close'].values
    smas = df['SMA_400'].values
    dists = df['Dist_Pct_400'].values
    asset_rets = df['Asset_Ret'].values
    n = len(df)
    
    pos = np.zeros(n)
    strat_ret = np.zeros(n)
    
    curr_trend = 0
    trade_eq = 1.0
    max_trade_eq = 1.0
    stopped = False
    
    stop_thresh = 1.0 - S2_STOP_PCT
    
    for i in range(n):
        trend = 1 if closes[i] > smas[i] else -1
        prox = dists[i] < S2_PROX_PCT
        tgt_w = 0.5 if prox else 1.0
        
        if trend != curr_trend:
            curr_trend = trend
            trade_eq = 1.0
            max_trade_eq = 1.0
            stopped = False
            pos[i] = curr_trend * tgt_w
        else:
            prev = pos[i-1] if i > 0 else 0
            pnl = prev * asset_rets[i]
            strat_ret[i] = pnl
            
            if prev != 0 and np.sign(prev) == curr_trend:
                trade_eq *= (1 + pnl)
                if trade_eq > max_trade_eq: max_trade_eq = trade_eq
                if not stopped and trade_eq < (max_trade_eq * stop_thresh):
                    stopped = True
            
            if stopped:
                if prox:
                    stopped = False
                    trade_eq = 1.0
                    max_trade_eq = 1.0
                    pos[i] = curr_trend * tgt_w
                else:
                    pos[i] = 0.0
            else:
                pos[i] = curr_trend * tgt_w
                
    return pos, strat_ret

# --- Helper Stats ---
def calc_stats(series):
    tot = (series.iloc[-1] - 1) * 100
    daily = series.pct_change().dropna()
    mean = daily.mean()
    std = daily.std()
    shp = (mean/std)*np.sqrt(365) if std != 0 else 0
    
    rmax = series.cummax()
    dd = (series - rmax) / rmax
    mdd = dd.min() * 100
    return tot, shp, mdd

@app.route('/')
def dashboard():
    df = fetch_data()
    
    # 1. Run Individual Strategies
    pos1, ret1 = run_s1(df)
    pos2, ret2 = run_s2(df)
    
    df['Pos_1'] = pos1
    df['Pos_2'] = pos2
    
    # 2. Run Combined Strategy 3
    # Net Position is simple sum. Compounding happens on the Net.
    df['Pos_3'] = df['Pos_1'] + df['Pos_2']
    
    # Enter next day
    df['Active_Pos_3'] = df['Pos_3'].shift(1).fillna(0.0)
    df['Strat_3_Ret'] = df['Active_Pos_3'] * df['Asset_Ret']
    
    # Calculate Equities
    df['Eq_1'] = (1 + pd.Series(ret1, index=df.index)).cumprod()
    df['Eq_2'] = (1 + pd.Series(ret2, index=df.index)).cumprod()
    df['Eq_3'] = (1 + df['Strat_3_Ret']).cumprod()
    
    # Stats
    s1_tot, s1_shp, s1_dd = calc_stats(df['Eq_1'])
    s2_tot, s2_shp, s2_dd = calc_stats(df['Eq_2'])
    s3_tot, s3_shp, s3_dd = calc_stats(df['Eq_3'])
    
    # --- Visualization ---
    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    gs = fig.add_gridspec(4, 1)
    
    # Plot 1: Equity Curves
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_title('Equity Curve: S1 vs S2 vs Combined S3 (Leveraged)', fontsize=14, fontweight='bold')
    ax1.plot(df.index, df['Eq_1'], color='orange', lw=1.5, alpha=0.8, label=f'S1: SMA 120 (Tot: {s1_tot:.0f}%, DD: {s1_dd:.0f}%)')
    ax1.plot(df.index, df['Eq_2'], color='blue', lw=1.5, alpha=0.8, label=f'S2: SMA 400 (Tot: {s2_tot:.0f}%, DD: {s2_dd:.0f}%)')
    ax1.plot(df.index, df['Eq_3'], color='green', lw=2.5, label=f'S3: Combined (Tot: {s3_tot:.0f}%, DD: {s3_dd:.0f}%)')
    ax1.axhline(1.0, color='black', linestyle='--')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Equity Multiple')
    
    # Plot 2: Leverage / Position Size of S3
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_title('Strategy 3 Net Leverage (Pos 1 + Pos 2)', fontsize=12)
    # Use fill_between to show leverage zones
    ax2.fill_between(df.index, df['Pos_3'], 0, where=(df['Pos_3']>0), color='green', alpha=0.3, label='Net Long')
    ax2.fill_between(df.index, df['Pos_3'], 0, where=(df['Pos_3']<0), color='red', alpha=0.3, label='Net Short')
    ax2.plot(df.index, df['Pos_3'], color='black', lw=0.5)
    
    # Highlight max leverage moments
    ax2.axhline(2.0, color='gray', linestyle=':', lw=1)
    ax2.axhline(-2.0, color='gray', linestyle=':', lw=1)
    ax2.set_ylabel('Leverage (x)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot 3: Annual Returns
    ax3 = fig.add_subplot(gs[3, 0])
    ann_1 = (1+pd.Series(ret1, index=df.index)).resample('Y').prod() - 1
    ann_2 = (1+pd.Series(ret2, index=df.index)).resample('Y').prod() - 1
    ann_3 = (1+df['Strat_3_Ret']).resample('Y').prod() - 1
    
    x = np.arange(len(ann_1))
    w = 0.25
    years = ann_1.index.year
    
    ax3.bar(x - w, ann_1*100, w, label='S1', color='orange', alpha=0.7)
    ax3.bar(x, ann_2*100, w, label='S2', color='blue', alpha=0.7)
    ax3.bar(x + w, ann_3*100, w, label='S3 (Combined)', color='green', alpha=0.9)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.set_title('Annual Returns Comparison (%)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Combined Strategy</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; padding: 20px; text-align: center; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stat-box {{ display: flex; justify-content: space-around; gap: 15px; margin-bottom: 20px; }}
            .card {{ flex: 1; padding: 15px; border-radius: 8px; color: white; }}
            .c1 {{ background: linear-gradient(135deg, #f09819 0%, #edde5d 100%); }}
            .c2 {{ background: linear-gradient(135deg, #5b86e5 0%, #36d1dc 100%); }}
            .c3 {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border: 2px solid #000; transform: scale(1.05); }}
            .val {{ font-size: 1.5em; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Combined Leveraged Strategy (S3)</h1>
            <p>S3 = S1 (120 SMA, 13% Stop) + S2 (400 SMA, 27% Stop). Max Leverage 2x.</p>
            
            <div class="stat-box">
                <div class="card c1">
                    <h3>Strategy 1</h3>
                    <div class="val">{s1_tot:,.0f}%</div>
                    <div>Return</div>
                    <div>Sharpe: {s1_shp:.2f}</div>
                    <div>Max DD: {s1_dd:.1f}%</div>
                </div>
                <div class="card c2">
                    <h3>Strategy 2</h3>
                    <div class="val">{s2_tot:,.0f}%</div>
                    <div>Return</div>
                    <div>Sharpe: {s2_shp:.2f}</div>
                    <div>Max DD: {s2_dd:.1f}%</div>
                </div>
                <div class="card c3">
                    <h3>Strategy 3 (Combined)</h3>
                    <div class="val">{s3_tot:,.0f}%</div>
                    <div>Return</div>
                    <div>Sharpe: {s3_shp:.2f}</div>
                    <div>Max DD: {s3_dd:.1f}%</div>
                </div>
            </div>
            
            <img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;" />
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting Combined S3 analysis on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
