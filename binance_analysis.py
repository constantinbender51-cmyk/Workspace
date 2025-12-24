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

# ==========================================
#        STRATEGY CONFIGURATION
# ==========================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_YEAR = 2018
PORT = 8080

# --- Strategy 1: Tactical Trend (SMA 120) ---
S1_SMA = 120
S1_DECAY_DAYS = 40
S1_STOP_PCT = 0.13    # 13% Trailing Stop

# --- Strategy 2: Core Trend (SMA 400) ---
S2_SMA = 400
S2_PROX_PCT = 0.05    # 5% Proximity Threshold
S2_STOP_PCT = 0.27    # 27% Trailing Stop

app = Flask(__name__)

# ==========================================
#            DATA ENGINE
# ==========================================
def fetch_data():
    print(f"Fetching {SYMBOL} data from Binance starting {START_YEAR}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds(): break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Pre-calculate Indicators
    df['Asset_Ret'] = df['close'].pct_change().fillna(0.0)
    df['SMA_120'] = df['close'].rolling(window=S1_SMA).mean()
    df['SMA_400'] = df['close'].rolling(window=S2_SMA).mean()
    df['Dist_Pct_400'] = (df['close'] - df['SMA_400']).abs() / df['SMA_400']
    
    df.dropna(inplace=True)
    return df

# ==========================================
#          STRATEGY LOGIC KERNELS
# ==========================================

def run_s1_logic(df):
    """
    Strategy 1: 
    - Entry: SMA 120 Crossover
    - Exit: Weighted Decay (40 days) OR 13% Trailing Stop
    """
    closes = df['close'].values
    smas = df['SMA_120'].values
    asset_rets = df['Asset_Ret'].values
    n = len(df)
    
    pos = np.zeros(n)
    
    curr_trend = 0
    last_sig = -9999
    trade_eq = 1.0
    max_trade_eq = 1.0
    stopped = False
    
    # Pre-calc decay curve
    d_weights = np.zeros(S1_DECAY_DAYS + 1)
    for d in range(1, S1_DECAY_DAYS + 1):
        d_weights[d] = 1 - (d / float(S1_DECAY_DAYS))**2
        
    stop_thresh = 1.0 - S1_STOP_PCT

    for i in range(n):
        # 1. Update Trade State (PnL from yesterday)
        prev = pos[i-1] if i > 0 else 0
        pnl = prev * asset_rets[i]
        
        if prev != 0 and np.sign(prev) == curr_trend:
            trade_eq *= (1 + pnl)
            if trade_eq > max_trade_eq: max_trade_eq = trade_eq
            if not stopped and trade_eq < (max_trade_eq * stop_thresh):
                stopped = True
        
        # 2. Logic for Today
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
            elif 1 <= dt <= S1_DECAY_DAYS:
                pos[i] = curr_trend * d_weights[dt]
            else:
                pos[i] = 0.0
                
    return pos

def run_s2_logic(df):
    """
    Strategy 2:
    - Entry: SMA 400 Crossover
    - Sizing: 0.5x if price within 5% of SMA, else 1.0x
    - Exit: 27% Trailing Stop
    - Re-entry: If stopped, re-enter if price returns to < 5% proximity
    """
    closes = df['close'].values
    smas = df['SMA_400'].values
    dists = df['Dist_Pct_400'].values
    asset_rets = df['Asset_Ret'].values
    n = len(df)
    
    pos = np.zeros(n)
    
    curr_trend = 0
    trade_eq = 1.0
    max_trade_eq = 1.0
    stopped = False
    
    stop_thresh = 1.0 - S2_STOP_PCT

    for i in range(n):
        trend = 1 if closes[i] > smas[i] else -1
        prox = dists[i] < S2_PROX_PCT
        tgt_w = 0.5 if prox else 1.0
        
        # 1. New Trend
        if trend != curr_trend:
            curr_trend = trend
            trade_eq = 1.0
            max_trade_eq = 1.0
            stopped = False
            pos[i] = curr_trend * tgt_w
            
        else:
            # 2. Manage Existing Trend
            prev = pos[i-1] if i > 0 else 0
            pnl = prev * asset_rets[i]
            
            if prev != 0 and np.sign(prev) == curr_trend:
                trade_eq *= (1 + pnl)
                if trade_eq > max_trade_eq: max_trade_eq = trade_eq
                if not stopped and trade_eq < (max_trade_eq * stop_thresh):
                    stopped = True
            
            if stopped:
                # Re-entry Check
                if prox:
                    stopped = False
                    trade_eq = 1.0
                    max_trade_eq = 1.0
                    pos[i] = curr_trend * tgt_w
                else:
                    pos[i] = 0.0
            else:
                pos[i] = curr_trend * tgt_w
                
    return pos

# ==========================================
#          ANALYTICS & PLOTTING
# ==========================================

def calc_metrics(series):
    tot_ret = (series.iloc[-1] - 1) * 100
    daily_rets = series.pct_change().dropna()
    mean = daily_rets.mean()
    std = daily_rets.std()
    sharpe = (mean/std)*np.sqrt(365) if std != 0 else 0
    
    rmax = series.cummax()
    dd = (series - rmax) / rmax
    max_dd = dd.min() * 100
    
    # Win Rate (Annual)
    ann_rets = series.resample('Y').last().pct_change().dropna()
    win_years = len(ann_rets[ann_rets > 0])
    tot_years = len(ann_rets)
    
    return {
        'Return': tot_ret,
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'CAGR': ((series.iloc[-1])**(1/(len(series)/365)) - 1)*100
    }

def generate_plots(df):
    # Plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(5, 2)
    
    # 1. Main Equity Curves (Linear)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['Eq_1'], color='#f39c12', label='Tactical (S1)', alpha=0.7, lw=1.5)
    ax1.plot(df.index, df['Eq_2'], color='#3498db', label='Strategic (S2)', alpha=0.7, lw=1.5)
    ax1.plot(df.index, df['Eq_3'], color='#2ecc71', label='Combined (S3)', lw=2.5)
    ax1.set_title('Strategy Equity Growth (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_ylabel('Equity Multiple')
    
    # 2. Leverage Analysis
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(df.index, df['Pos_3'], 0, where=df['Pos_3']>0, color='#2ecc71', alpha=0.3, label='Net Long')
    ax2.fill_between(df.index, df['Pos_3'], 0, where=df['Pos_3']<0, color='#e74c3c', alpha=0.3, label='Net Short')
    ax2.plot(df.index, df['Pos_3'], color='#2c3e50', lw=0.5)
    ax2.set_title('System Leverage Profile (-2x to +2x)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Net Exposure')
    ax2.legend(loc='upper right')
    
    # 3. Drawdown Curves
    ax3 = fig.add_subplot(gs[2, :])
    dd1 = (df['Eq_1'] / df['Eq_1'].cummax()) - 1
    dd2 = (df['Eq_2'] / df['Eq_2'].cummax()) - 1
    dd3 = (df['Eq_3'] / df['Eq_3'].cummax()) - 1
    ax3.plot(df.index, dd3, color='#e74c3c', lw=1, label='Combined Drawdown')
    ax3.fill_between(df.index, dd3, 0, color='#e74c3c', alpha=0.1)
    ax3.set_title('Drawdown Risk Profile', fontsize=14, fontweight='bold')
    ax3.set_ylabel('% Drawdown')
    
    # 4. Annual Returns
    ax4 = fig.add_subplot(gs[3, :])
    ann_ret = df['Eq_3'].resample('Y').last().pct_change()
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in ann_ret]
    ax4.bar(ann_ret.index.year, ann_ret.values * 100, color=colors, alpha=0.8)
    ax4.axhline(0, color='black', lw=1)
    ax4.set_title('Annual Returns (Combined)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Return %')
    
    # 5. Correlation Heatmap (Rolling 6M)
    ax5 = fig.add_subplot(gs[4, 0])
    roll_corr = df['Strat_1_Ret'].rolling(180).corr(df['Strat_2_Ret'])
    ax5.plot(df.index, roll_corr, color='purple', lw=1)
    ax5.set_title('6-Month Rolling Correlation (S1 vs S2)', fontsize=12)
    ax5.axhline(0, color='black', linestyle='--')
    ax5.set_ylim(-1, 1)

    # 6. Scatter
    ax6 = fig.add_subplot(gs[4, 1])
    ax6.scatter(df['Strat_1_Ret'], df['Strat_2_Ret'], alpha=0.1, s=10, color='gray')
    ax6.set_title('Daily Return Scatter (Diversification Check)', fontsize=12)
    ax6.set_xlabel('S1 Return')
    ax6.set_ylabel('S2 Return')

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getbuffer()).decode("ascii")

# ==========================================
#              WEB SERVER
# ==========================================

@app.route('/')
def dashboard():
    df = fetch_data()
    
    # Run Strategies
    df['Pos_1'] = run_s1_logic(df)
    df['Pos_2'] = run_s2_logic(df)
    
    # Combined
    df['Pos_3'] = df['Pos_1'] + df['Pos_2']
    
    # Returns
    df['Strat_1_Ret'] = df['Pos_1'].shift(1).fillna(0) * df['Asset_Ret']
    df['Strat_2_Ret'] = df['Pos_2'].shift(1).fillna(0) * df['Asset_Ret']
    df['Strat_3_Ret'] = df['Pos_3'].shift(1).fillna(0) * df['Asset_Ret']
    
    # Equity
    df['Eq_1'] = (1 + df['Strat_1_Ret']).cumprod()
    df['Eq_2'] = (1 + df['Strat_2_Ret']).cumprod()
    df['Eq_3'] = (1 + df['Strat_3_Ret']).cumprod()
    
    # Metrics
    m1 = calc_metrics(df['Eq_1'])
    m2 = calc_metrics(df['Eq_2'])
    m3 = calc_metrics(df['Eq_3'])
    
    plot_data = generate_plots(df)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Strategy Whitepaper: Combined Alpha</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
            body {{ font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #1e293b; }}
            .metric-card {{ transition: transform 0.2s; }}
            .metric-card:hover {{ transform: translateY(-2px); }}
        </style>
    </head>
    <body class="bg-gray-50">
        
        <!-- Header -->
        <header class="bg-indigo-900 text-white py-12">
            <div class="container mx-auto px-6">
                <div class="flex flex-col md:flex-row justify-between items-center">
                    <div>
                        <h1 class="text-4xl font-bold mb-2">Dual-Horizon Trend System</h1>
                        <p class="text-indigo-200 text-lg">Tactical Decay (SMA 120) + Strategic Core (SMA 400)</p>
                    </div>
                    <div class="mt-6 md:mt-0 text-right">
                        <div class="text-sm uppercase tracking-wide opacity-75">Symbol</div>
                        <div class="text-2xl font-bold">{SYMBOL}</div>
                        <div class="text-sm opacity-75 mt-1">{START_YEAR} - Present</div>
                    </div>
                </div>
            </div>
        </header>

        <main class="container mx-auto px-6 -mt-8">
            
            <!-- KPI Grid -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                <!-- S1 Card -->
                <div class="bg-white rounded-lg shadow-lg p-6 metric-card border-t-4 border-yellow-500">
                    <h3 class="text-gray-500 text-sm font-bold uppercase tracking-wider">Tactical Layer (S1)</h3>
                    <div class="mt-2 flex justify-between items-end">
                        <div class="text-3xl font-bold text-gray-800">{m1['Return']:,.0f}%</div>
                        <div class="text-green-600 font-semibold">{m1['Sharpe']:.2f} Sharpe</div>
                    </div>
                    <div class="mt-4 text-sm text-gray-600">
                        <div class="flex justify-between py-1 border-b"><span>Max Drawdown</span> <span class="text-red-500">{m1['MaxDD']:.1f}%</span></div>
                        <div class="flex justify-between py-1"><span>Risk Model</span> <span>40d Decay + 13% Stop</span></div>
                    </div>
                </div>

                <!-- S2 Card -->
                <div class="bg-white rounded-lg shadow-lg p-6 metric-card border-t-4 border-blue-500">
                    <h3 class="text-gray-500 text-sm font-bold uppercase tracking-wider">Core Layer (S2)</h3>
                    <div class="mt-2 flex justify-between items-end">
                        <div class="text-3xl font-bold text-gray-800">{m2['Return']:,.0f}%</div>
                        <div class="text-green-600 font-semibold">{m2['Sharpe']:.2f} Sharpe</div>
                    </div>
                    <div class="mt-4 text-sm text-gray-600">
                        <div class="flex justify-between py-1 border-b"><span>Max Drawdown</span> <span class="text-red-500">{m2['MaxDD']:.1f}%</span></div>
                        <div class="flex justify-between py-1"><span>Risk Model</span> <span>5% Prox + 27% Stop</span></div>
                    </div>
                </div>

                <!-- Combined Card -->
                <div class="bg-white rounded-lg shadow-lg p-6 metric-card border-t-4 border-green-500 transform scale-105 z-10">
                    <h3 class="text-gray-500 text-sm font-bold uppercase tracking-wider">Combined System (S3)</h3>
                    <div class="mt-2 flex justify-between items-end">
                        <div class="text-4xl font-bold text-gray-900">{m3['Return']:,.0f}%</div>
                        <div class="text-green-600 font-bold bg-green-100 px-2 py-1 rounded">{m3['Sharpe']:.2f} Sharpe</div>
                    </div>
                    <div class="mt-4 text-sm text-gray-600">
                        <div class="flex justify-between py-1 border-b"><span>Max Drawdown</span> <span class="font-bold text-red-600">{m3['MaxDD']:.1f}%</span></div>
                        <div class="flex justify-between py-1"><span>CAGR</span> <span class="font-bold text-gray-800">{m3['CAGR']:.1f}%</span></div>
                    </div>
                </div>
            </div>

            <!-- Strategy Logic Section -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                <div class="bg-white rounded-lg shadow p-8">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Strategy Logic</h2>
                    <ul class="space-y-4">
                        <li class="flex items-start">
                            <span class="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-indigo-100 text-indigo-600 rounded-full font-bold mr-3">1</span>
                            <div>
                                <h4 class="font-bold text-gray-800">Dual-Timeframe Alpha</h4>
                                <p class="text-gray-600 text-sm">Combines a faster tactical trend (SMA 120) with a slower structural trend (SMA 400). The divergence between these signals creates natural hedging opportunities (e.g., Long S2 + Short S1 = Neutral).</p>
                            </div>
                        </li>
                        <li class="flex items-start">
                            <span class="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-indigo-100 text-indigo-600 rounded-full font-bold mr-3">2</span>
                            <div>
                                <h4 class="font-bold text-gray-800">Tactical Decay (S1)</h4>
                                <p class="text-gray-600 text-sm">Exploits post-breakout momentum. Position size starts at 100% and decays parabolically to 0% over 40 days ($W = 1 - (d/40)^2$). Includes a tight 13% trailing stop to cut false breakouts early.</p>
                            </div>
                        </li>
                        <li class="flex items-start">
                            <span class="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-indigo-100 text-indigo-600 rounded-full font-bold mr-3">3</span>
                            <div>
                                <h4 class="font-bold text-gray-800">Strategic Re-entry (S2)</h4>
                                <p class="text-gray-600 text-sm">Captures macro trends. Uses a 27% wide stop to survive volatility. If stopped out, it monitors the SMA 400 daily; if price returns within 5% proximity, it re-enters at 0.5x leverage to catch the recovery.</p>
                            </div>
                        </li>
                    </ul>
                </div>

                <div class="bg-white rounded-lg shadow p-8">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Risk Management</h2>
                    <div class="space-y-6">
                        <div>
                            <div class="flex justify-between text-sm font-semibold mb-1">
                                <span>Leverage Profile</span>
                                <span class="text-indigo-600">-2.0x to +2.0x</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2">
                                <div class="bg-indigo-600 h-2 rounded-full" style="width: 100%"></div>
                            </div>
                            <p class="text-xs text-gray-500 mt-1">Maximum leverage occurs only when both tactical and strategic trends align.</p>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-4">
                            <div class="bg-red-50 p-4 rounded border border-red-100">
                                <span class="block text-red-600 font-bold text-sm">S1 Stop Loss</span>
                                <span class="text-2xl font-bold text-gray-800">13%</span>
                                <span class="block text-xs text-gray-500">Trailing from Peak</span>
                            </div>
                            <div class="bg-red-50 p-4 rounded border border-red-100">
                                <span class="block text-red-600 font-bold text-sm">S2 Stop Loss</span>
                                <span class="text-2xl font-bold text-gray-800">27%</span>
                                <span class="block text-xs text-gray-500">Trailing from Peak</span>
                            </div>
                        </div>
                        
                         <div class="bg-indigo-50 p-4 rounded border border-indigo-100">
                            <span class="block text-indigo-600 font-bold text-sm">Hedging Mechanism</span>
                            <p class="text-sm text-gray-700 mt-1">If S1 flips Short while S2 remains Long, the net position becomes Neutral (0.0x) or partially hedged, protecting capital during choppy reversals.</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Plots -->
            <div class="bg-white rounded-lg shadow-lg p-4 mb-12">
                <img src="data:image/png;base64,{plot_data}" class="w-full h-auto rounded" />
            </div>

        </main>
        
        <footer class="bg-gray-800 text-gray-400 py-8 text-center">
            <p>&copy; 2025 Algorithmic Trading Systems. Generated for Internal Review.</p>
        </footer>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    print(f"Starting Whitepaper Server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
