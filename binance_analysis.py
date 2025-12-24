import os
import io
import base64
import ccxt
import pandas as pd
import matplotlib
# Force non-interactive backend for Cloud/Headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template_string

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2017-01-01T00:00:00Z'
INITIAL_CAPITAL = 10000.0

# Strategy Params (Must match planner.py)
S1_SMA = 120
S1_DECAY_DAYS = 40
S1_STOP_PCT = 0.13
S2_SMA = 400
S2_PROX_PCT = 0.05
S2_STOP_PCT = 0.27

# --- Data Fetching ---
def fetch_binance_data():
    """Fetches data, caching it to disk to speed up subsequent reloads"""
    data_file = 'binance_data.csv'
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        # Simple check to ensure we have enough data, otherwise re-download
        if len(df) > 100: 
            return df

    exchange = ccxt.binance()
    since = exchange.parse8601(START_DATE)
    all_ohlcv = []
    
    # Fetch chunked data
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1000: break
        except Exception as e:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_csv(data_file)
    return df

# --- Strategy Logic ---
def calculate_decay_weight(entry_date, current_date):
    if pd.isna(entry_date): return 1.0
    days_since = (current_date - entry_date).days
    if days_since >= S1_DECAY_DAYS: return 0.0
    weight = 1.0 - (days_since / S1_DECAY_DAYS) ** 2
    return max(0.0, weight)

class Backtester:
    def __init__(self, df):
        self.df = df.copy()
        self.equity = INITIAL_CAPITAL
        self.equity_curve = []
        self.leverage_curve = []
        self.drawdown_curve = []
        self.state = {
            "s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False},
            "s2": {"peak_equity": 0.0, "stopped": False}
        }
        
        # Calculate Indicators
        self.df['SMA120'] = self.df['close'].rolling(window=S1_SMA).mean()
        self.df['SMA400'] = self.df['close'].rolling(window=S2_SMA).mean()
        
    def update_stops(self, current_equity):
        # S1 Stop
        s1 = self.state["s1"]
        if current_equity > s1["peak_equity"]: s1["peak_equity"] = current_equity
        if s1["peak_equity"] > 0:
            dd = (s1["peak_equity"] - current_equity) / s1["peak_equity"]
            if dd > S1_STOP_PCT: s1["stopped"] = True; s1["entry_date"] = None
            
        # S2 Stop
        s2 = self.state["s2"]
        if current_equity > s2["peak_equity"]: s2["peak_equity"] = current_equity
        if s2["peak_equity"] > 0:
            dd = (s2["peak_equity"] - current_equity) / s2["peak_equity"]
            if dd > S2_STOP_PCT: s2["stopped"] = True

    def run(self):
        start_idx = max(S1_SMA, S2_SMA)
        current_leverage = 0.0
        
        for i in range(start_idx, len(self.df)):
            date = self.df.index[i]
            price = self.df['close'].iloc[i]
            prev_price = self.df['close'].iloc[i-1]
            
            # 1. PnL Calculation
            ret = (price - prev_price) / prev_price
            self.equity *= (1 + (ret * current_leverage))
            
            # 2. Update Stops
            self.update_stops(self.equity)
            
            # 3. Generate Next Signal
            sma120 = self.df['SMA120'].iloc[i]
            sma400 = self.df['SMA400'].iloc[i]
            
            # S1 Logic
            s1_lev = 0.0
            if price > sma120:
                if not self.state["s1"]["stopped"]:
                    if self.state["s1"]["entry_date"] is None: self.state["s1"]["entry_date"] = date
                    s1_lev = 1.0 * calculate_decay_weight(self.state["s1"]["entry_date"], date)
            else:
                if self.state["s1"]["stopped"]: 
                    self.state["s1"]["stopped"] = False
                    self.state["s1"]["peak_equity"] = 0.0
                self.state["s1"]["entry_date"] = date
                s1_lev = -1.0 * calculate_decay_weight(self.state["s1"]["entry_date"], date)

            # S2 Logic
            s2_lev = 0.0
            if price > sma400:
                if not self.state["s2"]["stopped"]: s2_lev = 1.0
                else:
                    if ((price - sma400)/sma400) < S2_PROX_PCT:
                        self.state["s2"]["stopped"] = False; self.state["s2"]["peak_equity"] = 0.0; s2_lev = 0.5
            else:
                if self.state["s2"]["stopped"]: self.state["s2"]["stopped"] = False
            
            current_leverage = max(-2.0, min(2.0, s1_lev + s2_lev))
            
            self.equity_curve.append(self.equity)
            self.leverage_curve.append(current_leverage)
            
            peak = max(self.equity_curve)
            self.drawdown_curve.append((peak - self.equity) / peak)

        res_df = self.df.iloc[start_idx:].copy()
        res_df['Equity'] = self.equity_curve
        res_df['Leverage'] = self.leverage_curve
        res_df['Drawdown'] = self.drawdown_curve
        return res_df

# --- Web Route ---
@app.route('/')
def home():
    # 1. Run Backtest
    df = fetch_binance_data()
    bt = Backtester(df)
    res = bt.run()
    
    # 2. Calculate Stats
    start_eq = INITIAL_CAPITAL
    end_eq = res['Equity'].iloc[-1]
    years = (res.index[-1] - res.index[0]).days / 365.25
    cagr = (end_eq / start_eq) ** (1/years) - 1
    max_dd = res['Drawdown'].max()
    
    # 3. Generate Plot Image
    plt.style.use('bmh')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price
    ax1.plot(res.index, res['close'], color='black', alpha=0.3, label='BTC Price')
    ax1.plot(res.index, res['SMA120'], color='orange', linestyle='--', label='SMA120')
    ax1.plot(res.index, res['SMA400'], color='blue', linestyle='--', label='SMA400')
    ax1.set_yscale('log')
    ax1.set_title('Strategy Backtest: BTC/USDT')
    ax1.legend()
    
    # Equity
    ax2.plot(res.index, res['Equity'], color='green')
    ax2.set_title('Equity ($)')
    
    # Leverage
    ax3.plot(res.index, res['Leverage'], color='purple')
    ax3.fill_between(res.index, res['Leverage'], 0, alpha=0.1, color='purple')
    ax3.set_title('Net Leverage')
    ax3.set_ylim(-2.5, 2.5)
    
    plt.tight_layout()
    
    # Convert plot to Base64 String
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # 4. Render HTML
    html = f"""
    <html>
        <head>
            <title>Strategy Backtest Report</title>
            <style>
                body {{ font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; background: #f5f5f5; padding: 20px; border-radius: 8px; }}
                .stat-box {{ text-align: center; }}
                .value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Strategy Performance Report</h1>
            <div class="stats">
                <div class="stat-box">
                    <div class="value">${start_eq:,.0f}</div>
                    <div class="label">Initial Capital</div>
                </div>
                <div class="stat-box">
                    <div class="value">${end_eq:,.0f}</div>
                    <div class="label">Final Equity</div>
                </div>
                <div class="stat-box">
                    <div class="value">{cagr*100:.2f}%</div>
                    <div class="label">CAGR</div>
                </div>
                <div class="stat-box">
                    <div class="value" style="color: red">-{max_dd*100:.2f}%</div>
                    <div class="label">Max Drawdown</div>
                </div>
            </div>
            <br>
            <img src="data:image/png;base64,{plot_url}" style="width: 100%; height: auto;">
        </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
