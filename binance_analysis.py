import gdown
import pandas as pd
import numpy as np
import os
import time
import sys
import http.server
import socketserver
import base64
import urllib.parse
import random
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Global variables
DATA_1H = None
DATA_1D = None
DATA_5M = None     # All Data (minus first 4 years)
DATA_5M_2Y = None  # Last 2 Years of DATA_5M

# GA Results Cache
GA_BEST_ALL = 1.0
GA_BEST_2Y = 1.0

# Gainer Configuration
GAINER_PARAMS = {
    "GA_WEIGHTS": {"MACD_1H": 0.8, "MACD_1D": 0.4, "SMA_1D": 0.4},
    "MACD_1H": {
        'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 
        'weights': [0.45, 0.43, 0.01]
    },
    "MACD_1D": {
        'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 
        'weights': [0.87, 0.92, 0.73]
    },
    "SMA_1D": {
        'params': [40, 120, 390], 
        'weights': [0.6, 0.8, 0.4]
    }
}

def delayed_print(text):
    print(text)
    sys.stdout.flush()
    time.sleep(0.05)

def download_data(file_id, output_filename='ohlcv.csv'):
    url = f'https://drive.google.com/uc?id={file_id}'
    if os.path.exists(output_filename): return
    delayed_print("[INFO] Downloading dataset...")
    gdown.download(url, output_filename, quiet=True)

def prepare_data(csv_file):
    global DATA_1H, DATA_1D, DATA_5M, DATA_5M_2Y
    delayed_print("[PROCESS] Loading and slicing data...")
    
    df = pd.read_csv(csv_file)
    
    # Skip first 4 years (1-min bars)
    skip_rows = 60 * 24 * 365 * 4
    delayed_print(f"[INFO] Skipping first {skip_rows} raw 1-min candles.")
    df = df.iloc[skip_rows:].reset_index(drop=True)
    
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    
    delayed_print("[PROCESS] Resampling (Anti-Lookahead)...")
    
    # Resample with label='right' to prevent lookahead
    DATA_5M = df.resample('5min', label='right', closed='right').agg({'close': 'last'}).dropna()
    DATA_1H = df.resample('1H', label='right', closed='right').agg({'close': 'last'}).dropna()
    DATA_1D = df.resample('1D', label='right', closed='right').agg({'close': 'last'}).dropna()
    
    # Create 2-Year subset
    last_date = DATA_5M.index[-1]
    start_2y = last_date - timedelta(days=365*2)
    DATA_5M_2Y = DATA_5M[DATA_5M.index >= start_2y].copy()
    
    delayed_print(f"[SUCCESS] Full Data: {len(DATA_5M)} bars | 2-Year Data: {len(DATA_5M_2Y)} bars")

# --- Optimized Vectorized Signals ---
def calc_vectorized_macd_signal(df, config):
    params, weights = config['params'], config['weights']
    total_signal = pd.Series(0.0, index=df.index)
    for (f, s, sp), w in zip(params, weights):
        fast = df['close'].ewm(span=f, adjust=False).mean()
        slow = df['close'].ewm(span=s, adjust=False).mean()
        macd = fast - slow
        sig_line = macd.ewm(span=sp, adjust=False).mean()
        total_signal += np.where(macd > sig_line, 1.0, -1.0) * w
    return total_signal / sum(weights)

def calc_vectorized_sma_signal(df, config):
    params, weights = config['params'], config['weights']
    total_signal = pd.Series(0.0, index=df.index)
    for p, w in zip(params, weights):
        sma = df['close'].rolling(window=p).mean()
        total_signal += np.where(df['close'] > sma, 1.0, -1.0) * w
    return total_signal / sum(weights)

def get_base_signal_series(target_index):
    # Calculate on 1H/1D
    sig_1h = calc_vectorized_macd_signal(DATA_1H, GAINER_PARAMS["MACD_1H"])
    sig_1d_macd = calc_vectorized_macd_signal(DATA_1D, GAINER_PARAMS["MACD_1D"])
    sig_1d_sma = calc_vectorized_sma_signal(DATA_1D, GAINER_PARAMS["SMA_1D"])
    
    # Broadcast to target 5m index
    # label='right' allows safe ffill
    s_1h = sig_1h.reindex(target_index, method='ffill').fillna(0)
    s_1d_macd = sig_1d_macd.reindex(target_index, method='ffill').fillna(0)
    s_1d_sma = sig_1d_sma.reindex(target_index, method='ffill').fillna(0)
    
    w = GAINER_PARAMS["GA_WEIGHTS"]
    raw = (s_1h * w["MACD_1H"] + s_1d_macd * w["MACD_1D"] + s_1d_sma * w["SMA_1D"]) / sum(w.values())
    
    # SHIFT 1 to enforce No-Lookahead execution
    return raw.shift(1).fillna(0)

# --- Genetic Algorithm ---
def run_genetic_optimization():
    global GA_BEST_ALL, GA_BEST_2Y
    delayed_print("[GA] Starting Extensive Genetic Optimization...")
    delayed_print("[GA] Config: Pop=50 | Gens=15 | Slices=5 | Range=[5m, 20d]")
    
    # Pre-calculate full signal series to avoid re-computing indicators in GA loop
    full_sigs = get_base_signal_series(DATA_5M.index)
    
    def fitness_fn(hold_days, data_df, signal_series):
        # Constraint: Use a random 2-month (60 day) slice
        # 60 days * 288 bars = 17280 bars
        SLICE_SIZE = 17280
        
        if len(data_df) < SLICE_SIZE:
            slice_df = data_df
            slice_sig = signal_series
        else:
            max_start = len(data_df) - SLICE_SIZE
            start_idx = random.randint(0, max_start)
            slice_df = data_df.iloc[start_idx : start_idx + SLICE_SIZE].copy()
            slice_sig = signal_series.iloc[start_idx : start_idx + SLICE_SIZE].copy()
            
        hold_bars = max(1, int(hold_days * 288))
        
        # Fast Vectorized Backtest on Slice
        exposure = slice_sig.rolling(window=hold_bars).sum()
        fees = exposure.diff().abs().fillna(0) * 0.0002
        
        fut_close = slice_df['close'].shift(-hold_bars)
        ret = (slice_sig * (fut_close - slice_df['close']) / slice_df['close']) - fees
        
        cum_ret = ret.cumsum().ffill()
        # Daily Sharpe approximation on slice
        daily = cum_ret.iloc[::288] # Sample every day
        if len(daily) < 2 or daily.diff().std() == 0: return -999.0
        return (daily.diff().mean() / daily.diff().std()) * np.sqrt(365)

    def optimize_dataset(name, df, full_signal_idx):
        delayed_print(f"[GA] Optimizing {name}...")
        
        # GA Configuration
        POP_SIZE = 50
        GENERATIONS = 15
        SLICES_PER_EVAL = 5
        ELITISM_COUNT = 3
        
        # Initial Population: Random float between 5 mins (approx 0.0035 days) and 20 days
        population = [random.uniform(1.0/288, 20.0) for _ in range(POP_SIZE)]
        
        # Subset signals for this DF
        df_sigs = full_sigs.loc[df.index]
        
        best_overall_sharpe = -999.0
        best_overall_gene = 1.0

        for gen in range(GENERATIONS):
            scores = []
            for indiv in population:
                # Average fitness over multiple random slices to ensure robustness
                sharpe_avg = np.mean([fitness_fn(indiv, df, df_sigs) for _ in range(SLICES_PER_EVAL)])
                scores.append((sharpe_avg, indiv))
            
            scores.sort(reverse=True)
            
            # Track best ever
            if scores[0][0] > best_overall_sharpe:
                best_overall_sharpe = scores[0][0]
                best_overall_gene = scores[0][1]
            
            # Logging every few generations
            if (gen + 1) % 3 == 0 or gen == 0:
                delayed_print(f"  > Gen {gen+1}/{GENERATIONS}: Best Sharpe={best_overall_sharpe:.3f} (Hold={best_overall_gene:.3f}d)")
            
            # Selection: Keep top 20% parents
            survivor_count = int(POP_SIZE * 0.2)
            survivors = [x[1] for x in scores[:survivor_count]]
            
            # Elitism: Directly carry over top performers
            new_pop = [x[1] for x in scores[:ELITISM_COUNT]]
            
            # Breeding
            while len(new_pop) < POP_SIZE:
                p1, p2 = random.sample(survivors, 2)
                # Blend crossover
                alpha = random.random()
                child = p1 * alpha + p2 * (1 - alpha)
                
                # Mutation (40% chance)
                if random.random() < 0.4:
                    # Variable mutation strength
                    mutation_strength = random.uniform(0.1, 2.0)
                    child += random.gauss(0, mutation_strength)
                
                # Clamp boundaries
                child = max(1.0/288, min(child, 20.0))
                new_pop.append(child)
                
            population = new_pop
            
        delayed_print(f"[GA] Result for {name}: {best_overall_gene:.4f} days (Sharpe: {best_overall_sharpe:.2f})")
        return best_overall_gene

    GA_BEST_ALL = optimize_dataset("Full History", DATA_5M, DATA_5M.index)
    GA_BEST_2Y = optimize_dataset("Last 2 Years", DATA_5M_2Y, DATA_5M_2Y.index)


# --- Simulation Engine ---
def run_simulation(hold_days, dataset_df, dataset_name):
    hold_period_5m = max(1, int(round(hold_days * 288)))
    
    # Generate signals (cached logic)
    raw_signal = get_base_signal_series(dataset_df.index)
    
    df = dataset_df.copy()
    df['signal'] = raw_signal # Already shifted
    
    df['net_exposure'] = df['signal'].rolling(window=hold_period_5m).sum()
    df['fees'] = df['net_exposure'].diff().abs().fillna(0) * 0.0002
    
    future_close = df['close'].shift(-hold_period_5m)
    pct_change = (future_close - df['close']) / df['close']
    
    df['ret'] = (df['signal'] * pct_change) - df['fees']
    
    res = df.dropna(subset=['ret', 'close']).copy()
    res['equity'] = res['ret'].cumsum()
    
    # Metrics
    daily = res['equity'].resample('D').last().dropna()
    diff = daily.diff().dropna()
    sharpe = (diff.mean() / diff.std()) * np.sqrt(365) if len(diff) > 1 and diff.std() != 0 else 0
    max_dd = (res['equity'].cummax() - res['equity']).max() * 100

    metrics = {
        "dataset": dataset_name,
        "hold_days": hold_days,
        "sharpe": sharpe,
        "total_return": res['equity'].iloc[-1] * 100,
        "max_dd": max_dd,
        "max_exposure": res['net_exposure'].abs().max()
    }
    
    # Plot Generation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(res.index, res['equity']*100, color='#1a73e8', lw=1.5)
    ax1.set_title(f'{dataset_name}: Net Return % (Hold: {hold_days:.2f}d)')
    ax1.grid(True, alpha=0.15)
    ax1.set_ylabel("Return %")
    
    ax2.fill_between(res.index, res['net_exposure'], color='#1a73e8', alpha=0.1)
    ax2.plot(res.index, res['net_exposure'], color='#1a73e8', lw=0.8)
    ax2.set_ylabel("Units")
    ax2.grid(True, alpha=0.15)
    
    plt.tight_layout()
    buf = BytesIO(); plt.savefig(buf, format='png', dpi=80); plt.close()
    return metrics, base64.b64encode(buf.getvalue()).decode('utf-8')


# --- Server ---
class SimulatorHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        hold_val = float(params.get('hold', [1.0])[0])
        hold_val = max(1/288, min(hold_val, 20.0))
        
        # Run 2 Simulations
        m_all, p_all = run_simulation(hold_val, DATA_5M, "Full History")
        m_2y, p_2y = run_simulation(hold_val, DATA_5M_2Y, "Last 2 Years")
        
        # Formatter
        def fmt_hold(v): return f"{v:.2f} Days"

        html = f"""
        <!DOCTYPE html><html><head><title>Dual Backtest & GA</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #eaeff2; margin:0; padding:20px; color: #202124; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            h1 {{ margin:0; font-size: 1.5rem; color: #1a73e8; }}
            .control {{ display: flex; align-items: center; gap: 20px; margin-top: 15px; }}
            input[type=range] {{ flex-grow: 1; cursor: pointer; }}
            
            .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .panel {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            .panel h2 {{ margin-top:0; font-size: 1.1rem; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            
            .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }}
            .stat {{ background: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 0.9rem; }}
            .stat span {{ display: block; font-weight: bold; font-size: 1.1rem; }}
            
            .ga-box {{ background: #e8f0fe; padding: 15px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #d2e3fc; }}
            .ga-title {{ color: #1967d2; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; }}
            img {{ width: 100%; border: 1px solid #eee; border-radius: 4px; }}
        </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Gainer Strategy: Dual Backtest + Genetic Optimization</h1>
                    <div class="control">
                        <label>Manual Hold:</label>
                        <input type="range" min="{1/288}" max="20" step="{1/288}" value="{hold_val}" onchange="window.location.href='?hold='+this.value">
                        <span style="font-weight:bold; min-width:80px">{fmt_hold(hold_val)}</span>
                    </div>
                </div>
                
                <div class="dashboard">
                    <!-- FULL DATA PANEL -->
                    <div class="panel">
                        <h2>Full History (Minus 4 Yrs)</h2>
                        <div class="ga-box">
                            <div class="ga-title">GENETIC ALGORITHM SUGGESTION</div>
                            Optimal Hold: <b>{GA_BEST_ALL:.2f} Days</b><br/>
                            <small>Optimized on random 2-month slices</small>
                        </div>
                        <div class="stats-grid">
                            <div class="stat">Sharpe <span>{m_all['sharpe']:.2f}</span></div>
                            <div class="stat">Return <span>{m_all['total_return']:.0f}%</span></div>
                            <div class="stat">Max DD <span style="color:#d93025">{m_all['max_dd']:.1f}%</span></div>
                            <div class="stat">Max Exp <span>{m_all['max_exposure']:.0f}</span></div>
                        </div>
                        <img src="data:image/png;base64,{p_all}">
                    </div>
                    
                    <!-- 2 YEAR PANEL -->
                    <div class="panel">
                        <h2>Last 2 Years</h2>
                        <div class="ga-box">
                            <div class="ga-title">GENETIC ALGORITHM SUGGESTION</div>
                            Optimal Hold: <b>{GA_BEST_2Y:.2f} Days</b><br/>
                            <small>Optimized on random 2-month slices</small>
                        </div>
                        <div class="stats-grid">
                            <div class="stat">Sharpe <span>{m_2y['sharpe']:.2f}</span></div>
                            <div class="stat">Return <span>{m_2y['total_return']:.0f}%</span></div>
                            <div class="stat">Max DD <span style="color:#d93025">{m_2y['max_dd']:.1f}%</span></div>
                            <div class="stat">Max Exp <span>{m_2y['max_exposure']:.0f}</span></div>
                        </div>
                        <img src="data:image/png;base64,{p_2y}">
                    </div>
                </div>
            </div>
        </body></html>
        """
        self.send_response(200); self.send_header("Content-type", "text/html"); self.end_headers()
        self.wfile.write(html.encode())

if __name__ == "__main__":
    FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'; FILENAME = 'ohlcv.csv'
    download_data(FILE_ID, FILENAME)
    prepare_data(FILENAME)
    
    # Run GA Once at Startup
    run_genetic_optimization()
    
    with socketserver.TCPServer(("", 8080), SimulatorHandler) as httpd:
        delayed_print("[INFO] Server live at http://localhost:8080")
        httpd.serve_forever()