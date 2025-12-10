import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request
import io
import base64
import time
from numba import njit, prange

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE_STR = '2017-08-17 00:00:00'
PORT = 8080
GRID_SIZE = 10  # 10^6 combinations

app = Flask(__name__)

# --- Global Storage ---
global_store = {
    'results': None,
    'best_params': None,
    'is_optimizing': False
}

# --- Data Fetching ---
def fetch_data(symbol, timeframe, start_str):
    print(f"Fetching {symbol} data from Binance starting {start_str}...")
    exchange = ccxt.binance()
    try:
        start_ts = exchange.parse8601(start_str)
    except:
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        
    ohlcv_list = []
    current_ts = start_ts
    now_ts = exchange.milliseconds()
    
    while current_ts < now_ts:
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcvs:
                break
            current_ts = ohlcvs[-1][0] + 1
            ohlcv_list += ohlcvs
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

# --- Pre-Computation ---
def prepare_features(df):
    data = df.copy()
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1)).fillna(0)
    
    # SMAs
    data['sma_40'] = data['close'].rolling(window=40).mean().fillna(0)
    data['sma_120'] = data['close'].rolling(window=120).mean().fillna(0)
    data['sma_360'] = data['close'].rolling(window=360).mean().fillna(0)
    
    # Precompute III matrix
    iii_windows = np.linspace(5, 50, GRID_SIZE).astype(int)
    iii_matrix = np.zeros((len(data), GRID_SIZE))
    
    for idx, w in enumerate(iii_windows):
        rolling_net = data['log_ret'].rolling(window=w).sum().abs()
        rolling_abs = data['log_ret'].abs().rolling(window=w).sum()
        iii = (rolling_net / rolling_abs).fillna(0)
        iii_matrix[:, idx] = iii.values

    return data, iii_matrix, iii_windows

# --- Numba Kernel: Global Optimization ---
@njit(parallel=True, fastmath=True)
def grid_search_global(
    opens, closes, sma40, sma120, sma360, 
    iii_matrix, 
    p_a, p_b, p_c, p_d, p_e, p_iii_idx
):
    n_params = len(p_a)
    n_days = len(closes)
    
    # Store Sharpe for every parameter set
    sharpe_results = np.zeros(n_params)
    
    # Start after SMA 360 stabilizes
    start_idx = 360
    
    for i in prange(n_params):
        lev_a = p_a[i]
        lev_b = p_b[i]
        lev_thresh = p_c[i]
        flat_thresh = p_d[i]
        band_pct = p_e[i]
        iii_idx = int(p_iii_idx[i])
        
        sum_daily_ret = 0.0
        sum_sq_daily_ret = 0.0
        is_flat = False
        count = 0
        
        for t in range(start_idx, n_days):
            prev_c = closes[t-1]
            prev_s40 = sma40[t-1]
            prev_s120 = sma120[t-1]
            prev_s360 = sma360[t-1]
            prev_iii = iii_matrix[t-1, iii_idx]
            
            # Flat Regime Logic
            if prev_iii < flat_thresh:
                is_flat = True
            
            # Release Logic
            if is_flat:
                dist40 = abs(prev_c - prev_s40)
                dist120 = abs(prev_c - prev_s120)
                dist360 = abs(prev_c - prev_s360)
                th40 = prev_s40 * band_pct
                th120 = prev_s120 * band_pct
                th360 = prev_s360 * band_pct
                
                if (dist40 <= th40) or (dist120 <= th120) or (dist360 <= th360):
                    is_flat = False
            
            # Leverage Logic
            lev = 0.0
            signal = 0
            if not is_flat:
                lev = lev_a if prev_iii < lev_thresh else lev_b
                
                if prev_c > prev_s40 and prev_c > prev_s120 and prev_c > prev_s360:
                    signal = 1
                elif prev_c < prev_s40 and prev_c < prev_s120 and prev_c < prev_s360:
                    signal = -1
            
            # PnL Calculation
            daily_ret = 0.0
            if signal != 0:
                raw_ret = (closes[t] - opens[t]) / opens[t]
                daily_ret = raw_ret * signal * lev
            
            sum_daily_ret += daily_ret
            sum_sq_daily_ret += (daily_ret * daily_ret)
            count += 1
            
        # Calculate Sharpe
        if count > 0:
            mean_ret = sum_daily_ret / count
            var_ret = (sum_sq_daily_ret / count) - (mean_ret * mean_ret)
            if var_ret < 0: var_ret = 0.0
            std_ret = np.sqrt(var_ret)
            
            if std_ret > 1e-9:
                sharpe_results[i] = (mean_ret / std_ret) * np.sqrt(365)
            else:
                sharpe_results[i] = 0.0
        else:
            sharpe_results[i] = 0.0
            
    return sharpe_results

def run_backtest(
    df, best_idx, 
    p_a, p_b, p_c, p_d, p_e, p_iii_idx, 
    iii_windows, iii_matrix
):
    lev_a = p_a[best_idx]
    lev_b = p_b[best_idx]
    lev_thresh = p_c[best_idx]
    flat_thresh = p_d[best_idx]
    band_pct = p_e[best_idx]
    iii_idx = int(p_iii_idx[best_idx])
    
    opens = df['open'].values
    closes = df['close'].values
    s40 = df['sma_40'].values
    s120 = df['sma_120'].values
    s360 = df['sma_360'].values
    
    returns = np.zeros(len(closes))
    is_flat = False
    start_idx = 360
    
    for t in range(start_idx, len(closes)):
        prev_c = closes[t-1]
        prev_s40 = s40[t-1]
        prev_s120 = s120[t-1]
        prev_s360 = s360[t-1]
        prev_iii = iii_matrix[t-1, iii_idx]
        
        if prev_iii < flat_thresh:
            is_flat = True
        
        if is_flat:
            dist40 = abs(prev_c - prev_s40)
            dist120 = abs(prev_c - prev_s120)
            dist360 = abs(prev_c - prev_s360)
            th40 = prev_s40 * band_pct
            th120 = prev_s120 * band_pct
            th360 = prev_s360 * band_pct
            if (dist40 <= th40) or (dist120 <= th120) or (dist360 <= th360):
                is_flat = False
        
        lev = 0.0
        signal = 0
        if not is_flat:
            lev = lev_a if prev_iii < lev_thresh else lev_b
            if prev_c > prev_s40 and prev_c > prev_s120 and prev_c > prev_s360:
                signal = 1
            elif prev_c < prev_s40 and prev_c < prev_s120 and prev_c < prev_s360:
                signal = -1
        
        if signal != 0:
            returns[t] = ((closes[t] - opens[t]) / opens[t]) * signal * lev

    res_df = df.iloc[start_idx:].copy()
    res_df['strategy_ret'] = returns[start_idx:]
    res_df['cum_strategy'] = (1 + res_df['strategy_ret']).cumprod()
    res_df['bnh_ret'] = df['log_ret'].iloc[start_idx:]
    res_df['cum_bnh'] = (1 + res_df['bnh_ret']).cumprod()
    
    return res_df

# --- Main Driver ---
def perform_optimization():
    global global_store
    global_store['is_optimizing'] = True
    
    try:
        df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
        if df.empty: return

        df, iii_matrix, iii_windows = prepare_features(df)
        
        # Grid Generation (10^6)
        print("Generating 1M parameter combinations...")
        g_lev_a = np.linspace(0, 5, GRID_SIZE)
        g_lev_b = np.linspace(0, 5, GRID_SIZE)
        g_lev_th = np.linspace(0, 0.6, GRID_SIZE)
        g_flat_th = np.linspace(0, 0.5, GRID_SIZE)
        g_band = np.linspace(0, 0.07, GRID_SIZE)
        g_iii_idx = np.arange(GRID_SIZE)
        
        mesh = np.array(np.meshgrid(
            g_lev_a, g_lev_b, g_lev_th, g_flat_th, g_band, g_iii_idx
        ))
        flat_mesh = mesh.reshape(6, -1)
        
        n_a = np.ascontiguousarray(flat_mesh[0])
        n_b = np.ascontiguousarray(flat_mesh[1])
        n_c = np.ascontiguousarray(flat_mesh[2])
        n_d = np.ascontiguousarray(flat_mesh[3])
        n_e = np.ascontiguousarray(flat_mesh[4])
        n_iii = np.ascontiguousarray(flat_mesh[5])

        print("Starting Global Optimization (This may take 10-20 seconds)...")
        start_time = time.time()
        
        # Run Kernel
        sharpe_array = grid_search_global(
            df['open'].values, df['close'].values, 
            df['sma_40'].values, df['sma_120'].values, df['sma_360'].values,
            iii_matrix,
            n_a, n_b, n_c, n_d, n_e, n_iii
        )
        
        print(f"Optimization finished in {time.time() - start_time:.2f}s")
        
        # Get Best
        best_idx = np.argmax(sharpe_array)
        best_sharpe = sharpe_array[best_idx]
        
        best_params = {
            'lev_a': n_a[best_idx],
            'lev_b': n_b[best_idx],
            'lev_thresh': n_c[best_idx],
            'flat_thresh': n_d[best_idx],
            'band': n_e[best_idx],
            'iii_win': iii_windows[int(n_iii[best_idx])],
            'sharpe': best_sharpe
        }
        
        global_store['best_params'] = best_params
        print(f"Best Sharpe: {best_sharpe:.2f}")
        print(f"Params: {best_params}")
        
        # Run Backtest with Best Params to get Equity Curve
        res_df = run_backtest(df, best_idx, n_a, n_b, n_c, n_d, n_e, n_iii, iii_windows, iii_matrix)
        global_store['results'] = res_df
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    global_store['is_optimizing'] = False

# --- Web Server ---
@app.route('/')
def index():
    status = "Idle"
    if global_store['is_optimizing']:
        status = "Running Global Optimization..."
    
    res = global_store['results']
    best = global_store['best_params']
    plot_url = ""
    stats_html = ""
    
    if res is not None and not res.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(res.index, res['cum_strategy'], label=f'Optimized Strategy (Sharpe {best["sharpe"]:.2f})', color='green')
        plt.plot(res.index, res['cum_bnh'], label='Buy & Hold', color='gray', alpha=0.5)
        plt.yscale('log')
        plt.title('Global Optimization Result (Whole Dataset)')
        plt.ylabel('Cumulative Return (Log)')
        plt.legend()
        plt.grid(True, which="both", alpha=0.2)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        total_ret = res['cum_strategy'].iloc[-1]
        
        stats_html = f"""
        <div class="stats">
            <div class="stat-box"><strong>Total Return:</strong><br>{total_ret:.2f}x</div>
            <div class="stat-box"><strong>Max Sharpe:</strong><br>{best['sharpe']:.2f}</div>
        </div>
        <h3>Optimal Parameters (Global Best)</h3>
        <table border="1" cellpadding="10" style="border-collapse: collapse; margin:auto;">
            <tr><td>Lev Tier A</td><td>{best['lev_a']:.1f}x</td></tr>
            <tr><td>Lev Tier B</td><td>{best['lev_b']:.1f}x</td></tr>
            <tr><td>Lev Threshold</td><td>{best['lev_thresh']:.2f}</td></tr>
            <tr><td>Flat Threshold</td><td>{best['flat_thresh']:.2f}</td></tr>
            <tr><td>SMA Band</td><td>{best['band']:.3f}</td></tr>
            <tr><td>III Window</td><td>{best['iii_win']} days</td></tr>
        </table>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Global Optimization</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; max-width: 1000px; margin: auto; background: #f9f9f9; text-align: center; }}
            .container {{ background: white; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; justify-content: center; }}
            .stat-box {{ padding: 20px; background: #eef; border-radius: 8px; min-width: 150px; }}
            button {{ padding: 12px 24px; font-size: 16px; cursor: pointer; background: #d9534f; color: white; border: none; border-radius: 5px; }}
            button:disabled {{ background: #ccc; }}
        </style>
        <meta http-equiv="refresh" content="10">
    </head>
    <body>
        <div class="container">
            <h1>Global Strategy Optimization</h1>
            <p><strong>Status:</strong> {status}</p>
            <form action="/run" method="post">
                <button type="submit" {'disabled' if global_store['is_optimizing'] else ''}>
                    { 'Running...' if global_store['is_optimizing'] else 'Run Global Optimization (Whole History)' }
                </button>
            </form>
            <hr>
            {'<img src="data:image/png;base64,' + plot_url + '">' if plot_url else '<p>No results available.</p>'}
            {stats_html}
        </div>
    </body>
    </html>
    """
    return html

@app.route('/run', methods=['POST'])
def run_opt():
    if not global_store['is_optimizing']:
        perform_optimization()
    return render_template_string("Optimization started. <a href='/'>Return</a>")

if __name__ == '__main__':
    print(f"Server starting on {PORT}")
    app.run(host='0.0.0.0', port=PORT)
