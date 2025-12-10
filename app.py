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

# Optimization Settings
OOS_STEP_SIZE = 60       # Re-optimize every 60 days
TRAINING_WINDOW = 1460   # 4-year fixed lookback
GRID_SIZE = 8            # Resolution

app = Flask(__name__)

# --- Global Storage ---
global_store = {
    'data': None,
    'results': None,
    'optimization_log': [],
    'is_optimizing': False,
    'oos_sharpe': 0.0
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

# --- Numba Kernel ---
@njit(parallel=True, fastmath=True)
def grid_search_kernel(
    opens, closes, highs, lows, 
    sma40, sma120, sma360, 
    iii_matrix, 
    p_a, p_b, p_c, p_d, p_e, p_iii_idx,
    current_time_idx,
    training_window
):
    n_params = len(p_a)
    sharpe_results = np.zeros(n_params)
    
    for i in prange(n_params):
        lev_a = p_a[i]
        lev_b = p_b[i]
        lev_thresh = p_c[i]
        flat_thresh = p_d[i]
        band_pct = p_e[i]
        iii_idx = int(p_iii_idx[i])
        
        start_idx = current_time_idx - training_window
        end_idx = current_time_idx
        
        if start_idx < 360: 
            sharpe_results[i] = -999.0
            continue
            
        sum_daily_ret = 0.0
        sum_sq_daily_ret = 0.0
        is_flat = False
        n_days = 0
        
        for t in range(start_idx, end_idx):
            prev_c = closes[t-1]
            prev_s40 = sma40[t-1]
            prev_s120 = sma120[t-1]
            prev_s360 = sma360[t-1]
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
            
            daily_ret = 0.0
            if signal != 0:
                raw_ret = (closes[t] - opens[t]) / opens[t]
                daily_ret = raw_ret * signal * lev
            
            sum_daily_ret += daily_ret
            sum_sq_daily_ret += (daily_ret * daily_ret)
            n_days += 1
            
        if n_days > 1:
            mean_ret = sum_daily_ret / n_days
            var_ret = (sum_sq_daily_ret / n_days) - (mean_ret * mean_ret)
            if var_ret < 0: var_ret = 0.0
            std_ret = np.sqrt(var_ret)
            if std_ret > 1e-9:
                sharpe_results[i] = (mean_ret / std_ret) * np.sqrt(365)
            else:
                sharpe_results[i] = 0.0
        else:
            sharpe_results[i] = 0.0
            
    return sharpe_results

# --- Robust Selector Logic ---
def select_robust_parameters(sharpe_array, p_a, p_b, p_c, p_d, p_e, p_iii_idx):
    """
    Instead of argmax (Best Peak), we select the MEDIAN of the Top 5%.
    This forces the strategy to pick a 'Plateau' of stability.
    """
    # 1. Filter out garbage (negative sharpes)
    valid_indices = np.where(sharpe_array > 0.1)[0]
    if len(valid_indices) == 0:
        return np.argmax(sharpe_array) # Fallback
        
    valid_sharpes = sharpe_array[valid_indices]
    
    # 2. Get Top 5% threshold
    percentile_95 = np.percentile(valid_sharpes, 95)
    
    # 3. Get indices of the "Elite" group
    elite_indices = valid_indices[valid_sharpes >= percentile_95]
    
    if len(elite_indices) == 0:
        return np.argmax(sharpe_array)
        
    # 4. Calculate Median Parameter Values of the Elite Group
    median_a = np.median(p_a[elite_indices])
    median_b = np.median(p_b[elite_indices])
    median_c = np.median(p_c[elite_indices])
    median_d = np.median(p_d[elite_indices])
    median_e = np.median(p_e[elite_indices])
    median_iii = np.median(p_iii_idx[elite_indices])
    
    # 5. Find the single parameter set in our Elite Group closest to this "Center of Mass"
    # Normalize simply by dividing by range if needed, but Euclidean dist on raw is okay for approximation
    # Actually, we just want the index in `elite_indices` that minimizes distance to medians.
    
    best_dist = 999999.0
    robust_idx = elite_indices[0]
    
    for idx in elite_indices:
        dist = (
            (p_a[idx] - median_a)**2 + 
            (p_b[idx] - median_b)**2 + 
            (p_c[idx] - median_c)**2 + 
            (p_d[idx] - median_d)**2 + 
            (p_e[idx] - median_e)**2 +
            (p_iii_idx[idx] - median_iii)**2
        )
        if dist < best_dist:
            best_dist = dist
            robust_idx = idx
            
    return robust_idx

# --- Run Single OOS Period ---
def run_oos_period(
    opens, closes, sma40, sma120, sma360, iii_matrix, 
    best_idx, 
    p_a, p_b, p_c, p_d, p_e, p_iii_idx,
    start_idx, end_idx
):
    lev_a = p_a[best_idx]
    lev_b = p_b[best_idx]
    lev_thresh = p_c[best_idx]
    flat_thresh = p_d[best_idx]
    band_pct = p_e[best_idx]
    iii_idx = int(p_iii_idx[best_idx])
    
    returns = []
    is_flat = False
    
    for t in range(start_idx, end_idx):
        if t >= len(closes): break
        
        prev_c = closes[t-1]
        prev_s40 = sma40[t-1]
        prev_s120 = sma120[t-1]
        prev_s360 = sma360[t-1]
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
        
        daily_ret = 0.0
        if signal != 0:
            daily_ret = ((closes[t] - opens[t]) / opens[t]) * signal * lev
            
        returns.append(daily_ret)
        
    return np.array(returns)

# --- Main Optimization Driver ---
def perform_optimization():
    global global_store
    global_store['is_optimizing'] = True
    global_store['optimization_log'] = [] 
    
    try:
        df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
        if df.empty: 
            global_store['is_optimizing'] = False
            return

        df, iii_matrix, iii_windows = prepare_features(df)
        
        # Create Grid
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

        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        s40 = df['sma_40'].values
        s120 = df['sma_120'].values
        s360 = df['sma_360'].values
        
        oos_results = []
        start_t = TRAINING_WINDOW + 360
        total_len = len(closes)
        
        for t in range(start_t, total_len, OOS_STEP_SIZE):
            oos_end = min(t + OOS_STEP_SIZE, total_len)
            if t >= total_len: break
            
            # 1. Get All Sharpes for this window
            sharpe_array = grid_search_kernel(
                opens, closes, highs, lows,
                s40, s120, s360,
                iii_matrix,
                n_a, n_b, n_c, n_d, n_e, n_iii,
                t, TRAINING_WINDOW
            )
            
            # 2. SELECT ROBUST PARAMETERS (Median of Top 5%)
            robust_idx = select_robust_parameters(sharpe_array, n_a, n_b, n_c, n_d, n_e, n_iii)
            robust_sharpe = sharpe_array[robust_idx]
            
            # 3. Run OOS
            oos_ret = run_oos_period(
                opens, closes, s40, s120, s360, iii_matrix,
                robust_idx,
                n_a, n_b, n_c, n_d, n_e, n_iii,
                t, oos_end
            )
            
            oos_dates = df.index[t:oos_end]
            chunk_df = pd.DataFrame({'strategy_ret': oos_ret}, index=oos_dates)
            oos_results.append(chunk_df)
            
            best_params = {
                'lev_a': n_a[robust_idx],
                'lev_b': n_b[robust_idx],
                'lev_thresh': n_c[robust_idx],
                'flat_thresh': n_d[robust_idx],
                'band': n_e[robust_idx],
                'iii_win': iii_windows[int(n_iii[robust_idx])]
            }
            
            global_store['optimization_log'].append({
                'date': df.index[t],
                'is_sharpe': robust_sharpe,
                'params': best_params
            })

        if oos_results:
            full_res = pd.concat(oos_results)
            full_res['bnh_ret'] = df['log_ret'].loc[full_res.index]
            full_res['cum_strategy'] = (1 + full_res['strategy_ret']).cumprod()
            full_res['cum_bnh'] = (1 + full_res['bnh_ret']).cumprod()
            
            mean_oos = full_res['strategy_ret'].mean()
            std_oos = full_res['strategy_ret'].std()
            oos_sharpe = (mean_oos / std_oos) * np.sqrt(365) if std_oos > 0 else 0
            
            global_store['oos_sharpe'] = oos_sharpe
            global_store['results'] = full_res
        
    except Exception as e:
        print(f"Error: {e}")
        
    global_store['is_optimizing'] = False

# --- Web Server Routes ---
@app.route('/')
def index():
    status = "Idle"
    if global_store['is_optimizing']:
        status = "Running Robust Optimization... (Check Console)"
    
    res = global_store['results']
    plot_url = ""
    stats_html = ""
    
    if res is not None and not res.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(res.index, res['cum_strategy'], label='Robust Median Strategy', color='green')
        plt.plot(res.index, res['cum_bnh'], label='Buy & Hold', color='gray', alpha=0.5)
        plt.yscale('log')
        plt.title('Robust (Cluster-Based) Walk-Forward Results')
        plt.legend()
        plt.grid(True, which="both", alpha=0.2)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        total_ret = res['cum_strategy'].iloc[-1]
        oos_sharpe = global_store.get('oos_sharpe', 0.0)
        
        log_rows = ""
        for entry in global_store['optimization_log'][-15:]:
            p = entry['params']
            p_str = (f"A:{p['lev_a']:.1f}, B:{p['lev_b']:.1f}, "
                     f"Th:{p['lev_thresh']:.2f}, Flat:{p['flat_thresh']:.2f}")
            log_rows += f"<tr><td>{entry['date'].date()}</td><td>{entry['is_sharpe']:.2f}</td><td>{p_str}</td></tr>"

        stats_html = f"""
        <div class="stats">
            <div class="stat-box"><strong>Total Return:</strong><br>{total_ret:.2f}x</div>
            <div class="stat-box"><strong>OOS Sharpe Ratio:</strong><br>{oos_sharpe:.2f}</div>
        </div>
        <h3>Walk-Forward Log (Median of Top 5%)</h3>
        <div style="overflow-x: auto;">
        <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%; font-size: 0.9em;">
            <tr><th>Rebalance Date</th><th>Training Sharpe</th><th>Robust Parameters</th></tr>
            {log_rows}
        </table>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robust Strategy</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; max-width: 1000px; margin: auto; background: #f9f9f9; }}
            .container {{ background: white; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; justify-content: center; }}
            .stat-box {{ padding: 20px; background: #eef; border-radius: 8px; text-align: center; min-width: 150px; }}
            button {{ padding: 12px 24px; font-size: 16px; cursor: pointer; background: #28a745; color: white; border: none; border-radius: 5px; }}
            button:disabled {{ background: #ccc; }}
        </style>
        <meta http-equiv="refresh" content="10">
    </head>
    <body>
        <div class="container">
            <h1 style="text-align:center;">Robust Grid Search: {SYMBOL}</h1>
            <p style="text-align:center;"><strong>Status:</strong> {status}</p>
            <div style="text-align:center;">
                <form action="/run" method="post">
                    <button type="submit" {'disabled' if global_store['is_optimizing'] else ''}>
                        { 'Running Simulation...' if global_store['is_optimizing'] else 'Run Robust Optimization' }
                    </button>
                </form>
            </div>
            <hr>
            {'<img src="data:image/png;base64,' + plot_url + '">' if plot_url else '<p style="text-align:center;">No results available.</p>'}
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
    return render_template_string("Optimization started. <a href='/'>Return to Dashboard</a>")

if __name__ == '__main__':
    print(f"Server starting on {PORT}")
    app.run(host='0.0.0.0', port=PORT)
