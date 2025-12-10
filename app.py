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
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

# Optimization Settings
OOS_STEP_SIZE = 60      # How often we re-optimize (Out-of-Sample horizon)
GRID_SIZE = 8           # Resolution for strategy params (8^6 ~ 262k combos)
WINDOW_GRID_STEPS = 5   # Resolution for Window Size (100 to 1000)
# Total Iterations per Step = 262k * 5 ~= 1.3 Million (Safe for 30s timeout)

app = Flask(__name__)

# --- Global Storage ---
global_store = {
    'data': None,
    'results': None,
    'optimization_log': [],
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
    print(f"Fetched {len(df)} rows.")
    return df

# --- Pre-Computation ---
def prepare_features(df):
    data = df.copy()
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1)).fillna(0)
    
    # SMAs (Max lookback 360, but window grid goes to 1000, so we need enough history)
    data['sma_40'] = data['close'].rolling(window=40).mean().fillna(0)
    data['sma_120'] = data['close'].rolling(window=120).mean().fillna(0)
    data['sma_360'] = data['close'].rolling(window=360).mean().fillna(0)
    
    # Precompute III for all GRID_SIZE possible iii_windows (5 to 50)
    iii_windows = np.linspace(5, 50, GRID_SIZE).astype(int)
    iii_matrix = np.zeros((len(data), GRID_SIZE))
    
    for idx, w in enumerate(iii_windows):
        rolling_net = data['log_ret'].rolling(window=w).sum().abs()
        rolling_abs = data['log_ret'].abs().rolling(window=w).sum()
        iii = (rolling_net / rolling_abs).fillna(0)
        iii_matrix[:, idx] = iii.values

    return data, iii_matrix, iii_windows

# --- Numba Optimized Strategy Kernel ---
@njit(parallel=True, fastmath=True)
def grid_search_kernel_variable_window(
    opens, closes, highs, lows, 
    sma40, sma120, sma360, 
    iii_matrix, 
    param_grid_a, param_grid_b, param_grid_c, param_grid_d, param_grid_e, param_grid_window,
    current_time_idx
):
    """
    Runs combinations for variable lookback windows.
    Returns: Array of Sharpes
    """
    n_params = len(param_grid_a) 
    sharpe_results = np.zeros(n_params) 
    
    # Parallel loop
    for p in prange(n_params):
        
        # 1. Decode Parameters
        lev_a = param_grid_a[p]
        lev_b = param_grid_b[p]
        lev_thresh = param_grid_c[p]
        flat_thresh = param_grid_d[p]
        band_pct = param_grid_e[p]
        lookback_window = int(param_grid_window[p])
        
        # Map p to iii_matrix column (Implicit: we constructed meshgrid so last dim iterates fastest)
        # Note: We must ensure the meshgrid construction aligns with this modulo logic.
        # Actually, if we pass explicit arrays from meshgrid, we don't need modulo math if we included the iii_idx in the mesh.
        # But here param_grid_iii_idx is not passed, so we rely on grid structure. 
        # To be safe, let's assume iii index is derived.
        # However, we have 7 dimensions now. It's safer to use the 'f' parameter from the grid if possible.
        # For now, let's assume the iii_index is the modulo GRID_SIZE of p? 
        # No, with 7 dims that's risky. 
        # Let's Rely on `param_grid_iii_idx` passed in (added to args).
        # WAIT: I will add `param_grid_f_idx` to arguments for safety.
        
        # Temporary fallback: We will assume p iterates [a, b, c, d, e, win, f_idx] 
        # actually, let's just pass the iii index array.
        pass 

    return sharpe_results

# Re-defining Kernel with explicit index array for safety
@njit(parallel=True, fastmath=True)
def grid_search_kernel_safe(
    opens, closes, highs, lows, 
    sma40, sma120, sma360, 
    iii_matrix, 
    p_a, p_b, p_c, p_d, p_e, p_win, p_iii_idx,
    current_time_idx
):
    n_params = len(p_a)
    sharpe_results = np.zeros(n_params)
    
    for i in prange(n_params):
        lev_a = p_a[i]
        lev_b = p_b[i]
        lev_thresh = p_c[i]
        flat_thresh = p_d[i]
        band_pct = p_e[i]
        window_size = int(p_win[i])
        iii_idx = int(p_iii_idx[i])
        
        # Define Time Range
        start_idx = current_time_idx - window_size
        end_idx = current_time_idx
        
        # Check bounds
        if start_idx < 120: # Ensure enough data for SMAs
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
            
            # Flat Regime
            if prev_iii < flat_thresh:
                is_flat = True
            
            if is_flat:
                # Check bands (Release logic)
                dist40 = abs(prev_c - prev_s40)
                dist120 = abs(prev_c - prev_s120)
                dist360 = abs(prev_c - prev_s360)
                th40 = prev_s40 * band_pct
                th120 = prev_s120 * band_pct
                th360 = prev_s360 * band_pct
                
                if (dist40 <= th40) or (dist120 <= th120) or (dist360 <= th360):
                    is_flat = False
            
            # Leverage & Signal
            lev = 0.0
            signal = 0
            
            if not is_flat:
                lev = lev_a if prev_iii < lev_thresh else lev_b
                
                if prev_c > prev_s40 and prev_c > prev_s120 and prev_c > prev_s360:
                    signal = 1
                elif prev_c < prev_s40 and prev_c < prev_s120 and prev_c < prev_s360:
                    signal = -1
            
            # PnL
            daily_ret = 0.0
            if signal != 0:
                raw_ret = (closes[t] - opens[t]) / opens[t]
                daily_ret = raw_ret * signal * lev
            
            sum_daily_ret += daily_ret
            sum_sq_daily_ret += (daily_ret * daily_ret)
            n_days += 1
            
        # Sharpe Calc
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

# --- Run Single OOS Period ---
def run_oos_period(
    opens, closes, sma40, sma120, sma360, iii_matrix, 
    best_idx, 
    p_a, p_b, p_c, p_d, p_e, p_iii_idx,
    start_idx, end_idx
):
    # Retrieve best params
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
    global_store['optimization_log'] = [] # Reset log
    
    try:
        # 1. Fetch
        df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
        if df.empty: return

        # 2. Pre-process
        df, iii_matrix, iii_windows = prepare_features(df)
        
        # 3. Create 7D Parameter Grid
        # Using GRID_SIZE=8 for params and 5 for window gives 8^6 * 5 ~ 1.3M iterations
        print("Generating parameter grid...")
        g_lev_a = np.linspace(0, 5, GRID_SIZE)
        g_lev_b = np.linspace(0, 5, GRID_SIZE)
        g_lev_th = np.linspace(0, 0.6, GRID_SIZE)
        g_flat_th = np.linspace(0, 0.5, GRID_SIZE)
        g_band = np.linspace(0, 0.07, GRID_SIZE)
        g_iii_idx = np.arange(GRID_SIZE) # Indices for iii windows
        
        # New Dimension: Lookback Window Size
        g_win_size = np.linspace(100, 1000, WINDOW_GRID_STEPS) 
        
        # Meshgrid (Order: a, b, c, d, e, win, iii_idx)
        mesh = np.array(np.meshgrid(
            g_lev_a, g_lev_b, g_lev_th, g_flat_th, g_band, g_win_size, g_iii_idx
        ))
        flat_mesh = mesh.reshape(7, -1)
        
        # Convert to contiguous arrays for Numba
        n_a = np.ascontiguousarray(flat_mesh[0])
        n_b = np.ascontiguousarray(flat_mesh[1])
        n_c = np.ascontiguousarray(flat_mesh[2])
        n_d = np.ascontiguousarray(flat_mesh[3])
        n_e = np.ascontiguousarray(flat_mesh[4])
        n_win = np.ascontiguousarray(flat_mesh[5])
        n_iii = np.ascontiguousarray(flat_mesh[6])
        
        print(f"Grid size: {len(n_a)} combinations per step.")

        # 4. Rolling Window Loop
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        s40 = df['sma_40'].values
        s120 = df['sma_120'].values
        s360 = df['sma_360'].values
        
        oos_results = []
        
        # Start time must allow for max lookback (1000) + SMAs (360) -> ~1360
        start_t = 1400 
        total_len = len(closes)
        
        if total_len < start_t:
            print("Not enough data for 1000 day lookback.")
            global_store['is_optimizing'] = False
            return
            
        print(f"Starting Rolling Optimization from index {start_t}...")
        
        for t in range(start_t, total_len, OOS_STEP_SIZE):
            
            # OOS Horizon
            oos_end = min(t + OOS_STEP_SIZE, total_len)
            if t >= total_len: break
            
            print(f" optimizing for OOS [{t}:{oos_end}]...")
            
            # A. OPTIMIZE (Find best window + best params)
            sharpe_array = grid_search_kernel_safe(
                opens, closes, highs, lows,
                s40, s120, s360,
                iii_matrix,
                n_a, n_b, n_c, n_d, n_e, n_win, n_iii,
                t # Current time (end of training, start of OOS)
            )
            
            best_idx = np.argmax(sharpe_array)
            best_sharpe = sharpe_array[best_idx]
            
            # B. RUN OOS
            oos_ret = run_oos_period(
                opens, closes, s40, s120, s360, iii_matrix,
                best_idx,
                n_a, n_b, n_c, n_d, n_e, n_iii,
                t, oos_end
            )
            
            oos_dates = df.index[t:oos_end]
            chunk_df = pd.DataFrame({'strategy_ret': oos_ret}, index=oos_dates)
            oos_results.append(chunk_df)
            
            # Log
            best_params = {
                'lev_a': n_a[best_idx],
                'lev_b': n_b[best_idx],
                'lev_thresh': n_c[best_idx],
                'flat_thresh': n_d[best_idx],
                'band': n_e[best_idx],
                'lookback': n_win[best_idx],
                'iii_win': iii_windows[int(n_iii[best_idx])]
            }
            
            global_store['optimization_log'].append({
                'date': df.index[t],
                'is_sharpe': best_sharpe,
                'params': best_params
            })

        if oos_results:
            full_res = pd.concat(oos_results)
            full_res['bnh_ret'] = df['log_ret'].loc[full_res.index]
            full_res['cum_strategy'] = (1 + full_res['strategy_ret']).cumprod()
            full_res['cum_bnh'] = (1 + full_res['bnh_ret']).cumprod()
            
            # Calculate Aggregate OOS Sharpe
            mean_oos = full_res['strategy_ret'].mean()
            std_oos = full_res['strategy_ret'].std()
            oos_sharpe = (mean_oos / std_oos) * np.sqrt(365) if std_oos > 0 else 0
            
            global_store['oos_sharpe'] = oos_sharpe
            global_store['results'] = full_res
            print(f"Optimization Complete. OOS Sharpe: {oos_sharpe:.2f}")
        else:
            print("No OOS results generated.")

    except Exception as e:
        print(f"Optimization Failed: {e}")
        import traceback
        traceback.print_exc()
        
    global_store['is_optimizing'] = False

# --- Web Server Routes ---
@app.route('/')
def index():
    status = "Idle"
    if global_store['is_optimizing']:
        status = "Running Optimization... (Check Console)"
    
    res = global_store['results']
    plot_url = ""
    stats_html = ""
    
    if res is not None and not res.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(res.index, res['cum_strategy'], label='Adaptive Rolling Strategy', color='blue')
        plt.plot(res.index, res['cum_bnh'], label='Buy & Hold', color='gray', alpha=0.5)
        plt.yscale('log')
        plt.title(f'Walk-Forward Optimization Results')
        plt.ylabel('Cumulative Return (Log)')
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
            p_str = (f"Lookback:{int(p['lookback'])}d, A:{p['lev_a']:.1f}, B:{p['lev_b']:.1f}, "
                     f"Th:{p['lev_thresh']:.2f}, Flat:{p['flat_thresh']:.2f}, "
                     f"Band:{p['band']:.3f}, iii_W:{p['iii_win']}")
            log_rows += f"<tr><td>{entry['date'].date()}</td><td>{entry['is_sharpe']:.2f}</td><td>{p_str}</td></tr>"

        stats_html = f"""
        <div class="stats">
            <div class="stat-box"><strong>Total Return:</strong><br>{total_ret:.2f}x</div>
            <div class="stat-box"><strong>OOS Sharpe Ratio:</strong><br>{oos_sharpe:.2f}</div>
        </div>
        <h3>Walk-Forward Log (Re-optimized every {OOS_STEP_SIZE} days)</h3>
        <div style="overflow-x: auto;">
        <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%; font-size: 0.9em;">
            <tr><th>Rebalance Date</th><th>Training Sharpe</th><th>Optimized Parameters</th></tr>
            {log_rows}
        </table>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Adaptive Strategy</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; max-width: 1000px; margin: auto; background: #f9f9f9; }}
            .container {{ background: white; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; justify-content: center; }}
            .stat-box {{ padding: 20px; background: #eef; border-radius: 8px; text-align: center; min-width: 150px; }}
            button {{ padding: 12px 24px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; transition: 0.2s; }}
            button:disabled {{ background: #ccc; cursor: not-allowed; }}
            button:hover:not(:disabled) {{ background: #0056b3; }}
        </style>
        <meta http-equiv="refresh" content="10">
    </head>
    <body>
        <div class="container">
            <h1 style="text-align:center;">Adaptive Grid Search: {SYMBOL}</h1>
            <p style="text-align:center;"><strong>Status:</strong> {status}</p>
            <div style="text-align:center;">
                <form action="/run" method="post">
                    <button type="submit" {'disabled' if global_store['is_optimizing'] else ''}>
                        { 'Running Simulation...' if global_store['is_optimizing'] else 'Run Walk-Forward Optimization' }
                    </button>
                </form>
            </div>
            
            <hr>
            
            {'<img src="data:image/png;base64,' + plot_url + '">' if plot_url else '<p style="text-align:center;">No results available. Please run optimization.</p>'}
            
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
