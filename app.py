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
WINDOW_SIZE = 60  # Days for In-Sample training AND Out-of-Sample execution
GRID_SIZE = 10    # Number of steps per parameter

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
        # Fallback for simple date strings
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
    """
    Precomputes indicators to speed up the Numba loop.
    Returns numpy arrays required for the simulation.
    """
    data = df.copy()
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1)).fillna(0)
    
    # SMAs
    data['sma_40'] = data['close'].rolling(window=40).mean().fillna(0)
    data['sma_120'] = data['close'].rolling(window=120).mean().fillna(0)
    data['sma_360'] = data['close'].rolling(window=360).mean().fillna(0)
    
    # Precompute III for all 10 possible window sizes to avoid re-calc in loop
    # Window range: 5 to 50
    iii_windows = np.linspace(5, 50, GRID_SIZE).astype(int)
    iii_matrix = np.zeros((len(data), GRID_SIZE))
    
    log_ret_vals = data['log_ret'].values
    
    # Simple rolling loop for III pre-calc
    for idx, w in enumerate(iii_windows):
        # iii = sum(abs(log_ret)) / abs(sum(log_ret)) ... wait, usually Efficiency Ratio is Abs(Net Change) / Sum(Abs Change)
        # Formula: |Sum(r)| / Sum(|r|)
        rolling_net = data['log_ret'].rolling(window=w).sum().abs()
        rolling_abs = data['log_ret'].abs().rolling(window=w).sum()
        iii = (rolling_net / rolling_abs).fillna(0)
        iii_matrix[:, idx] = iii.values

    return data, iii_matrix, iii_windows

# --- Numba Optimized Strategy Kernel ---
@njit(parallel=True, fastmath=True)
def grid_search_kernel(
    opens, closes, highs, lows, 
    sma40, sma120, sma360, 
    iii_matrix, 
    param_grid_a, param_grid_b, param_grid_c, param_grid_d, param_grid_e, 
    start_idx, end_idx
):
    """
    Runs 10^6 combinations for the specific time slice [start_idx : end_idx].
    Returns the index of the best parameter set based on Sharpe Ratio.
    """
    n_params = len(param_grid_a) # Should be 10^6
    n_days = end_idx - start_idx
    
    best_sharpe = -999.0
    best_idx = 0
    
    # Parallel loop over all parameter combinations
    for p in prange(n_params):
        
        # Unpack parameters for this iteration
        lev_a = param_grid_a[p]
        lev_b = param_grid_b[p]
        lev_thresh = param_grid_c[p]
        flat_thresh = param_grid_d[p]
        band_pct = param_grid_e[p]
        iii_col_idx = int(p % 10) # The last parameter varies fastest in our generation logic
        
        # Local variables for the run
        total_return = 0.0
        sum_daily_ret = 0.0
        sum_sq_daily_ret = 0.0
        
        # State
        is_flat = False
        
        # Inner loop: Time (Simulation)
        # We start at start_idx. Note: Data arrays are full length.
        for t in range(start_idx, end_idx):
            
            # --- 0. Get Data (Yesterday's close for signal, Today's OHLC for PnL) ---
            # Signal is based on t-1 data
            prev_c = closes[t-1]
            prev_s40 = sma40[t-1]
            prev_s120 = sma120[t-1]
            prev_s360 = sma360[t-1]
            
            # III is shifted by 1 (calculated on T-1 close)
            prev_iii = iii_matrix[t-1, iii_col_idx]
            
            # --- 1. Flat Regime Logic ---
            # Trigger
            if prev_iii < flat_thresh:
                is_flat = True
            
            # Release (check proximity to ANY sma)
            if is_flat:
                # Check bands
                dist40 = abs(prev_c - prev_s40)
                dist120 = abs(prev_c - prev_s120)
                dist360 = abs(prev_c - prev_s360)
                
                # Thresholds
                th40 = prev_s40 * band_pct
                th120 = prev_s120 * band_pct
                th360 = prev_s360 * band_pct
                
                if (dist40 <= th40) or (dist120 <= th120) or (dist360 <= th360):
                    is_flat = False
            
            # --- 2. Leverage Logic ---
            lev = 0.0
            if not is_flat:
                if prev_iii < lev_thresh:
                    lev = lev_a
                else:
                    lev = lev_b
            
            # --- 3. Signal Generation ---
            signal = 0
            if not is_flat:
                # Long: Price > All SMAs
                if prev_c > prev_s40 and prev_c > prev_s120 and prev_c > prev_s360:
                    signal = 1
                # Short: Price < All SMAs
                elif prev_c < prev_s40 and prev_c < prev_s120 and prev_c < prev_s360:
                    signal = -1
            
            # --- 4. Calculate PnL ---
            daily_ret = 0.0
            if signal != 0:
                o = opens[t]
                c = closes[t]
                
                # Simplified Daily Return (Close-to-Close or Open-to-Close approx)
                # Using (Close - Open) / Open for daily capture
                raw_ret = (c - o) / o
                daily_ret = raw_ret * signal * lev
                
            # Accumulate stats for Sharpe
            sum_daily_ret += daily_ret
            sum_sq_daily_ret += (daily_ret * daily_ret)
            
        # --- End Time Loop ---
        
        # Calculate Sharpe for this period
        if n_days > 0:
            mean_ret = sum_daily_ret / n_days
            var_ret = (sum_sq_daily_ret / n_days) - (mean_ret * mean_ret)
            std_ret = np.sqrt(var_ret) if var_ret > 0 else 0.0
            
            sharpe = 0.0
            if std_ret > 1e-9:
                sharpe = (mean_ret / std_ret) * np.sqrt(365) # Annualized
            
            # Update Best
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_idx = p
                
    return best_idx, best_sharpe

# --- Run Single OOS Period ---
def run_oos_period(
    opens, closes, highs, lows, 
    sma40, sma120, sma360, 
    iii_matrix, 
    best_idx, 
    param_grid_a, param_grid_b, param_grid_c, param_grid_d, param_grid_e,
    start_idx, end_idx
):
    """
    Runs the strategy for the Out-of-Sample period using the single best parameter set.
    Returns array of daily returns.
    """
    # Extract params
    lev_a = param_grid_a[best_idx]
    lev_b = param_grid_b[best_idx]
    lev_thresh = param_grid_c[best_idx]
    flat_thresh = param_grid_d[best_idx]
    band_pct = param_grid_e[best_idx]
    iii_col_idx = int(best_idx % 10)
    
    returns = []
    is_flat = False
    
    for t in range(start_idx, end_idx):
        if t >= len(closes): break
        
        prev_c = closes[t-1]
        prev_s40 = sma40[t-1]
        prev_s120 = sma120[t-1]
        prev_s360 = sma360[t-1]
        prev_iii = iii_matrix[t-1, iii_col_idx]
        
        # Flat Logic
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
        
        # Leverage
        lev = 0.0
        if not is_flat:
            lev = lev_a if prev_iii < lev_thresh else lev_b
            
        # Signal
        signal = 0
        if not is_flat:
            if prev_c > prev_s40 and prev_c > prev_s120 and prev_c > prev_s360:
                signal = 1
            elif prev_c < prev_s40 and prev_c < prev_s120 and prev_c < prev_s360:
                signal = -1
        
        # PnL
        daily_ret = 0.0
        if signal != 0:
            daily_ret = ((closes[t] - opens[t]) / opens[t]) * signal * lev
            
        returns.append(daily_ret)
        
    return np.array(returns)

# --- Main Optimization Driver ---
def perform_optimization():
    global global_store
    global_store['is_optimizing'] = True
    
    try:
        # 1. Fetch
        df = fetch_data(SYMBOL, TIMEFRAME, START_DATE_STR)
        if df.empty:
            print("No data.")
            return

        # 2. Pre-process
        df, iii_matrix, iii_windows = prepare_features(df)
        
        # 3. Create Parameter Grid (Flattened for Numba)
        print("Generating parameter grid...")
        # Grid definitions
        g_lev_a = np.linspace(0, 5, GRID_SIZE)
        g_lev_b = np.linspace(0, 5, GRID_SIZE)
        g_lev_th = np.linspace(0, 0.6, GRID_SIZE)
        g_flat_th = np.linspace(0, 0.5, GRID_SIZE)
        g_band = np.linspace(0, 0.07, GRID_SIZE)
        # iii_window is handled by index 0..9 (last dim of grid)
        
        # Meshgrid to get all combinations (10^5 spatial, 10 for window)
        # To avoid massive memory for 6x10^6 float64 arrays, we can construct them smarter, 
        # but for 1M items, 6 arrays is only ~48MB. It's fine.
        
        # Using numpy meshgrid to generate cartesian product
        # Order: a, b, c, d, e, f_idx
        mesh = np.array(np.meshgrid(g_lev_a, g_lev_b, g_lev_th, g_flat_th, g_band, np.arange(10)))
        # Reshape to (6, 1000000)
        flat_mesh = mesh.reshape(6, -1)
        
        p_a = flat_mesh[0]
        p_b = flat_mesh[1]
        p_c = flat_mesh[2]
        p_d = flat_mesh[3]
        p_e = flat_mesh[4]
        # p_f_idx is implicit in loop logic or we can pass it, but checking modulo is faster in kernel? 
        # Actually modulo in kernel is slower than reading array. Let's pass it if needed, 
        # but grid_search_kernel uses modulo on index 'p' if we structure loops correctly.
        # Wait, meshgrid order depends on indexing. 
        # To ensure p%10 maps to window correctly, window must be the last axis in meshgrid.
        # It is (axis 5). So flattening keeps it varying fastest. Correct.
        
        # Arrays for Numba
        numba_a = np.ascontiguousarray(p_a)
        numba_b = np.ascontiguousarray(p_b)
        numba_c = np.ascontiguousarray(p_c)
        numba_d = np.ascontiguousarray(p_d)
        numba_e = np.ascontiguousarray(p_e)
        
        print(f"Grid size: {len(numba_a)} combinations.")

        # 4. Rolling Window Loop
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        s40 = df['sma_40'].values
        s120 = df['sma_120'].values
        s360 = df['sma_360'].values
        
        oos_results = []
        sharpe_history = []
        date_indices = []
        
        # Start after max SMA (360) + Window (60)
        start_t = 360 + WINDOW_SIZE
        total_len = len(closes)
        
        print("Starting Rolling Optimization...")
        
        # Loop steps of 60 days
        for t in range(start_t, total_len, WINDOW_SIZE):
            
            # In-Sample (IS): [t-60 : t]
            is_start = t - WINDOW_SIZE
            is_end = t
            
            # Out-of-Sample (OOS): [t : t+60]
            oos_start = t
            oos_end = min(t + WINDOW_SIZE, total_len)
            
            if oos_start >= total_len:
                break
                
            print(f"Processing window: IS [{is_start}:{is_end}] -> OOS [{oos_start}:{oos_end}]")
            
            # A. OPTIMIZE on IS data
            best_idx, best_sharpe = grid_search_kernel(
                opens, closes, highs, lows,
                s40, s120, s360,
                iii_matrix,
                numba_a, numba_b, numba_c, numba_d, numba_e,
                is_start, is_end
            )
            
            sharpe_history.append(best_sharpe)
            
            # B. RUN OOS with best params
            oos_ret = run_oos_period(
                opens, closes, highs, lows,
                s40, s120, s360,
                iii_matrix,
                best_idx,
                numba_a, numba_b, numba_c, numba_d, numba_e,
                oos_start, oos_end
            )
            
            # Store OOS result
            oos_dates = df.index[oos_start:oos_end]
            chunk_df = pd.DataFrame({'strategy_ret': oos_ret}, index=oos_dates)
            oos_results.append(chunk_df)
            
            # Log params
            best_params = {
                'lev_a': numba_a[best_idx],
                'lev_b': numba_b[best_idx],
                'lev_thresh': numba_c[best_idx],
                'flat_thresh': numba_d[best_idx],
                'band': numba_e[best_idx],
                'iii_win': iii_windows[int(best_idx % 10)]
            }
            global_store['optimization_log'].append({
                'date': df.index[t],
                'sharpe': best_sharpe,
                'params': best_params
            })

        # 5. Stitch Results
        if oos_results:
            full_res = pd.concat(oos_results)
            # Align with original DF to get Buy & Hold
            full_res['bnh_ret'] = df['log_ret'].loc[full_res.index]
            
            full_res['cum_strategy'] = (1 + full_res['strategy_ret']).cumprod()
            full_res['cum_bnh'] = (1 + full_res['bnh_ret']).cumprod()
            
            global_store['results'] = full_res
            print("Optimization Complete.")
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
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(res.index, res['cum_strategy'], label='Rolling Walk-Forward Strategy', color='blue')
        plt.plot(res.index, res['cum_bnh'], label='Buy & Hold', color='gray', alpha=0.5)
        plt.yscale('log')
        plt.title(f'Walk-Forward Optimization: {SYMBOL}')
        plt.ylabel('Cumulative Return (Log)')
        plt.legend()
        plt.grid(True, which="both", alpha=0.2)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Stats
        total_ret = res['cum_strategy'].iloc[-1]
        
        # Log Table
        log_rows = ""
        for entry in global_store['optimization_log'][-10:]: # Show last 10
            p = entry['params']
            p_str = f"A:{p['lev_a']:.1f}, B:{p['lev_b']:.1f}, Th:{p['lev_thresh']:.2f}, Flat:{p['flat_thresh']:.2f}, Band:{p['band']:.3f}, Win:{p['iii_win']}"
            log_rows += f"<tr><td>{entry['date'].date()}</td><td>{entry['sharpe']:.2f}</td><td>{p_str}</td></tr>"

        stats_html = f"""
        <div class="stats">
            <div class="stat-box"><strong>Total Return:</strong><br>{total_ret:.2f}x</div>
        </div>
        <h3>Recent Optimization Windows</h3>
        <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">
            <tr><th>Date</th><th>In-Sample Sharpe</th><th>Best Params</th></tr>
            {log_rows}
        </table>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rolling Grid Search</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; max-width: 1000px; margin: auto; }}
            .container {{ background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; }}
            .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
            .stat-box {{ padding: 15px; background: #f0f0f0; border-radius: 5px; }}
            button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }}
            button:disabled {{ background: #ccc; }}
        </style>
        <meta http-equiv="refresh" content="10"> <!-- Auto refresh to check status -->
    </head>
    <body>
        <div class="container">
            <h1>Rolling Window Grid Search: {SYMBOL}</h1>
            <p><strong>Status:</strong> {status}</p>
            
            <form action="/run" method="post">
                <button type="submit" {'disabled' if global_store['is_optimizing'] else ''}>
                    { 'Optimization Running...' if global_store['is_optimizing'] else 'Run Optimization (High CPU)' }
                </button>
            </form>
            
            <hr>
            
            {'<img src="data:image/png;base64,' + plot_url + '">' if plot_url else '<p>No results yet. Click Run.</p>'}
            
            {stats_html}
        </div>
    </body>
    </html>
    """
    return html

@app.route('/run', methods=['POST'])
def run_opt():
    if not global_store['is_optimizing']:
        # Run in background via simple non-blocking way? 
        # For this file constraint, we'll run blocking but the UI won't update until done.
        # Actually, let's just run it blocking for simplicity as requested "one file". 
        # In production, use Celery/Thread.
        perform_optimization()
    return render_template_string("Optimization finished. <a href='/'>Go Back</a>")

if __name__ == '__main__':
    # Optimization takes time, so we don't run on import.
    print(f"Server starting on {PORT}")
    app.run(host='0.0.0.0', port=PORT)
