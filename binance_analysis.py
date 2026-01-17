import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import http.server
import socketserver
import time
import os
import glob
import threading
from datetime import datetime, timedelta
from collections import Counter

# --- CONFIGURATION ---
PORT = 8080
SEQ_LENGTH = 5
SYMBOL = 'ETH/USDT'
TIMEFRAME = '30m'

# Grid Search Parameters
GRID_MIN = 0.005
GRID_MAX = 0.1
GRID_STEPS = 100

# Ensemble Threshold
ENSEMBLE_ACC_THRESHOLD = 70.0

# Global State for Server
SERVER_DATA = {
    'image': '',
    'steps': [],
    'accs': [],
    'counts': [],
    'cmb_acc': 0,
    'cmb_count': 0,
    'last_update': 'Initializing...',
    'prediction': 'Waiting for next candle...',
    'next_check': 'Calculating...'
}

def fetch_initial_data(start_date='2020-01-01T00:00:00Z'):
    print(f"Fetching initial history for {SYMBOL}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date)
    all_ohlcv = []
    
    # Fetch up to "now"
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            # Stop if we reached close to current time
            if len(ohlcv) < 1000: break
            
            print(f"Fetched up to {datetime.fromtimestamp(ohlcv[-1][0]/1000)}...", end='\r')
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df.drop_duplicates(subset='timestamp').reset_index(drop=True)

def get_grid_indices(df, step_size):
    """Converts price series to log-integer grid indices."""
    close_array = df['close'].to_numpy()
    
    # Absolute Price Index calculation
    pct_change = np.zeros(len(close_array))
    pct_change[1:] = np.diff(close_array) / close_array[:-1]
    multipliers = 1.0 + pct_change
    abs_price_raw = np.cumprod(multipliers)
    
    # Log Rounding
    abs_price_log = np.log(abs_price_raw)
    grid_indices = np.floor(abs_price_log / step_size).astype(int)
    return grid_indices

class LiveProcessor:
    def __init__(self):
        self.df = fetch_initial_data()
        self.high_acc_configs = [] # Stores {step_size, patterns, accuracy}
        self.exchange = ccxt.binance()
        
    def run_training(self):
        print(f"\n\n--- RUNNING INITIAL GRID SEARCH ({GRID_STEPS} Steps) ---")
        step_sizes = np.linspace(GRID_MIN, GRID_MAX, GRID_STEPS)
        results = []
        
        # We define a helper to train on the current DF
        # Note: We use the *entire* DF for training initially to get best patterns
        # In a real rigorous backtest you might split train/test, but here we 
        # follow the user's logic to find the best 'current' configurations.
        
        for step in step_sizes:
            res = self.evaluate_config(step)
            results.append(res)
            
        # Filter best configs
        self.high_acc_configs = [r for r in results if r['accuracy'] > ENSEMBLE_ACC_THRESHOLD]
        print(f"Qualifying Configs (> {ENSEMBLE_ACC_THRESHOLD}%): {len(self.high_acc_configs)}")
        
        # Calculate Backtest Metrics
        cmb_acc, cmb_count = self.calculate_combined_metric(self.high_acc_configs)
        
        # Update Server Data
        unique_id = int(time.time())
        filename = f"grid_search_{unique_id}.png"
        self.generate_plot(results, cmb_acc, cmb_count, step_sizes, filename)
        
        SERVER_DATA.update({
            'image': filename,
            'steps': step_sizes,
            'accs': [r['accuracy'] for r in results],
            'counts': [r['trade_count'] for r in results],
            'cmb_acc': cmb_acc,
            'cmb_count': cmb_count
        })
        generate_html()

    def evaluate_config(self, step_size):
        grid_indices = get_grid_indices(self.df, step_size)
        
        # 90/10 Split for validation
        total_len = len(grid_indices)
        idx_split = int(total_len * 0.90)
        
        train_seq = grid_indices[:idx_split]
        val_seq = grid_indices[idx_split:]
        
        # Build Patterns
        patterns = {}
        for i in range(len(train_seq) - SEQ_LENGTH):
            seq = tuple(train_seq[i : i + SEQ_LENGTH])
            target = train_seq[i + SEQ_LENGTH]
            if seq not in patterns: patterns[seq] = []
            patterns[seq].append(target)
            
        # Evaluate
        move_correct = 0
        move_total_valid = 0
        
        for i in range(len(val_seq) - SEQ_LENGTH):
            current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
            current_level = current_seq[-1]
            actual_next_level = val_seq[i + SEQ_LENGTH]
            
            if current_seq in patterns:
                history = patterns[current_seq]
                predicted_level = Counter(history).most_common(1)[0][0]
                pred_diff = predicted_level - current_level
                actual_diff = actual_next_level - current_level
                
                if pred_diff != 0 and actual_diff != 0:
                    move_total_valid += 1
                    same_direction = (pred_diff > 0 and actual_diff > 0) or \
                                     (pred_diff < 0 and actual_diff < 0)
                    if same_direction:
                        move_correct += 1
        
        accuracy = (move_correct / move_total_valid * 100) if move_total_valid > 0 else 0.0
        
        return {
            'step_size': step_size,
            'accuracy': accuracy,
            'trade_count': move_total_valid,
            'patterns': patterns,
            'val_seq': val_seq # stored for combined backtest
        }

    def calculate_combined_metric(self, configs):
        # Simply run the logic over the validation set of the first config 
        # (assuming time alignment is identical, which it is)
        if not configs: return 0.0, 0
        
        # Use indices from the first config to iterate time
        ref_seq_len = len(configs[0]['val_seq'])
        combined_correct = 0
        combined_total = 0
        
        for i in range(ref_seq_len - SEQ_LENGTH):
            up_votes = 0
            down_votes = 0
            
            # Gather Votes
            for cfg in configs:
                val_seq = cfg['val_seq']
                current_seq = tuple(val_seq[i : i + SEQ_LENGTH])
                
                if current_seq in cfg['patterns']:
                    history = cfg['patterns'][current_seq]
                    pred_lvl = Counter(history).most_common(1)[0][0]
                    curr_lvl = current_seq[-1]
                    diff = pred_lvl - curr_lvl
                    
                    if diff > 0: up_votes += 1
                    elif diff < 0: down_votes += 1
            
            # Logic: At least 1 UP, 0 DOWN -> Trade UP (and vice versa)
            prediction = 0 # 0: None, 1: UP, -1: DOWN
            if up_votes > 0 and down_votes == 0:
                prediction = 1
            elif down_votes > 0 and up_votes == 0:
                prediction = -1
                
            if prediction != 0:
                # Check outcome using the first config as reference for direction
                # (Direction is universal across step sizes)
                ref_seq = configs[0]['val_seq']
                actual_move = ref_seq[i + SEQ_LENGTH] - ref_seq[i + SEQ_LENGTH - 1]
                
                if actual_move != 0:
                    combined_total += 1
                    if (prediction == 1 and actual_move > 0) or \
                       (prediction == -1 and actual_move < 0):
                        combined_correct += 1
                        
        acc = (combined_correct / combined_total * 100) if combined_total > 0 else 0.0
        return acc, combined_total

    def generate_plot(self, results, cmb_acc, cmb_count, step_sizes, filename):
        accuracies = [r['accuracy'] for r in results]
        trade_counts = [r['trade_count'] for r in results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Log Step Size')
        ax1.set_ylabel('Directional Accuracy (%)', color='tab:blue')
        ax1.plot(step_sizes, accuracies, marker='o', color='tab:blue', label='Accuracy')
        ax1.axhline(y=ENSEMBLE_ACC_THRESHOLD, color='r', linestyle=':', label='Threshold')
        
        ax2 = ax1.twinx() 
        ax2.set_ylabel('Number of Trades', color='tab:orange')
        ax2.plot(step_sizes, trade_counts, marker='x', linestyle='--', color='tab:orange', label='Trades')
        
        plt.title(f'Ensemble Accuracy: {cmb_acc:.2f}% ({cmb_count} trades)')
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()

    def update_live_candle(self):
        """Fetches only the most recent completed candle."""
        print("Fetching new candle...")
        try:
            # Fetch last 3 candles to be safe, but we only need the latest closed one
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=3)
            new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
            
            # Filter for new data only
            last_known_time = self.df['timestamp'].iloc[-1]
            new_rows = new_df[new_df['timestamp'] > last_known_time]
            
            if not new_rows.empty:
                print(f"Added {len(new_rows)} new candle(s). Last: {new_rows['timestamp'].iloc[-1]}")
                self.df = pd.concat([self.df, new_rows]).reset_index(drop=True)
                return True
            else:
                print("No new candles found yet.")
                return False
        except Exception as e:
            print(f"Update failed: {e}")
            return False

    def make_prediction(self):
        if not self.high_acc_configs:
            return "Not enough high-accuracy configs found."

        up_votes = 0
        down_votes = 0
        details = []

        # We must regenerate indices on the full updated dataframe to ensure consistency
        # However, we only need the tails for prediction
        
        for cfg in self.high_acc_configs:
            step = cfg['step_size']
            # Get fresh indices for the whole history
            indices = get_grid_indices(self.df, step)
            
            # Get the very last sequence (representing "Now")
            if len(indices) < SEQ_LENGTH: continue
            
            current_seq = tuple(indices[-SEQ_LENGTH:])
            current_level = current_seq[-1]
            
            if current_seq in cfg['patterns']:
                history = cfg['patterns'][current_seq]
                predicted_lvl = Counter(history).most_common(1)[0][0]
                diff = predicted_lvl - current_level
                
                if diff > 0: 
                    up_votes += 1
                elif diff < 0: 
                    down_votes += 1
                    
        # Combined Logic
        decision = "NEUTRAL / NO TRADE"
        color = "gray"
        
        if up_votes > 0 and down_votes == 0:
            decision = "BULLISH (UP)"
            color = "green"
        elif down_votes > 0 and up_votes == 0:
            decision = "BEARISH (DOWN)"
            color = "red"
            
        return decision, up_votes, down_votes, color

    def live_loop(self):
        while True:
            # 1. Calculate time to next 30m mark
            now = datetime.utcnow()
            minutes = now.minute
            seconds = now.second
            
            # Find next 00 or 30
            if minutes < 30:
                wait_min = 30 - minutes
            else:
                wait_min = 60 - minutes
            
            # Target time
            wait_seconds = (wait_min * 60) - seconds + 10 # +10s buffer for exchange
            
            next_check_time = now + timedelta(seconds=wait_seconds)
            SERVER_DATA['next_check'] = next_check_time.strftime('%H:%M:%S UTC')
            generate_html()
            
            print(f"Waiting {wait_seconds:.0f}s for next candle close ({SERVER_DATA['next_check']})...")
            time.sleep(wait_seconds)
            
            # 2. Update Data
            if self.update_live_candle():
                # 3. Predict
                decision, ups, downs, color = self.make_prediction()
                
                timestamp_str = self.df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')
                pred_str = f"<span style='color:{color}; font-weight:bold; font-size:1.5em'>{decision}</span><br>"
                pred_str += f"(Based on candle: {timestamp_str})<br>"
                pred_str += f"Votes: <span style='color:green'>{ups} UP</span> vs <span style='color:red'>{downs} DOWN</span>"
                
                SERVER_DATA['prediction'] = pred_str
                SERVER_DATA['last_update'] = datetime.utcnow().strftime('%H:%M:%S UTC')
                generate_html()
            else:
                print("Skipping prediction (no new data).")

def generate_html():
    # Helper to clean old images
    for f in glob.glob("grid_search_*.png"):
        if f != SERVER_DATA['image']:
            try: os.remove(f)
            except: pass

    # Build Table
    table_rows = ""
    # Sort by accuracy for display
    sorted_metrics = sorted(zip(SERVER_DATA['steps'], SERVER_DATA['accs'], SERVER_DATA['counts']), key=lambda x: x[1], reverse=True)
    
    for s, a, c in sorted_metrics:
        bg_style = "background-color: #e6fffa;" if a > ENSEMBLE_ACC_THRESHOLD else ""
        table_rows += f"<tr style='{bg_style}'><td>{s:.5f}</td><td>{a:.2f}%</td><td>{c}</td></tr>"

    html = f"""
    <html>
    <head>
        <title>Crypto Grid Ensemble</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding: 20px; background: #f0f2f5; }}
            .container {{ background: white; padding: 30px; display: inline-block; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 900px; }}
            .live-box {{ background: #2d3748; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 2px solid #4a5568; }}
            .metric-box {{ background: #ebf8ff; color: #2c5282; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            table {{ margin: 20px auto; border-collapse: collapse; width: 100%; font-size: 0.9em; }}
            th {{ background: #4a5568; color: white; padding: 12px; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            img {{ max-width: 100%; border-radius: 8px; margin-top: 20px; border: 1px solid #ddd; }}
            .status {{ font-size: 0.8em; color: #cbd5e0; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="live-box">
                <h2>LIVE SIGNAL: {SYMBOL}</h2>
                <div>{SERVER_DATA['prediction']}</div>
                <div class="status">
                    Last Update: {SERVER_DATA['last_update']} <br>
                    Next Check: {SERVER_DATA['next_check']}
                </div>
            </div>

            <div class="metric-box">
                <h3>Backtest Performance (Combined)</h3>
                <p style="font-size: 1.2em; margin: 5px;">Accuracy: <strong>{SERVER_DATA['cmb_acc']:.2f}%</strong> | Trades: {SERVER_DATA['cmb_count']}</p>
            </div>
            
            <img src="{SERVER_DATA['image']}" alt="Analysis Plot">
            
            <h3>Individual Grid Configs</h3>
            <table>
                <tr><th>Step Size</th><th>Accuracy</th><th>Trades</th></tr>
                {table_rows}
            </table>
        </div>
    </body>
    </html>
    """
    
    with open("index.html", "w") as f:
        f.write(html)

class ThreadedHTTPServer(object):
    def __init__(self, host, port):
        self.server = socketserver.TCPServer((host, port), http.server.SimpleHTTPRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True

    def start(self):
        self.server_thread.start()
        print(f"Server running at http://localhost:{PORT}")

if __name__ == "__main__":
    # 1. Start Web Server
    try:
        server = ThreadedHTTPServer("", PORT)
        server.start()
    except OSError:
        print(f"Port {PORT} is busy. Server not started, but script will run.")

    # 2. Init Processor & Train
    processor = LiveProcessor()
    processor.run_training()
    
    # 3. Enter Live Loop
    try:
        processor.live_loop()
    except KeyboardInterrupt:
        print("Stopping...")
