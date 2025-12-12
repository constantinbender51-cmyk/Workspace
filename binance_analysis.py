import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import http.server
import socketserver
import webbrowser
import time
from datetime import datetime, timedelta, timezone

# --- Configuration ---
PORT = 8080
DAYS_BACK = 720
TIMEFRAME = '1h'  # 1-hour candles for Binance
SYMBOL_BINANCE = 'BTCUSDT'
SYMBOL_KRAKEN = 'PF_XBTUSD'
FUNDING_THRESHOLD = 0.5  # As requested

# --- Data Fetching Functions ---

def fetch_binance_ohlcv(symbol, interval, days):
    """Fetches OHLCV data from Binance API (public)."""
    print(f"Fetching Binance data for {symbol}...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        try:
            r = requests.get(base_url, params=params)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_data.extend(data)
            # Update start time to the last timestamp fetched + 1ms
            current_start = data[-1][0] + 1
            
            # Rate limit respect
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    print(f"Binance data fetched: {len(df)} rows.")
    return df

def fetch_kraken_funding(symbol, days):
    """Fetches historical funding rates from Kraken Futures API."""
    print(f"Fetching Kraken funding rates for {symbol}...")
    base_url = "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates"
    
    # Kraken usually works with ISO strings or ms timestamps depending on the specific v3 endpoint docs. 
    # v3 historical-funding-rates takes a 'before' param (timestamp in ms).
    
    all_rates = []
    # Start from now, working backwards
    # Note: If the API doesn't support easy pagination backwards, we might need a different approach.
    # Standard pattern: query 'before' current time, get list, take oldest, query 'before' oldest.
    
    # Using a simplified forward iteration or single grab approach is risky for 720 days.
    # We will iterate backwards.
    
    current_pointer = int(datetime.now(timezone.utc).timestamp() * 1000)
    min_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    
    while current_pointer > min_time:
        params = {
            'symbol': symbol,
            'before': current_pointer
        }
        
        try:
            r = requests.get(base_url, params=params)
            r.raise_for_status()
            data = r.json()
            
            rates = data.get('rates', [])
            if not rates:
                break
                
            all_rates.extend(rates)
            
            # Timestamps in Kraken v3 are usually ISO or ms. Let's check the first item.
            # Response format: {"rates": [{"timestamp": "2023-...", "fundingRate": 0.0001, ...}]}
            # We need to parse the oldest timestamp to update current_pointer
            
            oldest_ts_str = rates[-1]['timestamp']
            # Parse ISO to ms timestamp
            dt_obj = datetime.fromisoformat(oldest_ts_str.replace('Z', '+00:00'))
            oldest_ts = int(dt_obj.timestamp() * 1000)
            
            if oldest_ts >= current_pointer:
                # Avoid infinite loop if timestamps aren't moving
                current_pointer -= 86400000 # Force move back 1 day
            else:
                current_pointer = oldest_ts
            
            if current_pointer <= min_time:
                break
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching Kraken data: {e}")
            break
            
    df = pd.DataFrame(all_rates)
    # Filter columns and parse
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['fundingRate'] = df['fundingRate'].astype(float)
        # Sort index to be chronological
        df.sort_index(inplace=True)
        # Filter for the last 720 days only (since we fetched backwards, we might have extra or gaps)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        df = df[df.index >= cutoff]
    
    print(f"Kraken funding rates fetched: {len(df)} rows.")
    return df

# --- Processing & Strategy ---

def process_data(df_price, df_funding):
    # Resample funding to match price index (1H)
    # Funding rates typically occur every 4h or 8h. We forward fill the rate to the hours in between.
    df = df_price.join(df_funding['fundingRate'], how='left')
    df['fundingRate'] = df['fundingRate'].ffill().fillna(0)
    
    # 1. Calculate SMA 400
    df['sma_400'] = df['close'].rolling(window=400).mean()
    
    # 2. Strategy Logic
    # Start with flat (0)
    df['position'] = 0
    
    # Long if Price > SMA 400
    df.loc[df['close'] > df['sma_400'], 'position'] = 1
    
    # Short if Price < SMA 400
    df.loc[df['close'] < df['sma_400'], 'position'] = -1
    
    # Flat if Funding Rate > Threshold (Overwrites previous signals)
    df.loc[df['fundingRate'] > FUNDING_THRESHOLD, 'position'] = 0
    
    # 3. Calculate Equity
    # Shift position by 1 to simulate executing on the next open/close after signal
    df['strategy_ret'] = df['position'].shift(1) * df['close'].pct_change()
    df['equity'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
    
    # Buy & Hold for comparison
    df['bnh_ret'] = df['close'].pct_change()
    df['bnh_equity'] = (1 + df['bnh_ret'].fillna(0)).cumprod()
    
    return df

# --- Plotting ---

def create_dashboard(df):
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price & SMA 400", "Equity Curve", "Funding Rate")
    )

    # Row 1: Price
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='BTC Price', line=dict(color='gray', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_400'], name='SMA 400', line=dict(color='orange', width=2)), row=1, col=1)

    # Row 2: Equity
    fig.add_trace(go.Scatter(x=df.index, y=df['equity'], name='Strategy Equity', line=dict(color='green', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bnh_equity'], name='Buy & Hold', line=dict(color='blue', width=1, dash='dot')), row=2, col=1)

    # Row 3: Funding
    # Color red if above threshold
    colors = ['red' if val > FUNDING_THRESHOLD else 'purple' for val in df['fundingRate']]
    fig.add_trace(go.Bar(x=df.index, y=df['fundingRate'], name='Funding Rate', marker_color=colors), row=3, col=1)
    
    # Add Threshold line
    fig.add_hline(y=FUNDING_THRESHOLD, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Threshold")

    fig.update_layout(
        title=f"Strategy Analysis ({DAYS_BACK} Days): SMA 400 Filter + Funding Cutoff",
        height=900,
        template="plotly_dark",
        hovermode="x unified"
    )
    
    return fig

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Fetch
    df_ohlcv = fetch_binance_ohlcv(SYMBOL_BINANCE, TIMEFRAME, DAYS_BACK)
    df_funding = fetch_kraken_funding(SYMBOL_KRAKEN, DAYS_BACK)
    
    if df_ohlcv.empty:
        print("No OHLCV data found. Exiting.")
        exit()

    # 2. Process
    df_final = process_data(df_ohlcv, df_funding)
    
    # 3. Plot
    fig = create_dashboard(df_final)
    filename = "strategy_dashboard.html"
    fig.write_html(filename)
    print(f"Dashboard saved to {filename}")

    # 4. Serve
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.path = f'/{filename}'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

    print(f"Starting server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    
    # Open browser automatically
    webbrowser.open(f"http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.server_close()
