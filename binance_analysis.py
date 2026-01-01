import os
import time
import threading
import requests
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template_string
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1M'  # Monthly candles
API_URL = 'https://api.binance.com/api/v3/klines'
UPDATE_INTERVAL_SECONDS = 3600  # Update data every hour

# Global storage for the latest analysis
analysis_result = {
    "last_updated": None,
    "current_price": 0,
    "current_phase": "WAITING",
    "months_since_peak": 0,
    "next_phase_date": "Calculating...",
    "peak_price": 0,
    "peak_date": None,
    "graph_json": None
}

app = Flask(__name__)

# --- Data Fetching & Strategy Logic ---

def fetch_binance_data():
    """Fetches monthly kline data from Binance public API."""
    try:
        # Limit 1000 months is plenty for BTC history
        params = {'symbol': SYMBOL, 'interval': INTERVAL, 'limit': 1000}
        response = requests.get(API_URL, params=params)
        data = response.json()
        
        # DataFrame columns: Open Time, Open, High, Low, Close, Volume, ...
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        
        # Convert types
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def apply_strategy(df):
    """
    Applies the strategy:
    1. Identify Global Peaks (High Water Mark on Monthly Close).
    2. Logic:
       - 0-12 months since peak: SHORT
       - 13-36 months since peak: FLAT (2 years)
       - 37+ months: BUY (until new peak resets cycle)
    """
    if df.empty:
        return None

    df['peak_close'] = df['close'].cummax()
    df['is_peak'] = df['close'] == df['peak_close']
    
    # Calculate months since last peak
    last_peak_idx = 0
    months_since = []
    signals = []
    
    for i in range(len(df)):
        if df['is_peak'].iloc[i]:
            last_peak_idx = i
            # If we just made a new peak, we are technically in a "Euploria/Top" phase
            # But the strategy usually implies Shorting *after* the peak is established.
            # We will mark the Peak month itself as the start of the count (0)
        
        # Calculate distance in months (approximate by index since it's monthly data)
        months_diff = i - last_peak_idx
        months_since.append(months_diff)
        
        # Determine Signal
        # Strategy: Short 1y (1-12), Flat 2y (13-36), Buy 1y/Repeat (37+)
        if months_diff == 0:
            signal = "PEAK" # The top
        elif 1 <= months_diff <= 12:
            signal = "SHORT"
        elif 13 <= months_diff <= 36:
            signal = "FLAT"
        else:
            signal = "BUY"
            
        signals.append(signal)

    df['months_since_peak'] = months_since
    df['signal'] = signals
    return df

def generate_plot(df):
    """Generates a Plotly Figure JSON for the web interface."""
    if df.empty:
        return "{}"

    # Create figure
    fig = go.Figure()

    # Add Candlestick
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='BTCUSDT'
    ))

    # Add colored background regions based on signal
    # We do this by creating shapes or a scatter plot with fill. 
    # For simplicity/performance, we'll use a scatter marker overlay or colored bars.
    # Let's use a separate trace for signals to make it clear.
    
    # Filter for signals to plot markers
    short_df = df[df['signal'] == 'SHORT']
    flat_df = df[df['signal'] == 'FLAT']
    buy_df = df[df['signal'] == 'BUY']
    peak_df = df[df['signal'] == 'PEAK']

    fig.add_trace(go.Scatter(
        x=short_df['date'], y=short_df['close'],
        mode='markers', marker=dict(color='red', size=4), name='Short Phase'
    ))
    fig.add_trace(go.Scatter(
        x=flat_df['date'], y=flat_df['close'],
        mode='markers', marker=dict(color='gray', size=4), name='Flat Phase'
    ))
    fig.add_trace(go.Scatter(
        x=buy_df['date'], y=buy_df['close'],
        mode='markers', marker=dict(color='green', size=6), name='Buy Phase'
    ))
    fig.add_trace(go.Scatter(
        x=peak_df['date'], y=peak_df['close'],
        mode='markers', marker=dict(color='purple', size=10, symbol='star'), name='ATH Peak'
    ))

    fig.update_layout(
        title='BTCUSDT Monthly Strategy (Anchor: ATH Close)',
        yaxis_title='Price (USDT)',
        xaxis_title='Date',
        template='plotly_dark',
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    return fig.to_json()

def update_loop():
    """Background thread to update data and run strategy."""
    global analysis_result
    print("Background thread started.")
    while True:
        print(f"[{datetime.now()}] Fetching data...")
        df = fetch_binance_data()
        
        if not df.empty:
            df_processed = apply_strategy(df)
            
            # Get current state (last completed month/candle)
            # Note: The last row in Binance is the *current* unfinished month.
            # Strategy logic usually applies to closed candles, but for visualization
            # we want to see where we are *right now*.
            current_state = df_processed.iloc[-1]
            
            # Determine next phase logic for display
            msp = current_state['months_since_peak']
            curr_sig = current_state['signal']
            
            next_event_msg = ""
            if curr_sig == "PEAK":
                next_event_msg = "Market at ATH. Short Phase starts next month."
            elif curr_sig == "SHORT":
                months_left = 12 - msp
                next_event_msg = f"{months_left} months until FLAT phase."
            elif curr_sig == "FLAT":
                months_left = 36 - msp
                next_event_msg = f"{months_left} months until BUY phase."
            elif curr_sig == "BUY":
                next_event_msg = "In BUY zone until new ATH breaks."

            graph = generate_plot(df_processed)
            
            analysis_result = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_price": f"${current_state['close']:,.2f}",
                "current_phase": curr_sig,
                "months_since_peak": int(msp),
                "next_phase_msg": next_event_msg,
                "peak_price": f"${current_state['peak_close']:,.2f}",
                "graph_json": graph
            }
            print("Data updated successfully.")
        
        time.sleep(UPDATE_INTERVAL_SECONDS)

# --- Web Server Routes ---

@app.route('/')
def dashboard():
    # Simple HTML template with embedded Plotly
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Strategy Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            h1 { color: #f0f0f0; }
            .stat-label { font-size: 0.9em; color: #aaa; }
            .stat-value { font-size: 1.5em; font-weight: bold; }
            .SHORT { color: #ff4d4d; }
            .FLAT { color: #aaaaaa; }
            .BUY { color: #00cc66; }
            .PEAK { color: #d633ff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BTC Monthly Strategy (Peak Anchor)</h1>
            
            <div class="grid">
                <div class="card">
                    <div class="stat-label">Current Phase</div>
                    <div class="stat-value {{ data.current_phase }}">{{ data.current_phase }}</div>
                </div>
                <div class="card">
                    <div class="stat-label">Months Since Peak</div>
                    <div class="stat-value">{{ data.months_since_peak }}</div>
                </div>
                <div class="card">
                    <div class="stat-label">Current Price</div>
                    <div class="stat-value">{{ data.current_price }}</div>
                </div>
                <div class="card">
                    <div class="stat-label">Outlook</div>
                    <div class="stat-value" style="font-size: 1rem;">{{ data.next_phase_msg }}</div>
                </div>
            </div>

            <div class="card">
                <div id="chart"></div>
            </div>
            
            <p style="text-align: center; color: #555;">Last Updated: {{ data.last_updated }} | Strategy: Short 1y, Flat 2y, Buy 1y (Repeat)</p>
        </div>

        <script>
            var graphData = {{ data.graph_json | safe }};
            Plotly.newPlot('chart', graphData.data, graphData.layout);
        </script>
    </body>
    </html>
    """
    return render_template_string(html, data=analysis_result)

# --- Entry Point ---

def start_background_thread():
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()

if __name__ == '__main__':
    # Start the data fetching loop
    start_background_thread()
    
    # Run Flask Server
    # Railway provides the PORT environment variable.
    # We must listen on 0.0.0.0 to be accessible.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)