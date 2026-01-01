import threading
import time
import os
import ccxt
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template_string
from datetime import datetime

# --- Global Data Storage (In-Memory) ---
# Since Railway filesystems are ephemeral, we store the latest data in memory.
# The script thread writes to this, the web thread reads from it.
market_data = {
    "symbol": "BTC/USDT",
    "df": pd.DataFrame(),
    "last_updated": None,
    "status": "Initializing..."
}

# --- Configuration ---
SYMBOL = "BTC/USDT"
TIMEFRAME = "1M"  # Monthly candles
UPDATE_INTERVAL = 3600  # How often the script runs (in seconds) - e.g., every hour

# --- The Web Server (Visualization Layer) ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Binance Monthly OHLCV</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .status { font-size: 0.9em; color: #888; }
        .card { background: #2d2d2d; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1 { margin: 0; color: #f0b90b; } /* Binance Yellow */
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ symbol }} Monthly Data</h1>
            <div class="status">
                Last Script Update: {{ last_updated }}<br>
                Status: {{ status }}
            </div>
        </div>
        
        <div class="card">
            {% if chart_json %}
                <div id="chart"></div>
                <script>
                    var graphs = {{ chart_json | safe }};
                    Plotly.newPlot('chart', graphs.data, graphs.layout);
                </script>
            {% else %}
                <p>Waiting for script to fetch data...</p>
            {% endif %}
        </div>
    </div>
    <!-- Auto-refresh page every 5 minutes to see new script updates -->
    <script>setTimeout(function(){ location.reload(); }, 300000);</script>
</body>
</html>
"""

@app.route('/')
def index():
    global market_data
    
    chart_json = None
    if not market_data["df"].empty:
        df = market_data["df"]
        
        # Create Plotly Candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'])])

        fig.update_layout(
            title=f'{market_data["symbol"]} Monthly Chart',
            yaxis_title=f'{market_data["symbol"]} Price',
            xaxis_title='Date',
            template="plotly_dark",
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Convert to JSON for embedding
        import json
        import plotly.utils
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template_string(
        HTML_TEMPLATE, 
        symbol=market_data["symbol"],
        last_updated=market_data["last_updated"],
        status=market_data["status"],
        chart_json=chart_json
    )

def run_web_server():
    """
    Runs the Flask server in a separate thread.
    Railway provides the PORT via environment variables.
    """
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Web Server on port {port}...")
    # host='0.0.0.0' is CRITICAL for Railway to expose the port
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# --- The "Script" Logic ---
def fetch_binance_data():
    """
    The core logic. Downloads data using CCXT.
    """
    global market_data
    print(f"[{datetime.now()}] Script: Connecting to Binance...")
    
    try:
        exchange = ccxt.binance()
        # Fetch monthly candles (1M)
        # Limit 100 ensures we get enough history without overloading in one call
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
        
        if ohlcv:
            # Process Data
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Update Global Storage
            market_data["df"] = df
            market_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            market_data["status"] = "Active"
            print(f"[{datetime.now()}] Script: Success! Loaded {len(df)} months of data.")
        else:
            market_data["status"] = "No data received"
            print("Script: Warning - No data received.")
            
    except Exception as e:
        market_data["status"] = f"Error: {str(e)}"
        print(f"Script Error: {e}")

def main_script_loop():
    """
    Simulates the 'Normal Application' running locally.
    It runs an infinite loop doing its job, regardless of the web server.
    """
    print("--- Crypto Script Started ---")
    
    # 1. Start the Web Server as a daemon thread
    # Daemon means it will automatically close if the main script crashes/exits
    server_thread = threading.Thread(target=run_web_server, daemon=True)
    server_thread.start()
    
    # 2. Give the server a second to spin up
    time.sleep(1)
    
    # 3. Enter the main script execution loop
    while True:
        fetch_binance_data()
        
        print(f"Script: Sleeping for {UPDATE_INTERVAL} seconds...")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main_script_loop()