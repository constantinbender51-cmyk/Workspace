import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import datetime
import threading
import json
import time
import websocket # requires: pip install websocket-client

# =============================================================================
# GLOBAL SETTINGS & SHARED STATE
# =============================================================================
LIVE_WHALES = []
CURRENT_BTC_PRICE = 90000.0  # Fallback price, updated on startup
WHALE_THRESHOLD_USD = 100000 # Filter for live transactions > $100k

# =============================================================================
# DATA FETCHING (HISTORICAL)
# =============================================================================

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1d", start_year=2018):
    """
    Fetches daily OHLCV data from Binance since start_year.
    """
    print(f"[Init] Fetching Binance data for {symbol} since {start_year}...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    start_ts = int(datetime.datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(datetime.datetime.now().timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    # Pagination loop
    while start_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "limit": limit}
        try:
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data or not isinstance(data, list): break
            all_data.extend(data)
            start_ts = data[-1][6] + 1 # Last Close Time + 1ms
            time.sleep(0.05) # Rate limit courtesy
        except Exception as e:
            print(f"Error fetching Binance: {e}")
            break
            
    if not all_data: return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume", 
        "Close Time", "Quote Asset Vol", "Number of Trades", 
        "Taker buy base vol", "Taker buy quote vol", "Ignore"
    ])
    
    df["Date"] = pd.to_datetime(df["Open Time"], unit='ms')
    for col in ["Open", "High", "Low", "Close", "Volume", "Quote Asset Vol"]:
        df[col] = df[col].astype(float)
        
    return df[["Date", "Close", "Volume", "Quote Asset Vol"]]

def fetch_blockchain_charts():
    """
    Fetches historical proxy metrics from Blockchain.com.
    We use 'avg-transaction-value' as the best free proxy for 'Large Tx Volume'.
    """
    print("[Init] Fetching Blockchain.com chart data...")
    
    def get_chart(chart_name):
        url = f"https://api.blockchain.info/charts/{chart_name}?timespan=8years&format=json"
        try:
            r = requests.get(url).json()
            df = pd.DataFrame(r['values'])
            df['Date'] = pd.to_datetime(df['x'], unit='s')
            df = df.rename(columns={'y': chart_name})
            return df[['Date', chart_name]]
        except:
            return pd.DataFrame()

    # 1. Average Transaction Value (The Whale Proxy)
    df_avg = get_chart("avg-transaction-value")
    # 2. Total Estimated Transaction Volume (Overall Network Activity)
    df_vol = get_chart("estimated-transaction-volume-usd")

    if df_avg.empty or df_vol.empty:
        return pd.DataFrame()

    # Merge
    df_final = pd.merge(df_avg, df_vol, on="Date", how="outer")
    return df_final.sort_values("Date")

# =============================================================================
# LIVE WHALE LISTENER (BACKGROUND THREAD)
# =============================================================================

class WhaleListener(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True # Kills thread when main app exits
        self.ws_url = "wss://ws.blockchain.info/inv"
        self.running = True

    def run(self):
        print("[Listener] Starting Live Whale Monitor...")
        # Update price once before starting
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
            global CURRENT_BTC_PRICE
            CURRENT_BTC_PRICE = float(r.json()['price'])
            print(f"[Listener] Current BTC Price: ${CURRENT_BTC_PRICE:,.2f}")
        except:
            pass

        while self.running:
            try:
                ws = websocket.WebSocket()
                ws.connect(self.ws_url)
                ws.send(json.dumps({"op": "unconfirmed_sub"}))
                
                while self.running:
                    result = ws.recv()
                    data = json.loads(result)
                    
                    if data.get("op") == "utx":
                        x = data["x"]
                        # Calculate total output value of transaction
                        total_satoshi = sum([out["value"] for out in x["out"]])
                        btc_amount = total_satoshi / 100_000_000
                        usd_value = btc_amount * CURRENT_BTC_PRICE
                        
                        if usd_value > WHALE_THRESHOLD_USD:
                            timestamp = datetime.datetime.fromtimestamp(x["time"]).strftime('%H:%M:%S')
                            hash_short = x["hash"][:8] + "..." + x["hash"][-8:]
                            
                            entry = {
                                "Time": timestamp,
                                "Hash": hash_short,
                                "BTC": f"{btc_amount:.4f}",
                                "Value (USD)": f"${usd_value:,.2f}",
                                "Size": "🐳 WHALE" if usd_value > 1_000_000 else "🐟 Large"
                            }
                            
                            # Prepend to global list (keep last 50)
                            global LIVE_WHALES
                            LIVE_WHALES.insert(0, entry)
                            LIVE_WHALES = LIVE_WHALES[:50]
                            
            except Exception as e:
                print(f"[Listener] Connection lost ({e}). Reconnecting in 5s...")
                time.sleep(5)

# Start the listener immediately
listener = WhaleListener()
listener.start()

# =============================================================================
# DATA PREPARATION (MAIN THREAD)
# =============================================================================

df_binance = fetch_binance_ohlcv()
df_blockchain = fetch_blockchain_charts()

# Merge daily data
if not df_binance.empty and not df_blockchain.empty:
    # Normalize dates to midnight
    df_binance['Date'] = df_binance['Date'].dt.normalize()
    df_blockchain['Date'] = df_blockchain['Date'].dt.normalize()
    
    df = pd.merge(df_binance, df_blockchain, on="Date", how="inner")
    df = df.sort_values("Date")
else:
    df = pd.DataFrame()
    print("Warning: Could not fetch historical data.")

# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, title="Whale Watcher Pro")
server = app.server

app.layout = html.Div(style={'backgroundColor': '#0b0c10', 'minHeight': '100vh', 'color': '#c5c6c7', 'fontFamily': 'Arial, sans-serif'}, children=[
    
    # Header
    html.Div(style={'padding': '20px', 'backgroundColor': '#1f2833', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'}, children=[
        html.H1("🐋 Bitcoin Whale Watcher & Market Analysis", style={'color': '#66fcf1', 'margin': '0'}),
        html.P("Combining Historical Binance Data with Blockchain.com On-Chain Metrics", style={'margin': '5px 0 0 0', 'fontSize': '14px'})
    ]),

    # Main Content Area
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '20px'}, children=[
        
        # LEFT COLUMN: Historical Charts
        html.Div(style={'flex': '2', 'minWidth': '600px', 'marginRight': '20px'}, children=[
            
            # Chart 1: Price vs Avg Transaction Value
            html.Div(style={'backgroundColor': '#1f2833', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'}, children=[
                html.H3("Historical Whale Activity Proxy", style={'color': '#45a29e'}),
                html.P("Metric: Average Transaction Value (USD). Spikes indicate periods where whales moved large funds, dragging the average up significantly.", 
                       style={'fontSize': '12px', 'color': '#888'}),
                dcc.Graph(id='historical-chart')
            ]),
        ]),

        # RIGHT COLUMN: Live Feed
        html.Div(style={'flex': '1', 'minWidth': '300px'}, children=[
            html.Div(style={'backgroundColor': '#1f2833', 'padding': '15px', 'borderRadius': '10px', 'height': '80vh', 'overflowY': 'hidden', 'display': 'flex', 'flexDirection': 'column'}, children=[
                html.H3(f"Live Scanner (>{WHALE_THRESHOLD_USD/1000}k)", style={'color': '#ff4d4d', 'display': 'inline-block'}),
                html.Span(" • Live Mempool Feed", style={'fontSize': '12px', 'color': '#00ff00', 'marginLeft': '10px', 'animation': 'blink 2s infinite'}),
                
                html.Div(id='live-whale-feed', style={'flex': '1', 'overflowY': 'auto', 'marginTop': '10px'}),
                
                dcc.Interval(id='interval-component', interval=2000, n_intervals=0) # Update every 2s
            ])
        ])
    ])
])

@app.callback(
    Output('historical-chart', 'figure'),
    Input('interval-component', 'n_intervals') # Just to trigger once on load really
)
def update_charts(n):
    if df.empty:
        return go.Figure()

    # Create Dual Axis Chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Bar Chart: Average Transaction Value (The Whale Indicator)
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['avg-transaction-value'], 
            name="Avg Tx Value (USD)",
            marker=dict(color='#45a29e', opacity=0.5),
        ),
        secondary_y=False,
    )

    # 2. Line Chart: BTC Price
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Close'], 
            name="BTC Price",
            line=dict(color='#ffffff', width=2)
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=40, r=40, t=20, b=40),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Avg Transaction Value ($)", secondary_y=False, showgrid=False, gridcolor='#333')
    fig.update_yaxes(title_text="BTC Price ($)", secondary_y=True, showgrid=True, gridcolor='#333')

    return fig

@app.callback(
    Output('live-whale-feed', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_feed(n):
    if not LIVE_WHALES:
        return html.Div("Waiting for large transactions...", style={'textAlign': 'center', 'marginTop': '20px', 'color': '#888'})
    
    # Convert list of dicts to a nice HTML list
    rows = []
    for whale in LIVE_WHALES:
        # Determine color based on size
        row_color = '#ff4d4d' if "🐳" in whale['Size'] else '#45a29e'
        
        row = html.Div(style={
            'borderBottom': '1px solid #333', 
            'padding': '10px 0', 
            'display': 'flex', 
            'justifyContent': 'space-between', 
            'alignItems': 'center'
        }, children=[
            html.Div([
                html.Div(whale['Size'], style={'fontSize': '14px', 'fontWeight': 'bold', 'color': row_color}),
                html.Div(whale['Time'], style={'fontSize': '11px', 'color': '#888'}),
            ]),
            html.Div([
                html.Div(whale['Value (USD)'], style={'fontSize': '15px', 'fontWeight': 'bold', 'color': '#fff', 'textAlign': 'right'}),
                html.Div(f"{whale['BTC']} BTC", style={'fontSize': '11px', 'color': '#aaa', 'textAlign': 'right'}),
            ])
        ])
        rows.append(row)
    
    return rows

if __name__ == '__main__':
    # Important: Setting debug=False because we are using threads
    app.run_server(debug=True, port=8050)
