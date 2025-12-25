import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import datetime
import threading
import json
import time
import websocket
import ssl

# =============================================================================
# GLOBAL SETTINGS & SHARED STATE
# =============================================================================

# Data Store for Live Transactions
LIVE_WHALES = []
DATA_LOCK = threading.Lock()

# Connection Status
WS_CONNECTED = False

# Thresholds
WHALE_THRESHOLD_USD = 100000  # $100k Limit
CURRENT_BTC_PRICE = 95000.0   # Default fallback

# Headers to bypass 403 Forbidden errors
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# =============================================================================
# 1. HISTORICAL DATA FETCHING
# =============================================================================

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1d", start_year=2018):
    """
    Fetches daily OHLCV data from Binance.
    """
    print(f"[Init] Fetching Binance data for {symbol}...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    start_ts = int(datetime.datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(datetime.datetime.now().timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while start_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "limit": limit}
        try:
            r = requests.get(base_url, params=params, headers=HEADERS)
            data = r.json()
            
            if not data or not isinstance(data, list):
                break
                
            all_data.extend(data)
            start_ts = data[-1][6] + 1
            time.sleep(0.05) 
        except Exception as e:
            print(f"Error fetching Binance: {e}")
            break
            
    if not all_data: 
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume", 
        "Close Time", "Quote Asset Vol", "Number of Trades", 
        "Taker buy base vol", "Taker buy quote vol", "Ignore"
    ])
    
    df["Date"] = pd.to_datetime(df["Open Time"], unit='ms')
    for col in ["Close", "Volume"]:
        df[col] = df[col].astype(float)
        
    return df[["Date", "Close", "Volume"]]

def fetch_blockchain_charts():
    """
    Fetches historical proxy metrics. 
    Since 'avg-transaction-value' is 404, we calculate it manually:
    Avg = Total USD Volume / Number of Transactions
    """
    print("[Init] Fetching Blockchain.com data...")
    
    def get_chart(chart_name, col_name):
        # Using 3years to ensure stability, or 8years if available
        url = f"https://api.blockchain.info/charts/{chart_name}?timespan=8years&format=json"
        try:
            r = requests.get(url, headers=HEADERS)
            if r.status_code != 200:
                print(f"Failed to fetch {chart_name}: Status {r.status_code}")
                return pd.DataFrame()
                
            data = r.json()
            if 'values' not in data:
                return pd.DataFrame()

            df = pd.DataFrame(data['values'])
            df['Date'] = pd.to_datetime(df['x'], unit='s')
            df = df.rename(columns={'y': col_name})
            return df[['Date', col_name]]
        except Exception as e:
            print(f"Error fetching {chart_name}: {e}")
            return pd.DataFrame()

    # 1. Total Estimated Volume (The numerator)
    df_vol = get_chart("estimated-transaction-volume-usd", "total_volume_usd")
    
    # 2. Number of Transactions (The denominator)
    df_count = get_chart("n-transactions", "tx_count")

    if df_vol.empty or df_count.empty:
        print("Error: One of the Blockchain.com charts returned empty.")
        return pd.DataFrame()

    # Merge
    df_final = pd.merge(df_vol, df_count, on="Date", how="inner")
    
    # Clean Data: Remove rows with 0 volume or 0 transactions to avoid skewed averages or div/0
    df_final = df_final[df_final["tx_count"] > 0]
    df_final = df_final[df_final["total_volume_usd"] > 0]

    # Calculate Average
    df_final["avg-transaction-value"] = df_final["total_volume_usd"] / df_final["tx_count"]

    # Re-index to ensure continuous timeline (Fills missing dates with NaN so they show as gaps)
    # This prevents the chart from looking "gappy" due to missing index rows
    if not df_final.empty:
        df_final.set_index("Date", inplace=True)
        df_final = df_final.resample('D').mean() # Resample to Daily frequency
        df_final.reset_index(inplace=True)
    
    print(f"[Init] Successfully calculated Avg Tx Value for {len(df_final)} days.")
    return df_final.sort_values("Date")

# =============================================================================
# 2. LIVE WEBSOCKET LISTENER
# =============================================================================

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get("op") == "utx":
            x = data["x"]
            # Sum all output values
            total_satoshi = sum([out.get("value", 0) for out in x.get("out", [])])
            btc_amount = total_satoshi / 100_000_000
            usd_value = btc_amount * CURRENT_BTC_PRICE
            
            if usd_value > WHALE_THRESHOLD_USD:
                timestamp = datetime.datetime.fromtimestamp(x["time"]).strftime('%H:%M:%S')
                hash_short = x["hash"][:6] + "..." + x["hash"][-6:]
                
                # Determine Whale Type
                if usd_value > 1_000_000:
                    icon = "🐋 WHALE"
                    color = "#ff4d4d" # Red
                elif usd_value > 500_000:
                    icon = "🦈 Shark"
                    color = "#ffa600" # Orange
                else:
                    icon = "🐟 Large"
                    color = "#00cc96" # Green

                entry = {
                    "Time": timestamp,
                    "Hash": hash_short,
                    "BTC": f"{btc_amount:.2f}",
                    "Value": f"${usd_value:,.0f}",
                    "Type": icon,
                    "Color": color,
                    "id": x["hash"]
                }
                
                with DATA_LOCK:
                    global LIVE_WHALES
                    LIVE_WHALES.insert(0, entry)
                    if len(LIVE_WHALES) > 50:
                        LIVE_WHALES.pop()
    except Exception as e:
        print(f"Parse Error: {e}")

def on_error(ws, error):
    global WS_CONNECTED
    WS_CONNECTED = False
    print(f"[WebSocket] Error: {error}")

def on_close(ws, close_status_code, close_msg):
    global WS_CONNECTED
    WS_CONNECTED = False
    print("[WebSocket] Closed. Reconnecting in 5s...")
    
def on_open(ws):
    global WS_CONNECTED
    WS_CONNECTED = True
    print("[WebSocket] Connected! Subscribing to transactions...")
    ws.send(json.dumps({"op": "unconfirmed_sub"}))

def start_socket():
    # Update Price First
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", headers=HEADERS)
        global CURRENT_BTC_PRICE
        CURRENT_BTC_PRICE = float(r.json()['price'])
        print(f"[Init] Live BTC Price: ${CURRENT_BTC_PRICE:,.2f}")
    except:
        print("[Init] Could not fetch price, using default.")

    while True:
        ws = websocket.WebSocketApp(
            "wss://ws.blockchain.info/inv",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        time.sleep(5) 

# Start WebSocket in Background Thread
t = threading.Thread(target=start_socket)
t.daemon = True
t.start()

# =============================================================================
# 3. PREPARE DATA
# =============================================================================

df_binance = fetch_binance_ohlcv()
df_chain = fetch_blockchain_charts()

if not df_binance.empty and not df_chain.empty:
    df_binance['Date'] = df_binance['Date'].dt.normalize()
    df_chain['Date'] = df_chain['Date'].dt.normalize()
    df_main = pd.merge(df_binance, df_chain, on="Date", how="inner").sort_values("Date")
else:
    print("Warning: Historical charts may be empty.")
    df_main = pd.DataFrame()

# =============================================================================
# 4. DASHBOARD LAYOUT
# =============================================================================

app = dash.Dash(__name__, title="Whale Monitor Final")
server = app.server

app.layout = html.Div(style={'backgroundColor': '#111', 'minHeight': '100vh', 'color': '#eee', 'fontFamily': 'sans-serif', 'padding': '20px'}, children=[
    
    html.Div([
        html.H1("Whale Activity Dashboard", style={'display': 'inline-block', 'marginRight': '20px'}),
        html.Span(id='status-indicator', style={'fontSize': '14px', 'fontWeight': 'bold'})
    ], style={'borderBottom': '1px solid #333', 'paddingBottom': '10px'}),

    html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '20px'}, children=[
        
        # LEFT: Historical Charts
        html.Div(style={'flex': '2'}, children=[
            html.Div(style={'backgroundColor': '#222', 'padding': '15px', 'borderRadius': '8px'}, children=[
                html.H3("Avg Transaction Value (Calculated)", style={'marginTop': 0}),
                html.P("Derived from Total On-Chain Volume / Transaction Count", style={'color': '#888', 'fontSize': '12px'}),
                dcc.Graph(id='main-chart', style={'height': '500px'}),
            ])
        ]),
        
        # RIGHT: Live Feed
        html.Div(style={'flex': '1', 'backgroundColor': '#222', 'padding': '15px', 'borderRadius': '8px', 'height': '600px', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H3(f"Live Feed (>${WHALE_THRESHOLD_USD/1000:,.0f}k)", style={'marginTop': 0, 'color': '#00cc96'}),
            html.Div(style={'borderBottom': '1px solid #444', 'paddingBottom': '5px', 'marginBottom': '10px', 'display': 'flex', 'fontSize': '12px', 'color': '#888'}, children=[
                html.Div("Type", style={'width': '25%'}),
                html.Div("Value (USD)", style={'width': '35%', 'textAlign': 'right'}),
                html.Div("Time", style={'width': '20%', 'textAlign': 'right'}),
                html.Div("Hash", style={'flex': '1', 'textAlign': 'right'}),
            ]),
            html.Div(id='live-list', style={'overflowY': 'auto', 'flex': '1'})
        ])
    ]),
    
    dcc.Interval(id='update-interval', interval=1000, n_intervals=0)
])

# =============================================================================
# 5. CALLBACKS
# =============================================================================

@app.callback(
    Output('main-chart', 'figure'),
    Input('update-interval', 'n_intervals') 
)
def update_chart(n):
    if n > 0 or df_main.empty: 
        if df_main.empty: return go.Figure()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar: Avg Transaction Value (Calculated)
        fig.add_trace(go.Bar(
            x=df_main['Date'], 
            y=df_main['avg-transaction-value'], 
            name="Avg Tx Size ($)",
            marker_color='#00cc96', opacity=0.3
        ), secondary_y=False)

        # Line: BTC Price
        fig.add_trace(go.Scatter(
            x=df_main['Date'], 
            y=df_main['Close'], 
            name="Price ($)",
            line=dict(color='#ff4d4d', width=2)
        ), secondary_y=True)

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.05)
        )
        return fig
    return dash.no_update

@app.callback(
    [Output('live-list', 'children'), Output('status-indicator', 'children'), Output('status-indicator', 'style')],
    Input('update-interval', 'n_intervals')
)
def update_feed(n):
    status_text = "🟢 Connected to Blockchain.com" if WS_CONNECTED else "🔴 Connecting..."
    status_style = {'color': '#00cc96' if WS_CONNECTED else '#ff4d4d', 'paddingTop': '5px'}

    children = []
    with DATA_LOCK:
        current_data = list(LIVE_WHALES)
    
    if not current_data:
        children.append(html.Div(f"Waiting for >${WHALE_THRESHOLD_USD/1000:,.0f}k transactions...", style={'textAlign': 'center', 'color': '#666', 'marginTop': '20px'}))
    else:
        for item in current_data:
            row = html.Div(style={'display': 'flex', 'padding': '8px 0', 'borderBottom': '1px solid #333', 'alignItems': 'center', 'animation': 'fadeIn 0.5s'}, children=[
                html.Div(item['Type'], style={'width': '25%', 'fontWeight': 'bold', 'color': item['Color'], 'fontSize': '13px'}),
                html.Div(item['Value'], style={'width': '35%', 'textAlign': 'right', 'color': '#fff', 'fontWeight': 'bold', 'fontSize': '14px'}),
                html.Div(item['Time'], style={'width': '20%', 'textAlign': 'right', 'color': '#888', 'fontSize': '12px'}),
                html.Div(item['Hash'], style={'flex': '1', 'textAlign': 'right', 'fontFamily': 'monospace', 'color': '#555', 'fontSize': '11px'}),
            ])
            children.append(row)
            
    return children, status_text, status_style

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
