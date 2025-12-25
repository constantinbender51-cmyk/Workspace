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
# GLOBAL STATE
# =============================================================================

LIVE_WHALES = []
DATA_LOCK = threading.Lock()
WS_CONNECTED = False

WHALE_THRESHOLD_USD = 100000 
CURRENT_BTC_PRICE = 96000.0

GLOBAL_DF = pd.DataFrame()
DATA_STATUS = "Initializing..."

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# =============================================================================
# 1. ROBUST DATA FETCHING
# =============================================================================

def fetch_binance_data():
    print("[Fetcher] Getting Binance Prices...")
    try:
        url = "https://api.binance.com/api/v3/klines"
        start_ts = int((datetime.datetime.now() - datetime.timedelta(days=365*5)).timestamp() * 1000)
        params = {"symbol": "BTCUSDT", "interval": "1d", "startTime": start_ts, "limit": 1000}
        
        all_data = []
        while True:
            r = requests.get(url, params=params, headers=HEADERS)
            data = r.json()
            if not data or not isinstance(data, list): break
            all_data.extend(data)
            if data[-1][6] > (datetime.datetime.now().timestamp() * 1000) - 86400000: break
            params['startTime'] = data[-1][6] + 1
            time.sleep(0.1)

        df = pd.DataFrame(all_data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume", 
            "Close Time", "Quote Asset Vol", "Num Trades", "Taker Base", "Taker Quote", "Ignore"
        ])
        df["Date"] = pd.to_datetime(df["Open Time"], unit='ms').dt.normalize()
        df["Close"] = df["Close"].astype(float)
        return df[["Date", "Close"]]
    except Exception as e:
        print(f"[Fetcher] Binance Error: {e}")
        return pd.DataFrame()

def fetch_blockchain_data():
    print("[Fetcher] Getting Blockchain.com Volume...")
    
    # 1. Get Volume
    df_vol = pd.DataFrame()
    try:
        url = "https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=5years&format=json"
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            if 'values' in data:
                df_vol = pd.DataFrame(data['values'])
                df_vol['Date'] = pd.to_datetime(df_vol['x'], unit='s').dt.normalize()
                df_vol['Volume'] = df_vol['y'].astype(float) # Force Float
    except Exception as e:
        print(f"[Fetcher] Volume Error: {e}")

    # 2. Get Count
    df_count = pd.DataFrame()
    try:
        url = "https://api.blockchain.info/charts/n-transactions?timespan=5years&format=json"
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            if 'values' in data:
                df_count = pd.DataFrame(data['values'])
                df_count['Date'] = pd.to_datetime(df_count['x'], unit='s').dt.normalize()
                df_count['Count'] = df_count['y'].astype(float) # Force Float
    except Exception as e:
        print(f"[Fetcher] Count Error: {e}")

    if df_vol.empty or df_count.empty:
        print("[Fetcher] One or both Blockchain datasets failed.")
        return pd.DataFrame()

    # Merge
    df_final = pd.merge(df_vol, df_count, on="Date", how="inner")
    
    # Calculate Average
    df_final = df_final[df_final['Count'] > 0]
    df_final['AvgTxValue'] = df_final['Volume'] / df_final['Count']
    
    # Debug Print
    print(f"[Fetcher] Calculated {len(df_final)} rows. Max Avg: ${df_final['AvgTxValue'].max():.2f}")
    
    return df_final[['Date', 'AvgTxValue']]

def update_data_thread():
    global GLOBAL_DF, DATA_STATUS
    while True:
        DATA_STATUS = "Fetching Data..."
        
        df_price = fetch_binance_data()
        df_chain = fetch_blockchain_data()
        
        if not df_price.empty:
            if not df_chain.empty:
                # Left merge ensures we keep all price dates. Fill missing chain data with NaN.
                GLOBAL_DF = pd.merge(df_price, df_chain, on="Date", how="left")
                DATA_STATUS = f"Active. Price: {len(df_price)} rows. Chain: {len(df_chain)} rows."
            else:
                GLOBAL_DF = df_price
                GLOBAL_DF['AvgTxValue'] = 0
                DATA_STATUS = "Partial. Showing Price Only (Blockchain API failed)."
        else:
            DATA_STATUS = "Error. Could not fetch Price data."
            
        time.sleep(3600)

t_data = threading.Thread(target=update_data_thread)
t_data.daemon = True
t_data.start()

# =============================================================================
# 2. WEBSOCKET LISTENER
# =============================================================================

def start_socket():
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get("op") == "utx":
                x = data["x"]
                total_satoshi = sum([out.get("value", 0) for out in x.get("out", [])])
                usd_val = (total_satoshi / 100_000_000) * CURRENT_BTC_PRICE
                
                if usd_val > WHALE_THRESHOLD_USD:
                    t_str = datetime.datetime.fromtimestamp(x["time"]).strftime('%H:%M:%S')
                    
                    if usd_val > 1_000_000:
                        icon, color = "🐋 WHALE", "#ff4d4d"
                    elif usd_val > 500_000:
                        icon, color = "🦈 Shark", "#ffa600"
                    else:
                        icon, color = "🐟 Large", "#00cc96"

                    entry = {
                        "Time": t_str,
                        "Hash": x["hash"][:8],
                        "Value": f"${usd_val:,.0f}",
                        "Type": icon,
                        "Color": color
                    }
                    with DATA_LOCK:
                        LIVE_WHALES.insert(0, entry)
                        if len(LIVE_WHALES) > 50: LIVE_WHALES.pop()
        except: pass

    def on_open(ws):
        global WS_CONNECTED
        WS_CONNECTED = True
        ws.send(json.dumps({"op": "unconfirmed_sub"}))

    def on_error(ws, err):
        global WS_CONNECTED
        WS_CONNECTED = False

    while True:
        ws = websocket.WebSocketApp("wss://ws.blockchain.info/inv", 
                                  on_open=on_open, on_message=on_message, on_error=on_error)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        time.sleep(5)

t_ws = threading.Thread(target=start_socket)
t_ws.daemon = True
t_ws.start()

# =============================================================================
# 3. DASH APP
# =============================================================================

app = dash.Dash(__name__, title="Whale Dashboard Visible")
server = app.server

app.layout = html.Div(style={'backgroundColor': '#111', 'minHeight': '100vh', 'color': '#ccc', 'fontFamily': 'sans-serif', 'padding': '20px'}, children=[
    
    html.Div([
        html.H2("Whale & Market Monitor", style={'color': '#fff', 'margin': 0}),
        html.Div(id='system-status', style={'color': '#888', 'fontSize': '12px', 'marginTop': '5px'})
    ], style={'borderBottom': '1px solid #333', 'paddingBottom': '15px'}),

    html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '20px'}, children=[
        html.Div(style={'flex': '2', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px', 'padding': '15px'}, children=[
            dcc.Graph(id='main-chart', style={'height': '600px'})
        ]),
        html.Div(style={'flex': '1', 'backgroundColor': '#1e1e1e', 'borderRadius': '8px', 'padding': '15px', 'height': '600px', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H3("Live Whales (>$100k)", style={'color': '#00cc96', 'margin': '0 0 15px 0'}),
            html.Div(id='live-feed', style={'flex': '1', 'overflowY': 'auto'})
        ])
    ]),
    
    dcc.Interval(id='timer', interval=2000)
])

@app.callback(
    [Output('main-chart', 'figure'), Output('live-feed', 'children'), Output('system-status', 'children')],
    Input('timer', 'n_intervals')
)
def update_ui(n):
    # 1. Update Chart
    if GLOBAL_DF.empty:
        fig = go.Figure()
        fig.update_layout(title="Loading Data...", template="plotly_dark")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Trace 1: Calculated Average Value (FILLED AREA)
        # Using Scatter with fill='tozeroy' ensures it's visible even if lines are thin
        fig.add_trace(go.Scatter(
            x=GLOBAL_DF['Date'], 
            y=GLOBAL_DF['AvgTxValue'],
            name="Avg Tx Size ($)",
            mode='lines',
            line=dict(width=0), # Hide line, show fill
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.4)', # Green with opacity
            connectgaps=True # Connect across missing dates
        ), secondary_y=False)
        
        # Trace 2: Price (Line)
        fig.add_trace(go.Scatter(
            x=GLOBAL_DF['Date'], 
            y=GLOBAL_DF['Close'],
            name="BTC Price",
            mode='lines',
            line=dict(color='#ff4d4d', width=2)
        ), secondary_y=True)
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", y=1.02, x=0),
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Avg Transaction Value ($)", secondary_y=False, showgrid=False)
        fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=True, gridcolor='#333')

    # 2. Update Feed
    with DATA_LOCK:
        feed_data = list(LIVE_WHALES)
        
    feed_items = []
    if not feed_data:
        feed_items.append(html.Div("Waiting for transactions...", style={'textAlign': 'center', 'marginTop': '20px'}))
    else:
        for x in feed_data:
            feed_items.append(html.Div(style={'borderBottom': '1px solid #333', 'padding': '10px 0', 'display': 'flex', 'justifyContent': 'space-between'}, children=[
                html.Div([
                    html.Div(x['Type'], style={'color': x['Color'], 'fontWeight': 'bold', 'fontSize': '14px'}),
                    html.Div(x['Time'], style={'fontSize': '12px', 'color': '#666'})
                ]),
                html.Div([
                    html.Div(x['Value'], style={'color': '#fff', 'fontWeight': 'bold'}),
                    html.Div(x['Hash'], style={'fontSize': '11px', 'color': '#444', 'fontFamily': 'monospace'})
                ], style={'textAlign': 'right'})
            ]))

    return fig, feed_items, f"System Status: {DATA_STATUS} | WebSocket: {'Connected' if WS_CONNECTED else 'Disconnected'}"

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
