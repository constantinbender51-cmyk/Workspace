import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
import datetime
import threading
import time
import os
import ssl

# =============================================================================
# GLOBAL STATE & CONFIG
# =============================================================================

GLOBAL_DF = pd.DataFrame()
DATA_STATUS = "Initializing..."
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# =============================================================================
# DATA FETCHING (HISTORICAL ONLY)
# =============================================================================

def fetch_binance_data():
    """Fetches daily BTCUSDT close prices since 2018."""
    try:
        url = "https://api.binance.com/api/v3/klines"
        start_ts = int(datetime.datetime(2018, 1, 1).timestamp() * 1000)
        params = {"symbol": "BTCUSDT", "interval": "1d", "startTime": start_ts, "limit": 1000}
        
        all_data = []
        while True:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            data = r.json()
            if not data or not isinstance(data, list): break
            all_data.extend(data)
            # Stop if we are within the last 24 hours
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
        print(f"Binance Fetch Error: {e}")
        return pd.DataFrame()

def fetch_blockchain_data():
    """Fetches and calculates Average Transaction Value (Volume / Tx Count)."""
    try:
        # Fetch Total Estimated Volume (USD)
        url_v = "https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=8years&format=json"
        r_v = requests.get(url_v, headers=HEADERS, timeout=15)
        df_vol = pd.DataFrame(r_v.json()['values'])
        df_vol['Date'] = pd.to_datetime(df_vol['x'], unit='s').dt.normalize()
        df_vol['Volume'] = df_vol['y'].astype(float)

        # Fetch Transaction Count
        url_c = "https://api.blockchain.info/charts/n-transactions?timespan=8years&format=json"
        r_c = requests.get(url_c, headers=HEADERS, timeout=15)
        df_count = pd.DataFrame(r_c.json()['values'])
        df_count['Date'] = pd.to_datetime(df_count['x'], unit='s').dt.normalize()
        df_count['Count'] = df_count['y'].astype(float)

        # Merge and Calculate
        df_final = pd.merge(df_vol, df_count, on="Date", how="inner")
        df_final = df_final[df_final['Count'] > 0]
        df_final['AvgTxValue'] = df_final['Volume'] / df_final['Count']
        
        return df_final[['Date', 'AvgTxValue']]
    except Exception as e:
        print(f"Blockchain Fetch Error: {e}")
        return pd.DataFrame()

def update_data_thread():
    """Background loop to keep the dataset updated every hour."""
    global GLOBAL_DF, DATA_STATUS
    while True:
        DATA_STATUS = "Updating historical data..."
        df_p = fetch_binance_data()
        df_c = fetch_blockchain_data()
        
        if not df_p.empty and not df_c.empty:
            # Join datasets on Date
            merged = pd.merge(df_p, df_c, on="Date", how="left")
            # Fill gaps for a clean area chart
            merged = merged.sort_values("Date").reset_index(drop=True)
            GLOBAL_DF = merged
            DATA_STATUS = f"Data Synced: {len(GLOBAL_DF)} days of history."
        else:
            DATA_STATUS = "Sync failed. Retrying in 1 minute..."
            time.sleep(60)
            continue
            
        time.sleep(3600)

# Start background sync
threading.Thread(target=update_data_thread, daemon=True).start()

# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, title="BTC Historical Whale Dashboard")
server = app.server # For Gunicorn/Railway

app.layout = html.Div(style={'backgroundColor': '#0b0c10', 'minHeight': '100vh', 'color': '#c5c6c7', 'fontFamily': 'sans-serif', 'padding': '40px'}, children=[
    
    html.Div(style={'marginBottom': '30px', 'borderBottom': '1px solid #1f2833', 'paddingBottom': '20px'}, children=[
        html.H1("Bitcoin Historical Whale Analysis", style={'color': '#66fcf1', 'margin': '0'}),
        html.P(id='status-text', style={'fontSize': '14px', 'color': '#45a29e', 'marginTop': '10px'})
    ]),

    html.Div(style={'backgroundColor': '#1f2833', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 10px 30px rgba(0,0,0,0.5)'}, children=[
        html.H3("Average Transaction Value vs. Market Price (Since 2018)", style={'color': '#fff', 'marginTop': '0'}),
        html.P("Spikes in Average Transaction Value indicate periods of high whale movement.", style={'fontSize': '12px', 'color': '#888'}),
        dcc.Graph(id='main-chart', style={'height': '70vh'})
    ]),
    
    dcc.Interval(id='ui-refresh', interval=5000)
])

@app.callback(
    [Output('main-chart', 'figure'), Output('status-text', 'children')],
    Input('ui-refresh', 'n_intervals')
)
def update_ui(n):
    if GLOBAL_DF.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="Fetching history from APIs...")
        return fig, DATA_STATUS

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Historical Whale Proxy: Avg Transaction Value (Area Chart)
    fig.add_trace(
        go.Scatter(
            x=GLOBAL_DF['Date'], 
            y=GLOBAL_DF['AvgTxValue'],
            name="Avg Tx Size (USD)",
            mode='lines',
            line=dict(width=0),
            fill='tozeroy',
            fillcolor='rgba(102, 252, 241, 0.2)', # Cyan area
            connectgaps=True
        ),
        secondary_y=False,
    )

    # 2. Market Price: BTCUSDT (Line Chart)
    fig.add_trace(
        go.Scatter(
            x=GLOBAL_DF['Date'], 
            y=GLOBAL_DF['Close'],
            name="BTC Price (Binance)",
            mode='lines',
            line=dict(color='#ff4d4d', width=2)
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(showgrid=False)
    )

    fig.update_yaxes(title_text="Avg Transaction Value ($)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=True, gridcolor='#333')

    return fig, DATA_STATUS

if __name__ == '__main__':
    # Railway Environment Port Binding
    port = int(os.environ.get("PORT", 8080))
    app.run_server(host='0.0.0.0', port=port)