import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import requests
import threading
import time
import os

# =============================================================================
# CONFIG
# =============================================================================

GLOBAL_DF = pd.DataFrame()
DATA_STATUS = "Initializing..."
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json'
}

# =============================================================================
# DATA FETCHING (BLOCKCHAIN.COM ONLY)
# =============================================================================

def fetch_blockchain_data():
    """
    Fetches raw volume and transaction count to calculate average value.
    Returns: DataFrame, ErrorMessage (str)
    """
    status_log = []
    try:
        # 1. Fetch Volume (USD)
        url_vol = "https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=8years&format=json"
        r_vol = requests.get(url_vol, headers=HEADERS, timeout=15)
        
        if r_vol.status_code != 200:
            return pd.DataFrame(), f"Volume API Failed: Status {r_vol.status_code}"
            
        data_vol = r_vol.json()
        if 'values' not in data_vol:
            return pd.DataFrame(), "Volume API returned no 'values' key."
            
        df_vol = pd.DataFrame(data_vol['values'])
        df_vol['Date'] = pd.to_datetime(df_vol['x'], unit='s').dt.normalize()
        df_vol['Volume'] = df_vol['y'].astype(float)
        status_log.append(f"Volume fetched ({len(df_vol)} rows)")

        # 2. Fetch Tx Count
        url_count = "https://api.blockchain.info/charts/n-transactions?timespan=8years&format=json"
        r_count = requests.get(url_count, headers=HEADERS, timeout=15)
        
        if r_count.status_code != 200:
            return pd.DataFrame(), f"Count API Failed: Status {r_count.status_code}"
            
        data_count = r_count.json()
        df_count = pd.DataFrame(data_count['values'])
        df_count['Date'] = pd.to_datetime(df_count['x'], unit='s').dt.normalize()
        df_count['Count'] = df_count['y'].astype(float)
        status_log.append(f"Count fetched ({len(df_count)} rows)")

        # 3. Merge & Calculate
        df = pd.merge(df_vol, df_count, on="Date", how="inner")
        
        # Filter out zero counts to avoid division by zero
        df = df[df['Count'] > 0]
        df['AvgTxValue'] = df['Volume'] / df['Count']
        
        status_msg = f"Success! {len(df)} days of data loaded. (Last Date: {df['Date'].max().date()})"
        return df[['Date', 'AvgTxValue']], status_msg

    except Exception as e:
        return pd.DataFrame(), f"Python Exception: {str(e)}"

def update_data_thread():
    global GLOBAL_DF, DATA_STATUS
    while True:
        DATA_STATUS = "Fetching from Blockchain.com..."
        df, msg = fetch_blockchain_data()
        
        if not df.empty:
            GLOBAL_DF = df.sort_values("Date")
        
        DATA_STATUS = msg
        time.sleep(3600) # Refresh hourly

# Start background sync
threading.Thread(target=update_data_thread, daemon=True).start()

# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, title="Blockchain.com Debugger")
server = app.server

app.layout = html.Div(style={'backgroundColor': '#111', 'minHeight': '100vh', 'color': '#ccc', 'fontFamily': 'monospace', 'padding': '20px'}, children=[
    
    html.H2("Blockchain.com Data Debugger"),
    html.Div(id='status-display', style={'border': '1px solid #333', 'padding': '10px', 'marginBottom': '20px', 'color': '#00ff00'}),
    
    dcc.Graph(id='main-chart', style={'height': '70vh'}),
    
    dcc.Interval(id='timer', interval=5000) # Check status every 5s
])

@app.callback(
    [Output('main-chart', 'figure'), Output('status-display', 'children')],
    Input('timer', 'n_intervals')
)
def update_view(n):
    # Style the status message based on success/failure
    status_style = {'color': '#ff4d4d'} if "Failed" in DATA_STATUS or "Exception" in DATA_STATUS else {'color': '#00ff00'}
    
    status_component = html.Span(DATA_STATUS, style=status_style)

    if GLOBAL_DF.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", 
            title="No Data Available",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        return fig, status_component

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=GLOBAL_DF['Date'],
        y=GLOBAL_DF['AvgTxValue'],
        name="Avg Tx Value",
        mode='lines',
        fill='tozeroy',
        line=dict(color='#00cc96')
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Average Transaction Value (USD)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        uirevision='constant' # Keeps zoom level
    )

    return fig, status_component

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run_server(host='0.0.0.0', port=port)