import requests
import pandas as pd
import datetime
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. Data Fetching Functions
# -----------------------------------------------------------------------------

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1d", start_date="2018-01-01"):
    """
    Fetches historical OHLCV data from Binance API (Free, Public).
    Handles pagination since Binance limits to 1000 candles per request.
    """
    print(f"Fetching Binance data for {symbol}...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start_date to milliseconds timestamp
    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.datetime.now().timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data or isinstance(data, dict) and 'code' in data:
                print("Error or end of data reached.")
                break
                
            all_data.extend(data)
            
            # Update start_ts to the last timestamp + 1ms to avoid duplicates
            last_close_time = data[-1][6]
            start_ts = last_close_time + 1
            
            # Respect API rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            break
            
    # Process data into DataFrame
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume", 
        "Close Time", "Quote Asset Vol", "Number of Trades", 
        "Taker buy base vol", "Taker buy quote vol", "Ignore"
    ])
    
    # Convert types
    df["Date"] = pd.to_datetime(df["Open Time"], unit='ms')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
        
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

def fetch_blockchain_data():
    """
    Fetches 'estimated-transaction-volume-usd' from Blockchain.com (Free, Public).
    Note: Exact filtering for >$100k txs is a premium feature on most APIs.
    This metric represents the total estimated on-chain transaction volume, 
    which is the best free proxy for whale activity.
    """
    print("Fetching Blockchain.com data...")
    # 'timespan=8years' covers 2018 to present
    url = "https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=8years&format=json"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "values" not in data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data["values"])
        df["Date"] = pd.to_datetime(df["x"], unit='s')
        df["OnChainVolumeUSD"] = df["y"]
        
        return df[["Date", "OnChainVolumeUSD"]]
        
    except Exception as e:
        print(f"Error fetching Blockchain.com data: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. Data Processing & Merging
# -----------------------------------------------------------------------------

def get_combined_data():
    # 1. Get Market Price (Binance)
    df_binance = fetch_binance_ohlcv()
    
    # 2. Get On-Chain Volume (Blockchain.com)
    df_chain = fetch_blockchain_data()
    
    if df_binance.empty or df_chain.empty:
        return pd.DataFrame()
    
    # Normalize dates to midnight for merging
    df_binance["Date"] = df_binance["Date"].dt.normalize()
    df_chain["Date"] = df_chain["Date"].dt.normalize()
    
    # Merge
    df_merged = pd.merge(df_binance, df_chain, on="Date", how="inner")
    df_merged = df_merged.sort_values("Date")
    
    return df_merged

# -----------------------------------------------------------------------------
# 3. Web Server & Dashboard (Dash)
# -----------------------------------------------------------------------------

# Initialize data (fetch once on startup)
df = get_combined_data()

app = dash.Dash(__name__, title="Crypto Whale Watch")
server = app.server

# Layout
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'fontFamily': 'sans-serif', 'padding': '20px'}, children=[
    
    html.H1("Bitcoin Market vs. On-Chain Volume", style={'textAlign': 'center'}),
    
    html.Div([
        html.P("Comparing Binance Price Action with Blockchain.com On-Chain Transaction Volume."),
        html.P("Note: Specifically filtering for individual transactions >$100k requires paid institutional APIs. "
               "The chart below uses Total Estimated On-Chain Transaction Volume (USD) as a proxy for high-value network throughput.",
               style={'color': '#888888', 'fontSize': '0.9em'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Graph(id='crypto-chart', style={'height': '80vh'}),
    
    html.Div(
        html.Button('Refresh Data', id='refresh-btn', n_clicks=0, 
                    style={'backgroundColor': '#00CC96', 'border': 'none', 'color': 'white', 'padding': '10px 20px', 'cursor': 'pointer'}),
        style={'textAlign': 'center', 'marginTop': '20px'}
    )
])

@app.callback(
    Output('crypto-chart', 'figure'),
    [Input('refresh-btn', 'n_clicks')]
)
def update_graph(n_clicks):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: On-Chain Volume (Bar Chart) - Represents "Large Transactions" proxy
    fig.add_trace(
        go.Bar(
            x=df['Date'], 
            y=df['OnChainVolumeUSD'], 
            name="On-Chain Tx Volume (USD)",
            marker_color='#00CC96',
            opacity=0.4
        ),
        secondary_y=False,
    )

    # Trace 2: Binance Close Price (Line Chart)
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Close'], 
            name="BTC Price (Binance)",
            line=dict(color='#EF553B', width=2)
        ),
        secondary_y=True,
    )

    # Layout Customization
    fig.update_layout(
        template="plotly_dark",
        title_text="Daily On-Chain Volume vs Price (2018 - Present)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            showgrid=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="#333",
                activecolor="#00CC96"
            )
        )
    )

    # Axis Labels
    fig.update_yaxes(title_text="On-Chain Volume (USD)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Price (USDT)", secondary_y=True, showgrid=True, gridcolor='#333333')

    return fig

if __name__ == '__main__':
    print("Starting server...")
    # debug=True allows hot-reloading during development
    app.run_server(debug=True, port=8050)
