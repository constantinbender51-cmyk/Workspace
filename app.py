import ccxt
import pandas as pd
import datetime
import time
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Fetch Data from Binance ---
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', since_year=2018):
    print(f"Fetching {symbol} OHLCV data from {since_year}...")
    exchange = ccxt.binance()
    
    # Binance requires timestamp in milliseconds
    since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
    
    all_ohlcv = []
    limit = 1000  # Binance limit per request
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1 timeframe duration to fetch next batch
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            
            # Rate limit sleep (good practice)
            time.sleep(exchange.rateLimit / 1000)
            
            # Break if we reached current time (roughly)
            if last_timestamp >= exchange.milliseconds():
                break
                
            print(f"Fetched up to {datetime.datetime.fromtimestamp(last_timestamp/1000)}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Data fetch complete. {len(df)} rows loaded.")
    return df

# Initialize Data
# (Fetching data on startup - this might take a few seconds)
df = fetch_binance_data()

# --- 2. Setup Dash Application ---
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Binance Price & SMA Distance Visualizer", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select SMA Window (x):"),
        dcc.Slider(
            id='sma-slider',
            min=1,
            max=360,
            step=1,
            value=200,  # Default value
            marks={i: str(i) for i in range(0, 361, 40)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px'}),

    dcc.Graph(id='price-graph', style={'height': '80vh'})
])

# --- 3. Callback for Interactivity ---
@app.callback(
    Output('price-graph', 'figure'),
    Input('sma-slider', 'value')
)
def update_graph(x):
    # Calculate SMA (smax) based on slider input
    dff = df.copy()
    dff['smax'] = dff['close'].rolling(window=x).mean()
    
    # Calculate Distance: (Price - smax) / smax
    dff['distance'] = (dff['close'] - dff['smax']) / dff['smax']

    # Create Subplots: Row 1 = Price, Row 2 = Distance
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"BTC/USDT Price vs SMA({x})", f"Distance: (Price - SMA{x}) / SMA{x}")
    )

    # Plot Price
    fig.add_trace(go.Scatter(
        x=dff.index, y=dff['close'], name='Close Price', line=dict(color='blue')
    ), row=1, col=1)

    # Plot SMA
    fig.add_trace(go.Scatter(
        x=dff.index, y=dff['smax'], name=f'SMA({x})', line=dict(color='orange')
    ), row=1, col=1)

    # Plot Distance
    fig.add_trace(go.Scatter(
        x=dff.index, y=dff['distance'], name='Distance', 
        line=dict(color='purple'), fill='tozeroy'
    ), row=2, col=1)
    
    # Add a zero line for distance
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Update Layout
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

# --- 4. Run Server ---
if __name__ == '__main__':
    # Starts server on port 8080
    app.run_server(debug=True, port=8080)
