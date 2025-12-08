import ccxt
import pandas as pd
import time
from dash import Dash, dcc, html
import plotly.graph_objects as go

# --- 1. Fetch Data ---
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', since_year=2018):
    print(f"Fetching {symbol} OHLCV data from {since_year}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
    all_ohlcv = []
    limit = 1000 
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            time.sleep(exchange.rateLimit / 1000)
            if last_timestamp >= exchange.milliseconds(): break
        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Loaded {len(df)} rows.")
    return df

df = fetch_binance_data()

# --- 2. Generate Plotly Figure ---
# We use Scattergl because we are plotting ~150 lines with 2000+ points each. 
# Standard SVG plotting would crash the browser.

fig = go.Figure()

print("Generating ribbons...")

# Helper to add a ribbon of SMAs
def add_ribbon(df, start, end, color, name_prefix):
    # We add the traces in reverse so the smaller SMAs (closer to price) are on top if needed, 
    # though with opacity it matters less.
    for x in range(start, end + 1):
        sma = df['close'].rolling(window=x).mean()
        fig.add_trace(go.Scattergl(
            x=df.index, 
            y=sma, 
            mode='lines',
            line=dict(color=color, width=1),
            opacity=0.3, # Low opacity creates the "Cloud" effect where lines overlap
            name=f'{name_prefix} SMA {x}',
            hoverinfo='skip', # Disable hover for individual ribbon lines to save performance
            showlegend=False
        ))
    # Add a dummy trace for the legend
    fig.add_trace(go.Scattergl(
        x=[df.index[0]], y=[df['close'].iloc[0]], 
        mode='lines', line=dict(color=color, width=4), name=f'{name_prefix} Cluster ({start}-{end})'
    ))

# 1. Unstable Ribbon (Red) - 241 to 315
add_ribbon(df, 241, 315, 'rgba(255, 50, 50, 0.4)', 'Unstable')

# 2. Stable Ribbon 1 (Green) - 109 to 150
add_ribbon(df, 109, 150, 'rgba(0, 255, 100, 0.4)', 'Stable 1')

# 3. Stable Ribbon 2 (Blue) - 320 to 344
add_ribbon(df, 320, 344, 'rgba(50, 100, 255, 0.4)', 'Stable 2')

# 4. Price (White/Bold)
fig.add_trace(go.Scattergl(
    x=df.index, 
    y=df['close'], 
    mode='lines',
    line=dict(color='white', width=2),
    name='BTC Price'
))

fig.update_layout(
    title="SMA Ribbons: Stable Zones vs Unstable Zones",
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price (USDT)",
    hovermode="x unified",
    height=800,
    margin=dict(l=50, r=50, t=50, b=50),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)"
    )
)

# --- 3. Run Server ---
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(figure=fig, style={'height': '95vh'})
])

if __name__ == '__main__':
    print("Server starting on port 8080...")
    app.run(debug=True, host='0.0.0.0', port=8080)
