import ccxt
import pandas as pd
import numpy as np
import datetime
import time
from dash import Dash, dcc, html
import plotly.graph_objects as go

# --- 1. Fetch Data from Binance ---
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', since_year=2018):
    print(f"Fetching {symbol} OHLCV data from {since_year}...")
    exchange = ccxt.binance()
    
    since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
    all_ohlcv = []
    limit = 1000 
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            time.sleep(exchange.rateLimit / 1000)
            if last_timestamp >= exchange.milliseconds():
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"Data fetch complete. {len(df)} rows loaded.")
    return df

# --- 2. Calculate Stability Ratios ---
def calculate_stability_matrix(df):
    results = []
    
    # Pre-calculate price series for speed
    price = df['close']
    
    print("Calculating stability metrics for SMAs 1-360...")
    
    for x in range(1, 361):
        # 1. Calculate SMA
        sma = price.rolling(window=x).mean()
        
        # We need to align data to avoid NaN issues at the start
        valid_idx = sma.first_valid_index()
        if valid_idx is None:
            continue
            
        s_slice = sma.loc[valid_idx:]
        p_slice = price.loc[valid_idx:]
        
        # 2. Count Crosses (Price vs SMA)
        # Logic: (Price > SMA) state flips
        price_above_sma = p_slice > s_slice
        # diff() gives True where state changes. sum() counts the Trues.
        crosses = price_above_sma.astype(int).diff().abs().sum()
        
        # 3. Count Direction Changes (SMA Slope)
        # Logic: (SMA_current > SMA_prev) state flips
        sma_diff = s_slice.diff()
        # We look for sign changes in the diff
        # np.sign(0) is 0, so we handle strict positive/negative changes
        sma_slope_pos = sma_diff > 0
        turns = sma_slope_pos.astype(int).diff().abs().sum()
        
        # 4. Calculate Ratio
        # Avoid division by zero if SMA never turns (e.g. perfect straight line)
        if turns == 0:
            ratio = np.nan 
        else:
            ratio = crosses / turns
            
        results.append({
            'sma': x,
            'crosses': crosses,
            'turns': turns,
            'ratio': ratio
        })
        
    return pd.DataFrame(results)

# Initialize Data & Calculations
df = fetch_binance_data()
stats_df = calculate_stability_matrix(df)

# --- 3. Setup Dash Application ---
app = Dash(__name__)

# Find the "best" SMA (lowest ratio)
best_sma = stats_df.loc[stats_df['ratio'].idxmin()]

app.layout = html.Div([
    html.H1("SMA Stability Analysis: Crosses vs. Turns", style={'textAlign': 'center'}),
    
    html.Div([
        html.P("Lower Ratio = Higher Stability (Good Support/Resistance)."),
        html.P("High Ratio = Noise (Price crosses often regardless of SMA direction)."),
        html.B(f"Most Stable SMA: {int(best_sma['sma'])} (Ratio: {best_sma['ratio']:.4f})")
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Graph(
        id='stability-graph',
        style={'height': '80vh'},
        figure={
            'data': [
                go.Scatter(
                    x=stats_df['sma'],
                    y=stats_df['ratio'],
                    mode='lines',
                    name='Stability Ratio',
                    line=dict(color='#00CC96', width=2)
                )
            ],
            'layout': go.Layout(
                title='Stability Ratio (Price Crosses / SMA Turns) per Window Size',
                xaxis={'title': 'SMA Window Size (1-360)'},
                yaxis={'title': 'Ratio (Crosses / Turns)'},
                template='plotly_dark',
                hovermode='x unified',
                annotations=[
                    dict(
                        x=best_sma['sma'],
                        y=best_sma['ratio'],
                        xref="x",
                        yref="y",
                        text=f"Best: SMA {int(best_sma['sma'])}",
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=-40
                    )
                ]
            )
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
