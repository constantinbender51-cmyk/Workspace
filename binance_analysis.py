import gdown
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from datetime import timedelta
import os

# --- 1. Data Download & Loading ---
def get_data():
    file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = 'ohlcv_data.csv'

    if not os.path.exists(output_file):
        print("Downloading data from Google Drive...")
        try:
            gdown.download(url, output_file, quiet=False)
        except Exception as e:
            print(f"Download failed: {e}")
            return pd.DataFrame(), None
    else:
        print("File already exists. Using local copy.")

    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame(), None

    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next((col for col in df.columns if 'date' in col or 'time' in col), None)
    if not date_col:
        raise ValueError("Could not identify a date/timestamp column in the CSV.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    return df, date_col

# --- 2. Regression Calculation Logic ---
def calculate_regression(df, date_col, lookback_days):
    """Fits y = mx + b for the last 'lookback_days' of data."""
    if df.empty:
        return None, None, None

    max_date = df[date_col].max()
    cutoff_date = max_date - timedelta(days=lookback_days)
    subset = df.loc[df[date_col] >= cutoff_date].copy()

    if subset.empty:
        return subset, None, None

    # Calculate X in minutes (Unix epoch seconds // 60)
    subset['x_minutes'] = subset[date_col].astype('int64') // 10**9 // 60
    price_col = 'close' if 'close' in subset.columns else subset.columns[1]
    
    X = subset['x_minutes'].values
    Y = subset[price_col].values

    # Formula: m = sum((x - avg_x)(y - avg_y)) / sum((x - avg_x)^2)
    avg_x, avg_y = np.mean(X), np.mean(Y)
    num = np.sum((X - avg_x) * (Y - avg_y))
    den = np.sum((X - avg_x) ** 2)
    
    m = num / den if den != 0 else 0
    b = avg_y - (m * avg_x)
    
    subset['y_pred'] = (m * X) + b
    
    stats = {
        'm': m,
        'b': b,
        'points': len(subset),
        'start': subset[date_col].min(),
        'end': subset[date_col].max(),
        'price_col': price_col
    }
    
    return subset, stats, date_col

# --- 3. Visualization Server ---
def start_server(full_df, date_column_name):
    app = dash.Dash(__name__, title="Dynamic OHLCV Regression")

    app.layout = html.Div(style={'backgroundColor': '#111', 'color': 'white', 'padding': '20px', 'fontFamily': 'Segoe UI, Arial'}, children=[
        html.H1("Dynamic Market Regression", style={'textAlign': 'center', 'marginBottom': '30px'}),
        
        # Control Panel
        html.Div([
            html.Label("Lookback Window (Days):", style={'fontSize': '18px', 'fontWeight': 'bold'}),
            dcc.Slider(
                id='day-slider',
                min=1,
                max=30,
                step=1,
                value=30,
                marks={i: f'{i}d' for i in [1, 5, 10, 15, 20, 25, 30]},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'padding': '20px', 'backgroundColor': '#222', 'borderRadius': '10px', 'marginBottom': '20px'}),

        # Stats Display
        html.Div(id='stats-container', style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
        
        # Graph
        dcc.Graph(id='regression-graph', style={'height': '70vh'}),
        
        dcc.Store(id='full-data-store')
    ])

    @app.callback(
        [Output('regression-graph', 'figure'),
         Output('stats-container', 'children')],
        [Input('day-slider', 'value')]
    )
    def update_graph(lookback_days):
        subset, stats, date_col = calculate_regression(full_df, date_column_name, lookback_days)
        
        if subset.empty or stats is None:
            return go.Figure(), html.Div("No data for this range.")

        # Create Figure
        fig = go.Figure()

        # Add OHLC/Price
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in subset.columns for col in ohlc_cols):
            fig.add_trace(go.Candlestick(
                x=subset[date_col],
                open=subset['open'], high=subset['high'],
                low=subset['low'], close=subset['close'],
                name='Market Data', opacity=0.4
            ))
        else:
            fig.add_trace(go.Scatter(x=subset[date_col], y=subset[stats['price_col']], name='Price', line=dict(color='#00d2ff')))

        # Add Regression Line
        fig.add_trace(go.Scatter(
            x=subset[date_col], y=subset['y_pred'],
            mode='lines', name=f'Fit ({lookback_days}d)',
            line=dict(color='#ffcc00', width=3)
        ))

        fig.update_layout(
            template='plotly_dark',
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_rangeslider_visible=False,
            title=f"Linear Regression: y = {stats['m']:.6f}x + {stats['b']:.2f}",
            hovermode='x unified'
        )

        # Stats Cards
        stats_html = [
            html.Div([html.Small("Slope (m)"), html.H3(f"{stats['m']:.8f}")], style={'flex': 1, 'padding': '10px', 'border': '1px solid #444'}),
            html.Div([html.Small("Intercept (b)"), html.H3(f"{stats['b']:.2f}")], style={'flex': 1, 'padding': '10px', 'border': '1px solid #444'}),
            html.Div([html.Small("Points"), html.H3(f"{stats['points']}")], style={'flex': 1, 'padding': '10px', 'border': '1px solid #444'}),
            html.Div([html.Small("Start Date"), html.H3(f"{stats['start'].strftime('%Y-%m-%d')}")], style={'flex': 1, 'padding': '10px', 'border': '1px solid #444'})
        ]

        return fig, stats_html

    print("\nStarting Web Server on http://0.0.0.0:8050")
    app.run(host='0.0.0.0', port=8050, debug=False)

if __name__ == '__main__':
    df, date_col = get_data()
    if not df.empty:
        start_server(df, date_col)