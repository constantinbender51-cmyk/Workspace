import gdown
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objects as go
from datetime import timedelta
import os

# --- 1. Data Download & Loading ---
def get_data():
    file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = 'ohlcv_data.csv'

    # Download if not exists (or force download if needed)
    if not os.path.exists(output_file):
        print("Downloading data from Google Drive...")
        gdown.download(url, output_file, quiet=False)
    else:
        print("File already exists. Using local copy.")

    # Load data
    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()

    # Normalize column names to lowercase to handle variations like "Date", "date", "Close", "close"
    df.columns = [c.lower() for c in df.columns]
    
    # Identify timestamp column
    date_col = next((col for col in df.columns if 'date' in col or 'time' in col), None)
    if not date_col:
        raise ValueError("Could not identify a date/timestamp column.")

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date just in case
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    return df, date_col

# --- 2. Data Processing & Regression ---
def process_last_30_days(df, date_col):
    if df.empty:
        return df, None, None, None

    # Determine the "last 30 days" based on the dataset's latest entry
    max_date = df[date_col].max()
    cutoff_date = max_date - timedelta(days=30)
    
    # Filter data
    mask = df[date_col] > cutoff_date
    subset = df.loc[mask].copy()

    if subset.empty:
        print("No data found in the last 30 days window.")
        return subset, None, None, None

    # Prepare X (minutes/time) and Y (price)
    # We use timestamp (seconds) / 60 to get "minutes" as requested
    subset['x_minutes'] = subset[date_col].astype('int64') // 10**9 // 60
    
    # Use 'close' price for Y, fallback to whatever is available if not
    price_col = 'close' if 'close' in subset.columns else subset.columns[1]
    
    X = subset['x_minutes'].values
    Y = subset[price_col].values

    # --- Manual Linear Regression Implementation ---
    # Formula: m = sum((x - x_avg)(y - y_avg)) / sum((x - x_avg)^2)
    #          b = y_avg - m * x_avg
    
    x_avg = np.mean(X)
    y_avg = np.mean(Y)
    
    numerator = np.sum((X - x_avg) * (Y - y_avg))
    denominator = np.sum((X - x_avg) ** 2)
    
    if denominator == 0:
        m = 0
    else:
        m = numerator / denominator
        
    b = y_avg - (m * x_avg)
    
    # Calculate regression line for plotting
    subset['y_pred'] = (m * X) + b
    
    stats = {
        'slope': m,
        'intercept': b,
        'x_avg': x_avg,
        'y_avg': y_avg
    }
    
    return subset, stats, date_col, price_col

# --- 3. Visualization Server ---
def start_server(df_30d, stats, date_col, price_col):
    app = dash.Dash(__name__, title="OHLCV Regression")

    # Create the figure
    fig = go.Figure()

    # Add OHLC(V) Candlesticks
    # Check for standard ohlc columns
    if all(col in df_30d.columns for col in ['open', 'high', 'low', 'close']):
        fig.add_trace(go.Candlestick(
            x=df_30d[date_col],
            open=df_30d['open'],
            high=df_30d['high'],
            low=df_30d['low'],
            close=df_30d['close'],
            name='OHLC Data'
        ))
    else:
        # Fallback to simple line if OHLC not fully present
        fig.add_trace(go.Scatter(x=df_30d[date_col], y=df_30d[price_col], mode='lines', name='Price'))

    # Add Regression Line
    fig.add_trace(go.Scatter(
        x=df_30d[date_col], 
        y=df_30d['y_pred'],
        mode='lines',
        name='Linear Fit (Last 30 Days)',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Formatting equation string
    eq_str = f"y = {stats['slope']:.5f}x + {stats['intercept']:.2f}"

    fig.update_layout(
        title=f"30-Day Price Trend Analysis<br><sub>Fit: {eq_str}</sub>",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark",
        height=800
    )

    # App Layout
    app.layout = html.Div([
        html.H1("Market Data Regression Analysis", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
        html.Div([
            html.P(f"Dataset Range: {df_30d[date_col].min()} to {df_30d[date_col].max()}"),
            html.P(f"Regression Slope (m): {stats['slope']}"),
            html.P(f"Regression Intercept (b): {stats['intercept']}"),
        ], style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'margin': '20px'}),
        dcc.Graph(figure=fig)
    ])

    print("\nStarting Dash Server...")
    app.run_server(debug=True, port=8050)

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Get Data
    full_df, date_column_name = get_data()
    
    # 2. Process
    df_30, regression_stats, date_col, price_col = process_last_30_days(full_df, date_column_name)
    
    if df_30 is not None and not df_30.empty:
        # 3. Visualize
        start_server(df_30, regression_stats, date_col, price_col)
    else:
        print("Could not process data for visualization.")