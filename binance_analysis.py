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

    # Download if not exists
    if not os.path.exists(output_file):
        print("Downloading data from Google Drive...")
        try:
            gdown.download(url, output_file, quiet=False)
        except Exception as e:
            print(f"Download failed: {e}")
            return pd.DataFrame(), None
    else:
        print("File already exists. Using local copy.")

    # Load data
    try:
        # We use low_memory=False to handle large datasets effectively
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame(), None

    # Normalize column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Identify timestamp column (common names: 'time', 'date', 'timestamp')
    date_col = next((col for col in df.columns if 'date' in col or 'time' in col), None)
    if not date_col:
        raise ValueError("Could not identify a date/timestamp column in the CSV.")

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date for time-series integrity
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

    # Prepare X (minutes) and Y (price)
    # We convert timestamps to unix epoch (seconds) and divide by 60 for minutes
    subset['x_minutes'] = subset[date_col].astype('int64') // 10**9 // 60
    
    # Use 'close' price for Y, fallback to the second column if 'close' isn't found
    price_col = 'close' if 'close' in subset.columns else subset.columns[1]
    
    X = subset['x_minutes'].values
    Y = subset[price_col].values

    # --- Manual Linear Regression Implementation ---
    # m = sum((x - avg_x)(y - avg_y)) / sum((x - avg_x)^2)
    # b = avg_y - m(avg_x)
    
    avg_x = np.mean(X)
    avg_y = np.mean(Y)
    
    numerator = np.sum((X - avg_x) * (Y - avg_y))
    denominator = np.sum((X - avg_x) ** 2)
    
    # Guard against division by zero (e.g., if all x values are the same)
    m = numerator / denominator if denominator != 0 else 0
    b = avg_y - (m * avg_x)
    
    # Calculate regression line values
    subset['y_pred'] = (m * X) + b
    
    stats = {
        'slope': m,
        'intercept': b,
        'avg_x': avg_x,
        'avg_y': avg_y
    }
    
    return subset, stats, date_col, price_col

# --- 3. Visualization Server ---
def start_server(df_30d, stats, date_col, price_col):
    # Initialize Dash App
    app = dash.Dash(__name__, title="OHLCV Regression")

    # Create Plotly figure
    fig = go.Figure()

    # 1. Add Price Data (Candlesticks if possible, else line)
    ohlc_cols = ['open', 'high', 'low', 'close']
    if all(col in df_30d.columns for col in ohlc_cols):
        fig.add_trace(go.Candlestick(
            x=df_30d[date_col],
            open=df_30d['open'],
            high=df_30d['high'],
            low=df_30d['low'],
            close=df_30d['close'],
            name='OHLC Data',
            opacity=0.6
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df_30d[date_col], 
            y=df_30d[price_col], 
            mode='lines', 
            name='Close Price',
            line=dict(color='cyan', width=1)
        ))

    # 2. Add Regression Line
    fig.add_trace(go.Scatter(
        x=df_30d[date_col], 
        y=df_30d['y_pred'],
        mode='lines',
        name='Linear Regression (Last 30 Days)',
        line=dict(color='#FFD700', width=3, dash='solid')
    ))

    # Regression formula display
    equation_text = f"Equation: y = {stats['slope']:.6f}x + {stats['intercept']:.2f}"

    fig.update_layout(
        title=dict(
            text=f"Market Trend Analysis (Last 30 Days)<br><sup>{equation_text}</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False
    )

    # Layout Definition
    app.layout = html.Div(style={'backgroundColor': '#111', 'color': 'white', 'padding': '20px', 'fontFamily': 'Arial'}, children=[
        html.H1("OHLCV Linear Regression Dashboard", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H4("Analysis Stats"),
                html.P(f"Data Points: {len(df_30d)}"),
                html.P(f"Slope (m): {stats['slope']:.8f}"),
                html.P(f"Intercept (b): {stats['intercept']:.4f}"),
            ], style={'flex': '1', 'padding': '15px', 'border': '1px solid #444', 'marginRight': '10px'}),
            
            html.Div([
                html.H4("Timeframe"),
                html.P(f"From: {df_30d[date_col].min()}"),
                html.P(f"To: {df_30d[date_col].max()}"),
            ], style={'flex': '1', 'padding': '15px', 'border': '1px solid #444', 'marginLeft': '10px'}),
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        dcc.Graph(figure=fig)
    ])

    print("\nStarting Web Server at http://127.0.0.1:8050/")
    # Replaced run_server with run as per the latest Dash versions
    app.run(debug=True, port=8050)

# --- Execution Entry Point ---
if __name__ == '__main__':
    # 1. Download and Parse
    full_df, date_column_name = get_data()
    
    if full_df is not None and not full_df.empty:
        # 2. Extract and Fit
        df_30, regression_stats, date_col, price_col = process_last_30_days(full_df, date_column_name)
        
        if df_30 is not None and not df_30.empty:
            # 3. Serve Visualization
            start_server(df_30, regression_stats, date_col, price_col)
        else:
            print("Processing failed. No data found for the specified period.")
    else:
        print("Data retrieval failed.")